from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from backend.models.pinn_model import FEATURE_COLUMNS, STATE_COLUMNS, TyphoonPINN

EARTH_ANGULAR_VELOCITY = 7.2921159e-5
EARTH_RADIUS_M = 6_371_000.0
SECONDS_PER_HOUR = 3600.0

DEFAULT_OUTPUT_PATH = (
    Path(__file__).resolve().parent.parent / "backend" / "models" / "weights" / "typhoon_pinn_v1.pth"
)


def load_dataset(dataset_path: str | Path | None = None) -> pd.DataFrame:
    """Load typhoon samples with lng, lat, time, wind speed and pressure columns."""
    if dataset_path is None:
        return _build_synthetic_dataset()

    path = Path(dataset_path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".json", ".jsonl"}:
        return pd.read_json(path)

    raise ValueError("Dataset must be a .csv, .json, or .jsonl file.")


def _build_synthetic_dataset() -> pd.DataFrame:
    """Small fallback dataset for smoke testing the training pipeline."""
    rows = []
    rng = np.random.default_rng(42)
    for storm_index in range(16):
        lng = 128.0 - storm_index * 0.35
        lat = 15.0 + storm_index * 0.45
        wind = 24.0 + storm_index * 0.8
        pressure = 995.0 - storm_index * 1.2
        for step in range(18):
            rows.append(
                {
                    "storm_id": f"synthetic-{storm_index:02d}",
                    "t_hours": step * 6.0,
                    "lng": lng - 0.22 * step + rng.normal(0.0, 0.015),
                    "lat": lat + 0.31 * step + rng.normal(0.0, 0.015),
                    "wind_speed": wind + 0.35 * step - 0.02 * step**2,
                    "pressure": pressure - 0.55 * step + 0.04 * step**2,
                }
            )
    return pd.DataFrame(rows)


def _prepare_dataframe(raw: pd.DataFrame) -> pd.DataFrame:
    frame = raw.copy()
    if "storm_id" not in frame.columns:
        frame["storm_id"] = "storm-0"
    if "t_hours" not in frame.columns:
        if "timestamp" not in frame.columns:
            raise ValueError("Dataset must include either `t_hours` or `timestamp`.")
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        frame["t_hours"] = frame.groupby("storm_id")["timestamp"].transform(
            lambda values: (values - values.min()).dt.total_seconds() / SECONDS_PER_HOUR
        )

    for column in ["lng", "lat", "wind_speed", "pressure"]:
        if column not in frame.columns:
            raise ValueError(f"Dataset is missing required column `{column}`.")
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame["t_hours"] = pd.to_numeric(frame["t_hours"], errors="coerce")
    frame = frame.dropna(subset=FEATURE_COLUMNS)
    frame = frame.sort_values(["storm_id", "t_hours"]).reset_index(drop=True)
    if frame.empty:
        raise ValueError("Dataset is empty after preprocessing.")
    return frame


@dataclass
class TensorScaler:
    ranges: dict[str, tuple[float, float]]

    @classmethod
    def fit(cls, frame: pd.DataFrame, columns: Iterable[str]) -> "TensorScaler":
        ranges = {}
        for column in columns:
            low = float(frame[column].min())
            high = float(frame[column].max())
            if np.isclose(low, high):
                high = low + 1.0
            ranges[column] = (low, high)
        return cls(ranges)

    def transform_array(self, values: np.ndarray, columns: list[str]) -> np.ndarray:
        transformed = np.empty_like(values, dtype=np.float32)
        for index, column in enumerate(columns):
            low, high = self.ranges[column]
            transformed[:, index] = 2.0 * (values[:, index] - low) / max(high - low, 1e-6) - 1.0
        return transformed

    def denormalize_tensor(self, values: torch.Tensor, columns: list[str]) -> torch.Tensor:
        lows = torch.tensor([self.ranges[column][0] for column in columns], dtype=values.dtype, device=values.device)
        highs = torch.tensor([self.ranges[column][1] for column in columns], dtype=values.dtype, device=values.device)
        return (values + 1.0) * 0.5 * (highs - lows) + lows

    def to_dict(self) -> dict[str, tuple[float, float]]:
        return {key: (float(value[0]), float(value[1])) for key, value in self.ranges.items()}


class TyphoonSequenceDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, scaler: TensorScaler, sequence_length: int) -> None:
        self.samples = []
        for _, group in frame.groupby("storm_id", sort=False):
            if len(group) <= sequence_length:
                continue
            raw_values = group[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
            normalized = scaler.transform_array(raw_values, FEATURE_COLUMNS)
            for index in range(sequence_length, len(group)):
                previous_index = max(index - 2, 0)
                self.samples.append(
                    {
                        "features": normalized[index - sequence_length : index].reshape(-1),
                        "target_state": normalized[index, 1:5],
                        "last_state_raw": raw_values[index - 1, 1:5],
                        "previous_state_raw": raw_values[previous_index, 1:5],
                        "dt_hours": max(raw_values[index, 0] - raw_values[index - 1, 0], 1e-3),
                        "previous_dt_hours": max(raw_values[index - 1, 0] - raw_values[previous_index, 0], 1e-3),
                    }
                )
        if not self.samples:
            raise ValueError("Not enough sequential samples. Add more points per storm or reduce sequence length.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[index]
        return {
            "features": torch.tensor(sample["features"], dtype=torch.float32),
            "target_state": torch.tensor(sample["target_state"], dtype=torch.float32),
            "last_state_raw": torch.tensor(sample["last_state_raw"], dtype=torch.float32),
            "previous_state_raw": torch.tensor(sample["previous_state_raw"], dtype=torch.float32),
            "dt_hours": torch.tensor(sample["dt_hours"], dtype=torch.float32),
            "previous_dt_hours": torch.tensor(sample["previous_dt_hours"], dtype=torch.float32),
        }


def _latlon_velocity_mps_torch(
    lng0: torch.Tensor,
    lat0: torch.Tensor,
    lng1: torch.Tensor,
    lat1: torch.Tensor,
    dt_hours: torch.Tensor,
) -> torch.Tensor:
    dt_seconds = torch.clamp(dt_hours * SECONDS_PER_HOUR, min=1e-6)
    lat_ref = torch.deg2rad((lat0 + lat1) * 0.5)
    dx = EARTH_RADIUS_M * torch.cos(lat_ref) * torch.deg2rad(lng1 - lng0)
    dy = EARTH_RADIUS_M * torch.deg2rad(lat1 - lat0)
    return torch.stack([dx / dt_seconds, dy / dt_seconds], dim=1)


def _coastline_lng_torch(lat: torch.Tensor) -> torch.Tensor:
    return 120.35 + 0.19 * (lat - 26.0) + 0.08 * torch.sin((lat - 26.0) * 1.6)


class PINNLoss(nn.Module):
    """Combined data and physics-informed losses for typhoon motion."""

    def __init__(
        self,
        scaler: TensorScaler,
        velocity_weight: float = 1e-3,
        inertia_weight: float = 1e4,
        coriolis_weight: float = 1e4,
        wind_pressure_weight: float = 0.05,
        nearshore_weight: float = 0.02,
    ) -> None:
        super().__init__()
        self.scaler = scaler
        self.velocity_weight = velocity_weight
        self.inertia_weight = inertia_weight
        self.coriolis_weight = coriolis_weight
        self.wind_pressure_weight = wind_pressure_weight
        self.nearshore_weight = nearshore_weight
        self.mse = nn.MSELoss()

    def forward(self, prediction: torch.Tensor, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        predicted_state = prediction[:, :4]
        predicted_velocity = prediction[:, 4:6]
        target_state = batch["target_state"]

        # Data loss forces the PINN to fit observed typhoon position and intensity.
        data_loss = self.mse(predicted_state, target_state)

        predicted_raw = self.scaler.denormalize_tensor(predicted_state, STATE_COLUMNS)
        last_raw = batch["last_state_raw"]
        previous_raw = batch["previous_state_raw"]
        dt_hours = batch["dt_hours"]
        previous_dt_hours = batch["previous_dt_hours"]

        velocity_from_position = _latlon_velocity_mps_torch(
            last_raw[:, 0],
            last_raw[:, 1],
            predicted_raw[:, 0],
            predicted_raw[:, 1],
            dt_hours,
        )
        previous_velocity = _latlon_velocity_mps_torch(
            previous_raw[:, 0],
            previous_raw[:, 1],
            last_raw[:, 0],
            last_raw[:, 1],
            previous_dt_hours,
        )

        # Velocity consistency encodes dx/dt = v, linking position derivatives to predicted velocity.
        velocity_loss = self.mse(predicted_velocity, velocity_from_position)

        dt_seconds = torch.clamp(dt_hours * SECONDS_PER_HOUR, min=1e-6).unsqueeze(1)
        acceleration = (predicted_velocity - previous_velocity) / dt_seconds

        # Inertia loss penalizes abrupt acceleration, reducing nonphysical sharp track reversals.
        inertia_loss = torch.mean(acceleration.pow(2))

        f_value = 2.0 * EARTH_ANGULAR_VELOCITY * torch.sin(torch.deg2rad(predicted_raw[:, 1]))
        coriolis_u = acceleration[:, 0] - f_value * predicted_velocity[:, 1]
        coriolis_v = acceleration[:, 1] + f_value * predicted_velocity[:, 0]

        # Coriolis loss is a weak large-scale balance prior for rotating Earth motion.
        coriolis_loss = torch.mean(coriolis_u.pow(2) + coriolis_v.pow(2))

        wind_tendency = (predicted_raw[:, 2] - last_raw[:, 2]) / torch.clamp(dt_hours, min=1e-6)
        pressure_tendency = (predicted_raw[:, 3] - last_raw[:, 3]) / torch.clamp(dt_hours, min=1e-6)

        # Wind-pressure loss encodes the empirical typhoon relation: pressure rises usually weaken wind.
        wind_pressure_loss = torch.mean(torch.relu(wind_tendency * pressure_tendency).pow(2))

        coast_lng = _coastline_lng_torch(predicted_raw[:, 1])
        coast_distance = predicted_raw[:, 0] - coast_lng
        near_coast = torch.clamp((0.45 - coast_distance) / 0.45, min=0.0, max=1.0)

        # Nearshore decay loss discourages strengthening after the storm approaches land.
        nearshore_decay_loss = torch.mean(
            near_coast
            * (
                torch.relu(predicted_raw[:, 2] - last_raw[:, 2]).pow(2)
                + torch.relu(last_raw[:, 3] - predicted_raw[:, 3]).pow(2)
            )
        )

        total_loss = (
            data_loss
            + self.velocity_weight * velocity_loss
            + self.inertia_weight * inertia_loss
            + self.coriolis_weight * coriolis_loss
            + self.wind_pressure_weight * wind_pressure_loss
            + self.nearshore_weight * nearshore_decay_loss
        )
        return total_loss, {
            "data_loss": float(data_loss.detach().cpu()),
            "velocity_loss": float(velocity_loss.detach().cpu()),
            "inertia_loss": float(inertia_loss.detach().cpu()),
            "coriolis_loss": float(coriolis_loss.detach().cpu()),
            "wind_pressure_loss": float(wind_pressure_loss.detach().cpu()),
            "nearshore_decay_loss": float(nearshore_decay_loss.detach().cpu()),
        }


def train(
    dataset_path: str | Path | None = None,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    sequence_length: int = 4,
    hidden_dim: int = 128,
    epochs: int = 200,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
) -> Path:
    frame = _prepare_dataframe(load_dataset(dataset_path))
    scaler = TensorScaler.fit(frame, FEATURE_COLUMNS)
    dataset = TyphoonSequenceDataset(frame, scaler, sequence_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = TyphoonPINN(input_dim=sequence_length * len(FEATURE_COLUMNS), hidden_dim=hidden_dim)
    criterion = PINNLoss(scaler)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        parts = {
            "data_loss": 0.0,
            "velocity_loss": 0.0,
            "inertia_loss": 0.0,
            "coriolis_loss": 0.0,
            "wind_pressure_loss": 0.0,
            "nearshore_decay_loss": 0.0,
        }
        for batch in loader:
            optimizer.zero_grad()
            prediction = model(batch["features"])
            loss, loss_parts = criterion(prediction, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach().cpu())
            for key in parts:
                parts[key] += loss_parts[key]

        if epoch == 1 or epoch % 20 == 0 or epoch == epochs:
            count = max(len(loader), 1)
            print(
                f"epoch={epoch:04d} loss={epoch_loss / count:.6f} "
                f"data={parts['data_loss'] / count:.6f} velocity={parts['velocity_loss'] / count:.6f} "
                f"inertia={parts['inertia_loss'] / count:.8f} coriolis={parts['coriolis_loss'] / count:.8f} "
                f"wind_pressure={parts['wind_pressure_loss'] / count:.6f} nearshore={parts['nearshore_decay_loss'] / count:.6f}"
            )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "sequence_length": sequence_length,
            "input_dim": sequence_length * len(FEATURE_COLUMNS),
            "hidden_dim": hidden_dim,
            "feature_columns": FEATURE_COLUMNS,
            "state_columns": STATE_COLUMNS,
            "scaler": scaler.to_dict(),
        },
        output,
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the typhoon PINN model.")
    parser.add_argument("--dataset", type=str, default=None, help="Path to a CSV/JSON typhoon dataset.")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_PATH), help="Output .pth file path.")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sequence-length", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    output = train(
        dataset_path=args.dataset,
        output_path=args.output,
        sequence_length=args.sequence_length,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )
    print(f"Saved PINN weights to {output}")


if __name__ == "__main__":
    main()
