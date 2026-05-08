from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
except ModuleNotFoundError as error:  # pragma: no cover - depends on local environment
    raise SystemExit(
        "PyTorch is required for training. Install backend\\requirements.txt in your active Python environment first."
    ) from error

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from backend.models.pinn_model import FEATURE_COLUMNS, STATE_COLUMNS, TyphoonPINN
from backend.utils.physics_engine import EARTH_ANGULAR_VELOCITY, EARTH_RADIUS_M, SECONDS_PER_HOUR

LOSS_PART_NAMES = (
    "data_loss",
    "velocity_loss",
    "inertia_loss",
    "coriolis_loss",
    "wind_pressure_loss",
    "nearshore_decay_loss",
)
DEFAULT_OUTPUT_PATH = (
    Path(__file__).resolve().parent.parent / "backend" / "models" / "weights" / "typhoon_pinn_v1.pth"
)


def load_dataset(dataset_path: str | Path | None = None) -> pd.DataFrame:
    """Load typhoon samples with lng, lat, time, wind speed and pressure columns."""
    if dataset_path is None:
        return _build_synthetic_dataset()

    path = Path(dataset_path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, encoding="utf-8-sig")
    if path.suffix.lower() == ".json":
        return pd.read_json(path)
    if path.suffix.lower() == ".jsonl":
        return pd.read_json(path, lines=True)

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
    frame.columns = [str(column).strip().lstrip("\ufeff") for column in frame.columns]
    if "storm_id" not in frame.columns:
        frame["storm_id"] = "storm-0"
    frame["storm_id"] = frame["storm_id"].astype(str)
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
    frame = frame.drop_duplicates(subset=["storm_id", "t_hours"], keep="last")
    frame = frame.sort_values(["storm_id", "t_hours"]).reset_index(drop=True)
    if frame.empty:
        raise ValueError("Dataset is empty after preprocessing.")
    return frame


def _set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available in the active PyTorch installation.")
    return torch.device(device_name)


def _split_frame_by_storm(
    frame: pd.DataFrame,
    *,
    val_ratio: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    if val_ratio <= 0:
        train_ids = frame["storm_id"].drop_duplicates().tolist()
        empty = frame.iloc[0:0].copy()
        return frame.copy(), empty, train_ids, []

    storm_ids = frame["storm_id"].drop_duplicates().tolist()
    if len(storm_ids) < 2:
        train_ids = storm_ids
        empty = frame.iloc[0:0].copy()
        return frame.copy(), empty, train_ids, []

    shuffled_ids = storm_ids.copy()
    np.random.default_rng(seed).shuffle(shuffled_ids)
    val_count = int(round(len(shuffled_ids) * val_ratio))
    val_count = min(max(val_count, 1), len(shuffled_ids) - 1)
    val_ids = sorted(shuffled_ids[:val_count])
    val_id_set = set(val_ids)
    train_ids = sorted([storm_id for storm_id in shuffled_ids if storm_id not in val_id_set])
    train_frame = frame[frame["storm_id"].isin(train_ids)].copy()
    val_frame = frame[frame["storm_id"].isin(val_ids)].copy()
    return train_frame, val_frame, train_ids, val_ids


def _summarize_frame(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {
            "points": 0,
            "storms": 0,
            "min_points_per_storm": 0,
            "max_points_per_storm": 0,
            "avg_points_per_storm": 0.0,
            "time_step_hours": {},
            "feature_ranges": {},
        }

    group_sizes = frame.groupby("storm_id", sort=False).size()
    intervals: Counter[float] = Counter()
    for _, group in frame.groupby("storm_id", sort=False):
        hours = group["t_hours"].to_numpy(dtype=float)
        if len(hours) < 2:
            continue
        for delta in np.diff(hours):
            if delta > 0:
                intervals[round(float(delta), 3)] += 1

    feature_ranges = {}
    for column in FEATURE_COLUMNS:
        feature_ranges[column] = {
            "min": round(float(frame[column].min()), 6),
            "max": round(float(frame[column].max()), 6),
        }

    return {
        "points": int(len(frame)),
        "storms": int(group_sizes.size),
        "min_points_per_storm": int(group_sizes.min()),
        "max_points_per_storm": int(group_sizes.max()),
        "avg_points_per_storm": round(float(group_sizes.mean()), 2),
        "time_step_hours": {str(key): int(value) for key, value in sorted(intervals.items())},
        "feature_ranges": feature_ranges,
    }


def _derive_report_path(output_path: str | Path) -> Path:
    return Path(output_path).with_suffix(".summary.json")


def _build_checkpoint_payload(
    *,
    model: TyphoonPINN,
    scaler: "TensorScaler",
    sequence_length: int,
    hidden_dim: int,
    best_epoch: int,
    best_metric: float,
    history: list[dict[str, Any]],
    dataset_summary: dict[str, Any],
    training_config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "state_dict": model.state_dict(),
        "sequence_length": sequence_length,
        "input_dim": sequence_length * len(FEATURE_COLUMNS),
        "hidden_dim": hidden_dim,
        "feature_columns": FEATURE_COLUMNS,
        "state_columns": STATE_COLUMNS,
        "scaler": scaler.to_dict(),
        "best_epoch": best_epoch,
        "best_metric": best_metric,
        "history": history,
        "dataset_summary": dataset_summary,
        "training_config": training_config,
        "saved_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }


def _empty_loss_totals() -> dict[str, float]:
    return {name: 0.0 for name in LOSS_PART_NAMES}


def _move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


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
        self.storm_ids: list[str] = []
        for storm_id, group in frame.groupby("storm_id", sort=False):
            if len(group) <= sequence_length:
                continue
            self.storm_ids.append(str(storm_id))
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
        self.storm_count = len(self.storm_ids)
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


def _run_epoch(
    *,
    model: TyphoonPINN,
    loader: DataLoader,
    criterion: PINNLoss,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    grad_clip: float | None = None,
) -> dict[str, Any]:
    is_training = optimizer is not None
    model.train(mode=is_training)

    total_loss = 0.0
    total_parts = _empty_loss_totals()
    total_samples = 0

    for batch in loader:
        batch = _move_batch_to_device(batch, device)
        batch_size = int(batch["features"].shape[0])

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            prediction = model(batch["features"])
            loss, loss_parts = criterion(prediction, batch)

        if is_training:
            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

        total_loss += float(loss.detach().cpu()) * batch_size
        total_samples += batch_size
        for key in total_parts:
            total_parts[key] += loss_parts[key] * batch_size

    if total_samples == 0:
        raise ValueError("No samples were produced for the current epoch.")

    metrics = {
        "loss": total_loss / total_samples,
        "samples": total_samples,
        "batches": len(loader),
    }
    for key, value in total_parts.items():
        metrics[key] = value / total_samples
    return metrics


def train(
    dataset_path: str | Path | None = None,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    sequence_length: int = 4,
    hidden_dim: int = 128,
    epochs: int = 200,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    val_ratio: float = 0.2,
    seed: int = 42,
    device: str = "auto",
    patience: int = 40,
    min_delta: float = 1e-4,
    grad_clip: float = 1.0,
    report_path: str | Path | None = None,
    log_every: int = 20,
    velocity_weight: float = 1e-3,
    inertia_weight: float = 1e4,
    coriolis_weight: float = 1e4,
    wind_pressure_weight: float = 0.05,
    nearshore_weight: float = 0.02,
) -> Path:
    _set_reproducible_seed(seed)
    device_obj = _resolve_device(device)

    frame = _prepare_dataframe(load_dataset(dataset_path))
    train_frame, val_frame, train_storm_ids, val_storm_ids = _split_frame_by_storm(
        frame,
        val_ratio=val_ratio,
        seed=seed,
    )
    scaler = TensorScaler.fit(train_frame, FEATURE_COLUMNS)
    train_dataset = TyphoonSequenceDataset(train_frame, scaler, sequence_length)
    val_dataset: TyphoonSequenceDataset | None = None
    if not val_frame.empty:
        try:
            val_dataset = TyphoonSequenceDataset(val_frame, scaler, sequence_length)
        except ValueError:
            print("Validation split has no usable sequential samples; continuing without validation.")
            val_frame = val_frame.iloc[0:0].copy()
            val_storm_ids = []

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = (
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0) if val_dataset is not None else None
    )
    model = TyphoonPINN(input_dim=sequence_length * len(FEATURE_COLUMNS), hidden_dim=hidden_dim).to(device_obj)
    criterion = PINNLoss(
        scaler,
        velocity_weight=velocity_weight,
        inertia_weight=inertia_weight,
        coriolis_weight=coriolis_weight,
        wind_pressure_weight=wind_pressure_weight,
        nearshore_weight=nearshore_weight,
    ).to(device_obj)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    dataset_summary = {
        "all": _summarize_frame(frame),
        "train": _summarize_frame(train_frame),
        "val": _summarize_frame(val_frame),
        "train_storm_ids": train_storm_ids,
        "val_storm_ids": val_storm_ids,
        "train_sequence_samples": len(train_dataset),
        "val_sequence_samples": len(val_dataset) if val_dataset is not None else 0,
    }
    training_config = {
        "dataset_path": str(dataset_path) if dataset_path is not None else "",
        "output_path": str(output_path),
        "sequence_length": sequence_length,
        "hidden_dim": hidden_dim,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "val_ratio": val_ratio,
        "seed": seed,
        "device": str(device_obj),
        "patience": patience,
        "min_delta": min_delta,
        "grad_clip": grad_clip,
        "velocity_weight": velocity_weight,
        "inertia_weight": inertia_weight,
        "coriolis_weight": coriolis_weight,
        "wind_pressure_weight": wind_pressure_weight,
        "nearshore_weight": nearshore_weight,
    }

    print(
        f"dataset_points={dataset_summary['all']['points']} storms={dataset_summary['all']['storms']} "
        f"train_samples={dataset_summary['train_sequence_samples']} val_samples={dataset_summary['val_sequence_samples']} "
        f"device={device_obj}"
    )
    print(
        f"time_steps={dataset_summary['all']['time_step_hours']} "
        f"train_storms={len(train_storm_ids)} val_storms={len(val_storm_ids)}"
    )

    best_metric_name = "val_loss" if val_loader is not None else "train_loss"
    best_metric = float("inf")
    best_epoch = 0
    best_state_dict = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    history: list[dict[str, Any]] = []
    epochs_without_improvement = 0
    early_stopped = False

    for epoch in range(1, epochs + 1):
        train_metrics = _run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device_obj,
            optimizer=optimizer,
            grad_clip=grad_clip,
        )
        val_metrics = (
            _run_epoch(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device_obj,
            )
            if val_loader is not None
            else None
        )

        monitor_value = val_metrics["loss"] if val_metrics is not None else train_metrics["loss"]
        improved = monitor_value < (best_metric - min_delta)
        if improved:
            best_metric = monitor_value
            best_epoch = epoch
            best_state_dict = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        history.append(
            {
                "epoch": epoch,
                "train": {key: round(float(value), 8) for key, value in train_metrics.items()},
                "val": {key: round(float(value), 8) for key, value in val_metrics.items()} if val_metrics else None,
                "monitor_metric": best_metric_name,
                "monitor_value": round(float(monitor_value), 8),
                "is_best": improved,
            }
        )

        should_log = epoch == 1 or epoch % max(log_every, 1) == 0 or epoch == epochs or improved
        if should_log:
            log_message = (
                f"epoch={epoch:04d} train_loss={train_metrics['loss']:.6f} "
                f"train_data={train_metrics['data_loss']:.6f} train_velocity={train_metrics['velocity_loss']:.6f}"
            )
            if val_metrics is not None:
                log_message += (
                    f" val_loss={val_metrics['loss']:.6f} val_data={val_metrics['data_loss']:.6f}"
                    f" best_{best_metric_name}={best_metric:.6f}"
                )
            else:
                log_message += f" best_{best_metric_name}={best_metric:.6f}"
            print(log_message)

        if patience > 0 and epochs_without_improvement >= patience:
            early_stopped = True
            print(
                f"Early stopping at epoch {epoch} after {epochs_without_improvement} epochs without "
                f"{best_metric_name} improvement."
            )
            break

    model.load_state_dict(best_state_dict)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_payload = _build_checkpoint_payload(
        model=model,
        scaler=scaler,
        sequence_length=sequence_length,
        hidden_dim=hidden_dim,
        best_epoch=best_epoch,
        best_metric=best_metric,
        history=history,
        dataset_summary=dataset_summary,
        training_config=training_config,
    )
    torch.save(checkpoint_payload, output)

    resolved_report_path = Path(report_path) if report_path is not None else _derive_report_path(output)
    resolved_report_path.parent.mkdir(parents=True, exist_ok=True)
    report_payload = {
        "checkpoint_path": str(output),
        "report_path": str(resolved_report_path),
        "best_epoch": best_epoch,
        "best_metric_name": best_metric_name,
        "best_metric": round(float(best_metric), 8),
        "epochs_completed": len(history),
        "early_stopped": early_stopped,
        "used_validation": val_loader is not None,
        "dataset_summary": dataset_summary,
        "training_config": training_config,
        "history": history,
        "saved_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    resolved_report_path.write_text(json.dumps(report_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved best PINN weights to {output}")
    print(f"Saved training summary to {resolved_report_path}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the typhoon PINN model.")
    parser.add_argument("--dataset", type=str, default=None, help="Path to a CSV/JSON/JSONL typhoon dataset.")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_PATH), help="Output .pth file path.")
    parser.add_argument("--report", type=str, default=None, help="Optional training summary JSON output path.")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sequence-length", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Holdout ratio by storm_id. Set to 0 to disable.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", help="auto, cpu, cuda, or cuda:0")
    parser.add_argument("--patience", type=int, default=40, help="Early-stopping patience in epochs. Set to 0 to disable.")
    parser.add_argument("--min-delta", type=float, default=1e-4, help="Minimum improvement required to reset patience.")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping max norm. Set to 0 to disable.")
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--velocity-weight", type=float, default=1e-3)
    parser.add_argument("--inertia-weight", type=float, default=1e4)
    parser.add_argument("--coriolis-weight", type=float, default=1e4)
    parser.add_argument("--wind-pressure-weight", type=float, default=0.05)
    parser.add_argument("--nearshore-weight", type=float, default=0.02)
    args = parser.parse_args()
    output = train(
        dataset_path=args.dataset,
        output_path=args.output,
        sequence_length=args.sequence_length,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        val_ratio=args.val_ratio,
        seed=args.seed,
        device=args.device,
        patience=args.patience,
        min_delta=args.min_delta,
        grad_clip=args.grad_clip,
        report_path=args.report,
        log_every=args.log_every,
        velocity_weight=args.velocity_weight,
        inertia_weight=args.inertia_weight,
        coriolis_weight=args.coriolis_weight,
        wind_pressure_weight=args.wind_pressure_weight,
        nearshore_weight=args.nearshore_weight,
    )
    print(f"Training complete. Best weights are available at {output}")


if __name__ == "__main__":
    main()
