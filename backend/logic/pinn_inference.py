from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

from backend.config import DEFAULT_FORECAST_STEPS, DEFAULT_TIME_STEP_HOURS
from backend.utils.physics_engine import (
    FeatureScaler,
    EARTH_RADIUS_M,
    coriolis_residual_mps2,
    latlon_velocity_mps,
)
from backend.utils.validation import normalize_observations, require_fields

try:
    import torch
    from backend.models.pinn_model import FEATURE_COLUMNS, STATE_COLUMNS, TyphoonPINN
except Exception:  # pragma: no cover - runtime fallback path
    torch = None
    TyphoonPINN = None
    FEATURE_COLUMNS = ["t_hours", "lng", "lat", "wind_speed", "pressure"]
    STATE_COLUMNS = ["lng", "lat", "wind_speed", "pressure"]

DEFAULT_WEIGHTS_PATH = Path(__file__).resolve().parent.parent / "models" / "weights" / "typhoon_pinn_v2.pth"


@dataclass(frozen=True)
class TyphoonObservation:
    lng: float
    lat: float
    timestamp: datetime
    wind_speed: float
    pressure: float


class TyphoonPredictor:
    """PINN-backed autoregressive predictor with a linear fallback path."""

    def __init__(self, weights_path: str | Path = DEFAULT_WEIGHTS_PATH) -> None:
        self.weights_path = Path(weights_path)
        self.sequence_length = 4
        self.hidden_dim = 128
        self.scaler = FeatureScaler()
        self.model: Any = None
        self.load_error = ""
        self._load_model()

    def _load_model(self) -> None:
        if torch is None or TyphoonPINN is None:
            self.load_error = "PyTorch is not installed; using linear fallback."
            return
        if not self.weights_path.exists():
            self.load_error = f"PINN weights not found at {self.weights_path}; using linear fallback."
            return

        try:
            checkpoint = torch.load(self.weights_path, map_location="cpu")
            self.sequence_length = int(checkpoint.get("sequence_length", self.sequence_length))
            self.hidden_dim = int(checkpoint.get("hidden_dim", self.hidden_dim))
            input_dim = int(checkpoint.get("input_dim", self.sequence_length * len(FEATURE_COLUMNS)))
            self.scaler = FeatureScaler.from_dict(checkpoint.get("scaler"))
            self.model = TyphoonPINN(input_dim=input_dim, hidden_dim=self.hidden_dim)
            self.model.load_state_dict(checkpoint["state_dict"])
            self.model.eval()
            self.load_error = ""
        except Exception as error:  # pragma: no cover - defensive fallback
            self.model = None
            self.load_error = f"Failed to load PINN weights: {error}"

    def build_prediction_response(self, payload: dict[str, Any]) -> dict[str, Any]:
        observations = self._extract_observations(payload)
        forecast_steps = max(1, min(int(payload.get("forecast_steps", DEFAULT_FORECAST_STEPS)), 24))
        time_step_hours = max(1, min(int(payload.get("time_step_hours", DEFAULT_TIME_STEP_HOURS)), 6))
        actual_track = self._extract_actual_track(payload)

        observed_track = [
            self._serialize_track_point(
                lng=observation.lng,
                lat=observation.lat,
                timestamp=observation.timestamp,
                wind_speed=observation.wind_speed,
                pressure=observation.pressure,
                source="observed",
                lead_hour=0,
            )
            for observation in observations
        ]
        baseline_track = [
            {**point, "source": "baseline"}
            for point in self._predict_with_linear_smoothing(observations, forecast_steps, time_step_hours)
        ]
        prediction_result = self.predict_steps(observations, forecast_steps, time_step_hours)
        predicted_track = prediction_result["predicted_track"]
        pinn_track = predicted_track if prediction_result["inference_mode"] == "pinn" else []
        combined_track = observed_track + predicted_track
        latest_point = combined_track[-1]
        losses = self._build_loss_summary(observations, predicted_track, actual_track)
        metrics = self._build_metrics(predicted_track, baseline_track, actual_track)
        model_name = "pinn" if prediction_result["inference_mode"] == "pinn" else "rule_baseline"

        return {
            "storm_id": payload["storm_id"],
            "storm_name": payload["storm_name"],
            "basin": payload.get("basin", "East China Sea"),
            "model_name": model_name,
            "model_type": "PINN-v1",
            "forecast_steps": forecast_steps,
            "time_step_hours": time_step_hours,
            "observed_track": observed_track,
            "predicted_track": predicted_track,
            "pinn_track": pinn_track,
            "baseline_track": baseline_track,
            "actual_track": actual_track,
            "combined_track": combined_track,
            "losses": losses,
            "metrics": metrics,
            "weather_context": {
                "center_lng": latest_point["lng"],
                "center_lat": latest_point["lat"],
                "max_wind_speed": latest_point["wind_speed"],
                "central_pressure": latest_point["pressure"],
            },
            "summary": {
                "input_points": len(observed_track),
                "predicted_points": len(predicted_track),
                "start_time": observed_track[0]["timestamp"],
                "end_time": combined_track[-1]["timestamp"],
                "max_wind_speed": max(point["wind_speed"] for point in combined_track),
                "min_pressure": min(point["pressure"] for point in combined_track),
                "model_name": model_name,
                "model_type": "PINN-v1",
                "inference_mode": prediction_result["inference_mode"],
                "physics_consistency_score": prediction_result["physics_consistency_score"],
                "data_loss": losses["data_loss"],
                "physics_loss": losses["physics_loss"],
                "metrics": metrics,
                "fallback_reason": prediction_result.get("fallback_reason", ""),
            },
        }

    def predict_steps(
        self,
        observations: list[TyphoonObservation],
        forecast_steps: int,
        time_step_hours: int,
    ) -> dict[str, Any]:
        if self.model is None:
            predicted_track = self._predict_with_linear_smoothing(observations, forecast_steps, time_step_hours)
            return {
                "predicted_track": predicted_track,
                "inference_mode": "linear_fallback",
                "physics_consistency_score": self._physics_consistency_score(observations, predicted_track),
                "fallback_reason": self.load_error,
            }

        try:
            predicted_track = self._predict_with_pinn(observations, forecast_steps, time_step_hours)
            return {
                "predicted_track": predicted_track,
                "inference_mode": "pinn",
                "physics_consistency_score": self._physics_consistency_score(observations, predicted_track),
                "fallback_reason": "",
            }
        except Exception as error:  # pragma: no cover - defensive fallback
            predicted_track = self._predict_with_linear_smoothing(observations, forecast_steps, time_step_hours)
            return {
                "predicted_track": predicted_track,
                "inference_mode": "linear_fallback",
                "physics_consistency_score": self._physics_consistency_score(observations, predicted_track),
                "fallback_reason": f"PINN inference failed: {error}",
            }

    def _predict_with_pinn(
        self,
        observations: list[TyphoonObservation],
        forecast_steps: int,
        time_step_hours: int,
    ) -> list[dict[str, Any]]:
        if torch is None or self.model is None:
            raise RuntimeError("PINN model is unavailable.")

        start_time = observations[0].timestamp
        history = [
            {
                "t_hours": self._hours_between(start_time, observation.timestamp),
                "lng": observation.lng,
                "lat": observation.lat,
                "wind_speed": observation.wind_speed,
                "pressure": observation.pressure,
                "timestamp": observation.timestamp,
            }
            for observation in observations
        ]
        predicted_track = []

        for step in range(1, forecast_steps + 1):
            features = self._build_model_features(history)
            with torch.no_grad():
                output = self.model(torch.tensor(features, dtype=torch.float32).unsqueeze(0)).squeeze(0).cpu().numpy()

            state = np.clip(output[:4], -1.0, 1.0)
            velocity_mps = output[4:6]
            lng, lat, wind_speed, pressure = self.scaler.denormalize_vector(state, STATE_COLUMNS)
            if not np.isfinite([lng, lat, wind_speed, pressure]).all():
                raise ValueError("PINN produced non-finite output.")

            previous = history[-1]
            jump_distance = float(np.hypot(lng - previous["lng"], lat - previous["lat"]))
            if jump_distance > 8.0:
                raise ValueError("PINN produced an unstable trajectory jump.")

            timestamp = previous["timestamp"] + timedelta(hours=time_step_hours)
            next_point = {
                "t_hours": previous["t_hours"] + time_step_hours,
                "lng": float(lng),
                "lat": float(lat),
                "wind_speed": float(np.clip(wind_speed, 5.0, 90.0)),
                "pressure": float(np.clip(pressure, 830.0, 1040.0)),
                "timestamp": timestamp,
            }
            history.append(next_point)
            predicted_track.append(
                self._serialize_track_point(
                    lng=next_point["lng"],
                    lat=next_point["lat"],
                    timestamp=timestamp,
                    wind_speed=next_point["wind_speed"],
                    pressure=next_point["pressure"],
                    source="forecast",
                    lead_hour=step * time_step_hours,
                    u_mps=float(velocity_mps[0]),
                    v_mps=float(velocity_mps[1]),
                )
            )

        return predicted_track

    def _predict_with_linear_smoothing(
        self,
        observations: list[TyphoonObservation],
        forecast_steps: int,
        time_step_hours: int,
    ) -> list[dict[str, Any]]:
        recent = observations[-min(len(observations), 5) :]
        rates = []
        for first, second in zip(recent[:-1], recent[1:]):
            dt_hours = max(self._hours_between(first.timestamp, second.timestamp), 1e-3)
            rates.append(
                np.array(
                    [
                        (second.lng - first.lng) / dt_hours,
                        (second.lat - first.lat) / dt_hours,
                        (second.wind_speed - first.wind_speed) / dt_hours,
                        (second.pressure - first.pressure) / dt_hours,
                    ],
                    dtype=float,
                )
            )
        if not rates:
            raise ValueError("At least two observations are required for fallback prediction.")

        weights = np.linspace(0.65, 1.0, len(rates))
        trend = np.average(np.vstack(rates), axis=0, weights=weights)
        current = np.array(
            [
                observations[-1].lng,
                observations[-1].lat,
                observations[-1].wind_speed,
                observations[-1].pressure,
            ],
            dtype=float,
        )
        current_time = observations[-1].timestamp
        predicted_track = []

        for step in range(1, forecast_steps + 1):
            previous_position = current.copy()
            damping = 0.92 ** (step - 1)
            current = current + trend * damping * time_step_hours
            current[2] = float(np.clip(current[2], 8.0, 90.0))
            current[3] = float(np.clip(current[3], 830.0, 1040.0))
            current_time = current_time + timedelta(hours=time_step_hours)
            u_mps, v_mps = latlon_velocity_mps(
                previous_position[0],
                previous_position[1],
                current[0],
                current[1],
                time_step_hours,
            )
            predicted_track.append(
                self._serialize_track_point(
                    lng=current[0],
                    lat=current[1],
                    timestamp=current_time,
                    wind_speed=current[2],
                    pressure=current[3],
                    source="forecast",
                    lead_hour=step * time_step_hours,
                    u_mps=float(u_mps),
                    v_mps=float(v_mps),
                )
            )

        return predicted_track

    def _build_model_features(self, history: list[dict[str, Any]]) -> np.ndarray:
        sequence = history[-self.sequence_length :]
        if len(sequence) < self.sequence_length:
            padding = [sequence[0]] * (self.sequence_length - len(sequence))
            sequence = padding + sequence

        normalized_rows = [
            self.scaler.normalize_vector(
                [point["t_hours"], point["lng"], point["lat"], point["wind_speed"], point["pressure"]],
                FEATURE_COLUMNS,
            )
            for point in sequence
        ]
        return np.asarray(normalized_rows, dtype=np.float32).reshape(-1)

    def _physics_consistency_score(
        self,
        observations: list[TyphoonObservation],
        predicted_track: list[dict[str, Any]],
    ) -> float:
        raw_points = [
            {
                "lng": observation.lng,
                "lat": observation.lat,
                "timestamp": observation.timestamp,
            }
            for observation in observations
        ] + [
            {
                "lng": point["lng"],
                "lat": point["lat"],
                "timestamp": self._to_datetime(point["timestamp"]),
            }
            for point in predicted_track
        ]
        if len(raw_points) < 3:
            return 1.0

        residuals = []
        for index in range(2, len(raw_points)):
            previous = raw_points[index - 2]
            current = raw_points[index - 1]
            next_point = raw_points[index]
            previous_dt = max(self._hours_between(previous["timestamp"], current["timestamp"]), 1e-3)
            next_dt = max(self._hours_between(current["timestamp"], next_point["timestamp"]), 1e-3)
            prev_u, prev_v = latlon_velocity_mps(
                previous["lng"],
                previous["lat"],
                current["lng"],
                current["lat"],
                previous_dt,
            )
            next_u, next_v = latlon_velocity_mps(
                current["lng"],
                current["lat"],
                next_point["lng"],
                next_point["lat"],
                next_dt,
            )
            residual_u, residual_v = coriolis_residual_mps2(
                float(prev_u),
                float(prev_v),
                float(next_u),
                float(next_v),
                float(next_point["lat"]),
                next_dt,
            )
            residuals.append(float(np.hypot(residual_u, residual_v)))

        mean_residual = float(np.mean(residuals)) if residuals else 0.0
        score = 1.0 / (1.0 + 650.0 * mean_residual)
        return round(float(np.clip(score, 0.0, 1.0)), 4)

    def _build_loss_summary(
        self,
        observations: list[TyphoonObservation],
        predicted_track: list[dict[str, Any]],
        actual_track: list[dict[str, Any]],
    ) -> dict[str, Any]:
        physics_parts = self._physics_loss_breakdown(observations, predicted_track)
        data_loss = self._data_loss(predicted_track, actual_track)
        return {
            "data_loss": data_loss,
            "physics_loss": physics_parts["physics_loss"],
            "velocity_consistency_loss": physics_parts["velocity_consistency_loss"],
            "inertia_loss": physics_parts["inertia_loss"],
            "coriolis_loss": physics_parts["coriolis_loss"],
            "speed_limit_loss": physics_parts["speed_limit_loss"],
            "wind_pressure_loss": physics_parts["wind_pressure_loss"],
            "nearshore_decay_loss": physics_parts["nearshore_decay_loss"],
        }

    def _physics_loss_breakdown(
        self,
        observations: list[TyphoonObservation],
        predicted_track: list[dict[str, Any]],
    ) -> dict[str, float]:
        points = [
            {
                "lng": observation.lng,
                "lat": observation.lat,
                "wind_speed": observation.wind_speed,
                "pressure": observation.pressure,
                "timestamp": observation.timestamp,
            }
            for observation in observations
        ] + [
            {
                "lng": point["lng"],
                "lat": point["lat"],
                "wind_speed": point["wind_speed"],
                "pressure": point["pressure"],
                "timestamp": self._to_datetime(point["timestamp"]),
                "u_mps": point.get("u_mps"),
                "v_mps": point.get("v_mps"),
            }
            for point in predicted_track
        ]
        if len(points) < 3:
            return {
                "physics_loss": 0.0,
                "velocity_consistency_loss": 0.0,
                "inertia_loss": 0.0,
                "coriolis_loss": 0.0,
                "speed_limit_loss": 0.0,
                "wind_pressure_loss": 0.0,
                "nearshore_decay_loss": 0.0,
            }

        segment_velocities = []
        velocity_losses = []
        speed_penalties = []
        wind_pressure_losses = []
        nearshore_losses = []
        for index in range(1, len(points)):
            previous = points[index - 1]
            current = points[index]
            dt_hours = max(self._hours_between(previous["timestamp"], current["timestamp"]), 1e-3)
            u_mps, v_mps = latlon_velocity_mps(
                previous["lng"],
                previous["lat"],
                current["lng"],
                current["lat"],
                dt_hours,
            )
            segment_velocities.append((float(u_mps), float(v_mps), dt_hours, float(current["lat"])))
            speed = float(np.hypot(u_mps, v_mps))
            speed_penalties.append(max(speed - 75.0, 0.0) ** 2)

            wind_tendency = (float(current["wind_speed"]) - float(previous["wind_speed"])) / dt_hours
            pressure_tendency = (float(current["pressure"]) - float(previous["pressure"])) / dt_hours
            wind_pressure_losses.append(max(wind_tendency * pressure_tendency, 0.0) ** 2)

            coast_distance = float(current["lng"]) - self._coastline_lng(float(current["lat"]))
            near_coast = float(np.clip((0.45 - coast_distance) / 0.45, 0.0, 1.0))
            nearshore_losses.append(
                near_coast
                * (
                    max(float(current["wind_speed"]) - float(previous["wind_speed"]), 0.0) ** 2
                    + max(float(previous["pressure"]) - float(current["pressure"]), 0.0) ** 2
                )
            )

            if current.get("u_mps") is not None and current.get("v_mps") is not None:
                velocity_losses.append((float(current["u_mps"]) - float(u_mps)) ** 2 + (float(current["v_mps"]) - float(v_mps)) ** 2)

        inertia_losses = []
        coriolis_losses = []
        for index in range(1, len(segment_velocities)):
            prev_u, prev_v, _, _ = segment_velocities[index - 1]
            next_u, next_v, dt_hours, latitude = segment_velocities[index]
            dt_seconds = max(dt_hours * 3600.0, 1e-6)
            acceleration_u = (next_u - prev_u) / dt_seconds
            acceleration_v = (next_v - prev_v) / dt_seconds
            inertia_losses.append(acceleration_u**2 + acceleration_v**2)
            residual_u, residual_v = coriolis_residual_mps2(prev_u, prev_v, next_u, next_v, latitude, dt_hours)
            coriolis_losses.append(residual_u**2 + residual_v**2)

        velocity_consistency_loss = float(np.mean(velocity_losses)) if velocity_losses else 0.0
        inertia_loss = float(np.mean(inertia_losses)) if inertia_losses else 0.0
        coriolis_loss = float(np.mean(coriolis_losses)) if coriolis_losses else 0.0
        speed_limit_loss = float(np.mean(speed_penalties)) if speed_penalties else 0.0
        wind_pressure_loss = float(np.mean(wind_pressure_losses)) if wind_pressure_losses else 0.0
        nearshore_decay_loss = float(np.mean(nearshore_losses)) if nearshore_losses else 0.0
        physics_loss = (
            1e-3 * velocity_consistency_loss
            + 1e4 * inertia_loss
            + 1e4 * coriolis_loss
            + 1e-3 * speed_limit_loss
            + 0.05 * wind_pressure_loss
            + 0.02 * nearshore_decay_loss
        )

        return {
            "physics_loss": round(float(physics_loss), 6),
            "velocity_consistency_loss": round(velocity_consistency_loss, 6),
            "inertia_loss": round(inertia_loss, 8),
            "coriolis_loss": round(coriolis_loss, 8),
            "speed_limit_loss": round(speed_limit_loss, 6),
            "wind_pressure_loss": round(wind_pressure_loss, 6),
            "nearshore_decay_loss": round(nearshore_decay_loss, 6),
        }

    def _build_metrics(
        self,
        predicted_track: list[dict[str, Any]],
        baseline_track: list[dict[str, Any]],
        actual_track: list[dict[str, Any]],
    ) -> dict[str, Any]:
        metrics: dict[str, Any] = {
            "baseline_vs_pinn_mean_km": self._mean_track_distance_km(predicted_track, baseline_track),
        }
        if actual_track:
            metrics.update(
                {
                    "track_mae_km": self._mean_track_distance_km(predicted_track, actual_track),
                    "final_position_error_km": self._final_position_error_km(predicted_track, actual_track),
                    "wind_mae_mps": self._mean_scalar_error(predicted_track, actual_track, "wind_speed"),
                    "pressure_mae_hpa": self._mean_scalar_error(predicted_track, actual_track, "pressure"),
                    "baseline_track_mae_km": self._mean_track_distance_km(baseline_track, actual_track),
                    "baseline_final_position_error_km": self._final_position_error_km(baseline_track, actual_track),
                }
            )
        else:
            metrics.update(
                {
                    "track_mae_km": None,
                    "final_position_error_km": None,
                    "wind_mae_mps": None,
                    "pressure_mae_hpa": None,
                    "baseline_track_mae_km": None,
                    "baseline_final_position_error_km": None,
                }
            )
        return metrics

    def _data_loss(self, predicted_track: list[dict[str, Any]], actual_track: list[dict[str, Any]]) -> float | None:
        if not actual_track:
            return None
        distances = self._paired_distances_km(predicted_track, actual_track)
        if not distances:
            return None
        return round(float(np.mean(np.square(distances))), 6)

    def _mean_track_distance_km(self, first_track: list[dict[str, Any]], second_track: list[dict[str, Any]]) -> float | None:
        distances = self._paired_distances_km(first_track, second_track)
        if not distances:
            return None
        return round(float(np.mean(distances)), 3)

    def _final_position_error_km(self, predicted_track: list[dict[str, Any]], actual_track: list[dict[str, Any]]) -> float | None:
        if not predicted_track or not actual_track:
            return None
        return round(float(self._distance_km(predicted_track[-1], actual_track[min(len(actual_track), len(predicted_track)) - 1])), 3)

    def _mean_scalar_error(
        self,
        predicted_track: list[dict[str, Any]],
        actual_track: list[dict[str, Any]],
        field: str,
    ) -> float | None:
        values = [
            abs(float(predicted[field]) - float(actual[field]))
            for predicted, actual in zip(predicted_track, actual_track)
            if field in predicted and field in actual
        ]
        if not values:
            return None
        return round(float(np.mean(values)), 3)

    def _paired_distances_km(self, first_track: list[dict[str, Any]], second_track: list[dict[str, Any]]) -> list[float]:
        return [self._distance_km(first, second) for first, second in zip(first_track, second_track)]

    @staticmethod
    def _distance_km(first: dict[str, Any], second: dict[str, Any]) -> float:
        first_lat = np.deg2rad(float(first["lat"]))
        second_lat = np.deg2rad(float(second["lat"]))
        delta_lat = second_lat - first_lat
        delta_lng = np.deg2rad(float(second["lng"]) - float(first["lng"]))
        a_value = np.sin(delta_lat / 2.0) ** 2 + np.cos(first_lat) * np.cos(second_lat) * np.sin(delta_lng / 2.0) ** 2
        a_value = float(np.clip(a_value, 0.0, 1.0))
        return float(2.0 * EARTH_RADIUS_M * np.arctan2(np.sqrt(a_value), np.sqrt(1.0 - a_value)) / 1000.0)

    @staticmethod
    def _coastline_lng(lat: float) -> float:
        return float(120.35 + 0.19 * (lat - 26.0) + 0.08 * np.sin((lat - 26.0) * 1.6))

    def _extract_observations(self, payload: dict[str, Any]) -> list[TyphoonObservation]:
        require_fields(payload, ["storm_id", "storm_name", "observations"])
        normalized = normalize_observations(payload["observations"])
        observations = [
            TyphoonObservation(
                lng=float(item["lng"]),
                lat=float(item["lat"]),
                timestamp=self._to_datetime(item["timestamp"]),
                wind_speed=float(item.get("wind_speed", 25.0)),
                pressure=float(item.get("pressure", 990.0)),
            )
            for item in normalized
        ]
        if len(observations) < 2:
            raise ValueError("At least two observations are required for trajectory prediction.")
        return observations

    def _extract_actual_track(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        for field in ("actual_track", "truth_track", "ground_truth", "future_observations"):
            if field not in payload:
                continue
            normalized = normalize_observations(payload[field])
            return [
                self._serialize_track_point(
                    lng=float(item["lng"]),
                    lat=float(item["lat"]),
                    timestamp=self._to_datetime(item["timestamp"]),
                    wind_speed=float(item.get("wind_speed", 25.0)),
                    pressure=float(item.get("pressure", 990.0)),
                    source="actual",
                    lead_hour=0,
                )
                for item in normalized
            ]
        return []

    @staticmethod
    def _to_datetime(timestamp: str | datetime) -> datetime:
        if isinstance(timestamp, datetime):
            parsed = timestamp
        else:
            parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    @staticmethod
    def _hours_between(first: datetime, second: datetime) -> float:
        return max((second - first).total_seconds() / 3600.0, 1e-6)

    @staticmethod
    def _serialize_track_point(
        *,
        lng: float,
        lat: float,
        timestamp: datetime,
        wind_speed: float,
        pressure: float,
        source: str,
        lead_hour: int,
        u_mps: float | None = None,
        v_mps: float | None = None,
    ) -> dict[str, Any]:
        point = {
            "lng": round(float(lng), 4),
            "lat": round(float(lat), 4),
            "timestamp": timestamp.isoformat(),
            "wind_speed": round(float(wind_speed), 2),
            "pressure": round(float(pressure), 2),
            "source": source,
            "lead_hour": int(lead_hour),
        }
        if u_mps is not None and v_mps is not None:
            point["u_mps"] = round(float(u_mps), 4)
            point["v_mps"] = round(float(v_mps), 4)
        return point
