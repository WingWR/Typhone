from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

EARTH_ANGULAR_VELOCITY = 7.2921159e-5
EARTH_RADIUS_M = 6_371_000.0
SECONDS_PER_HOUR = 3600.0

DEFAULT_FEATURE_RANGES: dict[str, tuple[float, float]] = {
    "t_hours": (0.0, 240.0),
    "lng": (100.0, 160.0),
    "lat": (0.0, 50.0),
    "wind_speed": (0.0, 80.0),
    "pressure": (850.0, 1030.0),
}


def coriolis_parameter(latitude_deg: float | np.ndarray) -> float | np.ndarray:
    """Return the Coriolis parameter f = 2 * Omega * sin(latitude)."""
    return 2.0 * EARTH_ANGULAR_VELOCITY * np.sin(np.deg2rad(latitude_deg))


def latlon_velocity_mps(
    lng0: float | np.ndarray,
    lat0: float | np.ndarray,
    lng1: float | np.ndarray,
    lat1: float | np.ndarray,
    dt_hours: float | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert two latitude/longitude positions into local east/north velocity."""
    dt_seconds = np.maximum(np.asarray(dt_hours, dtype=float) * SECONDS_PER_HOUR, 1e-6)
    lat_ref = np.deg2rad((np.asarray(lat0, dtype=float) + np.asarray(lat1, dtype=float)) * 0.5)
    dx = EARTH_RADIUS_M * np.cos(lat_ref) * np.deg2rad(np.asarray(lng1, dtype=float) - np.asarray(lng0, dtype=float))
    dy = EARTH_RADIUS_M * np.deg2rad(np.asarray(lat1, dtype=float) - np.asarray(lat0, dtype=float))
    return dx / dt_seconds, dy / dt_seconds


def coriolis_residual_mps2(
    prev_u: float,
    prev_v: float,
    next_u: float,
    next_v: float,
    latitude_deg: float,
    dt_hours: float,
) -> tuple[float, float]:
    """Return a weak inertial-motion residual under Coriolis acceleration."""
    dt_seconds = max(dt_hours * SECONDS_PER_HOUR, 1e-6)
    acceleration_u = (next_u - prev_u) / dt_seconds
    acceleration_v = (next_v - prev_v) / dt_seconds
    f_value = float(coriolis_parameter(latitude_deg))
    return acceleration_u - f_value * next_v, acceleration_v + f_value * next_u


@dataclass(frozen=True)
class FeatureScaler:
    """Min-max scaler that maps physical features to the [-1, 1] model range."""

    ranges: Mapping[str, tuple[float, float]] = field(default_factory=lambda: DEFAULT_FEATURE_RANGES.copy())

    @classmethod
    def from_dict(cls, payload: Mapping[str, Sequence[float]] | None) -> "FeatureScaler":
        if not payload:
            return cls()
        ranges = {
            key: (float(value[0]), float(value[1]))
            for key, value in payload.items()
            if isinstance(value, Sequence) and len(value) == 2
        }
        return cls({**DEFAULT_FEATURE_RANGES, **ranges})

    def normalize_feature(self, value: float, feature: str) -> float:
        low, high = self.ranges[feature]
        span = max(high - low, 1e-6)
        return 2.0 * (float(value) - low) / span - 1.0

    def denormalize_feature(self, value: float, feature: str) -> float:
        low, high = self.ranges[feature]
        return (float(value) + 1.0) * 0.5 * (high - low) + low

    def normalize_vector(self, values: Sequence[float], features: Sequence[str]) -> np.ndarray:
        return np.array(
            [self.normalize_feature(value, feature) for value, feature in zip(values, features)],
            dtype=np.float32,
        )

    def denormalize_vector(self, values: Sequence[float], features: Sequence[str]) -> np.ndarray:
        return np.array(
            [self.denormalize_feature(value, feature) for value, feature in zip(values, features)],
            dtype=np.float32,
        )
