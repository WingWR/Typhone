from __future__ import annotations

from typing import Any


def require_fields(payload: dict[str, Any], required_fields: list[str]) -> None:
    missing = [field for field in required_fields if field not in payload]
    if missing:
        raise ValueError(f"Missing required fields: {', '.join(missing)}")


def normalize_observations(observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(observations, list) or not observations:
        raise ValueError("`observations` must be a non-empty list.")

    required = ["lng", "lat", "timestamp"]
    normalized = []
    for index, observation in enumerate(observations):
        if not isinstance(observation, dict):
            raise ValueError(f"Observation at index {index} must be an object.")
        missing = [field for field in required if field not in observation]
        if missing:
            raise ValueError(f"Observation at index {index} is missing: {', '.join(missing)}")
        normalized.append(observation)

    return sorted(normalized, key=lambda item: item["timestamp"])
