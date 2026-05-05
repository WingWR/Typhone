from __future__ import annotations

import numpy as np

from backend.config import DOMAIN, GRID_SHAPE


def _grid_coords() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lats = np.linspace(DOMAIN["lat_min"], DOMAIN["lat_max"], GRID_SHAPE[0], dtype=np.float32)
    lngs = np.linspace(DOMAIN["lon_min"], DOMAIN["lon_max"], GRID_SHAPE[1], dtype=np.float32)
    lon_grid, lat_grid = np.meshgrid(lngs, lats)
    return lats, lngs, lon_grid, lat_grid


def build_weather_response(
    *,
    field: str,
    center_lng: float,
    center_lat: float,
    max_wind_speed: float,
    central_pressure: float,
) -> dict:
    lats, lngs, lon_grid, lat_grid = _grid_coords()

    radial_distance = np.sqrt(((lon_grid - center_lng) / 0.32) ** 2 + ((lat_grid - center_lat) / 0.26) ** 2)
    angle = np.arctan2(lat_grid - center_lat, lon_grid - center_lng)
    spiral_band = np.exp(-(radial_distance**2))
    asymmetric_band = np.exp(-((radial_distance - 0.72) ** 2) / 0.12) * (0.55 + 0.45 * np.cos(angle * 2.2 - radial_distance * 5.5))
    outer_rain = np.exp(-(((lon_grid - (center_lng - 0.42)) / 0.24) ** 2 + ((lat_grid - (center_lat + 0.28)) / 0.18) ** 2))
    rain_field = 2.0 + spiral_band * (max_wind_speed * 0.85)
    rain_field += asymmetric_band * (max_wind_speed * 0.42)
    rain_field += outer_rain * 10.5

    wind_core = 8 + max_wind_speed * np.exp(-((radial_distance - 0.95) ** 2) / 0.34)
    wind_shear = 4 * np.sin((lon_grid - DOMAIN["lon_min"]) * 3.1) + 2.4 * np.cos((lat_grid - DOMAIN["lat_min"]) * 4.8)
    wind_field = np.clip(wind_core + wind_shear, 4, None)

    pressure_field = central_pressure + 26 * (1 - np.exp(-(radial_distance**2)))
    pressure_field += 1.2 * np.sin((lon_grid - center_lng) * 3.0) - 0.9 * np.cos((lat_grid - center_lat) * 2.5)

    field_map = {
        "rain": {
            "values": rain_field,
            "units": "mm/h",
            "description": "Simulated precipitation intensity field driven by forecast typhoon center",
        },
        "wind": {
            "values": wind_field,
            "units": "m/s",
            "description": "Simulated near-surface wind speed field driven by forecast typhoon center",
        },
        "pressure": {
            "values": pressure_field,
            "units": "hPa",
            "description": "Simulated sea-level pressure field driven by forecast typhoon center",
        },
    }

    normalized_field = field if field in field_map else "rain"
    selected = field_map[normalized_field]
    values = np.round(selected["values"], 3)
    if normalized_field == "rain":
        mask = values >= max(6.5, float(values.max()) * 0.16)
    elif normalized_field == "wind":
        mask = values >= max(9.5, float(values.max()) * 0.36)
    else:
        mask = values <= min(1008.0, float(values.min()) + 7.0)

    points = np.column_stack((lon_grid[mask], lat_grid[mask], values[mask]))

    return {
        "field": normalized_field,
        "dims": ["lat", "lng"],
        "coords": {
            "lat": np.round(lats, 4).tolist(),
            "lng": np.round(lngs, 4).tolist(),
        },
        "values": values.tolist(),
        "points": np.round(points, 4).tolist(),
        "metadata": {
            "domain": {
                "lng": [DOMAIN["lon_min"], DOMAIN["lon_max"]],
                "lat": [DOMAIN["lat_min"], DOMAIN["lat_max"]],
            },
            "grid_shape": {"lat": int(values.shape[0]), "lng": int(values.shape[1])},
            "units": selected["units"],
            "description": selected["description"],
            "render_point_count": int(points.shape[0]),
            "storm_center": {
                "lng": round(float(center_lng), 4),
                "lat": round(float(center_lat), 4),
                "max_wind_speed": round(float(max_wind_speed), 2),
                "central_pressure": round(float(central_pressure), 2),
            },
            "xarray_hint": {
                "data_var": normalized_field,
                "dims": ["lat", "lng"],
                "coords": ["lat", "lng"],
            },
        },
    }
