from __future__ import annotations

from typing import Any

from backend.services.pinn_prediction_service import TyphoonPredictor
from backend.utils.request_parser import parse_typhoon_payload

_predictor = TyphoonPredictor()


def build_prediction_response(payload: dict[str, Any]) -> dict[str, Any]:
    """Backward-compatible entry point that now delegates to the PINN predictor."""
    return _predictor.build_prediction_response(payload)


__all__ = ["build_prediction_response", "parse_typhoon_payload"]
