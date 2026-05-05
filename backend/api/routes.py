from __future__ import annotations

import json
from json import JSONDecodeError
from pathlib import Path

from flask import Blueprint, jsonify, request

from backend.services.pinn_prediction_service import TyphoonPredictor
from backend.services.weather_service import build_weather_response
from backend.utils.request_parser import parse_typhoon_payload

api_blueprint = Blueprint("api", __name__)
SAMPLE_JSON_PATH = Path(__file__).resolve().parent.parent / "data" / "sample_typhoon_input.json"
typhoon_predictor = TyphoonPredictor()


@api_blueprint.get("/health")
def health_check():
    return jsonify({"status": "ok"})


@api_blueprint.get("/sample_typhoon_input")
def get_sample_typhoon_input():
    with SAMPLE_JSON_PATH.open("r", encoding="utf-8") as file:
        return jsonify(json.load(file))


@api_blueprint.post("/predict_typhoon")
def predict_typhoon():
    try:
        payload = parse_typhoon_payload(request)
        response = typhoon_predictor.build_prediction_response(payload)
        return jsonify(response)
    except (ValueError, JSONDecodeError) as error:
        return jsonify({"error": str(error)}), 400


@api_blueprint.get("/get_weather_conditions")
def get_weather_conditions():
    try:
        field = request.args.get("field", "rain").strip().lower()
        center_lng = float(request.args.get("center_lng", 121.78))
        center_lat = float(request.args.get("center_lat", 31.14))
        max_wind_speed = float(request.args.get("max_wind_speed", 36.0))
        central_pressure = float(request.args.get("central_pressure", 978.0))
        return jsonify(
            build_weather_response(
                field=field,
                center_lng=center_lng,
                center_lat=center_lat,
                max_wind_speed=max_wind_speed,
                central_pressure=central_pressure,
            )
        )
    except ValueError as error:
        return jsonify({"error": str(error)}), 400
