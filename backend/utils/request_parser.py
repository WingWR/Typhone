from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from flask import Request


def parse_typhoon_payload(request: "Request") -> dict[str, Any]:
    if "file" in request.files:
        uploaded = request.files["file"]
        if not uploaded.filename.lower().endswith(".json"):
            raise ValueError("Uploaded file must be a .json file.")
        return json.load(uploaded.stream)

    if request.is_json:
        payload = request.get_json(silent=True)
        if payload:
            return payload

    raise ValueError("Request must include a JSON file upload or JSON body.")
