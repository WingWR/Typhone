from __future__ import annotations


def create_app():
    from flask import Flask
    from flask_cors import CORS

    from backend.api.routes import api_blueprint

    app = Flask(__name__)
    CORS(app)
    app.register_blueprint(api_blueprint, url_prefix="/api")
    return app
