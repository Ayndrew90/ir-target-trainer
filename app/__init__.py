import os
from flask import Flask

def create_app():
    app = Flask(__name__)
    app.config["SECRET_KEY"] = "supergeheimesPasswort"
    base_dir = os.path.abspath(os.path.dirname(__file__))

    app.config["UPLOAD_FOLDER"] = os.path.join(base_dir, "static", "uploads")
    app.config["JSON_FOLDER"] = os.path.join(base_dir, "data")

    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    os.makedirs(app.config["JSON_FOLDER"], exist_ok=True)

    from .routes import bp_main
    from .laser_routes import bp_laser

    app.register_blueprint(bp_main)
    app.register_blueprint(bp_laser, url_prefix="/laser")

    return app
