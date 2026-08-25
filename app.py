import os
from flask import Flask, redirect, url_for
from flask_login import LoginManager

from config import Config
from models import db, Admin
from ml.predictor import predictor


def create_app(config_class: type = Config) -> Flask:
    app = Flask(__name__)
    app.config.from_object(config_class)

    os.makedirs(app.config["UPLOAD_FOLDER_ECG"], exist_ok=True)
    os.makedirs(app.config["UPLOAD_FOLDER_EEG"], exist_ok=True)
    os.makedirs(app.config["MODEL_DIR"], exist_ok=True)

    db.init_app(app)

    login_manager = LoginManager()
    login_manager.login_view = "auth.login"
    login_manager.init_app(app)

    @login_manager.user_loader
    def load_user(user_id: str):
        # user_id is stored as "admin:<pk>" (see models.Admin.get_id)
        if not user_id.startswith("admin:"):
            return None
        return Admin.query.get(int(user_id.split(":", 1)[1]))

    from routes.auth import auth_bp
    from routes.students import students_bp
    from routes.attendance import attendance_bp
    from routes.admin import admin_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(students_bp)
    app.register_blueprint(attendance_bp)
    app.register_blueprint(admin_bp)

    @app.route("/")
    def index():
        return redirect(url_for("attendance.attendance_page"))

    with app.app_context():
        db.create_all()
        predictor.load(app.config)  # no-op (returns False) until models are trained

    return app


app = create_app()

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
