"""Facial recognition student attendance - Flask application factory.

    from app import create_app
    app = create_app()              # APP_ENV=development (default) | production | testing

Configuration is layered: app/config.py class -> instance/config.py (optional, git-ignored)
-> environment variables read in config.py -> `overrides` (tests). The secret key comes from the
SECRET_KEY environment variable or instance/secret_key, which is created on first run.
"""
from __future__ import annotations

import os
import secrets
from pathlib import Path

from flask import Flask, flash, redirect, render_template, request, url_for
from flask_login import current_user
from flask_wtf.csrf import CSRFError

from .config import CONFIGS
from .extensions import csrf, db, login_manager
from .models import User


def create_app(config_name: str | None = None, overrides: dict | None = None) -> Flask:
    app = Flask(__name__, instance_relative_config=True)
    config_name = config_name or os.environ.get("APP_ENV", "development")
    app.config.from_object(CONFIGS[config_name])
    Path(app.instance_path).mkdir(parents=True, exist_ok=True)
    if config_name != "testing":
        app.config.from_pyfile("config.py", silent=True)       # instance/config.py
    if overrides:
        app.config.update(overrides)
    if not app.config.get("SQLALCHEMY_DATABASE_URI"):
        db_file = Path(app.instance_path) / "attendance.db"
        app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{db_file.as_posix()}"
    if not app.config.get("SECRET_KEY"):
        app.config["SECRET_KEY"] = _instance_secret_key(Path(app.instance_path))

    if app.config.get("TRUST_PROXY_HOPS"):
        from werkzeug.middleware.proxy_fix import ProxyFix
        n = app.config["TRUST_PROXY_HOPS"]
        app.wsgi_app = ProxyFix(app.wsgi_app, x_for=n, x_proto=n, x_host=n)

    db.init_app(app)
    csrf.init_app(app)
    login_manager.init_app(app)
    login_manager.login_view = "auth.login"
    login_manager.login_message = "Please log in."
    login_manager.session_protection = "strong"

    @login_manager.unauthorized_handler
    def _unauthorized():
        if request.blueprint == "api":
            return {"error": "login_required"}, 401
        flash(login_manager.login_message, "info")
        return redirect(url_for(login_manager.login_view))

    from .api import bp as api_bp
    from .auth import bp as auth_bp
    from .professor import bp as professor_bp
    from .student import bp as student_bp
    app.register_blueprint(auth_bp)
    app.register_blueprint(student_bp, url_prefix="/student")
    app.register_blueprint(professor_bp, url_prefix="/professor")
    app.register_blueprint(api_bp, url_prefix="/api")

    _register_hooks(app)
    return app


def _instance_secret_key(instance: Path) -> str:
    f = instance / "secret_key"
    if not f.exists():
        f.write_text(secrets.token_hex(32))
    return f.read_text().strip()


@login_manager.user_loader
def load_user(session_id: str) -> User | None:
    """Session id is "<user id>:<session version>". A password change/reset or disabling the account
    bumps the version, so every older session stops working immediately."""
    try:
        uid, version = (int(x) for x in session_id.split(":"))
    except (ValueError, AttributeError):
        return None
    user = db.session.get(User, uid)
    if user is None or not user.active or user.session_version != version:
        return None
    return user


# endpoints a logged-in user may use before changing the initial password
_ALWAYS_ALLOWED = {"auth.login", "auth.logout", "auth.change_password", "static", "index"}


def _register_hooks(app: Flask) -> None:
    @app.before_request
    def force_password_change():
        if (current_user.is_authenticated and current_user.must_change_password
                and request.endpoint not in _ALWAYS_ALLOWED and request.endpoint is not None):
            if request.blueprint == "api":
                return {"error": "password_change_required"}, 403
            return redirect(url_for("auth.change_password"))

    @app.after_request
    def security_headers(resp):
        resp.headers.setdefault("X-Content-Type-Options", "nosniff")
        resp.headers.setdefault("X-Frame-Options", "DENY")
        resp.headers.setdefault("Referrer-Policy", "same-origin")
        # camera only for this site; scripts/styles only from this site (no inline code)
        resp.headers.setdefault("Permissions-Policy", "camera=(self), microphone=(), geolocation=()")
        resp.headers.setdefault("Content-Security-Policy",
                                "default-src 'self'; img-src 'self' data: blob:; media-src 'self' blob:; "
                                "object-src 'none'; base-uri 'self'; frame-ancestors 'none'; form-action 'self'")
        if current_user.is_authenticated:
            resp.headers.setdefault("Cache-Control", "no-store")
        return resp

    @app.route("/")
    def index():
        if not current_user.is_authenticated:
            return redirect(url_for("auth.login"))
        return redirect(url_for("professor.dashboard" if current_user.is_professor else "student.home"))

    @app.errorhandler(CSRFError)
    def csrf_error(e):
        if request.is_json or request.blueprint == "api":
            return {"error": "csrf", "message": e.description}, 400
        return render_template("error.html", code=400, title="Form expired",
                               message="The form was open too long or was sent from somewhere else. "
                                       "Go back, reload the page and try again."), 400

    for code, title, message in ((403, "Not allowed", "You do not have access to this page."),
                                 (404, "Not found", "This page does not exist."),
                                 (413, "Too large", "The upload was too large.")):
        app.register_error_handler(code, _error_handler(code, title, message))

    @app.cli.command("init-db")
    def init_db_command():
        """Create all database tables (safe to run repeatedly)."""
        db.create_all()
        print(f"Database ready: {app.config['SQLALCHEMY_DATABASE_URI']}")


def _error_handler(code, title, message):
    def handler(e):
        if request.blueprint == "api":
            return {"error": title.lower().replace(" ", "_")}, code
        return render_template("error.html", code=code, title=title, message=message), code
    return handler
