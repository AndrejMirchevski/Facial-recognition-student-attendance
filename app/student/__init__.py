"""Student pages: home, enrollment (with consent), check-in, and deleting one's face data."""
from __future__ import annotations

import json

from flask import Blueprint, current_app, flash, g, redirect, render_template, request, send_from_directory, url_for
from flask_login import current_user
from sqlalchemy import select

from .. import attendance, vision_service
from ..extensions import db
from ..models import ROLE_STUDENT, Attendance, FaceTemplate
from ..security import role_required, to_local

bp = Blueprint("student", __name__)

TASKS_VISION_DIR = "vendor/mediapipe/tasks-vision-1.2.0"

# pages a student who is not enrolled yet may still open
_ALLOWED_WITHOUT_ENROLLMENT = {"student.enroll", "student.landmarker_model"}


@bp.before_request
@role_required(ROLE_STUDENT)
def _only_students():
    # Not enrolled -> must enroll before anything else (check-in is impossible without a template).
    if not current_user.is_enrolled and request.endpoint not in _ALLOWED_WITHOUT_ENROLLMENT:
        return redirect(url_for("student.enroll"))
    return None


def _capture_context(mode: str) -> dict:
    """Settings for static/js/capture.js. Thresholds come from config.py via vision_service."""
    g.csp_allow_wasm = True     # only the two capture pages may compile WebAssembly
    return {
        "capture_mode": mode,
        "capture_settings": json.dumps({
            "mode": mode,
            "uploadUrl": url_for("api.enroll" if mode == "enroll" else "api.checkin"),
            "doneUrl": url_for("student.home"),
            "bundleUrl": url_for("static", filename=f"{TASKS_VISION_DIR}/vision_bundle.mjs"),
            "wasmPath": url_for("static", filename=f"{TASKS_VISION_DIR}/wasm"),
            "modelUrl": url_for("student.landmarker_model"),
            "jpegQuality": current_app.config["CAPTURE_JPEG_QUALITY"],
            "maxWidth": current_app.config["CAPTURE_MAX_WIDTH"],
            "guide": vision_service.guidance_settings(),
        }),
    }


@bp.route("/")
def home():
    history = db.session.execute(
        select(Attendance).where(Attendance.user_id == current_user.id)
        .order_by(Attendance.date.desc())).scalars().all()
    tpl_created = db.session.execute(
        select(FaceTemplate.created_at).where(FaceTemplate.user_id == current_user.id)).scalar_one_or_none()
    return render_template("student/home.html", history=history, today=attendance.todays_attendance(current_user),
                           enrolled_at=to_local(tpl_created))


@bp.route("/enroll")
def enroll():
    if current_user.is_enrolled:
        return redirect(url_for("student.home"))
    return render_template("student/enroll.html", **_capture_context("enroll"))


@bp.route("/checkin")
def checkin():
    today = attendance.todays_attendance(current_user)
    if today is not None:
        flash(f"You already checked in today at {today.time.strftime('%H:%M')}.", "info")
        return redirect(url_for("student.home"))
    return render_template("student/checkin.html", **_capture_context("checkin"))


@bp.route("/face-data/delete", methods=["POST"])
def delete_face_data():
    attendance.delete_face_data(current_user)
    flash("Your face data was deleted. You cannot check in until you enroll again.", "info")
    return redirect(url_for("student.enroll"))


@bp.route("/assets/face_landmarker.task")
def landmarker_model():
    """The MediaPipe face landmark model for the browser (same file the server uses)."""
    return send_from_directory(vision_service.vision_config().models_dir, "face_landmarker.task",
                               mimetype="application/octet-stream", max_age=86400)
