"""JSON API used by the browser JavaScript.

Every POST here is CSRF-protected like the forms: the page puts the token in
<meta name="csrf-token"> and the JavaScript sends it in the X-CSRFToken header.
The browser only uploads pictures; every decision is made on the server (app/attendance.py).
"""
from __future__ import annotations

from flask import Blueprint, request
from flask_login import current_user, login_required

from .. import attendance, vision_service
from ..frames import FrameError, read_frames
from ..models import ROLE_STUDENT
from ..security import role_required

bp = Blueprint("api", __name__)


@bp.route("/me")
@login_required
def me():
    """Who is logged in and what they must do next (used by the frontend)."""
    return {
        "username": current_user.username,
        "full_name": current_user.full_name,
        "role": current_user.role,
        "must_change_password": current_user.must_change_password,
        "enrolled": current_user.is_enrolled if current_user.is_student else None,
    }


def _frames_or_error():
    try:
        return read_frames(request.files.getlist("frames"), vision_service.vision_config().num_frames), None
    except FrameError as e:
        return None, ({"error": "invalid_frames", "message": e.message}, 400)


@bp.route("/enroll", methods=["POST"])
@role_required(ROLE_STUDENT)
def enroll():
    frames, err = _frames_or_error()
    if err:
        return err
    consent = request.form.get("consent") == "yes"
    out = attendance.enroll(current_user._get_current_object(), frames, consent)
    return out.body, out.status


@bp.route("/checkin", methods=["POST"])
@role_required(ROLE_STUDENT)
def checkin():
    frames, err = _frames_or_error()
    if err:
        return err
    out = attendance.checkin(current_user._get_current_object(), frames)
    return out.body, out.status
