"""JSON API used by the browser JavaScript.

Every POST here is CSRF-protected like the forms: the page puts the token in
<meta name="csrf-token"> and the JavaScript sends it in the X-CSRFToken header.
The browser only uploads pictures; every decision is made on the server (app/attendance.py).
"""
from __future__ import annotations

import math
import time

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


def _client_capture_ms() -> float | None:
    """How long the guided capture took in the browser (camera ready -> 5th frame), for the
    end-to-end verification time in the paper. Reported by the browser, so only plausible values
    are kept; it never influences a decision."""
    try:
        v = float(request.form.get("capture_ms", ""))
    except ValueError:
        return None
    return round(v, 1) if math.isfinite(v) and 0 < v <= 120_000 else None


@bp.route("/enroll", methods=["POST"])
@role_required(ROLE_STUDENT)
def enroll():
    started = time.perf_counter()
    frames, err = _frames_or_error()
    if err:
        return err
    consent = request.form.get("consent") == "yes"
    out = attendance.enroll(current_user._get_current_object(), frames, consent,
                            timing=attendance.Timing(started, _client_capture_ms()))
    return out.body, out.status


@bp.route("/checkin", methods=["POST"])
@role_required(ROLE_STUDENT)
def checkin():
    started = time.perf_counter()
    frames, err = _frames_or_error()
    if err:
        return err
    out = attendance.checkin(current_user._get_current_object(), frames,
                             timing=attendance.Timing(started, _client_capture_ms()))
    return out.body, out.status
