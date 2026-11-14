"""JSON API used by the browser JavaScript (enrollment / check-in uploads come in stage 4).

Every POST here is CSRF-protected like the forms: the page puts the token in
<meta name="csrf-token"> and the JavaScript sends it in the X-CSRFToken header.
"""
from __future__ import annotations

from flask import Blueprint
from flask_login import current_user, login_required

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
