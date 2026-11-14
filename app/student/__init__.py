"""Student pages. Stage 3: home (status + attendance history) and an enrollment PLACEHOLDER.
Enrollment, check-in and "delete my face data" are built in stage 4."""
from __future__ import annotations

from flask import Blueprint, redirect, render_template, request, url_for
from flask_login import current_user
from sqlalchemy import select

from ..extensions import db
from ..models import ROLE_STUDENT, Attendance
from ..security import role_required

bp = Blueprint("student", __name__)

# pages a student who is not enrolled yet may still open
_ALLOWED_WITHOUT_ENROLLMENT = {"student.enroll"}


@bp.before_request
@role_required(ROLE_STUDENT)
def _only_students():
    # Not enrolled -> must enroll before anything else (check-in is impossible without a template).
    if not current_user.is_enrolled and request.endpoint not in _ALLOWED_WITHOUT_ENROLLMENT:
        return redirect(url_for("student.enroll"))
    return None


@bp.route("/")
def home():
    history = db.session.execute(
        select(Attendance).where(Attendance.user_id == current_user.id)
        .order_by(Attendance.date.desc())).scalars().all()
    return render_template("student/home.html", history=history)


@bp.route("/enroll")
def enroll():
    if current_user.is_enrolled:
        return redirect(url_for("student.home"))
    return render_template("student/enroll.html")
