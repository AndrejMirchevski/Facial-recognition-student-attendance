"""Professor pages: student account management (stage 3).

Attendance by date, the attempts log with labelling, and CSV export come in stage 5.
Professors NEVER see or download face templates or face images: these views only ever ask
whether a template exists (User.is_enrolled) and can delete it (reset enrollment).
"""
from __future__ import annotations

from flask import Blueprint, abort, flash, redirect, render_template, url_for
from flask_login import current_user
from flask_wtf import FlaskForm
from sqlalchemy import delete, select
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired, Length, Optional, Regexp

from ..extensions import db
from ..models import ROLE_PROFESSOR, ROLE_STUDENT, FaceTemplate, User
from ..security import generate_password, hash_password, normalise_username, password_problem, role_required

bp = Blueprint("professor", __name__)

STUDENT_ID_RE = r"^[A-Za-z0-9._/-]{3,32}$"
STUDENT_ID_HELP = "3-32 characters: letters, digits and . _ - /"


def _no_spaces(value: str | None) -> str | None:
    """Student IDs are often typed or pasted with spaces ("2023 / 0045 ") - remove them all."""
    return "".join(value.split()) if value else value


def _strip(value: str | None) -> str | None:
    return value.strip() if value else value


@bp.before_request
@role_required(ROLE_PROFESSOR)
def _only_professors():
    return None


class CreateStudentForm(FlaskForm):
    student_id = StringField("Student ID", filters=[_no_spaces],
                             validators=[DataRequired(), Regexp(STUDENT_ID_RE, message=STUDENT_ID_HELP)])
    full_name = StringField("Full name", filters=[_strip], validators=[DataRequired(), Length(min=2, max=128)])
    # A visible text field on purpose: the professor has to pass it on to the student anyway, and a
    # form WITHOUT a password field is not mistaken by the browser for a login form (otherwise it
    # autofills the professor's own username/password into Student ID / this field).
    initial_password = StringField("Initial password (leave empty to generate one)", filters=[_strip],
                                   validators=[Optional(), Length(max=128)])
    submit = SubmitField("Create student")


def _student_or_404(user_id: int) -> User:
    user = db.session.get(User, user_id)
    if user is None or user.role != ROLE_STUDENT:
        abort(404)
    return user


def _dashboard(form: CreateStudentForm, status: int = 200):
    students = db.session.execute(
        select(User).where(User.role == ROLE_STUDENT).order_by(User.username)).scalars().all()
    enrolled_ids = set(db.session.execute(select(FaceTemplate.user_id)).scalars().all())
    return render_template("professor/dashboard.html", students=students, enrolled_ids=enrolled_ids,
                           form=form), status


@bp.route("/")
def dashboard():
    return _dashboard(CreateStudentForm())


@bp.route("/students", methods=["POST"])
def create_student():
    form = CreateStudentForm()
    # On any error the page is shown again with the typed values and the error under its field.
    if not form.validate_on_submit():
        return _dashboard(form, 400)
    username = normalise_username(form.student_id.data)
    if db.session.execute(select(User.id).where(User.username == username)).first():
        form.student_id.errors.append(f"An account with ID {username} already exists.")
        return _dashboard(form, 400)
    password = form.initial_password.data or generate_password()
    if (problem := password_problem(password, username)) is not None:
        form.initial_password.errors.append(problem)
        return _dashboard(form, 400)
    student = User(role=ROLE_STUDENT, username=username, full_name=form.full_name.data.strip(),
                   password_hash=hash_password(password), must_change_password=True, active=True,
                   created_by_id=current_user.id)
    db.session.add(student)
    db.session.commit()
    # shown once, in this response only (not stored, not put in the session cookie)
    return render_template("professor/credentials.html", student=student, password=password,
                           action="created")


@bp.route("/students/<int:user_id>/reset-password", methods=["POST"])
def reset_password(user_id: int):
    student = _student_or_404(user_id)
    password = generate_password()
    student.password_hash = hash_password(password)
    student.must_change_password = True
    student.invalidate_sessions()
    db.session.commit()
    return render_template("professor/credentials.html", student=student, password=password,
                           action="password reset")


@bp.route("/students/<int:user_id>/toggle-active", methods=["POST"])
def toggle_active(user_id: int):
    student = _student_or_404(user_id)
    student.active = not student.active
    if not student.active:
        student.invalidate_sessions()
    db.session.commit()
    flash(f"{student.full_name} ({student.username}) is now {'enabled' if student.active else 'disabled'}.", "info")
    return redirect(url_for("professor.dashboard"))


@bp.route("/students/<int:user_id>/reset-enrollment", methods=["POST"])
def reset_enrollment(user_id: int):
    student = _student_or_404(user_id)
    deleted = db.session.execute(delete(FaceTemplate).where(FaceTemplate.user_id == student.id)).rowcount
    db.session.commit()
    flash(f"Face data of {student.full_name} ({student.username}) deleted - they must enroll again."
          if deleted else f"{student.full_name} was not enrolled.", "info")
    return redirect(url_for("professor.dashboard"))
