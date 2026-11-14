"""Login, logout and password change."""
from __future__ import annotations

import math

from flask import Blueprint, flash, redirect, render_template, request, url_for
from flask_login import current_user, login_required, login_user, logout_user
from flask_wtf import FlaskForm
from sqlalchemy import select
from wtforms import PasswordField, StringField, SubmitField
from wtforms.validators import DataRequired, EqualTo, Length

from ..extensions import db
from ..models import User, utcnow
from ..security import (clear_login_failures, client_ip, hash_password, lockout_remaining, normalise_username,
                        password_problem, record_login_failure, verify_password)

bp = Blueprint("auth", __name__)

GENERIC_LOGIN_ERROR = "Wrong username or password."


class LoginForm(FlaskForm):
    username = StringField("Student ID / username", validators=[DataRequired(), Length(max=64)])
    password = PasswordField("Password", validators=[DataRequired(), Length(max=128)])
    submit = SubmitField("Log in")


class ChangePasswordForm(FlaskForm):
    current_password = PasswordField("Current password", validators=[DataRequired(), Length(max=128)])
    new_password = PasswordField("New password", validators=[DataRequired(), Length(max=128)])
    confirm = PasswordField("Repeat new password",
                            validators=[DataRequired(), EqualTo("new_password", "The passwords do not match.")])
    submit = SubmitField("Change password")


def _minutes(td) -> int:
    return max(1, math.ceil(td.total_seconds() / 60))


@bp.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("index"))
    form = LoginForm()
    if form.validate_on_submit():
        username = normalise_username(form.username.data)
        ip = client_ip()
        locked = lockout_remaining(username, ip)
        if locked is not None:
            # do not even check the password while locked (also not a correct one)
            flash(f"Too many failed attempts. Try again in {_minutes(locked)} minute(s).", "error")
            return render_template("auth/login.html", form=form), 429
        user = db.session.execute(select(User).where(User.username == username)).scalar_one_or_none()
        if not verify_password(user, form.password.data):
            record_login_failure(username, ip)
            flash(GENERIC_LOGIN_ERROR, "error")
            return render_template("auth/login.html", form=form), 401
        if not user.active:
            flash("This account is disabled. Please contact your professor.", "error")
            return render_template("auth/login.html", form=form), 403
        clear_login_failures(username)
        user.last_login_at = utcnow()
        db.session.commit()
        login_user(user)
        return redirect(url_for("index"))
    return render_template("auth/login.html", form=form)


@bp.route("/logout", methods=["POST"])
def logout():
    logout_user()
    flash("You are logged out.", "info")
    return redirect(url_for("auth.login"))


@bp.route("/change-password", methods=["GET", "POST"])
@login_required
def change_password():
    form = ChangePasswordForm()
    forced = current_user.must_change_password
    if form.validate_on_submit():
        user: User = current_user._get_current_object()  # the real object, not the proxy
        if not verify_password(user, form.current_password.data):
            form.current_password.errors.append("The current password is wrong.")
        elif form.new_password.data == form.current_password.data:
            form.new_password.errors.append("Choose a password different from the current one.")
        elif (problem := password_problem(form.new_password.data, user.username)) is not None:
            form.new_password.errors.append(problem)
        else:
            user.password_hash = hash_password(form.new_password.data)
            user.must_change_password = False
            user.password_changed_at = utcnow()
            user.invalidate_sessions()          # log out other sessions ...
            db.session.commit()
            login_user(user)                    # ... but keep this one
            flash("Your password was changed.", "info")
            return redirect(url_for("index"))
    return render_template("auth/change_password.html", form=form, forced=forced), \
        (400 if request.method == "POST" else 200)
