"""Professor: the attempts log (filterable), one attempt in detail, labelling, and CSV export.

Labels (genuine / printed photo / screen replay + the attack type) are what the threshold report
(scripts/threshold_report.py) and the paper's Table I are computed from.

Only scores are shown - never face images (they are not stored) and never embeddings/templates
(face_templates is not read here at all).
"""
from __future__ import annotations

from urllib.parse import urlsplit

from flask import abort, current_app, flash, redirect, render_template, request, url_for
from flask_login import current_user
from sqlalchemy import select
from sqlalchemy.orm import aliased

from .. import vision_service
from ..attempt_log import (DECISIONS, MODES, REASONS, AttemptFilter, count, csv_header, csv_row,
                           label_filter_choices, students)
from ..csv_export import csv_response
from ..extensions import db
from ..models import Attempt, Attendance, User, label_choices, label_text, parse_label_value
from ..security import to_local
from ..vision.pipeline import MESSAGES
from . import bp

PER_PAGE = 100
MAX_NOTE = 1000


def _tz() -> str:
    return current_app.config["TIMEZONE"]


@bp.route("/attempts")
def attempts():
    f = AttemptFilter.from_args(request.args)
    for e in f.errors:
        flash(e, "error")
    stmt = f.statement(_tz())
    total = count(db.session, stmt)
    pages = max(1, -(-total // PER_PAGE))
    page = min(max(request.args.get("page", 1, type=int) or 1, 1), pages)
    rows = db.session.execute(stmt.limit(PER_PAGE).offset((page - 1) * PER_PAGE)).all()
    return render_template(
        "professor/attempts.html", rows=rows, f=f, total=total, page=page, pages=pages,
        students=students(db.session), modes=MODES, decisions=DECISIONS, reasons=REASONS,
        label_filters=label_filter_choices(), label_choices=label_choices(), label_text=label_text,
        cfg=vision_service.vision_config(), back=request.full_path.rstrip("?"))


@bp.route("/attempts/<int:attempt_id>")
def attempt_detail(attempt_id: int):
    a = db.session.get(Attempt, attempt_id)
    if a is None:
        abort(404)
    attendance = db.session.execute(select(Attendance).where(Attendance.attempt_id == a.id)).scalar_one_or_none()
    labeller = db.session.get(User, a.labelled_by_id) if a.labelled_by_id else None
    return render_template("professor/attempt.html", a=a, student=a.user, attendance=attendance, labeller=labeller,
                           label_choices=label_choices(), label_text=label_text,
                           message=MESSAGES.get(a.reason, ""), back=_safe_back(request.args.get("back")))


def _safe_back(url: str | None) -> str:
    """Only ever go back to a page of the attempts log on THIS site (no open redirect)."""
    default = url_for("professor.attempts")
    if not url:
        return default
    parts = urlsplit(url)
    if parts.scheme or parts.netloc or not parts.path.startswith(default) or "\\" in url:
        return default
    return url


@bp.route("/attempts/label", methods=["POST"])
def label_attempts():
    """Label one attempt (detail page) or many (checkboxes in the log) with ONE label + optional note.

    label = "genuine" | "photo" | "photo/printed_flat" | ... | "clear" (remove the label).
    Log (bulk): a label must be chosen; an empty note keeps each attempt's existing note.
    Detail page (single=1): the form shows the attempt's current label and note and saves both as
    they are, so an empty label means "not labelled" and an empty note removes the note.
    """
    back = _safe_back(request.form.get("back"))
    ids = sorted({i for i in request.form.getlist("ids", type=int)})
    value = request.form.get("label", "")
    note = (request.form.get("note") or "").strip()
    single = request.form.get("single") == "1"
    if not ids:
        flash("Select at least one attempt first.", "error")
        return redirect(back)
    if value == "clear" or (single and value == ""):
        label, attack_type = None, None
    else:
        try:
            label, attack_type = parse_label_value(value)
        except ValueError:
            label = None
        if label is None:
            flash("Choose a label.", "error")
            return redirect(back)
    if len(note) > MAX_NOTE:
        flash(f"The note is too long (at most {MAX_NOTE} characters).", "error")
        return redirect(back)

    rows = db.session.execute(select(Attempt).where(Attempt.id.in_(ids))).scalars().all()
    if len(rows) != len(ids):
        abort(404)
    for a in rows:
        if (a.label, a.attack_type) != (label, attack_type):   # unchanged label keeps who/when
            a.set_label(label, attack_type, by=current_user)
        if single:
            a.note = note or None
        elif note:
            a.note = note
    db.session.commit()
    if single:
        what = "saved" + (f" as {label_text(label, attack_type)}" if label else " (not labelled)")
    else:
        what = "label removed" if label is None else f"labelled as {label_text(label, attack_type)}"
    flash(f"{len(rows)} attempt{'s' if len(rows) != 1 else ''} {what}.", "info")
    return redirect(back)


@bp.route("/attempts.csv")
def attempts_csv():
    """Every attempt matching the filters (no paging): scores, checks, timings, decisions, labels.
    No biometric data. ?anonymise=1 replaces student ID + name with a stable participant code."""
    f = AttemptFilter.from_args(request.args)
    anonymise = request.args.get("anonymise") == "1"
    labeller = aliased(User)
    stmt = f.statement(_tz()).add_columns(labeller).outerjoin(labeller, Attempt.labelled_by_id == labeller.id)
    rows = db.session.execute(stmt.order_by(None).order_by(Attempt.id)).all()
    secret = current_app.config["SECRET_KEY"]
    name = "attempts" + ("_anonymised" if anonymise else "") + ".csv"
    return csv_response(name, csv_header(anonymise),
                        (csv_row(a, u, lb, to_local, anonymise, secret) for a, u, lb in rows))
