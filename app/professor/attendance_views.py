"""Professor: attendance by date and the attendance CSV export."""
from __future__ import annotations

from datetime import timedelta

from flask import abort, flash, render_template, request
from sqlalchemy import select

from ..attempt_log import parse_date, students
from ..csv_export import csv_response
from ..extensions import db
from ..models import Attendance, User
from ..security import local_now
from . import bp

MAX_EXPORT_DAYS = 3 * 366


@bp.route("/attendance")
def attendance():
    today = local_now().date()
    try:
        day = parse_date(request.args.get("date")) or today
    except ValueError:
        flash("That is not a valid date - showing today.", "error")
        day = today
    rows = db.session.execute(
        select(Attendance, User).join(User, Attendance.user_id == User.id)
        .where(Attendance.date == day).order_by(Attendance.time, User.username)).all()
    present_ids = {a.user_id for a, _ in rows}
    absent = [s for s in students(db.session) if s.active and s.id not in present_ids]
    return render_template("professor/attendance.html", day=day, today=today, rows=rows, absent=absent,
                           prev_day=day - timedelta(days=1), next_day=day + timedelta(days=1))


@bp.route("/attendance.csv")
def attendance_csv():
    """One row per check-in, for one day (?date=) or a range (?date_from=&date_to=), oldest first."""
    try:
        if request.args.get("date"):
            d_from = d_to = parse_date(request.args["date"])
        else:
            d_from, d_to = parse_date(request.args.get("date_from")), parse_date(request.args.get("date_to"))
    except ValueError:
        abort(400)
    if d_from is None or d_to is None:
        abort(400)
    if d_from > d_to:
        d_from, d_to = d_to, d_from
    if (d_to - d_from).days > MAX_EXPORT_DAYS:
        abort(400)
    rows = db.session.execute(
        select(Attendance, User).join(User, Attendance.user_id == User.id)
        .where(Attendance.date >= d_from, Attendance.date <= d_to)
        .order_by(Attendance.date, Attendance.time, User.username)).all()
    name = f"attendance_{d_from.isoformat()}" + ("" if d_from == d_to else f"_to_{d_to.isoformat()}") + ".csv"
    return csv_response(name, ["date", "time", "student_id", "full_name", "attempt_id"],
                        ([a.date.isoformat(), a.time.strftime("%H:%M:%S"), u.username, u.full_name, a.attempt_id]
                         for a, u in rows))
