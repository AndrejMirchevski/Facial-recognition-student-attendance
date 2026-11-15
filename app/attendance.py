"""Enrollment and check-in: everything the server decides after the browser uploads 5 frames.

Rules (CLAUDE.md):
* Every ANALYSED attempt - pass, fail or retry - is stored in `attempts` with all scores.
* Enrollment: consent required; liveness must pass; the template is stored with the consent time.
* Check-in: 1:1 against the logged-in student's own template only. Attendance is written only if
  the pipeline passed AND the student has no attendance today. The database's UNIQUE(user_id,
  date) constraint is the final guard (two simultaneous requests cannot both write a row).
* Requests refused BEFORE analysis (no consent, not enrolled, bad upload, too many attempts) are
  not attempts: nothing was analysed, nothing is stored.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta

import numpy as np
from flask import current_app
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from . import vision_service
from .extensions import db
from .models import Attempt, Attendance, FaceTemplate, User, utcnow
from .security import local_now


@dataclass
class Outcome:
    status: int                          # HTTP status
    body: dict = field(default_factory=dict)


def _refuse(status: int, error: str, message: str) -> Outcome:
    return Outcome(status, {"error": error, "message": message})


def todays_attendance(user: User) -> Attendance | None:
    today = local_now().date()
    return db.session.execute(select(Attendance).where(
        Attendance.user_id == user.id, Attendance.date == today)).scalar_one_or_none()


def too_many_attempts(user: User) -> bool:
    c = current_app.config
    since = utcnow() - timedelta(minutes=c["CAPTURE_RATE_WINDOW_MINUTES"])
    n = db.session.execute(select(func.count(Attempt.id)).where(
        Attempt.user_id == user.id, Attempt.created_at >= since)).scalar_one()
    return n >= c["CAPTURE_MAX_ATTEMPTS"]


def _log_attempt(user: User, result) -> Attempt:
    attempt = Attempt.from_summary(user.id, result.summary(), thresholds=vision_service.thresholds_snapshot())
    db.session.add(attempt)
    db.session.flush()           # get attempt.id
    return attempt


def _result_body(result, attempt: Attempt) -> dict:
    return {"decision": result.decision, "reason": result.reason, "message": result.message,
            "attempt_id": attempt.id}


def enroll(user: User, frames: list[np.ndarray], consent: bool) -> Outcome:
    if user.is_enrolled:
        return _refuse(409, "already_enrolled", "You are already enrolled.")
    if not consent:
        return _refuse(400, "consent_required", "Please read and accept the consent statement first.")
    if too_many_attempts(user):
        return _refuse(429, "too_many_attempts", "Too many attempts. Please wait a few minutes.")

    result = vision_service.analyse(frames, template=None)
    attempt = _log_attempt(user, result)
    body = _result_body(result, attempt)
    if result.passed and result.embedding is not None:
        now = utcnow()
        frame_embs = np.stack(result.frame_embeddings).astype(np.float32) if result.frame_embeddings else None
        db.session.add(FaceTemplate(
            user_id=user.id, template=vision_service.template_to_bytes(result.embedding),
            frame_embeddings=None if frame_embs is None else frame_embs.tobytes(),
            embedding_dim=int(result.embedding.shape[0]), model_name=result.model_name,
            consent_at=now, created_at=now, enroll_attempt_id=attempt.id))
        body["message"] = "Your face is enrolled. You can now check in."
    db.session.commit()
    return Outcome(200, body)


def checkin(user: User, frames: list[np.ndarray]) -> Outcome:
    tpl_row = db.session.execute(select(FaceTemplate).where(FaceTemplate.user_id == user.id)).scalar_one_or_none()
    if tpl_row is None:
        return _refuse(409, "not_enrolled", "You need to enroll your face before you can check in.")
    if too_many_attempts(user):
        return _refuse(429, "too_many_attempts", "Too many attempts. Please wait a few minutes.")

    # 1:1 verification against THIS student's template only
    template = vision_service.template_from_bytes(tpl_row.template)
    result = vision_service.analyse(frames, template=template)
    attempt = _log_attempt(user, result)
    body = _result_body(result, attempt)
    body["attendance"] = None

    if result.passed:
        existing = todays_attendance(user)
        if existing is None:
            now = local_now()
            row = Attendance(user_id=user.id, date=now.date(), time=now.time().replace(microsecond=0),
                             checked_in_at=utcnow(), attempt_id=attempt.id)
            try:
                with db.session.begin_nested():      # savepoint: a race only undoes this insert
                    db.session.add(row)
            except IntegrityError:
                existing = todays_attendance(user)
            else:
                body["attendance"] = {"date": row.date.isoformat(), "time": row.time.strftime("%H:%M")}
                body["message"] = f"Checked in at {row.time.strftime('%H:%M')}. Your attendance is recorded."
        if existing is not None:
            attempt.duplicate = True
            body["duplicate"] = True
            body["message"] = (f"Verified, but you already checked in today at {existing.time.strftime('%H:%M')}. "
                               "Nothing new was recorded.")
    db.session.commit()
    return Outcome(200, body)


def delete_face_data(user: User) -> bool:
    """Withdraw consent: delete the template. Attendance history and attempt scores (no biometric
    data) are kept. Check-in is impossible until the student enrolls again."""
    tpl = db.session.execute(select(FaceTemplate).where(FaceTemplate.user_id == user.id)).scalar_one_or_none()
    if tpl is None:
        return False
    db.session.delete(tpl)
    db.session.commit()
    return True
