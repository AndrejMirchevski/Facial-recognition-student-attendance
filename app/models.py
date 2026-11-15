"""Database tables (SQLAlchemy). SQLite now; PostgreSQL later only needs DATABASE_URL.

users           accounts (professor | student). Login name = student ID for students.
face_templates  biometric templates, ONE table on its own so it can be protected, deleted
                (consent withdrawal) and never joined into anything a professor sees.
attempts        EVERY enrollment / check-in attempt with all pipeline scores (no images),
                plus a label (genuine / photo / screen) and attack type for calibration.
attendance      successful check-ins only. UNIQUE (user_id, date) in the database = at most one
                check-in per student per calendar day. A nullable session_id can be added later
                when lectures/exams ("sessions") are introduced, without rewriting this table.
login_failures  failed logins, for the brute-force lockout.
"""
from __future__ import annotations

import datetime as dt
from datetime import datetime, timezone

from flask_login import UserMixin
from sqlalchemy import (JSON, Boolean, Date, DateTime, Float, ForeignKey, Integer, LargeBinary, String,
                        Text, Time, UniqueConstraint, select)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .extensions import db

ROLE_PROFESSOR = "professor"
ROLE_STUDENT = "student"
LABELS = ("genuine", "photo", "screen")
ATTACK_TYPES = {"photo": ("printed_flat", "printed_bent"),
                "screen": ("phone_photo", "phone_video", "monitor_photo")}
LABEL_NAMES = {"genuine": "Genuine (real student)", "photo": "Printed photo", "screen": "Screen replay"}
ATTACK_TYPE_NAMES = {"printed_flat": "printed, flat", "printed_bent": "printed, bent",
                     "phone_photo": "phone photo", "phone_video": "phone video", "monitor_photo": "monitor photo"}


def label_value(label: str | None, attack_type: str | None) -> str:
    """Label + attack type as ONE form value: "genuine", "photo", "photo/printed_flat", "" = none."""
    if not label:
        return ""
    return f"{label}/{attack_type}" if attack_type else label


def parse_label_value(value: str) -> tuple[str | None, str | None]:
    """Inverse of label_value(). Raises ValueError for anything that is not a known combination."""
    if not value:
        return None, None
    label, _, attack_type = value.partition("/")
    if label not in LABELS or (attack_type and attack_type not in ATTACK_TYPES.get(label, ())):
        raise ValueError(f"unknown label {value!r}")
    return label, attack_type or None


def label_choices() -> list[tuple[str, str]]:
    """(value, text) for every label a professor can set, attack types grouped under their label."""
    out = [("genuine", LABEL_NAMES["genuine"])]
    for label in ("photo", "screen"):
        out.append((label, f"{LABEL_NAMES[label]} - type not specified"))
        out += [(f"{label}/{t}", f"{LABEL_NAMES[label]} - {ATTACK_TYPE_NAMES[t]}") for t in ATTACK_TYPES[label]]
    return out


def label_text(label: str | None, attack_type: str | None) -> str:
    if not label:
        return ""
    return LABEL_NAMES[label] + (f" - {ATTACK_TYPE_NAMES.get(attack_type, attack_type)}" if attack_type else "")


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)  # stored as naive UTC


class User(UserMixin, db.Model):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    role: Mapped[str] = mapped_column(String(16), index=True)
    # Login name, stored lower-case. For students this is the student ID.
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    full_name: Mapped[str] = mapped_column(String(128))
    password_hash: Mapped[str] = mapped_column(String(255))
    must_change_password: Mapped[bool] = mapped_column(Boolean, default=True)
    active: Mapped[bool] = mapped_column(Boolean, default=True)   # False = disabled by a professor
    # Increased on password change / reset / disable: logs out every existing session of the user.
    session_version: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    created_by_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    password_changed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    @property
    def is_active(self) -> bool:  # used by Flask-Login
        return bool(self.active)

    @property
    def is_professor(self) -> bool:
        return self.role == ROLE_PROFESSOR

    @property
    def is_student(self) -> bool:
        return self.role == ROLE_STUDENT

    def get_id(self) -> str:  # Flask-Login: session id includes the version, see load_user()
        return f"{self.id}:{self.session_version}"

    def invalidate_sessions(self) -> None:
        self.session_version = (self.session_version or 0) + 1

    @property
    def is_enrolled(self) -> bool:
        """True if a face template exists. Only checks existence - never loads the template."""
        return db.session.execute(
            select(FaceTemplate.id).where(FaceTemplate.user_id == self.id)).first() is not None


class FaceTemplate(db.Model):
    """Biometric template of one student. Professors never see or download this table."""
    __tablename__ = "face_templates"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), unique=True, index=True)
    template: Mapped[bytes] = mapped_column(LargeBinary)                  # float32 aggregated embedding
    frame_embeddings: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)  # optional, 5 x dim
    embedding_dim: Mapped[int] = mapped_column(Integer)
    model_name: Mapped[str] = mapped_column(String(64))                   # e.g. arcface-w600k_r50
    model_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    consent_at: Mapped[datetime] = mapped_column(DateTime)                # when the student ticked consent
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    enroll_attempt_id: Mapped[int | None] = mapped_column(ForeignKey("attempts.id"), nullable=True)

    def __repr__(self) -> str:  # never print biometric bytes
        return f"<FaceTemplate user={self.user_id} model={self.model_name}>"


class Attempt(db.Model):
    """One enrollment or check-in attempt, pass or fail. Scores only - never images or embeddings.
    This table is the source of the paper's metrics (Table I) and of threshold calibration."""
    __tablename__ = "attempts"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)  # server time (UTC)
    mode: Mapped[str] = mapped_column(String(16))           # enroll | checkin
    decision: Mapped[str] = mapped_column(String(16), index=True)  # pass | fail | retry
    reason: Mapped[str] = mapped_column(String(64))
    failed_checks: Mapped[str] = mapped_column(String(255), default="")  # e.g. "texture;depth"
    # set when a check-in passed but the student already had attendance that day
    duplicate: Mapped[bool] = mapped_column(Boolean, default=False)

    # identity
    similarity: Mapped[float | None] = mapped_column(Float, nullable=True)
    consistency: Mapped[float | None] = mapped_column(Float, nullable=True)
    # liveness a: parallax (both methods)
    parallax_method: Mapped[str | None] = mapped_column(String(16), nullable=True)
    parallax_status: Mapped[str | None] = mapped_column(String(32), nullable=True)
    parallax_landmark_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    parallax_landmark_movement: Mapped[float | None] = mapped_column(Float, nullable=True)
    parallax_flow_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    parallax_flow_movement: Mapped[float | None] = mapped_column(Float, nullable=True)
    parallax_flow_points: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # liveness b: depth, c: texture
    depth_relief: Mapped[float | None] = mapped_column(Float, nullable=True)
    depth_nose: Mapped[float | None] = mapped_column(Float, nullable=True)
    texture_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    # per-check pass/fail (None = not evaluated), per-frame reports, timings - all JSON
    checks: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    frames: Mapped[list | None] = mapped_column(JSON, nullable=True)
    timings_ms: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    processing_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    model_name: Mapped[str | None] = mapped_column(String(64), nullable=True)
    # thresholds in force for this attempt (so old attempts stay interpretable after recalibration)
    thresholds: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # calibration labels, set by a professor / test page
    label: Mapped[str | None] = mapped_column(String(16), nullable=True, index=True)  # genuine|photo|screen
    attack_type: Mapped[str | None] = mapped_column(String(32), nullable=True)
    labelled_by_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    labelled_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    note: Mapped[str | None] = mapped_column(Text, nullable=True)

    user: Mapped[User] = relationship(foreign_keys=[user_id])

    # PipelineResult.summary() keys stored in their own columns
    SCORE_FIELDS = ("mode", "decision", "reason", "similarity", "consistency", "parallax_method",
                    "parallax_status", "parallax_landmark_score", "parallax_landmark_movement",
                    "parallax_flow_score", "parallax_flow_movement", "parallax_flow_points", "depth_relief",
                    "depth_nose", "texture_score", "checks", "frames", "timings_ms", "processing_ms",
                    "model_name")
    # summary() keys deliberately not stored as columns
    IGNORED_FIELDS = ("message",)

    @classmethod
    def from_summary(cls, user_id: int, summary: dict, thresholds: dict | None = None) -> "Attempt":
        """Build a row from PipelineResult.summary() (which contains no biometric data)."""
        a = cls(user_id=user_id, thresholds=thresholds)
        for f in cls.SCORE_FIELDS:
            setattr(a, f, summary.get(f))
        a.failed_checks = ";".join(summary.get("failed_checks") or [])
        return a

    def set_label(self, label: str | None, attack_type: str | None, by: User) -> None:
        if label is not None and label not in LABELS:
            raise ValueError(f"unknown label {label!r}")
        if attack_type is not None and attack_type not in ATTACK_TYPES.get(label or "", ()):
            raise ValueError(f"attack type {attack_type!r} does not fit label {label!r}")
        self.label, self.attack_type = label, attack_type
        if label is None:      # label removed
            self.labelled_by_id, self.labelled_at = None, None
        else:
            self.labelled_by_id, self.labelled_at = by.id, utcnow()


class Attendance(db.Model):
    """A successful check-in. At most ONE per student per calendar day (enforced by the database)."""
    __tablename__ = "attendance"
    __table_args__ = (UniqueConstraint("user_id", "date", name="uq_attendance_user_date"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    date: Mapped[dt.date] = mapped_column(Date, index=True)        # local calendar day (config TIMEZONE)
    time: Mapped[dt.time] = mapped_column(Time)                    # local time of the check-in
    checked_in_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)  # exact server time, UTC
    attempt_id: Mapped[int] = mapped_column(ForeignKey("attempts.id"))
    # session_id: added later (nullable FK to a future `sessions` table) - see module docstring.

    user: Mapped[User] = relationship(foreign_keys=[user_id])


class LoginFailure(db.Model):
    """One failed login. Used to lock a username / IP address after too many failures."""
    __tablename__ = "login_failures"

    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(String(64), index=True)
    ip: Mapped[str] = mapped_column(String(64), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)
