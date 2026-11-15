"""The attempts log: filters, CSV rows and pseudonyms - shared by the professor pages, the CSV export
and scripts/threshold_report.py.

No biometric data is ever read here: only the attempts table (scores, decisions, labels - never
images or embeddings) and the student's ID and name from users. The face_templates table is not
touched at all.
"""
from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from sqlalchemy import Select, func, select

from .models import ATTACK_TYPES, LABELS, ROLE_STUDENT, Attempt, User
from .vision.pipeline import MESSAGES

MODES = ("checkin", "enroll")
DECISIONS = ("pass", "fail", "retry")
REASONS = tuple(MESSAGES)


def label_filter_choices() -> list[tuple[str, str]]:
    """(value, text) for the label filter: any / unlabelled / labelled / each label / each attack type."""
    from .models import ATTACK_TYPE_NAMES, LABEL_NAMES
    out = [("", "any"), ("unlabelled", "not labelled yet"), ("labelled", "labelled (any)")]
    for lb in LABELS:
        out.append((lb, LABEL_NAMES[lb] + (" (all types)" if lb in ATTACK_TYPES else "")))
        out += [(f"{lb}/{t}", f"{LABEL_NAMES[lb]} - {ATTACK_TYPE_NAMES[t]}") for t in ATTACK_TYPES.get(lb, ())]
    return out


def parse_date(value: str | None) -> date | None:
    """YYYY-MM-DD -> date. Raises ValueError for anything else (empty -> None)."""
    if not value:
        return None
    return date.fromisoformat(value.strip())


def local_day_start_utc(d: date, tz: str) -> datetime:
    """Start of a local calendar day as naive UTC (how created_at is stored)."""
    return datetime.combine(d, time.min, tzinfo=ZoneInfo(tz)).astimezone(ZoneInfo("UTC")).replace(tzinfo=None)


@dataclass
class AttemptFilter:
    date_from: date | None = None      # local calendar days, inclusive
    date_to: date | None = None
    user_id: int | None = None
    mode: str = ""
    decision: str = ""
    reason: str = ""
    label: str = ""                    # see label_filter_choices()
    errors: list[str] = field(default_factory=list)

    @classmethod
    def from_args(cls, args) -> "AttemptFilter":
        """From a query string. Invalid values are ignored and reported in .errors (never trusted)."""
        f = cls()
        for name in ("date_from", "date_to"):
            try:
                setattr(f, name, parse_date(args.get(name)))
            except ValueError:
                f.errors.append(f"Ignored an invalid date: {args.get(name)!r} (use YYYY-MM-DD).")
        if args.get("student"):
            try:
                f.user_id = int(args.get("student"))
            except ValueError:
                f.errors.append("Ignored an invalid student.")
        for name, allowed in (("mode", MODES), ("decision", DECISIONS), ("reason", REASONS),
                              ("label", [v for v, _ in label_filter_choices()])):
            v = args.get(name, "")
            if v in allowed:
                setattr(f, name, v)
            elif v:
                f.errors.append(f"Ignored an invalid {name}: {v!r}.")
        return f

    def query_args(self) -> dict:
        """The active filters as query-string arguments (for links, paging and CSV)."""
        d = {"date_from": self.date_from.isoformat() if self.date_from else None,
             "date_to": self.date_to.isoformat() if self.date_to else None,
             "student": self.user_id, "mode": self.mode, "decision": self.decision,
             "reason": self.reason, "label": self.label}
        return {k: v for k, v in d.items() if v}

    @property
    def active(self) -> bool:
        return bool(self.query_args())

    def statement(self, tz: str) -> Select:
        """SELECT (Attempt, User) rows matching the filter (newest first)."""
        stmt = select(Attempt, User).join(User, Attempt.user_id == User.id)
        if self.date_from:
            stmt = stmt.where(Attempt.created_at >= local_day_start_utc(self.date_from, tz))
        if self.date_to:
            stmt = stmt.where(Attempt.created_at < local_day_start_utc(self.date_to + timedelta(days=1), tz))
        if self.user_id:
            stmt = stmt.where(Attempt.user_id == self.user_id)
        if self.mode:
            stmt = stmt.where(Attempt.mode == self.mode)
        if self.decision:
            stmt = stmt.where(Attempt.decision == self.decision)
        if self.reason:
            stmt = stmt.where(Attempt.reason == self.reason)
        if self.label == "unlabelled":
            stmt = stmt.where(Attempt.label.is_(None))
        elif self.label == "labelled":
            stmt = stmt.where(Attempt.label.is_not(None))
        elif self.label:
            lb, _, attack = self.label.partition("/")
            stmt = stmt.where(Attempt.label == lb)
            if attack:
                stmt = stmt.where(Attempt.attack_type == attack)
        return stmt.order_by(Attempt.id.desc())


def count(session, stmt: Select) -> int:
    return session.execute(select(func.count()).select_from(stmt.order_by(None).subquery())).scalar_one()


def students(session) -> list[User]:
    return session.execute(select(User).where(User.role == ROLE_STUDENT).order_by(User.username)).scalars().all()


# ------------------------------------------------------------------------------------- CSV
def pseudonym(user_id: int, secret: str) -> str:
    """Stable participant code for anonymised exports. Same student -> same code in every export of
    this installation; it cannot be turned back into the student without the server's secret key."""
    digest = hmac.new(secret.encode(), f"participant:{user_id}".encode(), hashlib.sha256).hexdigest()
    return "P-" + digest[:8]


CHECK_NAMES = ("consistency", "identity", "parallax", "parallax_landmarks", "parallax_flow", "depth", "texture")
TIMING_NAMES = ("detect", "embed", "parallax_landmarks", "parallax_flow", "depth", "texture")
# thresholds in force at the time of the attempt (they change after calibration)
THRESHOLD_NAMES = ("identity_threshold", "consistency_threshold", "parallax_method", "parallax_min_movement",
                   "parallax_threshold", "parallax_flow_min_movement", "parallax_flow_threshold",
                   "depth_relief_threshold", "depth_nose_threshold", "texture_threshold")
SCORE_COLUMNS = ("similarity", "consistency", "parallax_method", "parallax_status", "parallax_landmark_score",
                 "parallax_landmark_movement", "parallax_flow_score", "parallax_flow_movement",
                 "parallax_flow_points", "depth_relief", "depth_nose", "texture_score")


def csv_header(anonymise: bool) -> list[str]:
    who = ["participant"] if anonymise else ["student_id", "full_name"]
    return (["attempt_id", "time_local", "time_utc"] + who
            + ["mode", "decision", "reason", "failed_checks", "duplicate"] + list(SCORE_COLUMNS)
            + [f"check_{c}" for c in CHECK_NAMES]
            + ["processing_ms", "server_ms", "client_capture_ms", "end_to_end_ms"]
            + [f"time_{t}_ms" for t in TIMING_NAMES] + ["model_name"]
            + [f"thr_{t}" for t in THRESHOLD_NAMES]
            + ["label", "attack_type", "note"] + ([] if anonymise else ["labelled_by"]) + ["labelled_at_local"])


def _check(v) -> str:
    return "" if v is None else ("pass" if v else "fail")


def csv_row(a: Attempt, user: User, labeller: User | None, to_local, anonymise: bool, secret: str) -> list:
    t = a.timings_ms or {}
    e2e = None
    if t.get("client_capture_ms") is not None and t.get("server_ms") is not None:
        e2e = round(float(t["client_capture_ms"]) + float(t["server_ms"]), 1)
    checks, thr = a.checks or {}, a.thresholds or {}
    who = [pseudonym(user.id, secret)] if anonymise else [user.username, user.full_name]
    return ([a.id, to_local(a.created_at).strftime("%Y-%m-%d %H:%M:%S"), a.created_at.strftime("%Y-%m-%d %H:%M:%S")]
            + who + [a.mode, a.decision, a.reason, a.failed_checks, "yes" if a.duplicate else ""]
            + [getattr(a, c) for c in SCORE_COLUMNS]
            + [_check(checks.get(c)) for c in CHECK_NAMES]
            + [a.processing_ms, t.get("server_ms"), t.get("client_capture_ms"), e2e]
            + [t.get(n) for n in TIMING_NAMES] + [a.model_name]
            + [thr.get(n) for n in THRESHOLD_NAMES]
            + [a.label, a.attack_type, a.note]
            + ([] if anonymise else [labeller.username if labeller else None])
            + [to_local(a.labelled_at).strftime("%Y-%m-%d %H:%M:%S") if a.labelled_at else None])
