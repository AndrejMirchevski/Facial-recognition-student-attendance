"""Database rules: one attendance per student per day (enforced by the DB), attempts store every score."""
import datetime as dt

import pytest
from sqlalchemy.exc import IntegrityError

from app.models import ATTACK_TYPES, Attempt, Attendance
from app.vision.pipeline import FrameReport, PipelineResult

from .conftest import STUDENT_PW, make_user


def _attempt(db, user):
    a = Attempt(user_id=user.id, mode="checkin", decision="pass", reason="ok")
    db.session.add(a)
    db.session.commit()
    return a


def _attendance(user, attempt, day):
    return Attendance(user_id=user.id, date=day, time=dt.time(9, 0), attempt_id=attempt.id)


def test_second_attendance_same_day_is_rejected_by_database(db, student):
    day = dt.date(2026, 11, 14)
    db.session.add(_attendance(student, _attempt(db, student), day))
    db.session.commit()
    db.session.add(_attendance(student, _attempt(db, student), day))
    with pytest.raises(IntegrityError):
        db.session.commit()
    db.session.rollback()


def test_attendance_on_other_days_and_for_other_students_is_fine(db, student):
    other = make_user("student", "s1002", STUDENT_PW)
    day = dt.date(2026, 11, 14)
    db.session.add_all([
        _attendance(student, _attempt(db, student), day),
        _attendance(student, _attempt(db, student), day + dt.timedelta(days=1)),
        _attendance(other, _attempt(db, other), day),
    ])
    db.session.commit()
    assert db.session.query(Attendance).count() == 3


def _full_result() -> PipelineResult:
    r = PipelineResult(decision="fail", reason="liveness_failed", message="x", mode="checkin",
                       checks={"identity": True, "texture": False}, failed_checks=["texture"],
                       similarity=0.81, consistency=0.9, parallax_method="landmarks", parallax_status="pass",
                       parallax_landmark_score=0.041, parallax_landmark_movement=0.06,
                       parallax_flow_score=0.03, parallax_flow_movement=0.07, parallax_flow_points=120,
                       depth_relief=0.03, depth_nose=0.1, texture_score=0.2,
                       frames=[FrameReport(index=0, brightness=120.0)], timings_ms={"detect": 50.0},
                       processing_ms=700.0)
    return r


def test_every_pipeline_score_has_a_column():
    """If the pipeline gains a new score, this fails until the attempts table stores it too."""
    summary_keys = set(_full_result().summary())
    stored = set(Attempt.SCORE_FIELDS) | {"failed_checks"} | set(Attempt.IGNORED_FIELDS)
    assert summary_keys <= stored, f"not stored: {summary_keys - stored}"


def test_attempt_from_summary_round_trip(db, student):
    a = Attempt.from_summary(student.id, _full_result().summary(), thresholds={"texture_threshold": 0.5})
    db.session.add(a)
    db.session.commit()
    db.session.refresh(a)
    assert a.parallax_landmark_score == 0.041 and a.parallax_flow_points == 120
    assert a.depth_relief == 0.03 and a.texture_score == 0.2 and a.similarity == 0.81
    assert a.failed_checks == "texture" and a.checks["texture"] is False
    assert a.frames[0]["brightness"] == 120.0 and a.timings_ms["detect"] == 50.0
    assert a.thresholds == {"texture_threshold": 0.5}


def test_attempt_labels(db, student, professor):
    a = _attempt(db, student)
    a.set_label("screen", "phone_video", by=professor)
    db.session.commit()
    assert (a.label, a.attack_type, a.labelled_by_id) == ("screen", "phone_video", professor.id)
    with pytest.raises(ValueError):
        a.set_label("genuine", "phone_video", by=professor)
    with pytest.raises(ValueError):
        a.set_label("banana", None, by=professor)
    assert set(ATTACK_TYPES) == {"photo", "screen"}
