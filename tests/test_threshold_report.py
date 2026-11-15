"""Calibration analysis (app/calibration.py), Table I metrics, and scripts/threshold_report.py on a
real (temporary) database file."""
from __future__ import annotations

import hashlib
import importlib.util
from datetime import datetime
from pathlib import Path

import pytest

from app import create_app, vision_service
from app.attempt_log import AttemptFilter
from app.calibration import Rate, Record, analysis_lines, suggestions, table_one, threshold_changes
from app.config import VisionConfig
from app.extensions import db as _db
from app.models import ROLE_STUDENT, Attempt
from app.vision.pipeline import PipelineResult

from .conftest import STUDENT_PW, make_user

ROOT = Path(__file__).resolve().parent.parent
CFG = VisionConfig()


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def rec(label=None, attack=None, decision="pass", reason="ok", mode="checkin", identity=True, failed=(),
        lm=0.04, flow=0.03, relief=0.035, nose=0.1, tex=0.98, ms=800.0, e2e=None, user=1, thresholds=None):
    quality_stop = reason in ("face_too_close", "no_face")
    return Record(label=label, attack_type=attack, mode=mode, decision=decision, reason=reason,
                  failed_checks=list(failed), similarity=None if quality_stop or mode == "enroll" else 0.9,
                  identity_passed=None if quality_stop or mode == "enroll" else identity,
                  parallax_landmark_score=None if quality_stop else lm, parallax_flow_score=None if quality_stop else flow,
                  depth_relief=None if quality_stop else relief, depth_nose=None if quality_stop else nose,
                  texture_score=None if quality_stop else tex, processing_ms=ms, end_to_end_ms=e2e, user_id=user,
                  thresholds=thresholds)


# ------------------------------------------------------------------------------ suggestions / margins
def test_suggested_threshold_is_halfway_between_groups():
    rs = ([rec("genuine", tex=t) for t in (0.92, 0.95, 0.99)]
          + [rec("photo", "printed_flat", "fail", "liveness_failed", tex=t) for t in (0.10, 0.30)]
          + [rec("screen", "phone_video", "fail", "liveness_failed", tex=t) for t in (0.05, 0.62)])
    s = {x.field: x for x in suggestions(rs, CFG)}["texture_threshold"]
    assert s.margin == pytest.approx(0.92 - 0.62) and s.suggested == pytest.approx((0.92 + 0.62) / 2)
    assert s.per_group_margin == pytest.approx({"photo/printed_flat": 0.62, "screen/phone_video": 0.30})
    assert any("only 3 genuine" in w for w in s.warnings) and any("only 2 photo" in w for w in s.warnings)


def test_overlapping_groups_get_no_suggestion():
    rs = [rec("genuine", relief=r) for r in (0.028, 0.04)] + [rec("screen", "monitor_photo", relief=0.049)]
    s = {x.field: x for x in suggestions(rs, CFG)}["depth_relief_threshold"]
    assert s.suggested is None and s.margin < 0 and s.warnings[0].startswith("groups OVERLAP: 2/2 genuine")


def test_no_attacks_means_no_suggestion():
    s = suggestions([rec("genuine")], CFG)
    assert all(x.suggested is None and x.margin is None for x in s)


def test_analysis_text_has_all_sections():
    rs = [rec("genuine"), rec("photo", "printed_bent", "fail", "liveness_failed", failed=["texture"], tex=0.1),
          rec(None)]
    text = "\n".join(analysis_lines(rs, CFG))
    for part in ("Genuine (real student), n=1", "photo/printed_bent", "rejected by: {'texture': 1}",
                 "Unlabelled: 1 attempt(s)", "Score distributions", "Margin =", "Suggested thresholds",
                 "texture_threshold"):
        assert part in text, part


# ------------------------------------------------------------------------------ Table I
def test_wilson_interval():
    lo, hi = Rate(8, 10).wilson()
    assert lo == pytest.approx(0.4902, abs=1e-3) and hi == pytest.approx(0.9433, abs=1e-3)
    assert Rate(0, 0).wilson() is None and "n/a" in Rate(0, 0).text()
    lo, hi = Rate(10, 10).wilson()
    assert hi == 1.0 and lo == pytest.approx(0.722, abs=1e-3)


def test_table_one_numbers():
    rs = [
        # genuine check-ins: 4 accepted, 1 rejected by identity, 1 rejected by texture, 2 retries
        *[rec("genuine", user=u, e2e=3000.0 + 100 * u) for u in (1, 2, 3, 1)],
        rec("genuine", decision="fail", reason="identity_mismatch", identity=False, failed=["identity"], user=2),
        rec("genuine", decision="fail", reason="liveness_failed", failed=["texture"], tex=0.3, user=3),
        rec("genuine", decision="retry", reason="no_movement", user=1),
        rec("genuine", decision="retry", reason="face_too_close", ms=100.0, user=2),
        # genuine enrollment: no identity, counts for FRR / retry, not for identification or timing
        rec("genuine", mode="enroll", user=4),
        # attacks: photo 3 (1 accepted!), screen 2 (1 retry)
        rec("photo", "printed_flat", "fail", "liveness_failed", user=1),
        rec("photo", "printed_bent", "fail", "liveness_failed", user=1),
        rec("photo", "printed_flat", "pass", "ok", user=1),
        rec("screen", "phone_video", "fail", "liveness_failed", user=2),
        rec("screen", "monitor_photo", "retry", "no_movement", user=2),
        rec(None, user=5),                          # unlabelled: only counted as "any attempt"
    ]
    t = table_one(rs)
    assert (t.identification.k, t.identification.n) == (6, 7)        # 7 genuine check-ins with identity evaluated
    assert (t.photo_rejection.k, t.photo_rejection.n) == (2, 3)
    assert (t.screen_rejection.k, t.screen_rejection.n) == (2, 2) and t.attack_retry == {"photo": 0, "screen": 1}
    assert (t.false_rejection.k, t.false_rejection.n) == (2, 7)      # 2 fails among 5+2 decided genuine
    assert (t.genuine_retry.k, t.genuine_retry.n) == (2, 9)
    assert (t.genuine_accepted.k, t.genuine_accepted.n) == (5, 9)
    # timing: every check-in that ran the whole pipeline (labelled or not: 7 genuine + 5 attacks + 1);
    # the face_too_close one (100 ms) stopped early and is left out
    assert 100.0 not in t.server_time.values_ms and len(t.server_time.values_ms) == 13
    assert sorted(t.end_to_end_time.values_ms) == [3100.0, 3100.0, 3200.0, 3300.0]
    assert t.participants == 4 and t.students_any == 5


def test_record_from_pipeline_result_and_from_attempt():
    r = PipelineResult(decision="fail", reason="liveness_failed", message="", mode="checkin",
                       checks={"identity": True, "texture": False}, failed_checks=["texture"], similarity=0.8,
                       parallax_landmark_score=0.03, texture_score=0.2, processing_ms=700.0)
    x = Record.from_result(r, "screen", "phone_photo", name="cap1")
    assert (x.group, x.identity_passed, x.failed_checks, x.texture_score) == ("screen/phone_photo", True, ["texture"], 0.2)
    assert Record.from_result(r, "unlabelled", None).label is None

    a = Attempt(id=7, user_id=3, mode="checkin", decision="fail", reason="liveness_failed",
                failed_checks="texture;depth", checks={"identity": True}, label="photo", attack_type=None,
                timings_ms={"server_ms": 900.0, "client_capture_ms": 2500.0}, processing_ms=800.0)
    y = Record.from_attempt(a)
    assert (y.group, y.failed_checks, y.end_to_end_ms, y.name) == ("photo/unspecified", ["texture", "depth"], 3400.0, "#7")


def test_threshold_changes_are_reported():
    rs = [rec(thresholds={"texture_threshold": 0.5, "parallax_threshold": 0.025}),
          rec(thresholds={"texture_threshold": 0.7, "parallax_threshold": 0.025}),
          rec(thresholds=None)]
    assert threshold_changes(rs, {"texture_threshold": 0.5, "parallax_threshold": 0.025}) == {
        "texture_threshold": (1, {0.7})}
    assert threshold_changes(rs[:1], {"texture_threshold": 0.5, "parallax_threshold": 0.025}) == {}
    assert threshold_changes(rs, {"texture_threshold": 0.6, "parallax_threshold": 0.025}) == {
        "texture_threshold": (2, {0.5, 0.7})}


# ------------------------------------------------------------------------------ the report on the database
def add(db, user, label=None, attack=None, decision="pass", reason="ok", mode="checkin", tex=0.98, **kw):
    a = Attempt(user_id=user.id, created_at=kw.pop("created_at", datetime(2026, 11, 15, 9, 0)), mode=mode,
                decision=decision, reason=reason, label=label, attack_type=attack, similarity=0.9,
                parallax_landmark_score=0.04, parallax_flow_score=0.03, depth_relief=0.035, depth_nose=0.1,
                texture_score=tex, checks={"identity": True}, processing_ms=800.0,
                timings_ms={"server_ms": 850.0, "client_capture_ms": 2650.0},
                thresholds=vision_service.thresholds_snapshot(), **kw)
    db.session.add(a)
    db.session.commit()
    return a


def fill(db):
    s1 = make_user(ROLE_STUDENT, "2024/0001", STUDENT_PW)
    s2 = make_user(ROLE_STUDENT, "2024/0002", STUDENT_PW)
    for _ in range(3):
        add(db, s1, "genuine", tex=0.95)
    add(db, s2, "genuine", tex=0.91)
    add(db, s1, "photo", "printed_flat", "fail", "liveness_failed", tex=0.10, failed_checks="texture")
    add(db, s2, "screen", "phone_video", "fail", "liveness_failed", tex=0.61, failed_checks="texture")
    add(db, s2, "screen", "phone_video", "fail", "liveness_failed", tex=0.40, mode="enroll", failed_checks="texture")
    add(db, s2, None, created_at=datetime(2026, 11, 1, 9, 0))   # unlabelled, outside a date filter
    return s1, s2


def test_report_text(app, db):
    fill(db)
    load = load_script("threshold_report")
    text = load.report(db.session, "Europe/Skopje", AttemptFilter(), vision_service.vision_config(),
                       vision_service.thresholds_snapshot(), list_attempts=True)
    for part in ("Attempts in the selection: 8  (genuine 4, photo 1, screen 2, unlabelled 1)",
                 "texture_threshold          current 0.5      -> 0.7600",       # (0.91 + 0.61) / 2
                 "Verified identification accuracy           100.0%  (4/4;",
                 "Printed-photo attack rejection rate        100.0%  (1/1;",
                 "Screen-replay attack rejection rate        100.0%  (2/2;",
                 "False rejection rate (legitimate students)   0.0%  (0/4;",
                 "Average verification time (server)          0.80 s",
                 "Average end-to-end verification time        3.50 s",
                 "Pilot participants                         2 students",
                 "2024/0001", "Definitions"):
        assert part in text, part
    assert "NOTE: some attempts were made with other thresholds" not in text


def test_report_filters_and_empty_selection(app, db):
    s1, _ = fill(db)
    load = load_script("threshold_report")
    cfg, snap = vision_service.vision_config(), vision_service.thresholds_snapshot()
    text = load.report(db.session, "Europe/Skopje", AttemptFilter(mode="checkin", user_id=s1.id), cfg, snap)
    assert "Attempts in the selection: 4  (genuine 3, photo 1)" in text
    text = load.report(db.session, "Europe/Skopje", AttemptFilter(date_to=datetime(2026, 10, 1).date()), cfg, snap)
    assert "Attempts in the selection: 0" in text and "Nothing to analyse" in text


def test_report_notes_changed_thresholds(app, db):
    fill(db)
    load = load_script("threshold_report")
    snap = dict(vision_service.thresholds_snapshot(), texture_threshold=0.7)
    text = load.report(db.session, "Europe/Skopje", AttemptFilter(), vision_service.vision_config(), snap)
    assert "NOTE: some attempts were made with other thresholds" in text and "texture_threshold" in text


def test_script_end_to_end_on_a_database_file_is_read_only(tmp_path, capsys):
    db_file = tmp_path / "attendance.db"
    app = create_app("testing", overrides={"SQLALCHEMY_DATABASE_URI": f"sqlite:///{db_file.as_posix()}"})
    with app.app_context():
        _db.create_all()
        fill(_db)
        _db.session.remove()
        _db.engine.dispose()
    before = hashlib.sha256(db_file.read_bytes()).hexdigest()

    load = load_script("threshold_report")
    load.main(["--db", str(db_file), "--mode", "checkin", "--from", "2026-11-15", "--to", "2026-11-15"])
    out = capsys.readouterr().out
    assert "Threshold report - date_from=2026-11-15, date_to=2026-11-15, mode=checkin" in out
    assert "Attempts in the selection: 6  (genuine 4, photo 1, screen 1)" in out and "Table I" in out
    load.main(["--db", str(db_file), "--student", "2024/0002"])
    assert "Attempts in the selection: 4" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        load.main(["--db", str(db_file), "--student", "nobody"])
    with pytest.raises(SystemExit):
        load.main(["--db", str(tmp_path / "missing.db")])
    assert not (tmp_path / "missing.db").exists()
    assert hashlib.sha256(db_file.read_bytes()).hexdigest() == before        # nothing was written


def test_replay_captures_still_imports():
    mod = load_script("replay_captures")
    assert callable(mod.main) and mod.Record is Record
