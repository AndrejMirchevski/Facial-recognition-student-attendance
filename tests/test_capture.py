"""Enrollment, check-in, duplicate rule, wrong person, consent, face-data deletion, upload checks.

The vision pipeline is mocked: `camera.person` says who is in front of the camera, and the mock
compares that person's (fake) face embedding with the template it is given - so identity, the
1:1 rule and wrong-person cases behave like the real system without loading any model.
"""
from __future__ import annotations

import io
import struct
import zlib

import cv2
import numpy as np
import pytest
from sqlalchemy import select

from app import vision_service
from app.models import ROLE_STUDENT, Attempt, Attendance, FaceTemplate
from app.vision.pipeline import MESSAGES, FrameReport, PipelineResult

from .conftest import PROF_PW, STUDENT_PW, csrf_from, login, make_user


def face_vector(name: str) -> np.ndarray:
    v = np.random.default_rng(abs(hash(name)) % 2**32).normal(size=512).astype(np.float32)
    return v / np.linalg.norm(v)


class FakeCamera:
    """Who/what is in front of the camera, and what the liveness checks will say."""

    def __init__(self):
        self.person = "s1001"
        self.liveness = "pass"          # pass | fail | retry
        self.calls: list[dict] = []

    def analyse(self, frames, template):
        self.calls.append({"n": len(frames), "shape": frames[0].shape, "template": template})
        emb = face_vector(self.person)
        mode = "enroll" if template is None else "checkin"
        r = PipelineResult(decision="pass", reason="ok", message="", mode=mode,
                           consistency=0.9, parallax_method="landmarks", parallax_status="pass",
                           parallax_landmark_score=0.04, parallax_landmark_movement=0.06,
                           parallax_flow_score=0.03, parallax_flow_movement=0.07, parallax_flow_points=120,
                           depth_relief=0.035, depth_nose=0.1, texture_score=0.97,
                           frames=[FrameReport(index=i) for i in range(5)], timings_ms={"detect": 50.0},
                           processing_ms=650.0, embedding=emb, frame_embeddings=[emb] * 5)
        r.checks = {"consistency": True, "identity": None, "parallax": True, "depth": True, "texture": True}
        if template is not None:
            r.similarity = float(np.dot(emb, template))
            r.checks["identity"] = r.similarity >= vision_service.vision_config().identity_threshold
        if self.liveness == "fail":
            r.checks["texture"], r.texture_score = False, 0.1
        if self.liveness == "retry":
            r.checks["parallax"], r.parallax_status = None, "insufficient_movement"
        from app.vision.pipeline import decide
        decide(r, vision_service.vision_config())
        r.message = MESSAGES[r.reason]
        return r


@pytest.fixture
def camera(monkeypatch):
    cam = FakeCamera()
    monkeypatch.setattr(vision_service, "analyse", cam.analyse)
    return cam


def jpeg(w=640, h=480, seed=0) -> bytes:
    img = np.random.default_rng(seed).integers(0, 255, (h, w, 3), dtype=np.uint8)
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()


def frames_payload(n=5, **kw):
    return [(io.BytesIO(jpeg(seed=i, **kw)), f"frame_{i}.jpg") for i in range(n)]


def post_enroll(client, consent=True, frames=None):
    data = {"frames": frames if frames is not None else frames_payload()}
    if consent:
        data["consent"] = "yes"
    return client.post("/api/enroll", data=data, content_type="multipart/form-data")


def post_checkin(client, frames=None):
    return client.post("/api/checkin", data={"frames": frames if frames is not None else frames_payload()},
                       content_type="multipart/form-data")


@pytest.fixture
def new_student(app):
    """Logged-in student who changed the password but is NOT enrolled yet."""
    return make_user(ROLE_STUDENT, "s1001", STUDENT_PW, enrolled=False)


def attempts(db):
    return db.session.execute(select(Attempt).order_by(Attempt.id)).scalars().all()


# ------------------------------------------------------------------------------- enrollment
def test_enrollment_requires_consent(client, db, camera, new_student):
    login(client, "s1001", STUDENT_PW)
    r = post_enroll(client, consent=False)
    assert r.status_code == 400 and r.get_json()["error"] == "consent_required"
    assert not camera.calls and not attempts(db)
    assert db.session.execute(select(FaceTemplate)).first() is None


def test_enrollment_stores_template_with_consent_time(client, db, camera, new_student):
    login(client, "s1001", STUDENT_PW)
    r = post_enroll(client)
    body = r.get_json()
    assert r.status_code == 200 and body["decision"] == "pass"
    assert camera.calls[0]["template"] is None and camera.calls[0]["n"] == 5
    tpl = db.session.execute(select(FaceTemplate)).scalar_one()
    assert tpl.user_id == new_student.id and tpl.consent_at is not None
    assert np.allclose(np.frombuffer(tpl.template, np.float32), face_vector("s1001"))
    assert tpl.frame_embeddings is not None and len(tpl.frame_embeddings) == 5 * 512 * 4
    a = attempts(db)[0]
    assert (a.mode, a.decision, a.user_id) == ("enroll", "pass", new_student.id)
    assert tpl.enroll_attempt_id == a.id
    assert client.get("/student/").status_code == 200          # now allowed in


def test_enrollment_needs_liveness(client, db, camera, new_student):
    login(client, "s1001", STUDENT_PW)
    camera.liveness = "fail"
    body = post_enroll(client).get_json()
    assert body["decision"] == "fail" and body["reason"] == "liveness_failed"
    assert db.session.execute(select(FaceTemplate)).first() is None
    a = attempts(db)[0]
    assert a.decision == "fail" and "texture" in a.failed_checks


def test_enrollment_when_already_enrolled_is_refused(client, db, camera, student):
    login(client, "s1001", STUDENT_PW)
    r = post_enroll(client)
    assert r.status_code == 409 and not camera.calls


# ------------------------------------------------------------------------------- check-in
def enroll(client, camera, person="s1001"):
    camera.person = person
    assert post_enroll(client).get_json()["decision"] == "pass"


def test_checkin_writes_attendance(client, db, camera, new_student):
    login(client, "s1001", STUDENT_PW)
    enroll(client, camera)
    body = post_checkin(client).get_json()
    assert body["decision"] == "pass" and body["attendance"] is not None
    row = db.session.execute(select(Attendance)).scalar_one()
    a = attempts(db)[-1]
    assert row.user_id == new_student.id and row.attempt_id == a.id and a.mode == "checkin"
    assert a.similarity > 0.99 and not a.duplicate
    assert body["attendance"]["date"] == row.date.isoformat()


def test_second_checkin_same_day_is_not_recorded(client, db, camera, new_student):
    login(client, "s1001", STUDENT_PW)
    enroll(client, camera)
    post_checkin(client)
    body = post_checkin(client).get_json()
    assert body["decision"] == "pass" and body.get("duplicate") is True and body["attendance"] is None
    assert "already checked in" in body["message"]
    assert len(db.session.execute(select(Attendance)).all()) == 1
    assert attempts(db)[-1].duplicate is True
    # the check-in page itself sends the student back home
    r = client.get("/student/checkin")
    assert r.status_code == 302 and r.headers["Location"] == "/student/"


def test_duplicate_race_is_stopped_by_the_database(client, db, camera, new_student, monkeypatch):
    """Even if the "already checked in?" lookup misses (two requests at once), the UNIQUE
    constraint refuses the second row and the attempt is marked as duplicate."""
    login(client, "s1001", STUDENT_PW)
    enroll(client, camera)
    post_checkin(client)
    from app import attendance
    real = attendance.todays_attendance
    calls = {"n": 0}

    def first_call_misses(user):
        calls["n"] += 1
        return None if calls["n"] == 1 else real(user)
    monkeypatch.setattr(attendance, "todays_attendance", first_call_misses)
    body = post_checkin(client).get_json()
    assert body.get("duplicate") is True
    assert len(db.session.execute(select(Attendance)).all()) == 1


def test_wrong_person_is_rejected(client, db, camera, new_student):
    login(client, "s1001", STUDENT_PW)
    enroll(client, camera, person="s1001")
    camera.person = "someone-else"                   # another person uses this student's account
    body = post_checkin(client).get_json()
    assert body["decision"] == "fail" and body["reason"] == "identity_mismatch"
    assert db.session.execute(select(Attendance)).first() is None
    a = attempts(db)[-1]
    assert a.decision == "fail" and a.similarity < 0.4 and "identity" in a.failed_checks


def test_checkin_is_one_to_one_against_own_template(app, db, camera):
    """Each student is compared with THEIR template only - never searched among all students."""
    alice = make_user(ROLE_STUDENT, "alice", STUDENT_PW)
    make_user(ROLE_STUDENT, "bob", STUDENT_PW)
    ca, cb = app.test_client(), app.test_client()
    login(ca, "alice", STUDENT_PW)
    login(cb, "bob", STUDENT_PW)
    enroll(ca, camera, "alice")
    enroll(cb, camera, "bob")
    camera.person = "alice"                          # Alice stands in front of Bob's session
    body = post_checkin(cb).get_json()
    assert body["reason"] == "identity_mismatch"
    assert np.allclose(camera.calls[-1]["template"], face_vector("bob"))
    camera.person = "alice"
    assert post_checkin(ca).get_json()["decision"] == "pass"
    rows = db.session.execute(select(Attendance)).scalars().all()
    assert [r.user_id for r in rows] == [alice.id]


def test_retry_is_logged_without_attendance(client, db, camera, new_student):
    login(client, "s1001", STUDENT_PW)
    enroll(client, camera)
    camera.liveness = "retry"
    body = post_checkin(client).get_json()
    assert body["decision"] == "retry" and body["reason"] == "no_movement"
    assert db.session.execute(select(Attendance)).first() is None
    assert attempts(db)[-1].decision == "retry"


def test_checkin_without_enrollment_is_refused(client, db, camera, new_student):
    login(client, "s1001", STUDENT_PW)
    r = post_checkin(client)
    assert r.status_code == 409 and r.get_json()["error"] == "not_enrolled" and not camera.calls
    assert client.get("/student/checkin").headers["Location"] == "/student/enroll"


def test_every_attempt_stores_all_scores_and_thresholds(client, db, camera, new_student):
    login(client, "s1001", STUDENT_PW)
    enroll(client, camera)
    post_checkin(client)
    a = attempts(db)[-1]
    assert a.parallax_landmark_score == 0.04 and a.parallax_flow_score == 0.03 and a.parallax_flow_points == 120
    assert a.depth_relief == 0.035 and a.depth_nose == 0.1 and a.texture_score == 0.97
    assert a.timings_ms["detect"] == 50.0 and a.processing_ms == 650.0 and len(a.frames) == 5
    assert a.timings_ms["server_ms"] >= 0 and "client_capture_ms" not in a.timings_ms   # browser sent none
    assert a.thresholds["identity_threshold"] == vision_service.vision_config().identity_threshold
    assert a.label is None and a.attack_type is None


@pytest.mark.parametrize("sent, stored", [("2875.4", 2875.4), ("abc", None), ("-5", None), ("inf", None),
                                          ("nan", None), ("999999", None)])
def test_browser_capture_time_is_stored_only_when_plausible(client, db, camera, new_student, sent, stored):
    """The browser reports how long the guided capture took (end-to-end time for the paper)."""
    login(client, "s1001", STUDENT_PW)
    enroll(client, camera)
    client.post("/api/checkin", data={"frames": frames_payload(), "capture_ms": sent},
                content_type="multipart/form-data")
    t = attempts(db)[-1].timings_ms
    assert t.get("client_capture_ms") == stored and t["server_ms"] >= 0 and t["detect"] == 50.0


# ------------------------------------------------------------------------------- deletion
def test_delete_face_data_blocks_checkin_until_reenrolled(client, db, camera, new_student):
    login(client, "s1001", STUDENT_PW)
    enroll(client, camera)
    post_checkin(client)
    r = client.post("/student/face-data/delete")
    assert r.status_code == 302 and r.headers["Location"] == "/student/enroll"
    assert db.session.execute(select(FaceTemplate)).first() is None
    assert len(db.session.execute(select(Attendance)).all()) == 1           # history kept
    assert post_checkin(client).status_code == 409
    assert client.get("/student/").headers["Location"] == "/student/enroll"
    enroll(client, camera)                                                  # can enroll again
    assert db.session.execute(select(FaceTemplate)).scalar_one().user_id == new_student.id


def test_delete_face_data_needs_post_and_csrf(app, client, db, camera, student):
    login(client, "s1001", STUDENT_PW)
    assert client.get("/student/face-data/delete").status_code == 405
    app.config["WTF_CSRF_ENABLED"] = True
    assert client.post("/student/face-data/delete").status_code == 400
    assert db.session.execute(select(FaceTemplate)).first() is not None


# ------------------------------------------------------------------------------- upload checks
def test_wrong_number_of_frames(client, camera, new_student):
    login(client, "s1001", STUDENT_PW)
    r = post_enroll(client, frames=frames_payload(3))
    assert r.status_code == 400 and r.get_json()["error"] == "invalid_frames" and not camera.calls


def test_not_an_image(client, camera, new_student):
    login(client, "s1001", STUDENT_PW)
    bad = frames_payload(4) + [(io.BytesIO(b"<html>not an image</html>"), "frame_4.jpg")]
    assert post_enroll(client, frames=bad).status_code == 400 and not camera.calls


def test_huge_dimensions_rejected_before_decoding(client, camera, new_student):
    """A tiny PNG claiming 20000x20000 pixels (decompression bomb) is refused from its header."""
    def png(w, h):
        def chunk(t, d):
            return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d))
        return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                + chunk(b"IDAT", zlib.compress(b"\x00")) + chunk(b"IEND", b""))
    login(client, "s1001", STUDENT_PW)
    bad = frames_payload(4) + [(io.BytesIO(png(20000, 20000)), "frame_4.png")]
    r = post_enroll(client, frames=bad)
    assert r.status_code == 400 and "unsupported size" in r.get_json()["message"] and not camera.calls


def test_frames_of_different_sizes_rejected(client, camera, new_student):
    login(client, "s1001", STUDENT_PW)
    mixed = frames_payload(4) + [(io.BytesIO(jpeg(w=800, h=600)), "frame_4.jpg")]
    assert post_enroll(client, frames=mixed).status_code == 400


def test_rate_limit(app, client, db, camera, new_student):
    app.config["CAPTURE_MAX_ATTEMPTS"] = 3
    login(client, "s1001", STUDENT_PW)
    camera.liveness = "fail"
    for _ in range(3):
        assert post_enroll(client).status_code == 200
    r = post_enroll(client)
    assert r.status_code == 429 and len(camera.calls) == 3


# ------------------------------------------------------------------------------- access rules
def test_api_access_rules(client, camera, professor):
    assert post_checkin(client).status_code == 401                 # anonymous: JSON 401
    login(client, "prof", PROF_PW)
    r = post_checkin(client)
    assert r.status_code == 403 and r.is_json and not camera.calls  # professors cannot check in


def test_api_requires_csrf_token(app, client, camera, new_student):
    login(client, "s1001", STUDENT_PW)
    app.config["WTF_CSRF_ENABLED"] = True
    r = post_enroll(client)
    assert r.status_code == 400 and r.get_json()["error"] == "csrf" and not camera.calls
    token = csrf_from(client.get("/student/enroll").get_data(as_text=True))
    data = {"frames": frames_payload(), "consent": "yes"}
    r = client.post("/api/enroll", data=data, headers={"X-CSRFToken": token}, content_type="multipart/form-data")
    assert r.status_code == 200


def test_wasm_csp_only_on_capture_pages(client, new_student):
    login(client, "s1001", STUDENT_PW)
    enroll_page = client.get("/student/enroll")
    assert "'wasm-unsafe-eval'" in enroll_page.headers["Content-Security-Policy"]
    assert "'unsafe-eval'" not in enroll_page.headers["Content-Security-Policy"].replace("'wasm-unsafe-eval'", "")
    assert "'wasm-unsafe-eval'" not in client.get("/change-password").headers["Content-Security-Policy"]
    assert b'data-settings="' in enroll_page.data and b"guide_turn_target" in enroll_page.data


def test_landmarker_model_is_served_to_students_only(client, new_student):
    from pathlib import Path
    if not (Path(vision_service.vision_config().models_dir) / "face_landmarker.task").exists():
        pytest.skip("model not downloaded")
    assert client.get("/student/assets/face_landmarker.task").status_code == 302
    login(client, "s1001", STUDENT_PW)
    r = client.get("/student/assets/face_landmarker.task")
    assert r.status_code == 200 and len(r.data) > 1_000_000
    r.close()


# ------------------------------------------------------------------------------- professor form
def test_student_id_spaces_removed_and_slash_allowed(client, db, professor):
    from app.models import User
    login(client, "prof", PROF_PW)
    r = client.post("/professor/students", data={"student_id": " 2023 / 0045 ", "full_name": "  Ana Petrova "})
    assert r.status_code == 200
    u = db.session.execute(select(User).where(User.username == "2023/0045")).scalar_one()
    assert u.full_name == "Ana Petrova"


def test_form_errors_shown_at_field_with_values_kept(client, db, professor, student):
    login(client, "prof", PROF_PW)
    r = client.post("/professor/students", data={"student_id": "S1001", "full_name": "Typed Name",
                                                 "initial_password": "Abc-12345"})
    html = r.get_data(as_text=True)
    assert r.status_code == 400
    assert 'value="Typed Name"' in html and 'value="S1001"' in html and 'value="Abc-12345"' in html
    # the error is inside the Student ID field's label block
    sid_block = html.split('name="student_id"')[1].split("</label>")[0]
    assert "already exists" in sid_block
    r = client.post("/professor/students", data={"student_id": "s7777", "full_name": "X Y",
                                                 "initial_password": "short"})
    pw_block = r.get_data(as_text=True).split('name="initial_password"')[1].split("</label>")[0]
    assert r.status_code == 400 and "at least" in pw_block


def test_dashboard_form_cannot_be_autofilled_as_login(client, professor):
    login(client, "prof", PROF_PW)
    html = client.get("/professor/").get_data(as_text=True)
    form = html.split('action="/professor/students"')[1].split("</form>")[0]
    assert 'type="password"' not in form and 'autocomplete="off"' in form
