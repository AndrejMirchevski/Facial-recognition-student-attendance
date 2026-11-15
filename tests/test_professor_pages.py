"""Professor pages of stage 5: attendance by date, attendance CSV, attempts log (filters, paging,
detail), labelling (single + bulk, notes), attempts CSV (incl. anonymised) - and that none of it
ever exposes biometric data (templates / embeddings / images)."""
from __future__ import annotations

import base64
import csv
import io
from datetime import date, datetime, time

import numpy as np
import pytest
from sqlalchemy import event, select

from app.models import ROLE_STUDENT, Attempt, Attendance, FaceTemplate, User
from app.attempt_log import pseudonym

from .conftest import PROF_PW, STUDENT_PW, csrf_from, login, make_user

# November: Europe/Skopje = UTC+1


@pytest.fixture
def prof_client(client, professor):
    login(client, "prof", PROF_PW)
    return client


@pytest.fixture
def ana(app):
    return make_user(ROLE_STUDENT, "2024/0001", STUDENT_PW, enrolled=True)


@pytest.fixture
def bob(app):
    return make_user(ROLE_STUDENT, "2024/0002", STUDENT_PW, enrolled=True)


def add_attempt(db, user, created_at=datetime(2026, 11, 15, 9, 0), mode="checkin", decision="pass", reason="ok",
                label=None, attack_type=None, note=None, **scores):
    a = Attempt(user_id=user.id, created_at=created_at, mode=mode, decision=decision, reason=reason,
                failed_checks=scores.pop("failed_checks", ""), label=label, attack_type=attack_type, note=note,
                similarity=scores.pop("similarity", 0.9 if mode == "checkin" else None), consistency=0.95,
                parallax_method="landmarks", parallax_status="pass", parallax_landmark_score=0.0412,
                parallax_landmark_movement=0.07, parallax_flow_score=0.0311, parallax_flow_movement=0.08,
                parallax_flow_points=130, depth_relief=0.035, depth_nose=0.1, texture_score=0.98,
                checks={"consistency": True, "identity": True if mode == "checkin" else None, "parallax": True,
                        "parallax_landmarks": True, "parallax_flow": True, "depth": True, "texture": True},
                frames=[{"index": i, "problem": None, "faces": 1, "ignored_faces": 0, "interocular_px": 80.0,
                         "face_height_frac": 0.4, "yaw_proxy": 0.01, "brightness": 110.0, "sharpness": 500.0,
                         "similarity": 0.9, "texture": [0.97, 0.99], "depth_relief": 0.03, "depth_nose": 0.1}
                        for i in range(5)],
                timings_ms={"detect": 60.0, "depth": 400.0, "server_ms": 900.0, "client_capture_ms": 2600.0},
                processing_ms=850.0, model_name="arcface-w600k_r50",
                thresholds={"identity_threshold": 0.4, "texture_threshold": 0.5, "parallax_threshold": 0.025})
    for k, v in scores.items():
        setattr(a, k, v)
    db.session.add(a)
    db.session.commit()
    return a


def read_csv(resp) -> tuple[list[str], list[dict]]:
    text = resp.get_data(as_text=True)
    assert text.startswith("﻿")                     # BOM: Excel shows Cyrillic correctly
    rows = list(csv.reader(io.StringIO(text[1:])))
    return rows[0], [dict(zip(rows[0], r)) for r in rows[1:]]


# ------------------------------------------------------------------------------ roles
PROF_GETS = ["/professor/attendance", "/professor/attendance.csv?date=2026-11-15", "/professor/attempts",
             "/professor/attempts.csv", "/professor/attempts/1"]


@pytest.mark.parametrize("url", PROF_GETS)
def test_students_cannot_open_professor_pages(client, db, ana, url):
    add_attempt(db, ana)
    login(client, "2024/0001", STUDENT_PW)
    assert client.get(url).status_code == 403


def test_students_cannot_label(client, db, ana):
    a = add_attempt(db, ana)
    login(client, "2024/0001", STUDENT_PW)
    r = client.post("/professor/attempts/label", data={"ids": a.id, "label": "genuine"})
    assert r.status_code == 403 and db.session.get(Attempt, a.id).label is None


@pytest.mark.parametrize("url", PROF_GETS)
def test_anonymous_is_sent_to_login(client, url):
    r = client.get(url)
    assert r.status_code == 302 and "/login" in r.headers["Location"]


def test_professor_navigation(prof_client):
    html = prof_client.get("/professor/attempts").get_data(as_text=True)
    for href in ("/professor/", "/professor/attendance", "/professor/attempts"):
        assert f'href="{href}"' in html
    assert 'class="current" aria-current="page">Attempts' in html


# ------------------------------------------------------------------------------ attendance by date
def add_attendance(db, user, d=date(2026, 11, 15), t=time(9, 15, 7)):
    a = add_attempt(db, user, created_at=datetime.combine(d, t))
    db.session.add(Attendance(user_id=user.id, date=d, time=t, attempt_id=a.id))
    db.session.commit()
    return a


def test_attendance_by_date_shows_who_and_when(prof_client, db, ana, bob):
    add_attendance(db, ana)
    html = prof_client.get("/professor/attendance?date=2026-11-15").get_data(as_text=True)
    assert "Sunday, 15 November 2026" in html
    assert "09:15:07" in html and "2024/0001" in html
    assert "Not checked in (1)" in html and "2024/0002" in html.split("Not checked in")[1]
    assert 'href="/professor/attendance?date=2026-11-14"' in html and 'href="/professor/attendance?date=2026-11-16"' in html

    other = prof_client.get("/professor/attendance?date=2026-11-16").get_data(as_text=True)
    assert "Nobody checked in on this day." in other and "Not checked in (2)" in other


def test_attendance_bad_date_falls_back_to_today(prof_client):
    r = prof_client.get("/professor/attendance?date=15.11.2026")
    assert r.status_code == 200 and "not a valid date" in r.get_data(as_text=True)


def test_attendance_csv_one_day_and_range(prof_client, db, ana, bob):
    add_attendance(db, ana, date(2026, 11, 14), time(10, 0, 0))
    add_attendance(db, bob, date(2026, 11, 15), time(8, 30, 0))
    add_attendance(db, ana, date(2026, 11, 15), time(9, 0, 0))

    r = prof_client.get("/professor/attendance.csv?date=2026-11-15")
    assert r.mimetype == "text/csv" and 'filename="attendance_2026-11-15.csv"' in r.headers["Content-Disposition"]
    header, rows = read_csv(r)
    assert header == ["date", "time", "student_id", "full_name", "attempt_id"]
    assert [(x["time"], x["student_id"]) for x in rows] == [("08:30:00", "2024/0002"), ("09:00:00", "2024/0001")]

    # range, given the wrong way round: still works
    header, rows = read_csv(prof_client.get("/professor/attendance.csv?date_from=2026-11-15&date_to=2026-11-14"))
    assert [x["date"] for x in rows] == ["2026-11-14", "2026-11-15", "2026-11-15"]


@pytest.mark.parametrize("qs", ["", "?date=yesterday", "?date_from=2026-11-01", "?date_from=2020-01-01&date_to=2026-01-01"])
def test_attendance_csv_rejects_bad_ranges(prof_client, qs):
    assert prof_client.get("/professor/attendance.csv" + qs).status_code == 400


def test_csv_cells_cannot_become_spreadsheet_formulas(prof_client, db):
    evil = make_user(ROLE_STUDENT, "2024/0003", STUDENT_PW)
    evil.full_name = '=HYPERLINK("http://x","click")'
    db.session.commit()
    a = add_attendance(db, evil)
    a.note, a.depth_nose = "+cmd|' /C calc'!A0", -0.02
    db.session.commit()
    _, rows = read_csv(prof_client.get("/professor/attendance.csv?date=2026-11-15"))
    assert rows[0]["full_name"] == "'" + evil.full_name
    _, rows = read_csv(prof_client.get("/professor/attempts.csv"))
    assert rows[0]["full_name"].startswith("'=") and rows[0]["note"].startswith("'+")
    assert rows[0]["depth_nose"] == "-0.02"           # numbers stay numbers


# ------------------------------------------------------------------------------ attempts log
def ids_on_page(html: str) -> set[int]:
    import re
    return {int(x) for x in re.findall(r'name="ids" value="(\d+)"', html)}


def test_attempts_log_shows_every_score(prof_client, db, ana):
    a = add_attempt(db, ana, decision="fail", reason="liveness_failed", failed_checks="texture", texture_score=0.123,
                    checks={"identity": True, "texture": False, "parallax_landmarks": True}, note="dim room")
    html = prof_client.get("/professor/attempts").get_data(as_text=True)
    assert a.id in ids_on_page(html)
    for text in ("0.900", "0.950", "0.0412", "0.0311", "130", "0.035", "0.100", "0.123", "850",  # all scores + ms
                 "liveness failed", "dim room", "not labelled", "Parallax landmarks*", "Parallax flow"):
        assert text in html, text
    assert '<td class="num bad">0.123</td>' in html        # the failed check is marked


def test_attempts_filters(prof_client, db, ana, bob):
    a1 = add_attempt(db, ana, decision="pass", label="genuine")
    a2 = add_attempt(db, ana, decision="fail", reason="liveness_failed", label="photo", attack_type="printed_flat")
    a3 = add_attempt(db, bob, decision="retry", reason="no_movement", label="screen", attack_type="phone_video")
    a4 = add_attempt(db, bob, mode="enroll", decision="pass")
    # 23:30 UTC on the 14th is already the 15th in Skopje (UTC+1)
    a5 = add_attempt(db, bob, created_at=datetime(2026, 11, 14, 23, 30))
    a6 = add_attempt(db, bob, created_at=datetime(2026, 11, 14, 22, 30))   # 23:30 local on the 14th

    def ids(qs):
        return ids_on_page(prof_client.get("/professor/attempts?" + qs).get_data(as_text=True))

    assert ids("") == {a1.id, a2.id, a3.id, a4.id, a5.id, a6.id}
    assert ids(f"student={ana.id}") == {a1.id, a2.id}
    assert ids("decision=retry") == {a3.id}
    assert ids("reason=liveness_failed") == {a2.id}
    assert ids("mode=enroll") == {a4.id}
    assert ids("label=unlabelled") == {a4.id, a5.id, a6.id}
    assert ids("label=labelled") == {a1.id, a2.id, a3.id}
    assert ids("label=photo") == {a2.id}
    assert ids("label=screen/phone_video") == {a3.id}
    assert ids("label=screen/monitor_photo") == set()
    assert ids("date_from=2026-11-15&date_to=2026-11-15") == {a1.id, a2.id, a3.id, a4.id, a5.id}
    assert ids("date_to=2026-11-14") == {a6.id}


def test_attempts_invalid_filters_are_ignored(prof_client, db, ana):
    a = add_attempt(db, ana)
    r = prof_client.get("/professor/attempts?decision=drop+table&date_from=13/11&label=banana&student=x")
    html = r.get_data(as_text=True)
    assert r.status_code == 200 and ids_on_page(html) == {a.id}
    assert "Ignored an invalid date" in html and "Ignored an invalid decision" in html


def test_attempts_paging(prof_client, db, ana):
    made = [add_attempt(db, ana).id for _ in range(105)]
    p1 = prof_client.get("/professor/attempts").get_data(as_text=True)
    p2 = prof_client.get("/professor/attempts?page=2").get_data(as_text=True)
    assert "Page 1 of 2" in p1 and len(ids_on_page(p1)) == 100 and len(ids_on_page(p2)) == 5
    assert ids_on_page(p1) | ids_on_page(p2) == set(made)
    assert max(made) in ids_on_page(p1)               # newest first
    assert ids_on_page(prof_client.get("/professor/attempts?page=99").get_data(as_text=True)) == ids_on_page(p2)


def test_attempt_detail(prof_client, db, ana):
    a = add_attendance(db, ana)
    html = prof_client.get(f"/professor/attempts/{a.id}").get_data(as_text=True)
    for text in ("Attempt #", "2024/0001", "Check-in", "written: 2026-11-15 09:15:07", "0.0412", "0.0311",
                 "&ge; 0.4", "&ge; 0.025", "Frames", "80.0", "0.970 / 0.990", "End-to-end", "3500 ms", "Student saw"):
        assert text in html, text
    assert prof_client.get("/professor/attempts/9999").status_code == 404


# ------------------------------------------------------------------------------ labelling
def label(client, ids, value, note="", back=None, **extra):
    data = {"ids": ids, "label": value, "note": note, **extra}
    if back is not None:
        data["back"] = back
    return client.post("/professor/attempts/label", data=data)


def test_bulk_label_with_attack_type_and_note(prof_client, db, professor, ana):
    a, b, c = (add_attempt(db, ana) for _ in range(3))
    r = label(prof_client, [a.id, b.id], "screen/phone_video", note="iPhone, full brightness",
              back="/professor/attempts?decision=pass")
    assert r.status_code == 302 and r.headers["Location"] == "/professor/attempts?decision=pass"
    for x in (a, b):
        x = db.session.get(Attempt, x.id)
        assert (x.label, x.attack_type, x.note, x.labelled_by_id) == ("screen", "phone_video",
                                                                       "iPhone, full brightness", professor.id)
        assert x.labelled_at is not None
    assert db.session.get(Attempt, c.id).label is None
    html = prof_client.get("/professor/attempts").get_data(as_text=True)
    assert "2 attempts labelled as Screen replay - phone video." in html


def test_bulk_label_without_note_keeps_existing_notes(prof_client, db, ana):
    a = add_attempt(db, ana, note="glasses")
    label(prof_client, [a.id], "genuine")
    a = db.session.get(Attempt, a.id)
    assert (a.label, a.attack_type, a.note) == ("genuine", None, "glasses")


def test_label_attack_without_type(prof_client, db, ana):
    a = add_attempt(db, ana)
    label(prof_client, [a.id], "photo")
    assert (db.session.get(Attempt, a.id).label, db.session.get(Attempt, a.id).attack_type) == ("photo", None)


@pytest.mark.parametrize("value", ["", "banana", "genuine/phone_video", "photo/phone_video", "screen/printed_flat"])
def test_invalid_labels_change_nothing(prof_client, db, ana, value):
    a = add_attempt(db, ana, label="genuine")
    r = label(prof_client, [a.id], value)
    assert r.status_code == 302
    assert db.session.get(Attempt, a.id).label == "genuine"
    assert "Choose a label." in prof_client.get("/professor/attempts").get_data(as_text=True)


def test_label_needs_a_selection_and_existing_attempts(prof_client, db, ana):
    assert "Select at least one attempt" in prof_client.post(
        "/professor/attempts/label", data={"label": "genuine"}, follow_redirects=True).get_data(as_text=True)
    a = add_attempt(db, ana)
    assert label(prof_client, [a.id, 9999], "genuine").status_code == 404
    assert db.session.get(Attempt, a.id).label is None


def test_remove_label(prof_client, db, professor, ana):
    a = add_attempt(db, ana)
    label(prof_client, [a.id], "photo/printed_bent")
    label(prof_client, [a.id], "clear")
    a = db.session.get(Attempt, a.id)
    assert (a.label, a.attack_type, a.labelled_by_id, a.labelled_at) == (None, None, None, None)


def test_single_attempt_form_saves_label_and_note_as_shown(prof_client, db, ana):
    a = add_attempt(db, ana, note="old note")
    label(prof_client, [a.id], "screen/monitor_photo", note="", single="1")
    a = db.session.get(Attempt, a.id)
    assert (a.label, a.attack_type, a.note) == ("screen", "monitor_photo", None)    # empty note removes it
    first_time = a.labelled_at
    label(prof_client, [a.id], "screen/monitor_photo", note="27 inch monitor", single="1")
    a = db.session.get(Attempt, a.id)
    assert a.note == "27 inch monitor" and a.labelled_at == first_time            # same label: who/when kept
    label(prof_client, [a.id], "", note="27 inch monitor", single="1")              # "not labelled"
    assert db.session.get(Attempt, a.id).label is None


def test_detail_page_form_is_prefilled(prof_client, db, ana):
    a = add_attempt(db, ana, label="photo", attack_type="printed_bent", note="A4 matte")
    html = prof_client.get(f"/professor/attempts/{a.id}").get_data(as_text=True)
    assert '<option value="photo/printed_bent" selected>' in html and 'value="A4 matte"' in html
    assert 'name="single" value="1"' in html


def test_note_length_is_limited(prof_client, db, ana):
    a = add_attempt(db, ana)
    label(prof_client, [a.id], "genuine", note="x" * 1001)
    assert db.session.get(Attempt, a.id).label is None


@pytest.mark.parametrize("back", ["https://evil.example/professor/attempts", "//evil.example/professor/attempts",
                                  "/student/", "/professor/attempts\\@evil.example", "javascript:alert(1)"])
def test_label_redirect_stays_on_the_attempts_log(prof_client, db, ana, back):
    a = add_attempt(db, ana)
    r = label(prof_client, [a.id], "genuine", back=back)
    assert r.headers["Location"] == "/professor/attempts"


def test_labelling_requires_csrf(app, prof_client, db, ana):
    a = add_attempt(db, ana)
    app.config["WTF_CSRF_ENABLED"] = True
    assert label(prof_client, [a.id], "genuine").status_code == 400
    assert db.session.get(Attempt, a.id).label is None
    token = csrf_from(prof_client.get("/professor/attempts").get_data(as_text=True))
    assert label(prof_client, [a.id], "genuine", csrf_token=token).status_code == 302
    assert db.session.get(Attempt, a.id).label == "genuine"


# ------------------------------------------------------------------------------ attempts CSV
def test_attempts_csv_contents_and_filters(prof_client, db, professor, ana, bob):
    a = add_attempt(db, ana, label="photo", attack_type="printed_flat", note="flat A4")
    a.labelled_by_id = professor.id
    db.session.commit()
    add_attempt(db, bob, decision="retry", reason="no_movement")
    r = prof_client.get(f"/professor/attempts.csv?student={ana.id}")
    assert 'filename="attempts.csv"' in r.headers["Content-Disposition"]
    header, rows = read_csv(r)
    for col in ("attempt_id", "time_local", "time_utc", "student_id", "full_name", "mode", "decision", "reason",
                "similarity", "consistency", "parallax_landmark_score", "parallax_landmark_movement",
                "parallax_flow_score", "parallax_flow_movement", "parallax_flow_points", "depth_relief", "depth_nose",
                "texture_score", "check_texture", "processing_ms", "server_ms", "client_capture_ms", "end_to_end_ms",
                "thr_texture_threshold", "label", "attack_type", "note", "labelled_by"):
        assert col in header, col
    assert len(rows) == 1
    row = rows[0]
    assert row["student_id"] == "2024/0001" and row["time_local"] == "2026-11-15 10:00:00"
    assert row["time_utc"] == "2026-11-15 09:00:00"
    assert (row["parallax_landmark_score"], row["parallax_flow_score"], row["texture_score"]) == ("0.0412", "0.0311", "0.98")
    assert (row["check_texture"], row["end_to_end_ms"], row["thr_texture_threshold"]) == ("pass", "3500", "0.5")
    assert (row["label"], row["attack_type"], row["note"], row["labelled_by"]) == ("photo", "printed_flat", "flat A4", "prof")


def test_attempts_csv_anonymised(app, prof_client, db, ana, bob):
    add_attempt(db, ana)
    add_attempt(db, ana)
    add_attempt(db, bob)
    r = prof_client.get("/professor/attempts.csv?anonymise=1")
    text = r.get_data(as_text=True)
    header, rows = read_csv(r)
    assert "participant" in header and "student_id" not in header and "full_name" not in header
    assert "labelled_by" not in header
    for secret in ("2024/0001", "2024/0002", ana.full_name, bob.full_name):
        assert secret not in text
    codes = [x["participant"] for x in rows]
    assert codes[0] == codes[1] != codes[2] and all(c.startswith("P-") and len(c) == 10 for c in codes)
    assert codes[0] == pseudonym(ana.id, app.config["SECRET_KEY"])       # stable across exports


# ------------------------------------------------------------------------------ no biometric data, anywhere
TEMPLATE = np.linspace(0.1234567, 0.9876543, 512, dtype=np.float32)


@pytest.fixture
def enrolled_with_real_template(db, ana):
    tpl = db.session.execute(select(FaceTemplate).where(FaceTemplate.user_id == ana.id)).scalar_one()
    tpl.template = TEMPLATE.tobytes()
    tpl.frame_embeddings = np.tile(TEMPLATE, 5).tobytes()
    db.session.commit()
    add_attendance(db, ana)
    return ana


def biometric_fingerprints() -> list[str]:
    raw = TEMPLATE.tobytes()
    out = [raw.hex()[:64], base64.b64encode(raw).decode()[:40]]
    for v in TEMPLATE[:3]:
        out += [repr(float(v)), f"{v:.6g}", f"{v:.4f}"]
    return out


PROFESSOR_URLS = ["/professor/", "/professor/attendance?date=2026-11-15", "/professor/attendance.csv?date=2026-11-15",
                  "/professor/attempts", "/professor/attempts.csv", "/professor/attempts.csv?anonymise=1"]


def test_professor_pages_and_exports_contain_no_biometric_data(prof_client, db, enrolled_with_real_template):
    a = db.session.execute(select(Attempt)).scalars().first()
    for url in PROFESSOR_URLS + [f"/professor/attempts/{a.id}"]:
        r = prof_client.get(url)
        assert r.status_code == 200, url
        body = r.get_data(as_text=True)
        for fp in biometric_fingerprints():
            assert fp not in body, (url, fp)
        for word in ("embedding", "frame_embeddings", "template_blob"):
            assert word not in body.lower(), (url, word)
    for url in ("/professor/attempts.csv", "/professor/attempts.csv?anonymise=1"):
        header, _ = read_csv(prof_client.get(url))
        assert not [h for h in header if any(w in h for w in ("embedding", "template", "image", "frame"))], header


def test_professor_pages_never_read_template_columns(app, prof_client, db, enrolled_with_real_template):
    """Stronger than looking at the output: the SQL these pages run never selects the template or the
    per-frame embeddings (only whether a template exists)."""
    a = db.session.execute(select(Attempt)).scalars().first()
    statements: list[str] = []

    def record(conn, cursor, statement, *args):
        statements.append(statement)
    engine = db.engine
    event.listen(engine, "before_cursor_execute", record)
    try:
        for url in PROFESSOR_URLS + [f"/professor/attempts/{a.id}"]:
            prof_client.get(url)
        label(prof_client, [a.id], "genuine")
    finally:
        event.remove(engine, "before_cursor_execute", record)
    assert statements
    bad = [s for s in statements if "face_templates.template" in s or "frame_embeddings" in s]
    assert not bad, bad
    assert db.session.execute(select(User).where(User.username == "2024/0001")).scalar_one()
