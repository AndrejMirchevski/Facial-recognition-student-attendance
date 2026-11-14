"""Role checks, professor account management, enrollment redirect, face templates kept private."""
from sqlalchemy import select

from app.models import ROLE_STUDENT, FaceTemplate, User

from .conftest import PROF_PW, STUDENT_PW, login, make_user


def test_anonymous_is_sent_to_login(client):
    for path in ("/", "/professor/", "/student/", "/student/enroll"):
        r = client.get(path)
        assert r.status_code == 302 and r.headers["Location"].startswith("/login"), path
    assert client.get("/api/me").status_code == 401


def test_student_cannot_use_professor_pages(client, student, professor):
    login(client, "s1001", STUDENT_PW)
    assert client.get("/professor/").status_code == 403
    assert client.post("/professor/students", data={"student_id": "x123", "full_name": "X Y"}).status_code == 403
    assert client.post(f"/professor/students/{professor.id}/reset-password").status_code == 403
    assert client.post(f"/professor/students/{student.id}/toggle-active").status_code == 403


def test_professor_cannot_use_student_pages(client, professor):
    login(client, "prof", PROF_PW)
    assert client.get("/student/").status_code == 403
    assert client.get("/professor/").status_code == 200


def test_unenrolled_student_is_sent_to_enrollment(client, db):
    make_user(ROLE_STUDENT, "s4000", STUDENT_PW, enrolled=False)
    login(client, "s4000", STUDENT_PW)
    r = client.get("/student/")
    assert r.status_code == 302 and r.headers["Location"] == "/student/enroll"
    assert client.get("/student/enroll").status_code == 200


def test_enrolled_student_reaches_home(client, student):
    login(client, "s1001", STUDENT_PW)
    assert client.get("/student/").status_code == 200
    assert client.get("/student/enroll").status_code == 302   # already enrolled -> home


def test_professor_creates_student_with_given_password(client, db, professor):
    login(client, "prof", PROF_PW)
    r = client.post("/professor/students", data={"student_id": "S5000", "full_name": "Ana Petrova",
                                                 "initial_password": "Start-pass-1"})
    assert r.status_code == 200 and b"Start-pass-1" in r.data and b"s5000" in r.data
    u = db.session.execute(select(User).where(User.username == "s5000")).scalar_one()
    assert u.role == ROLE_STUDENT and u.must_change_password and u.active
    assert "Start-pass-1" not in u.password_hash


def test_professor_creates_student_with_generated_password(client, db, professor):
    login(client, "prof", PROF_PW)
    r = client.post("/professor/students", data={"student_id": "s5001", "full_name": "Marko Markov"})
    assert r.status_code == 200 and b"Initial password" in r.data
    new_client = client.application.test_client()
    import re
    pw = re.search(rb'class="mono big">([^<]+)<', r.data).group(1).decode()
    assert login(new_client, "s5001", pw).status_code == 302


def test_duplicate_student_id_is_refused(client, db, professor, student):
    login(client, "prof", PROF_PW)
    r = client.post("/professor/students", data={"student_id": "S1001", "full_name": "Dup"}, follow_redirects=True)
    assert b"already exists" in r.data
    assert len(db.session.execute(select(User).where(User.username == "s1001")).all()) == 1


def test_invalid_student_id_is_refused(client, db, professor):
    login(client, "prof", PROF_PW)
    r = client.post("/professor/students", data={"student_id": "a b<script>", "full_name": "Bad"},
                    follow_redirects=True)
    assert b"3-32 characters" in r.data


def test_disable_and_enable_student(client, db, professor, student):
    login(client, "prof", PROF_PW)
    client.post(f"/professor/students/{student.id}/toggle-active")
    db.session.refresh(student)
    assert not student.active
    client.post(f"/professor/students/{student.id}/toggle-active")
    db.session.refresh(student)
    assert student.active


def test_reset_password_forces_change(client, db, professor, student):
    login(client, "prof", PROF_PW)
    old_hash = student.password_hash
    assert client.post(f"/professor/students/{student.id}/reset-password").status_code == 200
    db.session.refresh(student)
    assert student.must_change_password and student.password_hash != old_hash


def test_professor_cannot_manage_professors(client, db, professor):
    other = make_user("professor", "prof2", PROF_PW)
    login(client, "prof", PROF_PW)
    assert client.post(f"/professor/students/{other.id}/toggle-active").status_code == 404
    assert client.post(f"/professor/students/{other.id}/reset-password").status_code == 404


def test_reset_enrollment_deletes_template_only(client, db, professor, student):
    login(client, "prof", PROF_PW)
    client.post(f"/professor/students/{student.id}/reset-enrollment")
    assert db.session.execute(select(FaceTemplate)).first() is None
    assert db.session.get(User, student.id) is not None


def test_professor_pages_never_contain_template_data(client, db, professor, student):
    tpl = db.session.execute(select(FaceTemplate)).scalar_one()
    tpl.template = b"SECRET-TEMPLATE-BYTES" * 10
    db.session.commit()
    login(client, "prof", PROF_PW)
    page = client.get("/professor/")
    assert page.status_code == 200
    assert b"SECRET-TEMPLATE-BYTES" not in page.data
    assert b"yes" in page.data   # only "enrolled: yes" is shown
