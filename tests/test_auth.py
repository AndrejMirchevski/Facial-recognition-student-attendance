"""Login, lockout, forced password change, disabled accounts, session invalidation, CSRF."""
from datetime import timedelta

from sqlalchemy import update

from app.models import ROLE_STUDENT, LoginFailure, User
from app.security import verify_password

from .conftest import PROF_PW, STUDENT_PW, csrf_from, login, make_user


# ---------------------------------------------------------------------------- login basics
def test_login_page_loads(client):
    r = client.get("/login")
    assert r.status_code == 200 and b"Log in" in r.data


def test_login_success_professor_goes_to_dashboard(client, professor):
    r = login(client, "prof", PROF_PW)
    assert r.status_code == 302 and r.headers["Location"] == "/"
    assert client.get("/").headers["Location"] == "/professor/"


def test_username_is_case_insensitive(client, professor):
    assert login(client, "  PROF ", PROF_PW).status_code == 302


def test_wrong_password_and_unknown_user_give_same_message(client, professor):
    a = login(client, "prof", "wrong-pass-1")
    b = login(client, "nobody", "wrong-pass-1")
    assert a.status_code == b.status_code == 401
    assert b"Wrong username or password." in a.data and b"Wrong username or password." in b.data


def test_passwords_are_hashed_with_argon2(app, professor):
    assert professor.password_hash.startswith("$argon2id$")
    assert PROF_PW not in professor.password_hash
    assert verify_password(professor, PROF_PW) and not verify_password(professor, "nope")


def test_logout_requires_post(client, professor):
    login(client, "prof", PROF_PW)
    assert client.get("/logout").status_code == 405
    client.post("/logout")
    assert client.get("/professor/").status_code == 302  # back to login


# ---------------------------------------------------------------------------- lockout
def test_lockout_after_max_failures_even_with_correct_password(app, client, professor):
    n = app.config["LOGIN_MAX_FAILURES"]
    for _ in range(n):
        assert login(client, "prof", "wrong-pass-1").status_code == 401
    r = login(client, "prof", PROF_PW)
    assert r.status_code == 429 and b"Too many failed attempts" in r.data


def test_lockout_expires(app, client, db, professor):
    for _ in range(app.config["LOGIN_MAX_FAILURES"]):
        login(client, "prof", "wrong-pass-1")
    assert login(client, "prof", PROF_PW).status_code == 429
    # move the failures back in time past the lockout window
    old = timedelta(minutes=app.config["LOGIN_LOCKOUT_MINUTES"] + 1)
    db.session.execute(update(LoginFailure).values(created_at=LoginFailure.created_at - old))
    db.session.commit()
    assert login(client, "prof", PROF_PW).status_code == 302


def test_lockout_is_per_username(app, client, professor, student):
    for _ in range(app.config["LOGIN_MAX_FAILURES"]):
        login(client, "prof", "wrong-pass-1")
    assert login(client, "s1001", STUDENT_PW).status_code == 302


def test_successful_login_clears_failures(app, client, professor):
    for _ in range(app.config["LOGIN_MAX_FAILURES"] - 1):
        login(client, "prof", "wrong-pass-1")
    assert login(client, "prof", PROF_PW).status_code == 302
    client.post("/logout")
    for _ in range(app.config["LOGIN_MAX_FAILURES"] - 1):
        login(client, "prof", "wrong-pass-1")
    assert login(client, "prof", PROF_PW).status_code == 302  # counter was reset


def test_ip_lockout_across_usernames(app, client, professor):
    app.config["LOGIN_MAX_FAILURES_PER_IP"] = 6
    for i in range(6):
        login(client, f"user{i}", "wrong-pass-1")   # different usernames, same IP
    assert login(client, "prof", PROF_PW).status_code == 429


# ---------------------------------------------------------------------------- forced password change
def test_first_login_forces_password_change(client, db):
    make_user(ROLE_STUDENT, "s2000", STUDENT_PW, must_change=True, enrolled=True)
    login(client, "s2000", STUDENT_PW)
    for path in ("/", "/student/", "/student/enroll", "/professor/"):
        r = client.get(path, follow_redirects=False)
        if path == "/":
            r = client.get(r.headers["Location"])
        assert r.status_code == 302 and r.headers["Location"] == "/change-password", path
    assert client.get("/api/me").status_code == 403


def test_password_change_flow(client, db):
    u = make_user(ROLE_STUDENT, "s2001", STUDENT_PW, must_change=True, enrolled=True)
    login(client, "s2001", STUDENT_PW)
    bad = client.post("/change-password", data={"current_password": "wrong-pass-9",
                                                "new_password": "New-pass-456", "confirm": "New-pass-456"})
    assert bad.status_code == 400 and b"current password is wrong" in bad.data
    same = client.post("/change-password", data={"current_password": STUDENT_PW,
                                                 "new_password": STUDENT_PW, "confirm": STUDENT_PW})
    assert same.status_code == 400
    weak = client.post("/change-password", data={"current_password": STUDENT_PW,
                                                 "new_password": "12345678901", "confirm": "12345678901"})
    assert weak.status_code == 400
    ok = client.post("/change-password", data={"current_password": STUDENT_PW,
                                               "new_password": "New-pass-456", "confirm": "New-pass-456"})
    assert ok.status_code == 302
    db.session.refresh(u)
    assert not u.must_change_password
    assert client.get("/student/").status_code == 200       # same session still works
    client.post("/logout")
    assert login(client, "s2001", "New-pass-456").status_code == 302


# ---------------------------------------------------------------------------- disabled / sessions
def test_disabled_user_cannot_log_in(client, db):
    make_user(ROLE_STUDENT, "s3000", STUDENT_PW, active=False)
    r = login(client, "s3000", STUDENT_PW)
    assert r.status_code == 403 and b"disabled" in r.data


def test_disabling_logs_out_existing_session(app, client, db, student):
    login(client, "s1001", STUDENT_PW)
    assert client.get("/student/").status_code == 200
    student.active = False
    student.invalidate_sessions()
    db.session.commit()
    assert client.get("/student/").status_code == 302


def test_password_reset_by_professor_logs_student_out(app, db, professor, student):
    s_client, p_client = app.test_client(), app.test_client()
    login(s_client, "s1001", STUDENT_PW)
    assert s_client.get("/student/").status_code == 200
    login(p_client, "prof", PROF_PW)
    assert p_client.post(f"/professor/students/{student.id}/reset-password").status_code == 200
    assert s_client.get("/student/").status_code == 302       # old session no longer valid


# ---------------------------------------------------------------------------- CSRF
def test_csrf_required_on_forms(app, client, professor):
    app.config["WTF_CSRF_ENABLED"] = True
    assert login(client, "prof", PROF_PW).status_code == 400       # no token
    token = csrf_from(client.get("/login").get_data(as_text=True))
    r = client.post("/login", data={"username": "prof", "password": PROF_PW, "csrf_token": token})
    assert r.status_code == 302


def test_csrf_required_on_json_posts(app, client, professor):
    app.config["WTF_CSRF_ENABLED"] = True

    @app.route("/api/_echo", methods=["POST"], endpoint="api._echo_test")
    def _echo():
        return {"ok": True}

    page = client.get("/login").get_data(as_text=True)
    token = csrf_from(page)
    r = client.post("/api/_echo", json={"x": 1})
    assert r.status_code == 400 and r.get_json()["error"] == "csrf"
    r = client.post("/api/_echo", json={"x": 1}, headers={"X-CSRFToken": token})
    assert r.status_code == 200


# ---------------------------------------------------------------------------- headers / cookies
def test_security_headers(client):
    r = client.get("/login")
    assert r.headers["X-Frame-Options"] == "DENY"
    assert "default-src 'self'" in r.headers["Content-Security-Policy"]
    assert r.headers["X-Content-Type-Options"] == "nosniff"


def test_cookie_settings():
    from app import create_app
    dev = create_app("testing").config
    prod = create_app("production", overrides={"SECRET_KEY": "x" * 32,
                                               "SQLALCHEMY_DATABASE_URI": "sqlite://"}).config
    assert dev["SESSION_COOKIE_HTTPONLY"] and dev["SESSION_COOKIE_SAMESITE"] == "Lax"
    assert prod["SESSION_COOKIE_SECURE"] and prod["REMEMBER_COOKIE_SECURE"]


def test_user_ids_in_session_carry_version(app, student):
    assert student.get_id() == f"{student.id}:{student.session_version}"
    assert isinstance(student, User)
