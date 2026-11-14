"""Shared fixtures for the web-app tests: an app with an in-memory database and helper logins."""
from __future__ import annotations

import re

import pytest
from flask import g

from app import create_app
from app.extensions import db as _db
from app.models import ROLE_PROFESSOR, ROLE_STUDENT, FaceTemplate, User, utcnow
from app.security import hash_password

PROF_PW = "Prof-pass-123"
STUDENT_PW = "Init-pass-123"


@pytest.fixture
def app():
    app = create_app("testing")

    # The tests keep one app context open (to use the database directly), so Flask's `g` would be
    # shared between requests and Flask-Login would reuse the previous request's user. A real server
    # has a fresh `g` per request; reproduce that by clearing the cached user first thing.
    def _fresh_user_per_request():
        g.pop("_login_user", None)
    app.before_request_funcs.setdefault(None, []).insert(0, _fresh_user_per_request)

    with app.app_context():
        _db.create_all()
        yield app
        _db.session.remove()
        _db.drop_all()


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def db(app):
    return _db


def make_user(role, username, password, must_change=False, active=True, enrolled=False):
    u = User(role=role, username=username, full_name=username.title(), password_hash=hash_password(password),
             must_change_password=must_change, active=active)
    _db.session.add(u)
    _db.session.commit()
    if enrolled:
        _db.session.add(FaceTemplate(user_id=u.id, template=b"\x00" * 2048, embedding_dim=512,
                                     model_name="test", consent_at=utcnow()))
        _db.session.commit()
    return u


@pytest.fixture
def professor(app):
    return make_user(ROLE_PROFESSOR, "prof", PROF_PW)


@pytest.fixture
def student(app):
    """A student who already changed the password and is enrolled."""
    return make_user(ROLE_STUDENT, "s1001", STUDENT_PW, enrolled=True)


def login(client, username, password, follow=False):
    return client.post("/login", data={"username": username, "password": password}, follow_redirects=follow)


def csrf_from(html: str) -> str:
    m = re.search(r'name="csrf-token" content="([^"]+)"', html) or re.search(
        r'name="csrf_token" type="hidden" value="([^"]+)"', html)
    assert m, "no CSRF token in page"
    return m.group(1)
