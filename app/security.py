"""Passwords (argon2id), login lockout, role checks, and server-side time helpers."""
from __future__ import annotations

import secrets
import string
from datetime import datetime, timedelta
from functools import wraps
from zoneinfo import ZoneInfo

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError
from flask import abort, current_app, request
from flask_login import current_user
from sqlalchemy import delete, func, select

from .extensions import db
from .models import LoginFailure, User, utcnow


# ------------------------------------------------------------------------------------ passwords
def _hasher() -> PasswordHasher:
    c = current_app.config
    return PasswordHasher(time_cost=c["ARGON2_TIME_COST"], memory_cost=c["ARGON2_MEMORY_COST"],
                          parallelism=c["ARGON2_PARALLELISM"])


def hash_password(password: str) -> str:
    return _hasher().hash(password)


# a valid hash of a random password, used so that unknown usernames take as long as known ones
_DUMMY_HASH: dict[tuple, str] = {}


def verify_password(user: User | None, password: str) -> bool:
    """Constant-effort check. Re-hashes transparently when the cost parameters were raised."""
    ph = _hasher()
    if user is None:
        key = (ph.time_cost, ph.memory_cost, ph.parallelism)
        if key not in _DUMMY_HASH:
            _DUMMY_HASH[key] = ph.hash(secrets.token_hex(16))
        try:
            ph.verify(_DUMMY_HASH[key], password)
        except VerificationError:
            pass
        return False
    try:
        ph.verify(user.password_hash, password)
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        return False
    if ph.check_needs_rehash(user.password_hash):
        user.password_hash = ph.hash(password)
    return True


def password_problem(password: str, username: str = "") -> str | None:
    """None if the password is acceptable, otherwise a message for the user."""
    min_len = current_app.config["PASSWORD_MIN_LENGTH"]
    if len(password) < min_len:
        return f"The password must be at least {min_len} characters long."
    if len(password) > 128:
        return "The password must be at most 128 characters long."
    if username and password.lower() == username.lower():
        return "The password must not be the same as the username."
    if password.isdigit() or password.isalpha():
        return "Use a mix of letters and numbers (or other characters)."
    return None


def generate_password(length: int = 10) -> str:
    """Readable random initial password (no 0/O, 1/l/I) with letters and digits."""
    alphabet = "".join(c for c in string.ascii_letters + string.digits if c not in "0O1lI")
    while True:
        pw = "".join(secrets.choice(alphabet) for _ in range(length))
        if any(c.isdigit() for c in pw) and any(c.isalpha() for c in pw):
            return pw


def normalise_username(username: str) -> str:
    return (username or "").strip().lower()


# ------------------------------------------------------------------------------------ lockout
def client_ip() -> str:
    # request.remote_addr is already the real client when ProxyFix is enabled (TRUST_PROXY_HOPS)
    return (request.remote_addr or "unknown")[:64]


def lockout_remaining(username: str, ip: str) -> timedelta | None:
    """How long this username / IP is still locked out, or None if login may be tried."""
    c = current_app.config
    window = timedelta(minutes=c["LOGIN_LOCKOUT_MINUTES"])
    since = utcnow() - window
    remaining = None
    for column, value, limit in ((LoginFailure.username, username, c["LOGIN_MAX_FAILURES"]),
                                 (LoginFailure.ip, ip, c["LOGIN_MAX_FAILURES_PER_IP"])):
        times = db.session.execute(
            select(LoginFailure.created_at).where(column == value, LoginFailure.created_at >= since)
            .order_by(LoginFailure.created_at.desc()).limit(limit)).scalars().all()
        if len(times) >= limit:
            # locked until `window` after the limit-th most recent failure
            left = times[-1] + window - utcnow()
            if left.total_seconds() > 0 and (remaining is None or left > remaining):
                remaining = left
    return remaining


def record_login_failure(username: str, ip: str) -> None:
    db.session.add(LoginFailure(username=username[:64], ip=ip))
    # keep the table small: drop entries older than one day
    db.session.execute(delete(LoginFailure).where(LoginFailure.created_at < utcnow() - timedelta(days=1)))
    db.session.commit()


def clear_login_failures(username: str) -> None:
    db.session.execute(delete(LoginFailure).where(LoginFailure.username == username))


def failure_count(username: str) -> int:
    since = utcnow() - timedelta(minutes=current_app.config["LOGIN_LOCKOUT_MINUTES"])
    return db.session.execute(select(func.count(LoginFailure.id)).where(
        LoginFailure.username == username, LoginFailure.created_at >= since)).scalar_one()


# ------------------------------------------------------------------------------------ roles
def role_required(role: str):
    """Decorator: logged in AND has `role`, otherwise 403 (anonymous users go to the login page)."""
    def decorator(view):
        @wraps(view)
        def wrapped(*args, **kwargs):
            if not current_user.is_authenticated:
                return current_app.login_manager.unauthorized()
            if current_user.role != role:
                abort(403)
            return view(*args, **kwargs)
        return wrapped
    return decorator


# ------------------------------------------------------------------------------------ time
def local_now() -> datetime:
    """Current server time in the configured time zone (attendance dates use this)."""
    return datetime.now(ZoneInfo(current_app.config["TIMEZONE"]))


def to_local(utc_naive: datetime | None) -> datetime | None:
    if utc_naive is None:
        return None
    return utc_naive.replace(tzinfo=ZoneInfo("UTC")).astimezone(ZoneInfo(current_app.config["TIMEZONE"]))
