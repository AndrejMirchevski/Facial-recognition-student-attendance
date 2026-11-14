"""Create a professor account (there is no self sign-up; this is how the first account is made).

Usage:  .venv\\Scripts\\python scripts\\create_professor.py

Asks for a username, full name and password (typed twice, not shown on screen).
Creates the database tables first if they do not exist yet.
"""
from __future__ import annotations

import getpass
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sqlalchemy import select  # noqa: E402

from app import create_app  # noqa: E402
from app.extensions import db  # noqa: E402
from app.models import ROLE_PROFESSOR, User, utcnow  # noqa: E402
from app.security import hash_password, normalise_username, password_problem  # noqa: E402


def ask(prompt: str) -> str:
    try:
        return input(prompt).strip()
    except EOFError:
        sys.exit("\nCancelled.")


def main() -> None:
    app = create_app()
    with app.app_context():
        db.create_all()
        print("Create a professor account\n")
        while True:
            username = normalise_username(ask("Username: "))
            if not re.match(r"^[a-z0-9._-]{3,64}$", username):
                print("  3-64 characters: letters, digits, . _ -")
            elif db.session.execute(select(User.id).where(User.username == username)).first():
                print("  That username already exists.")
            else:
                break
        full_name = ""
        while len(full_name) < 2:
            full_name = ask("Full name: ")
        while True:
            pw = getpass.getpass("Password (not shown): ")
            if (problem := password_problem(pw, username)) is not None:
                print(f"  {problem}")
                continue
            if getpass.getpass("Repeat password: ") != pw:
                print("  The passwords do not match.")
                continue
            break
        db.session.add(User(role=ROLE_PROFESSOR, username=username, full_name=full_name,
                            password_hash=hash_password(pw), must_change_password=False, active=True,
                            password_changed_at=utcnow()))
        db.session.commit()
        print(f"\nProfessor '{username}' created. Start the server with start_server.bat and log in.")


if __name__ == "__main__":
    main()
