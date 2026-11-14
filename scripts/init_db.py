"""Create the database tables (safe to run again: existing tables and data are kept).

Usage:  .venv\\Scripts\\python scripts\\init_db.py

The database is instance/attendance.db unless DATABASE_URL is set. Run by setup.bat.
Note: db.create_all() only creates MISSING tables; it does not alter existing ones. If a later
stage changes a table, that stage must say how to update an existing database.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sqlalchemy import inspect  # noqa: E402

from app import create_app  # noqa: E402
from app.extensions import db  # noqa: E402


def main() -> None:
    app = create_app()
    with app.app_context():
        db.create_all()
        tables = sorted(inspect(db.engine).get_table_names())
        print(f"Database ready: {db.engine.url.render_as_string(hide_password=True)}")
        print(f"Tables: {', '.join(tables)}")


if __name__ == "__main__":
    main()
