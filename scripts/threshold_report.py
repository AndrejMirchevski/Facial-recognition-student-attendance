"""Threshold report + the paper's Table I, from the attempts recorded by the web app.

Same analysis as replay_captures.py --suggest, but instead of re-running saved captures it reads
the `attempts` table (every enrollment / check-in with all scores) and the labels a professor set
on the Attempts page (genuine / printed photo / screen replay + attack type).

Usage:
    .venv\\Scripts\\python scripts\\threshold_report.py [options]

Options:
    --from YYYY-MM-DD / --to YYYY-MM-DD   only attempts on these local days (inclusive)
    --mode checkin|enroll|all             default: all (identity + timing always use check-ins only)
    --student ID                          only this student's attempts (student ID as used to log in)
    --list                                also list every attempt in the selection
    --db PATH|URL                         another database (default: the app's, instance/attendance.db)

Prints
  1. outcomes per label and attack type (accepted / rejected / asked to retry),
  2. score distributions (min / median / max) per label and attack type,
  3. margins (lowest genuine - highest attack) and suggested thresholds, with warnings,
  4. Table I of the paper: identification accuracy, printed-photo and screen-replay rejection
     rates, false rejection rate, genuine retry rate, average verification time, participants -
     with 95% confidence intervals and the exact definitions used.

Decisions are the ones the system made at the time of each attempt. If thresholds were changed
since, the report says which and for how many attempts. The database is opened READ-ONLY
(SQLite): nothing is changed. Nothing biometric is read (no templates, no images).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sqlalchemy import select  # noqa: E402

from app.attempt_log import AttemptFilter, parse_date  # noqa: E402
from app.calibration import (Record, analysis_lines, fmt, labels_of, table_one, table_one_lines,  # noqa: E402
                             threshold_changes)
from app.models import ROLE_STUDENT, Attempt, User  # noqa: E402
from app.security import to_local  # noqa: E402


def _read_only_sqlite(uri: str) -> str:
    """sqlite:///C:/x/attendance.db -> sqlite:///file:C:/x/attendance.db?mode=ro&uri=true"""
    prefix = "sqlite:///"
    if not uri.startswith(prefix) or uri == "sqlite://" or "mode=ro" in uri:
        return uri
    path = Path(uri[len(prefix):])
    if not path.exists():
        sys.exit(f"Database not found: {path}\nRun setup.bat (or scripts/init_db.py) first, or pass --db.")
    return f"{prefix}file:{path.as_posix()}?mode=ro&uri=true"


def make_app(db: str | None):
    from app import create_app
    overrides = {"VISION_PRELOAD": False}
    if db:
        overrides["SQLALCHEMY_DATABASE_URI"] = db if "://" in db else f"sqlite:///{Path(db).resolve().as_posix()}"
    app = create_app(overrides=overrides)
    app.config["SQLALCHEMY_DATABASE_URI"] = _read_only_sqlite(app.config["SQLALCHEMY_DATABASE_URI"])
    return app


def report(session, tz: str, f: AttemptFilter, cfg, current_thresholds: dict, list_attempts: bool = False,
           title: str = "") -> str:
    rows = session.execute(f.statement(tz).order_by(None).order_by(Attempt.id)).all()   # oldest first
    records = [Record.from_attempt(a) for a, _ in rows]
    users = {u.id: u for _, u in rows}
    by_label = labels_of(records)
    out = [title or "Threshold report - attempts table", ""]
    per_label = ", ".join(f"{lb} {len(by_label[lb])}" for lb in ("genuine", "photo", "screen", "unlabelled")
                          if by_label.get(lb))
    out.append(f"Attempts in the selection: {len(records)}" + (f"  ({per_label})" if per_label else ""))
    out.append(f"Parallax method that decides: {cfg.parallax_method}")
    if not records:
        out.append("\nNothing to analyse. Make attempts in the web app and label them on the Attempts page.")
        return "\n".join(out)
    if not any(r.label for r in records):
        out.append("\nNo attempt is labelled yet: open Attempts (professor), select attempts and label them as "
                   "genuine / printed photo / screen replay. Only labelled attempts are counted below.")

    changes = threshold_changes(records, current_thresholds)
    if changes:
        out.append("\nNOTE: some attempts were made with other thresholds than config.py has now "
                   "(their decisions used the old values):")
        for k, (n, vals) in sorted(changes.items()):
            out.append(f"  {k:<28} now {current_thresholds[k]!r:<10} {n} attempt(s) with "
                       + ", ".join(repr(v) for v in sorted(vals, key=str)))

    if list_attempts:
        out += ["", f"{'attempt':<8} {'time (local)':<17} {'student':<14} {'group':<22} {'mode':<7} {'decision':<8} "
                    f"{'reason':<17} {'sim':>5} {'par-LM':>7} {'par-FLOW':>8} {'relief':>6} {'nose':>6} {'tex':>5} {'ms':>5}"]
        for (a, _), r in zip(rows, records):
            out.append(f"{r.name:<8} {to_local(a.created_at).strftime('%Y-%m-%d %H:%M'):<17} {users[a.user_id].username:<14} "
                       f"{r.group:<22} {r.mode:<7} {r.decision:<8} {r.reason:<17} {fmt(r.similarity, '.2f'):>5} "
                       f"{fmt(r.parallax_landmark_score, '.4f'):>7} {fmt(r.parallax_flow_score, '.4f'):>8} "
                       f"{fmt(r.depth_relief):>6} {fmt(r.depth_nose):>6} {fmt(r.texture_score):>5} "
                       f"{fmt(r.processing_ms, '.0f'):>5}")

    out += [""] + analysis_lines(records, cfg, suggest=True) + [""] + table_one_lines(table_one(records))
    return "\n".join(out)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="date_from")
    ap.add_argument("--to", dest="date_to")
    ap.add_argument("--mode", choices=("checkin", "enroll", "all"), default="all")
    ap.add_argument("--student")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--db")
    args = ap.parse_args(argv)
    try:
        f = AttemptFilter(date_from=parse_date(args.date_from), date_to=parse_date(args.date_to),
                          mode="" if args.mode == "all" else args.mode)
    except ValueError:
        ap.error("dates must look like 2026-11-15")

    app = make_app(args.db)
    from app import vision_service
    from app.extensions import db
    with app.app_context():
        if args.student:
            u = db.session.execute(select(User).where(User.username == args.student.strip().lower(),
                                                      User.role == ROLE_STUDENT)).scalar_one_or_none()
            if u is None:
                sys.exit(f"No student with ID {args.student!r}.")
            f.user_id = u.id
        selection = ", ".join(f"{k}={v}" for k, v in f.query_args().items()) or "all attempts"
        text = report(db.session, app.config["TIMEZONE"], f, vision_service.vision_config(),
                      vision_service.thresholds_snapshot(), list_attempts=args.list,
                      title=f"Threshold report - {selection}")
    print(text)


if __name__ == "__main__":
    main()
