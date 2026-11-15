"""CSV downloads (attendance, attempts log).

* UTF-8 with a byte-order mark, so Excel shows Cyrillic names (e.g. Macedonian) correctly.
* Text cells that start with = + - @ (or a tab / carriage return) get a leading apostrophe:
  otherwise a spreadsheet would run them as a FORMULA ("CSV injection") - a student name or a note
  could be crafted to do that. Numbers are written unchanged (negative scores stay numbers).
"""
from __future__ import annotations

import csv
import io

from flask import Response

_FORMULA_START = ("=", "+", "-", "@", "\t", "\r")


def cell(v):
    if v is None:
        return ""
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return format(v, ".6g")
    if isinstance(v, str) and v.startswith(_FORMULA_START):
        return "'" + v
    return v


def csv_text(header: list[str], rows) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\r\n")
    w.writerow(header)
    for row in rows:
        w.writerow([cell(v) for v in row])
    return "﻿" + buf.getvalue()


def csv_response(filename: str, header: list[str], rows) -> Response:
    return Response(csv_text(header, rows), mimetype="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="{filename}"',
                             "Cache-Control": "no-store"})
