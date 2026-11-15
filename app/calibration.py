"""Threshold calibration and the paper's metrics (Table I) - pure Python, no Flask, no database.

Used by two scripts that only differ in where the attempts come from:
  * scripts/replay_captures.py  - re-runs the pipeline on captures saved by webcam_test.py
  * scripts/threshold_report.py - reads the attempts table of the web app (labelled by a professor)

Both turn their attempts into `Record`s and print the same analysis:
  1. outcomes per label / attack type (accepted, rejected, asked to retry),
  2. score distributions (min / median / max) per group,
  3. margins (lowest genuine - highest attack) and suggested thresholds,
  4. (threshold_report only) Table I of the paper.

All liveness scores are "higher = more likely real".
"""
from __future__ import annotations

import math
import statistics
from collections import defaultdict
from dataclasses import dataclass, field

ATTACK_LABELS = ("photo", "screen")
LABEL_ORDER = ("genuine", "photo", "screen", "unlabelled")
MIN_PER_LABEL = 5

# (Record attribute, display name, VisionConfig threshold field) - every one is "higher = more real"
SCORE_CHECKS = (
    ("parallax_landmark_score", "Parallax (landmarks)", "parallax_threshold"),
    ("parallax_flow_score", "Parallax (flow)", "parallax_flow_threshold"),
    ("depth_relief", "Depth relief", "depth_relief_threshold"),
    ("depth_nose", "Depth nose", "depth_nose_threshold"),
    ("texture_score", "Texture P(real)", "texture_threshold"),
)
# Reasons for which the pipeline stopped before the liveness checks (frame quality gates).
QUALITY_REASONS = ("wrong_frame_count", "no_face", "multiple_faces", "face_too_small", "face_too_close",
                   "too_dark", "too_blurry")


@dataclass
class Record:
    """One attempt with its label - the common format of both scripts."""
    label: str | None = None             # genuine | photo | screen | None (unlabelled)
    attack_type: str | None = None       # printed_flat, phone_video, ... (attacks only)
    mode: str = "checkin"                # checkin | enroll
    decision: str = ""                   # pass | fail | retry
    reason: str = ""
    failed_checks: list[str] = field(default_factory=list)
    similarity: float | None = None
    identity_passed: bool | None = None  # None = identity not evaluated (enrollment, or stopped early)
    parallax_landmark_score: float | None = None
    parallax_flow_score: float | None = None
    depth_relief: float | None = None
    depth_nose: float | None = None
    texture_score: float | None = None
    processing_ms: float | None = None   # server: pipeline only
    end_to_end_ms: float | None = None   # browser capture + server request (None if not recorded)
    user_id: int | None = None
    thresholds: dict | None = None       # thresholds in force when the attempt was made
    name: str = ""                       # for listings (capture folder / attempt id)

    @property
    def label_key(self) -> str:
        return self.label or "unlabelled"

    @property
    def group(self) -> str:
        """genuine | photo/printed_flat | screen/phone_video | ... | unlabelled"""
        if self.label in ATTACK_LABELS:
            return f"{self.label}/{self.attack_type or 'unspecified'}"
        return self.label_key

    @classmethod
    def from_result(cls, r, label: str | None, attack_type: str | None, name: str = "") -> "Record":
        """From a vision PipelineResult (replay_captures.py)."""
        return cls(label=None if label in (None, "unlabelled") else label, attack_type=attack_type,
                   mode=r.mode, decision=r.decision, reason=r.reason, failed_checks=list(r.failed_checks),
                   similarity=r.similarity, identity_passed=r.checks.get("identity"),
                   parallax_landmark_score=r.parallax_landmark_score, parallax_flow_score=r.parallax_flow_score,
                   depth_relief=r.depth_relief, depth_nose=r.depth_nose, texture_score=r.texture_score,
                   processing_ms=r.processing_ms, name=name)

    @classmethod
    def from_attempt(cls, a) -> "Record":
        """From a row of the attempts table (threshold_report.py)."""
        t = a.timings_ms or {}
        e2e = None
        if t.get("client_capture_ms") is not None and t.get("server_ms") is not None:
            e2e = float(t["client_capture_ms"]) + float(t["server_ms"])
        return cls(label=a.label, attack_type=a.attack_type, mode=a.mode, decision=a.decision, reason=a.reason,
                   failed_checks=[c for c in (a.failed_checks or "").split(";") if c],
                   similarity=a.similarity, identity_passed=(a.checks or {}).get("identity"),
                   parallax_landmark_score=a.parallax_landmark_score, parallax_flow_score=a.parallax_flow_score,
                   depth_relief=a.depth_relief, depth_nose=a.depth_nose, texture_score=a.texture_score,
                   processing_ms=a.processing_ms, end_to_end_ms=e2e, user_id=a.user_id,
                   thresholds=a.thresholds, name=f"#{a.id}")


# ------------------------------------------------------------------------------------- helpers
def fmt(v, spec=".3f") -> str:
    return "-" if v is None else format(v, spec)


def pct(k: int, n: int) -> str:
    return f"{k}/{n} ({100 * k / n:.0f}%)" if n else "0/0"


def _count(records, attr) -> dict:
    c: dict = defaultdict(int)
    for r in records:
        v = getattr(r, attr)
        for k in (v if isinstance(v, list) else [v]):
            c[k] += 1
    return dict(sorted(c.items(), key=lambda kv: -kv[1]))


def _group_sort_key(g: str):
    head = g.split("/")[0]
    return (LABEL_ORDER.index(head) if head in LABEL_ORDER else 99, g)


def groups_of(records) -> dict[str, list[Record]]:
    by: dict[str, list[Record]] = defaultdict(list)
    for r in records:
        by[r.group].append(r)
    return {g: by[g] for g in sorted(by, key=_group_sort_key)}


def labels_of(records) -> dict[str, list[Record]]:
    by: dict[str, list[Record]] = defaultdict(list)
    for r in records:
        by[r.label_key].append(r)
    return by


def _values(records, attr) -> list[float]:
    return [getattr(r, attr) for r in records if getattr(r, attr) is not None]


# ------------------------------------------------------------------------------------- 1. outcomes
def outcome_lines(records: list[Record], title: str = "Outcomes (decisions as made by the system)") -> list[str]:
    out = [title]
    by_label, by_group = labels_of(records), groups_of(records)
    gen = by_label.get("genuine", [])
    if gen:
        n = len(gen)
        acc = [r for r in gen if r.decision == "pass"]
        rej = [r for r in gen if r.decision == "fail"]
        ret = [r for r in gen if r.decision == "retry"]
        out.append(f"  Genuine (real student), n={n}:")
        out.append(f"    accepted                      {pct(len(acc), n)}")
        out.append(f"    REJECTED (false rejection)    {pct(len(rej), n)}" +
                   (f"   failed checks: {_count(rej, 'failed_checks')}" if rej else ""))
        out.append(f"    asked to RETRY                {pct(len(ret), n)}" +
                   (f"   reasons: {_count(ret, 'reason')}" if ret else ""))
        decided = len(acc) + len(rej)
        if decided:
            out.append(f"    false rejection rate among attempts that got a decision: {pct(len(rej), decided)}")
    attack_groups = [g for g in by_group if g.split("/")[0] in ATTACK_LABELS]
    if attack_groups:
        out.append("  Attacks (an ACCEPTED attack is a security failure):")
        for g in attack_groups:
            rs = by_group[g]
            n = len(rs)
            acc = sum(r.decision == "pass" for r in rs)
            rej = [r for r in rs if r.decision == "fail"]
            ret = [r for r in rs if r.decision == "retry"]
            out.append(f"    {g:<24} n={n:<3} ACCEPTED {pct(acc, n):<12} rejected {pct(len(rej), n):<12} "
                       f"retry {pct(len(ret), n)}")
            if rej:
                out.append(f"    {'':<24} rejected by: {_count(rej, 'failed_checks')}")
            if ret:
                out.append(f"    {'':<24} retry reasons: {_count(ret, 'reason')}")
    if by_label.get("unlabelled"):
        out.append(f"  Unlabelled: {len(by_label['unlabelled'])} attempt(s) (not counted above)")
    if not gen and not attack_groups:
        out.append("  No labelled attempts yet.")
    return out


# ------------------------------------------------------------------------------------- 2. distributions
def distribution_lines(records: list[Record], cfg) -> list[str]:
    by_group = groups_of(records)
    out = ["Score distributions (min / median / max, n = attempts that have this score):"]
    rows = [(attr, name, getattr(cfg, f)) for attr, name, f in SCORE_CHECKS]
    rows.append(("similarity", "Identity similarity (check-in; information only)", cfg.identity_threshold))
    for attr, name, thr in rows:
        out.append(f"  {name}  (current threshold {thr})")
        for g, rs in by_group.items():
            v = _values(rs, attr)
            if v:
                out.append(f"    {g:<24} n={len(v):<3} {min(v):8.4f} {statistics.median(v):8.4f} {max(v):8.4f}")
            else:
                out.append(f"    {g:<24} n=0")
    return out


# ------------------------------------------------------------------------------------- 3. margins
@dataclass
class Suggestion:
    attr: str
    name: str
    field: str
    current: float
    margin: float | None          # lowest genuine - highest attack (None = not enough data)
    suggested: float | None       # midpoint, only when the groups are separable
    warnings: list[str]
    per_group_margin: dict[str, float]


def suggestions(records: list[Record], cfg) -> list[Suggestion]:
    by_label, by_group = labels_of(records), groups_of(records)
    attack_groups = [g for g in by_group if g.split("/")[0] in ATTACK_LABELS]
    result = []
    for attr, name, f in SCORE_CHECKS:
        thr = getattr(cfg, f)
        gen = _values(by_label.get("genuine", []), attr)
        att = {lb: _values(by_label.get(lb, []), attr) for lb in ATTACK_LABELS}
        att_all = att["photo"] + att["screen"]
        if not gen or not att_all:
            result.append(Suggestion(attr, name, f, thr, None, None, ["need both genuine and attack attempts"], {}))
            continue
        margin = min(gen) - max(att_all)
        per = {g: min(gen) - max(_values(by_group[g], attr)) for g in attack_groups if _values(by_group[g], attr)}
        warnings = [f"only {n} {lb} attempt(s) with this score (want >= {MIN_PER_LABEL})"
                    for lb, n in [("genuine", len(gen)), ("photo", len(att["photo"])), ("screen", len(att["screen"]))]
                    if n < MIN_PER_LABEL]
        if margin > 0:
            suggested = (min(gen) + max(att_all)) / 2
        else:
            suggested = None
            hi_att, lo_gen = max(att_all), min(gen)
            warnings.insert(0, f"groups OVERLAP: {sum(g <= hi_att for g in gen)}/{len(gen)} genuine at or below the "
                               f"highest attack, {sum(a >= lo_gen for a in att_all)}/{len(att_all)} attacks at or "
                               f"above the lowest genuine - no threshold separates them")
        result.append(Suggestion(attr, name, f, thr, margin, suggested, warnings, per))
    return result


def margin_lines(records: list[Record], cfg, suggest: bool = True) -> list[str]:
    by_label = labels_of(records)
    out = ["Margin = lowest genuine - highest attack (positive = separable on this data):"]
    sugg = suggestions(records, cfg)
    for s in sugg:
        gen = _values(by_label.get("genuine", []), s.attr)
        att_all = sum((_values(by_label.get(lb, []), s.attr) for lb in ATTACK_LABELS), [])
        if s.margin is None:
            out.append(f"  {s.name:<22} not enough data (genuine n={len(gen)}, attacks n={len(att_all)})")
            continue
        per = "  ".join(f"{g}: {m:+.4f}" for g, m in s.per_group_margin.items())
        verdict = "separable" if s.margin > 0 else "OVERLAP"
        out.append(f"  {s.name:<22} {s.margin:+.4f}  {verdict:<9}  (lowest genuine {min(gen):.4f}, highest attack "
                   f"{max(att_all):.4f};  {per})")
    if suggest:
        out += ["", "Suggested thresholds (halfway between lowest genuine and highest attack):"]
        for s in sugg:
            v = "no suggestion" if s.suggested is None else f"{s.suggested:.4f}"
            out.append(f"  {s.field:<26} current {s.current:<8} -> {v}")
            out += [f"      WARNING: {w}" for w in s.warnings]
        out += ["", "Nothing was changed. Edit app/config.py to apply a value, then check it again."]
    return out


def analysis_lines(records: list[Record], cfg, suggest: bool = True,
                   outcome_title: str = "Outcomes (decisions as made by the system)") -> list[str]:
    return (outcome_lines(records, outcome_title) + [""] + distribution_lines(records, cfg) + [""]
            + margin_lines(records, cfg, suggest))


# ------------------------------------------------------------------------------------- 4. Table I
@dataclass
class Rate:
    k: int
    n: int

    @property
    def value(self) -> float | None:
        return self.k / self.n if self.n else None

    def wilson(self, z: float = 1.96) -> tuple[float, float] | None:
        """95% Wilson score interval - honest for small samples (unlike +- 1.96 * SE)."""
        if not self.n:
            return None
        p, n = self.k / self.n, self.n
        centre = (p + z * z / (2 * n)) / (1 + z * z / n)
        half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
        return max(0.0, centre - half), min(1.0, centre + half)

    def text(self) -> str:
        if not self.n:
            return "n/a (no attempts)"
        lo, hi = self.wilson()
        return f"{100 * self.value:5.1f}%  ({self.k}/{self.n}; 95% CI {100 * lo:.0f}-{100 * hi:.0f}%)"


@dataclass
class TimeStats:
    values_ms: list[float]

    def text(self) -> str:
        v = self.values_ms
        if not v:
            return "n/a (not recorded)"
        sd = statistics.stdev(v) / 1000 if len(v) > 1 else 0.0
        return (f"{statistics.mean(v) / 1000:5.2f} s  (SD {sd:.2f} s, median {statistics.median(v) / 1000:.2f} s, "
                f"n={len(v)})")


@dataclass
class TableOne:
    identification: Rate      # genuine check-ins: identity check passed / identity evaluated
    photo_rejection: Rate     # printed-photo attacks NOT accepted / all printed-photo attacks
    screen_rejection: Rate    # screen-replay attacks NOT accepted / all screen attacks
    false_rejection: Rate     # genuine rejected (fail) / genuine that got a decision (pass + fail)
    genuine_retry: Rate       # genuine asked to retry / all genuine
    genuine_accepted: Rate    # genuine accepted / all genuine (what a student experiences)
    server_time: TimeStats    # check-ins that ran the full pipeline: server processing
    end_to_end_time: TimeStats  # same check-ins: browser capture + server request
    participants: int         # distinct students with at least one labelled attempt
    students_any: int         # distinct students with any attempt in the selection
    attack_retry: dict[str, int] = field(default_factory=dict)  # attacks not accepted because of a retry


def table_one(records: list[Record]) -> TableOne:
    by_label = labels_of(records)
    gen = by_label.get("genuine", [])
    gen_checkin = [r for r in gen if r.mode == "checkin" and r.identity_passed is not None]

    def not_accepted(rs):
        return Rate(sum(r.decision != "pass" for r in rs), len(rs))

    full = [r for r in records if r.mode == "checkin" and r.reason not in QUALITY_REASONS]
    decided = [r for r in gen if r.decision in ("pass", "fail")]
    return TableOne(
        identification=Rate(sum(bool(r.identity_passed) for r in gen_checkin), len(gen_checkin)),
        photo_rejection=not_accepted(by_label.get("photo", [])),
        screen_rejection=not_accepted(by_label.get("screen", [])),
        false_rejection=Rate(sum(r.decision == "fail" for r in decided), len(decided)),
        genuine_retry=Rate(sum(r.decision == "retry" for r in gen), len(gen)),
        genuine_accepted=Rate(sum(r.decision == "pass" for r in gen), len(gen)),
        server_time=TimeStats([r.processing_ms for r in full if r.processing_ms is not None]),
        end_to_end_time=TimeStats([r.end_to_end_ms for r in full if r.end_to_end_ms is not None]),
        participants=len({r.user_id for r in records if r.label and r.user_id is not None}),
        students_any=len({r.user_id for r in records if r.user_id is not None}),
        attack_retry={lb: sum(r.decision == "retry" for r in by_label.get(lb, [])) for lb in ATTACK_LABELS},
    )


def table_one_lines(t: TableOne) -> list[str]:
    return [
        "Table I - preliminary evaluation metrics (from the labelled attempts)",
        f"  Verified identification accuracy           {t.identification.text()}",
        f"  Printed-photo attack rejection rate        {t.photo_rejection.text()}",
        f"  Screen-replay attack rejection rate        {t.screen_rejection.text()}",
        f"  False rejection rate (legitimate students) {t.false_rejection.text()}",
        f"  Genuine retry rate                         {t.genuine_retry.text()}",
        f"  Average verification time (server)         {t.server_time.text()}",
        f"  Average end-to-end verification time       {t.end_to_end_time.text()}",
        f"  Pilot participants                         {t.participants} students with labelled attempts "
        f"({t.students_any} with any attempt)",
        "",
        "  Also: genuine attempts accepted overall     " + t.genuine_accepted.text(),
        "",
        "  Definitions (use these wordings in the paper):",
        "  - identification accuracy: genuine CHECK-IN attempts whose identity check passed (cosine similarity",
        "    to the student's own template >= threshold), among those where identity was evaluated",
        "    (attempts stopped by a frame-quality problem have no identity score).",
        "  - attack rejection rate: attacks that were NOT accepted (rejected or asked to retry)",
        f"    = 1 - APCER. Not accepted because of a retry: photo {t.attack_retry.get('photo', 0)}, "
        f"screen {t.attack_retry.get('screen', 0)}.",
        "  - false rejection rate: genuine attempts REJECTED, among genuine attempts that got a decision",
        "    (accepted or rejected). Retries (turn your head, too dark, too close ...) are counted",
        "    separately as the genuine retry rate.",
        "  - verification time: check-in attempts that ran the full pipeline (not stopped by a frame-quality",
        "    gate). Server = analysis of the 5 frames. End-to-end = guided capture in the browser (from the",
        "    camera being ready to the 5th frame) + the server request; upload time over the network is not",
        "    included (negligible on localhost). Older attempts without browser timing are left out.",
        "  - 95% CI: Wilson score interval - with small pilots the intervals are wide; report them.",
    ]


# ------------------------------------------------------------------------------------- threshold changes
def threshold_changes(records: list[Record], current: dict) -> dict[str, tuple[int, set]]:
    """Thresholds that differ between the attempts' snapshots and the current config:
    {field: (number of attempts recorded with another value, the other values)}."""
    out: dict[str, tuple[int, set]] = {}
    for r in records:
        for k, v in (r.thresholds or {}).items():
            if k in current and current[k] != v:
                n, vals = out.get(k, (0, set()))
                vals.add(v if not isinstance(v, list) else tuple(v))
                out[k] = (n + 1, vals)
    return out
