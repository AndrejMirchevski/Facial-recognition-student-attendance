"""Re-run the CURRENT pipeline + thresholds on attempts saved by webcam_test.py --save-captures,
and show how well each check separates genuine attempts from attacks.

Usage:
    .venv\\Scripts\\python scripts\\replay_captures.py [captures_dir] [--suggest]

Prints
  1. every saved attempt with its decision and scores (both parallax methods side by side),
     then the outcomes: genuine attempts split into accepted / REJECTED (false rejection) /
     asked to RETRY (with reasons), and attacks per attack type (printed flat, phone video ...),
  2. per group (genuine, each attack type, unlabelled): min, median and max of every liveness score,
  3. per check: the MARGIN = lowest genuine score - highest attack score (photo + screen).
     Positive = the check separates them on this data; negative = they overlap.
  --suggest  also proposes a threshold per check, halfway between the lowest genuine and the
             highest attack score, with warnings when the groups overlap or a label has fewer
             than 5 scored attempts. Nothing is changed - copy values into app/config.py yourself.

All liveness scores are "higher = more likely real". Attempts stopped early (e.g. face too close)
have no liveness scores and only appear in the first table. Check-in captures are compared against
instance/webcam_test/template.npy (identity only; it does not affect the liveness scores).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from app.config import VisionConfig  # noqa: E402
from app.vision import VisionModels, run_pipeline  # noqa: E402

DEFAULT_DIR = ROOT / "instance" / "webcam_test" / "captures"
TEMPLATE_FILE = ROOT / "instance" / "webcam_test" / "template.npy"
ATTACK_LABELS = ("photo", "screen")
LABEL_ORDER = ("genuine", "photo", "screen", "unlabelled")
MIN_PER_LABEL = 5


def checks(cfg: VisionConfig):
    """(result attribute, name, current threshold, config field) - all higher = more real."""
    return [
        ("parallax_landmark_score", "Parallax (landmarks)", cfg.parallax_threshold, "parallax_threshold"),
        ("parallax_flow_score", "Parallax (flow)", cfg.parallax_flow_threshold, "parallax_flow_threshold"),
        ("depth_relief", "Depth relief", cfg.depth_relief_threshold, "depth_relief_threshold"),
        ("depth_nose", "Depth nose", cfg.depth_nose_threshold, "depth_nose_threshold"),
        ("texture_score", "Texture P(real)", cfg.texture_threshold, "texture_threshold"),
    ]


def load_frames(d: Path) -> list[np.ndarray]:
    files = sorted(d.glob("frame_*.jpg")) + sorted(d.glob("frame_*.png"))
    return [cv2.imdecode(np.fromfile(str(f), np.uint8), cv2.IMREAD_COLOR) for f in files]


def fmt(v, spec=".3f"):
    return "-" if v is None else format(v, spec)


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("captures_dir", nargs="?", default=str(DEFAULT_DIR))
    ap.add_argument("--suggest", action="store_true", help="propose thresholds halfway between the groups")
    args = ap.parse_args(argv)

    root = Path(args.captures_dir)
    dirs = sorted(p for p in root.iterdir() if p.is_dir()) if root.exists() else []
    if not dirs:
        sys.exit(f"No captures in {root}. Run webcam_test.py --save-captures first.")
    cfg = VisionConfig()
    models = VisionModels.load(cfg)
    template = np.load(TEMPLATE_FILE) if TEMPLATE_FILE.exists() else None

    # ---- 1. every attempt -------------------------------------------------------------------
    print(f"Parallax method that decides: {cfg.parallax_method}\n")
    print(f"{'capture':<50} {'group':<22} {'mode':<7} {'decision':<8} {'reason':<17} {'sim':>5} "
          f"{'par-LM':>7} {'par-FLOW':>8} {'pts':>4} {'relief':>6} {'nose':>6} {'tex':>5}")
    by_label = defaultdict(list)   # genuine / photo / screen / unlabelled
    by_group = defaultdict(list)   # genuine, photo/printed_flat, screen/phone_video, ...
    for d in dirs:
        meta = json.loads((d / "result.json").read_text()) if (d / "result.json").exists() else {}
        label, mode = meta.get("label", "unlabelled"), meta.get("mode", "checkin")
        group = f"{label}/{meta.get('attack_type') or 'unspecified'}" if label in ATTACK_LABELS else label
        tpl = template if mode == "checkin" else None
        r = run_pipeline(load_frames(d), models, cfg, template=tpl)
        by_label[label].append(r)
        by_group[group].append(r)
        print(f"{d.name:<50} {group:<22} {r.mode:<7} {r.decision:<8} {r.reason:<17} {fmt(r.similarity, '.2f'):>5} "
              f"{fmt(r.parallax_landmark_score, '.4f'):>7} {fmt(r.parallax_flow_score, '.4f'):>8} "
              f"{fmt(r.parallax_flow_points, 'd'):>4} {fmt(r.depth_relief):>6} {fmt(r.depth_nose):>6} "
              f"{fmt(r.texture_score):>5}")

    labels = [lb for lb in LABEL_ORDER if lb in by_label] + sorted(set(by_label) - set(LABEL_ORDER))
    groups = sorted(by_group, key=lambda g: (LABEL_ORDER.index(g.split("/")[0]) if g.split("/")[0] in LABEL_ORDER
                                             else 99, g))

    def count(rs, attr):
        c = defaultdict(int)
        for r in rs:
            for k in (getattr(r, attr) if isinstance(getattr(r, attr), list) else [getattr(r, attr)]):
                c[k] += 1
        return dict(sorted(c.items(), key=lambda kv: -kv[1]))

    def pct(k, n):
        return f"{k}/{n} ({100 * k / n:.0f}%)" if n else "0/0"

    # ---- outcomes -------------------------------------------------------------------------------
    print("\nOutcomes with the current thresholds")
    gen = by_label.get("genuine", [])
    if gen:
        n = len(gen)
        acc = [r for r in gen if r.decision == "pass"]
        rej = [r for r in gen if r.decision == "fail"]
        ret = [r for r in gen if r.decision == "retry"]
        print(f"  Genuine (real student), n={n}:")
        print(f"    accepted                      {pct(len(acc), n)}")
        print(f"    REJECTED (false rejection)    {pct(len(rej), n)}" +
              (f"   failed checks: {count(rej, 'failed_checks')}" if rej else ""))
        print(f"    asked to RETRY                {pct(len(ret), n)}" +
              (f"   reasons: {count(ret, 'reason')}" if ret else ""))
        decided = len(acc) + len(rej)
        if decided:
            print(f"    false rejection rate among attempts that got a decision: {pct(len(rej), decided)}")
    attack_groups = [g for g in groups if g.split("/")[0] in ATTACK_LABELS]
    if attack_groups:
        print("  Attacks (an ACCEPTED attack is a security failure):")
        for g in attack_groups:
            rs = by_group[g]
            n = len(rs)
            acc = sum(r.decision == "pass" for r in rs)
            rej = [r for r in rs if r.decision == "fail"]
            ret = [r for r in rs if r.decision == "retry"]
            print(f"    {g:<24} n={n:<3} ACCEPTED {pct(acc, n):<12} rejected {pct(len(rej), n):<12} "
                  f"retry {pct(len(ret), n)}")
            if rej:
                print(f"    {'':<24} rejected by: {count(rej, 'failed_checks')}")
            if ret:
                print(f"    {'':<24} retry reasons: {count(ret, 'reason')}")
    if by_label.get("unlabelled"):
        print(f"  Unlabelled: {len(by_label['unlabelled'])} attempt(s) (not counted above)")

    # ---- 2. score distributions --------------------------------------------------------------
    def values(label, attr):
        return [getattr(r, attr) for r in by_label.get(label, []) if getattr(r, attr) is not None]

    def gvalues(group, attr):
        return [getattr(r, attr) for r in by_group.get(group, []) if getattr(r, attr) is not None]

    print("\nScore distributions (min / median / max, n = attempts that have this score):")
    for attr, name, thr, _ in checks(cfg):
        print(f"  {name}  (current threshold {thr})")
        for g in groups:
            v = gvalues(g, attr)
            if v:
                print(f"    {g:<24} n={len(v):<3} {min(v):8.4f} {float(np.median(v)):8.4f} {max(v):8.4f}")
            else:
                print(f"    {g:<24} n=0")

    # ---- 3. margins + suggestions --------------------------------------------------------------
    print("\nMargin = lowest genuine - highest attack (positive = separable on this data):")
    suggestions = []
    for attr, name, thr, field in checks(cfg):
        gen = values("genuine", attr)
        att = {lb: values(lb, attr) for lb in ATTACK_LABELS}
        att_all = att["photo"] + att["screen"]
        if not gen or not att_all:
            print(f"  {name:<22} not enough data (genuine n={len(gen)}, attacks n={len(att_all)})")
            suggestions.append((name, field, thr, None, ["need both genuine and attack attempts"]))
            continue
        margin = min(gen) - max(att_all)
        per = "  ".join(f"{g}: {min(gen) - max(gvalues(g, attr)):+.4f}" for g in attack_groups if gvalues(g, attr))
        verdict = "separable" if margin > 0 else "OVERLAP"
        print(f"  {name:<22} {margin:+.4f}  {verdict:<9}  (lowest genuine {min(gen):.4f}, highest attack "
              f"{max(att_all):.4f};  {per})")
        warnings = [f"only {n} {lb} attempt(s) with this score (want >= {MIN_PER_LABEL})"
                    for lb, n in [("genuine", len(gen)), ("photo", len(att["photo"])), ("screen", len(att["screen"]))]
                    if n < MIN_PER_LABEL]
        if margin > 0:
            suggested = (min(gen) + max(att_all)) / 2
        else:
            suggested = None
            hi_att = max(att_all)
            lo_gen = min(gen)
            warnings.insert(0, f"groups OVERLAP: {sum(g <= hi_att for g in gen)}/{len(gen)} genuine at or below the "
                               f"highest attack, {sum(a >= lo_gen for a in att_all)}/{len(att_all)} attacks at or "
                               f"above the lowest genuine - no threshold separates them")
        suggestions.append((name, field, thr, suggested, warnings))

    if args.suggest:
        print("\nSuggested thresholds (halfway between lowest genuine and highest attack):")
        for name, field, thr, suggested, warnings in suggestions:
            s = "no suggestion" if suggested is None else f"{suggested:.4f}"
            print(f"  {field:<26} current {thr:<8} -> {s}")
            for w in warnings:
                print(f"      WARNING: {w}")
        print("\nNothing was changed. Edit app/config.py to apply a value, then run this again to check it.")


if __name__ == "__main__":
    main(sys.argv[1:])
