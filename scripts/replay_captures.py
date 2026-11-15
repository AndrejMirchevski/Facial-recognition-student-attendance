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

The analysis itself lives in app/calibration.py and is shared with scripts/threshold_report.py,
which does the same for the attempts recorded by the web app.

All liveness scores are "higher = more likely real". Attempts stopped early (e.g. face too close)
have no liveness scores and only appear in the first table. Check-in captures are compared against
instance/webcam_test/template.npy (identity only; it does not affect the liveness scores).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from app.calibration import Record, analysis_lines, fmt  # noqa: E402
from app.config import VisionConfig  # noqa: E402
from app.vision import VisionModels, run_pipeline  # noqa: E402

DEFAULT_DIR = ROOT / "instance" / "webcam_test" / "captures"
TEMPLATE_FILE = ROOT / "instance" / "webcam_test" / "template.npy"


def load_frames(d: Path) -> list[np.ndarray]:
    files = sorted(d.glob("frame_*.jpg")) + sorted(d.glob("frame_*.png"))
    return [cv2.imdecode(np.fromfile(str(f), np.uint8), cv2.IMREAD_COLOR) for f in files]


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
    records = []
    for d in dirs:
        meta = json.loads((d / "result.json").read_text()) if (d / "result.json").exists() else {}
        mode = meta.get("mode", "checkin")
        r = run_pipeline(load_frames(d), models, cfg, template=template if mode == "checkin" else None)
        rec = Record.from_result(r, meta.get("label"), meta.get("attack_type"), name=d.name)
        records.append(rec)
        print(f"{d.name:<50} {rec.group:<22} {r.mode:<7} {r.decision:<8} {r.reason:<17} {fmt(r.similarity, '.2f'):>5} "
              f"{fmt(r.parallax_landmark_score, '.4f'):>7} {fmt(r.parallax_flow_score, '.4f'):>8} "
              f"{fmt(r.parallax_flow_points, 'd'):>4} {fmt(r.depth_relief):>6} {fmt(r.depth_nose):>6} "
              f"{fmt(r.texture_score):>5}")

    # ---- 2.-4. outcomes, distributions, margins (+ suggestions): shared with threshold_report.py
    print("\n" + "\n".join(analysis_lines(records, cfg, suggest=args.suggest,
                                          outcome_title="Outcomes with the current thresholds")))


if __name__ == "__main__":
    main(sys.argv[1:])
