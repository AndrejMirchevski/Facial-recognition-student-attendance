"""Live webcam test of the vision pipeline - shows every score live, runs full 5-frame attempts.

Usage:
    .venv\\Scripts\\python scripts\\webcam_test.py [--camera 0] [--save-captures]

Keys (click the window first):
    E      enroll: guided 5-frame capture, run the pipeline, save your template (local test file)
    SPACE  check-in attempt: guided 5-frame capture, full pipeline against the template
    (Guided capture: look straight -> turn LEFT -> centre -> turn RIGHT -> centre. Each picture
     is taken when you have done the step; ESC cancels. A photo cannot turn, so its steps are
     taken after a few seconds anyway and the checks then judge it.)
    1 / 2 / 3 / 0   set the default label (genuine / photo / screen / none) - see below
    D      live depth on/off (it is the slowest model)
    S      save the current camera frame (for debugging)
    Q/ESC  quit

What you see live (single frames, refreshed continuously):
    face count, eye distance, face size, brightness, sharpness, letterbox trimming,
    similarity to your template, texture P(real) of both MiniFASNet models, depth relief/nose,
    and rolling parallax scores (landmark method and optical-flow method) over the last 2.5 s.
    The method marked * is the one that decides (config.parallax_method).
    Thumbnails show exactly what the texture models and the depth model look at.
A full attempt (E / SPACE) uses the real pipeline (app/vision/pipeline.py) and its decision.

After every attempt you are asked what it was: 1 genuine / 2 photo / 3 screen (Enter = the
label used last time, Esc = leave unlabelled), and for attacks the attack type:
    photo:  1 printed flat   2 printed bent
    screen: 1 phone photo    2 phone video    3 monitor photo
Every attempt is then appended to instance/webcam_test/attempts.csv with its label, attack type
and all scores.
With --save-captures the 5 frames of each attempt are also saved (instance/webcam_test/captures/),
so they can be re-analysed later with scripts/replay_captures.py after changing thresholds.
Everything stays in instance/, which is git-ignored. Delete the folder to remove it all.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from app.config import VisionConfig  # noqa: E402
from app.vision import VisionModels, run_pipeline  # noqa: E402
from app.vision.align import FIVE_POINTS, align_arcface, square_crop_box  # noqa: E402
from app.vision.detector import select_main_face  # noqa: E402
from app.vision.embedder import cosine  # noqa: E402
from app.vision.guidance import GuidedCapture  # noqa: E402
from app.vision.liveness_depth import depth_frame_scores  # noqa: E402
from app.vision.liveness_flow import flow_parallax_check  # noqa: E402
from app.vision.liveness_parallax import parallax_check  # noqa: E402
from app.vision.liveness_texture import FAS_MODELS, TextureAntiSpoof  # noqa: E402
from app.vision.pipeline import assess_frame, flow_params  # noqa: E402
from app.vision.quality import brightness, sharpness, trim_black_borders  # noqa: E402

OUT_DIR = ROOT / "instance" / "webcam_test"
TEMPLATE_FILE = OUT_DIR / "template.npy"
CSV_FILE = OUT_DIR / "attempts.csv"
LABEL_KEYS = {ord("0"): "unlabelled", ord("1"): "genuine", ord("2"): "photo", ord("3"): "screen"}
# attack sub-types, so the paper can report each attack separately
ATTACK_TYPES = {"photo": ["printed_flat", "printed_bent"],
                "screen": ["phone_photo", "phone_video", "monitor_photo"]}
ENTER_KEYS = (13, 10)

GREEN, RED, GREY, WHITE, YELLOW, BLUE = (90, 200, 90), (70, 70, 235), (150, 150, 150), (235, 235, 235), (0, 210, 240), (230, 160, 60)
FONT = cv2.FONT_HERSHEY_SIMPLEX
PANEL_W = 470

CSV_FIELDS = ["time", "label", "attack_type", "mode", "decision", "reason", "failed_checks", "similarity", "consistency",
              "parallax_method", "parallax_landmark_score", "parallax_landmark_movement",
              "parallax_flow_score", "parallax_flow_movement", "parallax_flow_points",
              "depth_relief", "depth_nose", "texture_score",
              "frame_texture", "frame_brightness", "frame_interocular", "processing_ms", "capture_dir"]


# ----------------------------------------------------------------------------- helpers
def open_camera(index: int, width: int, height: int) -> cv2.VideoCapture:
    for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY):
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            ok, _ = cap.read()
            if ok:
                return cap
            cap.release()
    sys.exit(f"Could not open camera {index}. Try --camera 1, or close other apps using the camera.")


def jpeg_roundtrip(img: np.ndarray, quality: int) -> tuple[np.ndarray, bytes | None]:
    """Encode/decode like the browser upload will (canvas -> JPEG). quality 0 = no-op."""
    if quality <= 0:
        return img, None
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR), enc.tobytes()


def put(img, text, org, color=WHITE, scale=0.52, thick=1, outline=False):
    """Draw text; `outline` adds a dark border for readability on top of the camera image."""
    if outline:
        cv2.putText(img, text, org, FONT, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, FONT, scale, color, thick, cv2.LINE_AA)


def judge(value, threshold, higher_is_better=True):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return GREY
    ok = value >= threshold if higher_is_better else value <= threshold
    return GREEN if ok else RED


def fit_height(img, h):
    return cv2.resize(img, (int(img.shape[1] * h / img.shape[0]), h))


def depth_vis(depth_map, mask, size):
    vis = cv2.applyColorMap((np.clip(depth_map, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    vis[mask == 0] = (vis[mask == 0] * 0.35).astype(np.uint8)
    return cv2.resize(vis, (size, size))


# ----------------------------------------------------------------------------- live analysis
class Live:
    def __init__(self, models: VisionModels, cfg: VisionConfig):
        self.m, self.cfg = models, cfg
        self.history: deque = deque(maxlen=cfg.num_frames)  # (t, grey image, face) every ~0.5 s

    def analyse(self, img, template, do_depth) -> dict:
        cfg, m = self.cfg, self.m
        info: dict = {}
        sel = select_main_face(m.detector.detect(img), cfg.duplicate_overlap, cfg.secondary_face_min_area_ratio)
        info["sel"] = sel
        if sel.main is None or sel.others:
            self.history.clear()
            return info
        f = sel.main
        al = align_arcface(img, f)
        info.update(iod=f.interocular, hfrac=f.height / img.shape[0], bright=brightness(al), sharp=sharpness(al),
                    tex=m.texture.real_probs(img, f),
                    tex_crops=[TextureAntiSpoof.crop(img, f, s) for _, s in FAS_MODELS])
        if template is not None:
            info["sim"] = cosine(m.embedder.embed(al), template)
        if do_depth:
            dr = depth_frame_scores(img, f, m.depth, cfg.depth_crop_scale)
            info.update(relief=dr.relief, nose=dr.nose, depth_vis=depth_vis(dr.depth_map, dr.mask, 130))
        now = time.time()
        if not self.history or now - self.history[-1][0] >= 0.5:
            self.history.append((now, cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), f))
        if len(self.history) == cfg.num_frames:
            grays, hfaces = [h[1] for h in self.history], [h[2] for h in self.history]
            info["parallax"] = parallax_check([hf.landmarks for hf in hfaces], [hf.interocular for hf in hfaces],
                                              min_movement=cfg.parallax_min_movement,
                                              threshold=cfg.parallax_threshold,
                                              ransac_reproj=cfg.parallax_ransac_reproj)
            if all(g.shape == grays[0].shape for g in grays):
                info["parallax_flow"] = flow_parallax_check(
                    grays, hfaces, min_movement=cfg.parallax_flow_min_movement,
                    threshold=cfg.parallax_flow_threshold, ransac_reproj=cfg.parallax_ransac_reproj,
                    params=flow_params(cfg))
        return info


def draw_faces(img, info):
    sel = info.get("sel")
    if sel is None:
        return
    for f in sel.ignored:
        x0, y0, x1, y1 = map(int, f.bbox)
        cv2.rectangle(img, (x0, y0), (x1, y1), YELLOW, 1)
    for f in sel.others:
        x0, y0, x1, y1 = map(int, f.bbox)
        cv2.rectangle(img, (x0, y0), (x1, y1), RED, 2)
    if sel.main is not None:
        f = sel.main
        x0, y0, x1, y1 = map(int, f.bbox)
        cv2.rectangle(img, (x0, y0), (x1, y1), GREEN, 2)
        # what MiniFASNetV2 sees (scale 2.7 crop, shifted inside the frame)
        cx, cy, side = f.square_box()
        bx = square_crop_box(img.shape[1], img.shape[0], cx, cy, side * 2.7, side * 2.7)
        cv2.rectangle(img, bx[:2], bx[2:], BLUE, 1)
        for p in f.landmarks[FIVE_POINTS]:
            cv2.circle(img, (int(p[0]), int(p[1])), 3, GREEN, -1)


def live_panel(h, info, cfg, state) -> np.ndarray:
    p = np.full((h, PANEL_W, 3), 32, np.uint8)
    y = 26

    def line(text, color=WHITE, scale=0.52, dy=23):
        nonlocal y
        put(p, text, (12, y), color, scale)
        y += dy

    line(f"LIVE   default label: {state['label'].upper()}   {state['fps']:.1f} fps", YELLOW, 0.58, 28)
    sel = info.get("sel")
    n_faces = 0 if sel is None or sel.main is None else 1 + len(sel.others)
    ign = f"  (+{len(sel.ignored)} ignored duplicate/small)" if sel and sel.ignored else ""
    line(f"Faces: {n_faces}{ign}", GREEN if n_faces == 1 else RED)
    t = state["trimmed"]
    if any(t):
        line(f"Letterbox removed: top {t[0]} bottom {t[1]} left {t[2]} right {t[3]} px", YELLOW)
    if "iod" in info:
        line(f"Eye distance {info['iod']:.0f} px   (min {cfg.min_interocular_px:.0f})", judge(info["iod"], cfg.min_interocular_px))
        line(f"Face height {info['hfrac']:.2f} of frame   (max {cfg.max_face_height_frac:.2f})",
             judge(info["hfrac"], cfg.max_face_height_frac, False))
        line(f"Brightness {info['bright']:.0f}   (min {cfg.min_face_brightness:.0f})", judge(info["bright"], cfg.min_face_brightness))
        line(f"Sharpness {info['sharp']:.0f}   (min {cfg.min_sharpness:.0f})", judge(info["sharp"], cfg.min_sharpness))
    y += 6
    line("IDENTITY", YELLOW)
    if state["template"] is None:
        line("not enrolled - press E", GREY)
    elif "sim" in info:
        line(f"Similarity {info['sim']:.3f}   (min {cfg.identity_threshold:.2f})", judge(info["sim"], cfg.identity_threshold))
    y += 6
    line("LIVENESS (this frame)", YELLOW)
    if "tex" in info:
        tex = float(np.mean(info["tex"]))
        line(f"Texture P(real) {tex:.3f}   (min {cfg.texture_threshold:.2f})", judge(tex, cfg.texture_threshold))
        line(f"    V2 {info['tex'][0]:.3f}   V1SE {info['tex'][1]:.3f}", GREY)
    if not state["depth_on"]:
        line("Depth: off (press D)", GREY)
    elif "relief" in info:
        line(f"Depth relief {info['relief']:.3f}   (min {cfg.depth_relief_threshold:.3f})",
             judge(info["relief"], cfg.depth_relief_threshold))
        line(f"Depth nose {info['nose']:+.3f}   (min {cfg.depth_nose_threshold:.3f})", judge(info["nose"], cfg.depth_nose_threshold))
    for key, label, thr, mv_min in (("parallax", "landmarks", cfg.parallax_threshold, cfg.parallax_min_movement),
                                    ("parallax_flow", "flow", cfg.parallax_flow_threshold, cfg.parallax_flow_min_movement)):
        star = "*" if cfg.parallax_method == label else ""
        par = info.get(key)
        if par is None:
            line(f"Parallax {label}{star}: collecting (2.5 s)...", GREY, 0.48, 22)
        elif par.score is None:
            why = "too few points" if par.status == "insufficient_points" else "turn your head"
            line(f"Parallax {label}{star}: {why}  (mv {par.movement:.3f} / {mv_min:.2f})", GREY, 0.48, 22)
        else:
            pts = f" pts {par.points}" if key == "parallax_flow" else ""
            line(f"Parallax {label}{star} {par.score:.4f} (min {thr:.3f}) mv {par.movement:.3f}{pts}",
                 judge(par.score, thr), 0.48, 22)

    # thumbnails: what the models see
    ty = h - 175
    put(p, "Texture V2 x2.7", (12, ty - 8), GREY, 0.42)
    put(p, "Texture V1SE x4", (157, ty - 8), GREY, 0.42)
    put(p, "Depth (face)", (302, ty - 8), GREY, 0.42)
    for i, crop in enumerate(info.get("tex_crops", [])):
        p[ty:ty + 130, 12 + 145 * i:12 + 145 * i + 130] = cv2.resize(crop, (130, 130), interpolation=cv2.INTER_NEAREST)
    if "depth_vis" in info:
        p[ty:ty + 130, 302:432] = info["depth_vis"]
    put(p, "E enroll  SPACE check-in  1/2/3/0 label  D depth  S save  Q quit", (12, h - 14), WHITE, 0.42)
    return p


# ----------------------------------------------------------------------------- full attempts
def capture(cap, models, cfg, args, title) -> list[np.ndarray] | None:
    """Guided capture: each of the 5 frames is taken when the student has done the step."""
    guide = GuidedCapture(cfg)
    frames: list[np.ndarray] = []
    mirrored = not args.no_mirror
    flash_until = 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            return None
        now = time.time()
        seen, _ = jpeg_roundtrip(frame, args.jpeg_quality)      # judge what the server will see
        img, face, _, rep = assess_frame(seen, models.detector, cfg)
        st = guide.update(now, face, rep.problem)
        if st.capture:
            frames.append(frame.copy())
            flash_until = now + 0.25
        if st.aborted:
            print("No face found - capture cancelled.")
            return None
        if st.done:
            return frames

        disp = img.copy()
        if face is not None:
            x0, y0, x1, y1 = map(int, face.bbox)
            cv2.rectangle(disp, (x0, y0), (x1, y1), GREEN if rep.problem is None else RED, 2)
        if mirrored:
            disp = cv2.flip(disp, 1)
        disp = fit_height(disp, args.display_height)
        h, w = disp.shape[:2]

        # instruction with an arrow pointing the way the student should turn (as they see it)
        text = st.instruction
        if st.direction:
            on_left = (st.direction == "left") == mirrored
            text = f"<<  {text}" if on_left else f"{text}  >>"
        head = f"{title}   picture {min(st.step + 1, args.frames)} of {args.frames}"
        put(disp, head, (20, 34), WHITE, 0.65, 2, outline=True)
        dots_x = 40 + cv2.getTextSize(head, FONT, 0.65, 2)[0][0]
        put(disp, text, (20, 74), YELLOW, 0.95, 2, outline=True)
        if st.hint:
            put(disp, st.hint, (20, 110), (80, 160, 255), 0.75, 2, outline=True)
        if st.step > 0:
            put(disp, f"{st.time_left:.0f}s", (w - 60, 34), GREY, 0.65, 2, outline=True)
        for i in range(args.frames):  # progress dots
            cv2.circle(disp, (dots_x + 26 * i, 28), 9, GREEN if i < st.step else GREY, -1 if i < st.step else 2)

        # head-turn bar: marker = current turn, ticks = target for this step
        if st.step > 0:
            cx, by, scale = w // 2, h - 40, (w * 0.35) / 0.30
            sgn = -1 if mirrored else 1           # + turn (student's left) is drawn on the left when mirrored
            cv2.line(disp, (int(cx - 0.30 * scale), by), (int(cx + 0.30 * scale), by), GREY, 3)
            if st.direction:
                side = 1 if st.direction == "left" else -1
                tx = int(cx + sgn * side * st.target * scale)
                cv2.line(disp, (tx, by - 18), (tx, by + 18), GREEN, 3)
            else:
                tol = int(cfg.guide_centre_tolerance * scale)
                cv2.rectangle(disp, (cx - tol, by - 14), (cx + tol, by + 14), GREEN, 2)
            mx = int(cx + sgn * float(np.clip(st.turn, -0.3, 0.3)) * scale)
            cv2.circle(disp, (mx, by), 11, YELLOW, -1)
        if now < flash_until:
            cv2.rectangle(disp, (0, 0), (w - 1, h - 1), GREEN, 12)
        cv2.imshow(WINDOW, disp)
        if cv2.waitKey(1) & 0xFF in (27, ord("q")):
            return None


def result_panel(h, res, cfg, title) -> np.ndarray:
    p = np.full((h, PANEL_W, 3), 32, np.uint8)
    y = 30

    def line(text, color=WHITE, scale=0.52, dy=23):
        nonlocal y
        put(p, text, (12, y), color, scale)
        y += dy

    col = {"pass": GREEN, "fail": RED, "retry": YELLOW}[res.decision]
    line(f"{title}: {res.decision.upper()}", col, 0.9, 32)
    words, cur = res.message.split(), ""
    for wd in words:  # wrap the message
        if len(cur) + len(wd) > 52:
            line(cur, col, 0.47, 20)
            cur = ""
        cur += wd + " "
    line(cur, col, 0.47, 28)

    def chk(name):
        v = res.checks.get(name)
        return {True: ("PASS", GREEN), False: ("FAIL", RED), None: ("n/a", GREY)}[v]

    for name, text in [
        ("consistency", f"Same person in all frames: {res.consistency}" if res.consistency is not None else "Same person: -"),
        ("identity", f"Identity similarity: {res.similarity}  (min {cfg.identity_threshold})"),
        ("parallax_landmarks", f"Parallax landmarks{'*' if res.parallax_method == 'landmarks' else ''}: "
                               f"{res.parallax_landmark_score} (min {cfg.parallax_threshold}), mv {res.parallax_landmark_movement}"),
        ("parallax_flow", f"Parallax flow{'*' if res.parallax_method == 'flow' else ''}: {res.parallax_flow_score} "
                          f"(min {cfg.parallax_flow_threshold}), mv {res.parallax_flow_movement}, pts {res.parallax_flow_points}"),
        ("depth", f"Depth relief {res.depth_relief} (min {cfg.depth_relief_threshold}) nose {res.depth_nose} (min {cfg.depth_nose_threshold})"),
        ("texture", f"Texture P(real): {res.texture_score}  (min {cfg.texture_threshold})"),
    ]:
        s, c = chk(name)
        line(f"[{s}] {text}", c, 0.43)
    y += 8
    line("Per frame: problem | eye px | bright | tex(V2,V1SE) | relief", YELLOW, 0.43)
    for f in res.frames:
        tex = "-" if f.texture is None else f"{f.texture[0]:.2f},{f.texture[1]:.2f}"
        line(f"{f.index + 1}: {f.problem or 'ok'} | {f.interocular_px} | {f.brightness} | {tex} | {f.depth_relief}",
             RED if f.problem else WHITE, 0.43, 20)
    y += 8
    line(f"Time: {res.processing_ms:.0f} ms  (* = parallax method that decides)", GREY, 0.40, 18)
    line(" ".join(f"{k.replace('parallax_', 'par_')} {v:.0f}" for k, v in res.timings_ms.items()), GREY, 0.38)
    return p


RENAMED_COLUMNS = {"parallax_score": "parallax_landmark_score", "parallax_movement": "parallax_landmark_movement"}


def migrate_csv():
    """Rewrite an attempts.csv written by an older version of this script with the current columns."""
    if not CSV_FILE.exists():
        return
    with CSV_FILE.open(newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames == CSV_FIELDS:
            return
        rows = [{RENAMED_COLUMNS.get(k, k): v for k, v in row.items()} for row in reader]
    with CSV_FILE.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"Updated {CSV_FILE.relative_to(ROOT)} to the current column layout.")


def log_attempt(res, label, attack_type, capture_dir):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    migrate_csv()
    new = not CSV_FILE.exists()
    with CSV_FILE.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        if new:
            w.writeheader()
        w.writerow({
            "time": datetime.now().isoformat(timespec="seconds"), "label": label,
            "attack_type": attack_type or "", "mode": res.mode,
            "decision": res.decision, "reason": res.reason, "failed_checks": ";".join(res.failed_checks),
            "similarity": res.similarity, "consistency": res.consistency, "parallax_method": res.parallax_method,
            "parallax_landmark_score": res.parallax_landmark_score,
            "parallax_landmark_movement": res.parallax_landmark_movement,
            "parallax_flow_score": res.parallax_flow_score, "parallax_flow_movement": res.parallax_flow_movement,
            "parallax_flow_points": res.parallax_flow_points,
            "depth_relief": res.depth_relief, "depth_nose": res.depth_nose,
            "texture_score": res.texture_score,
            "frame_texture": json.dumps([f.texture for f in res.frames]),
            "frame_brightness": json.dumps([f.brightness for f in res.frames]),
            "frame_interocular": json.dumps([f.interocular_px for f in res.frames]),
            "processing_ms": res.processing_ms, "capture_dir": capture_dir or "",
        })


def print_result(res, title):
    print(f"\n=== {title}: {res.decision.upper()} ({res.reason}) - {res.message}")
    print(f"    checks {res.checks}")
    print(f"    similarity {res.similarity}  consistency {res.consistency}  "
          f"parallax landmarks {res.parallax_landmark_score} (mv {res.parallax_landmark_movement})  "
          f"flow {res.parallax_flow_score} (mv {res.parallax_flow_movement}, pts {res.parallax_flow_points}) "
          f"[{res.parallax_method} decides]  depth relief {res.depth_relief} nose {res.depth_nose}  "
          f"texture {res.texture_score}  [{res.processing_ms:.0f} ms]")


def ask_label(view: np.ndarray, state) -> tuple[str, str | None]:
    """Ask in the window what this attempt was. Returns (label, attack_type)."""
    h = view.shape[0]
    x = view.shape[1] - PANEL_W + 12

    def prompt(lines):
        img = view.copy()
        cv2.rectangle(img, (x - 12, h - 30 - 24 * len(lines)), (view.shape[1], h), (60, 45, 20), -1)
        for i, (text, col) in enumerate(lines):
            put(img, text, (x, h - 18 - 24 * (len(lines) - 1 - i)), col, 0.5)
        cv2.imshow(WINDOW, img)
        return cv2.waitKey(0) & 0xFF

    default = state["label"] if state["label"] in ("genuine", "photo", "screen") else None
    while True:
        key = prompt([("What was this attempt?", YELLOW),
                      ("1 genuine (real you)   2 photo   3 screen", WHITE),
                      (f"Enter = {default or '(no default)'}    Esc = leave unlabelled", GREY)])
        if key == 27:
            return "unlabelled", None
        if key in (ord("1"), ord("2"), ord("3")):
            label = LABEL_KEYS[key]
            break
        if key in ENTER_KEYS and default:
            label = default
            break
    state["label"] = label
    if label not in ATTACK_TYPES:
        return label, None

    types = ATTACK_TYPES[label]
    last = state["attack_type"].get(label)
    options = "   ".join(f"{i + 1} {t.replace('_', ' ')}" for i, t in enumerate(types))
    while True:
        key = prompt([(f"Which kind of {label} attack?", YELLOW), (options, WHITE),
                      (f"Enter = {last.replace('_', ' ') if last else '(no default)'}    Esc = not specified", GREY)])
        if key == 27:
            return label, None
        if ord("1") <= key < ord("1") + len(types):
            attack = types[key - ord("1")]
            break
        if key in ENTER_KEYS and last:
            attack = last
            break
    state["attack_type"][label] = attack
    return label, attack


def run_attempt(cap, models, cfg, args, state, enroll: bool):
    title = "ENROLL" if enroll else "CHECK-IN"
    if not enroll and state["template"] is None:
        print("Not enrolled yet - press E first.")
        return
    raw = capture(cap, models, cfg, args, title)
    if raw is None:
        return
    when = datetime.now()
    frames, jpegs = [], []
    for f in raw:
        img, enc = jpeg_roundtrip(f, args.jpeg_quality)
        frames.append(img)
        jpegs.append(enc)
    res = run_pipeline(frames, models, cfg, template=None if enroll else state["template"], keep_debug=True)
    print_result(res, title)

    disp = frames[len(frames) // 2].copy()
    if not args.no_mirror:
        disp = cv2.flip(disp, 1)
    disp = fit_height(disp, args.display_height)
    view = np.hstack([disp, result_panel(disp.shape[0], res, cfg, title)])

    # ask what it was, THEN log
    label, attack = ask_label(view, state)
    print(f"    labelled: {label}" + (f" / {attack}" if attack else ""))
    capture_dir = None
    if args.save_captures:
        name = f"{when:%Y%m%d_%H%M%S}_{res.mode}_{label}" + (f"_{attack}" if attack else "")
        d = OUT_DIR / "captures" / name
        d.mkdir(parents=True, exist_ok=True)
        for i, (img, enc) in enumerate(zip(frames, jpegs)):
            if enc is not None:
                (d / f"frame_{i}.jpg").write_bytes(enc)
            else:
                cv2.imwrite(str(d / f"frame_{i}.png"), img)
        meta = {"label": label, "attack_type": attack, **res.summary()}
        (d / "result.json").write_text(json.dumps(meta, indent=1, default=float))
        capture_dir = str(d.relative_to(ROOT))
    log_attempt(res, label, attack, capture_dir)

    if enroll and res.embedding is not None:
        save = res.passed
        if not res.passed and res.checks.get("consistency"):
            img = view.copy()
            h = img.shape[0]
            x = img.shape[1] - PANEL_W + 12
            cv2.rectangle(img, (x - 12, h - 80), (img.shape[1], h), (40, 30, 90), -1)
            put(img, "Liveness failed - the real app would refuse this.", (x, h - 50), WHITE, 0.47)
            put(img, "Y = save the template anyway (test only), other key = no", (x, h - 22), WHITE, 0.47)
            cv2.imshow(WINDOW, img)
            save = (cv2.waitKey(0) & 0xFF) in (ord("y"), ord("Y"))
        if save:
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            np.save(TEMPLATE_FILE, res.embedding)
            state["template"] = res.embedding
            print(f"Template saved to {TEMPLATE_FILE.relative_to(ROOT)}")


# ----------------------------------------------------------------------------- main
WINDOW = "Face attendance - webcam test"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, default=0, help="camera index (default 0)")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--display-height", type=int, default=600)
    ap.add_argument("--jpeg-quality", type=int, default=90,
                    help="simulate the browser's JPEG upload (0 = off). Default 90.")
    ap.add_argument("--save-captures", action="store_true",
                    help="save the 5 frames of every attempt to instance/webcam_test/captures/")
    ap.add_argument("--no-mirror", action="store_true", help="do not mirror the preview")
    args = ap.parse_args()

    cfg = VisionConfig()
    args.frames = cfg.num_frames
    print("Loading models ...")
    models = VisionModels.load(cfg)
    cap = open_camera(args.camera, args.width, args.height)
    print(f"Camera {args.camera}: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    template = np.load(TEMPLATE_FILE) if TEMPLATE_FILE.exists() else None
    if template is not None:
        print(f"Loaded template from {TEMPLATE_FILE.relative_to(ROOT)} (press E to re-enroll)")
    state = {"label": "unlabelled", "attack_type": {}, "template": template, "depth_on": True, "fps": 0.0,
             "trimmed": (0, 0, 0, 0)}
    live = Live(models, cfg)
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    last = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Camera read failed.")
            break
        img, _ = jpeg_roundtrip(frame, args.jpeg_quality)
        img, state["trimmed"] = trim_black_borders(img)
        info = live.analyse(img, state["template"], state["depth_on"])
        disp = img.copy()
        draw_faces(disp, info)
        if not args.no_mirror:
            disp = cv2.flip(disp, 1)
        disp = fit_height(disp, args.display_height)
        now = time.time()
        state["fps"] = 0.8 * state["fps"] + 0.2 / max(now - last, 1e-3)
        last = now
        cv2.imshow(WINDOW, np.hstack([disp, live_panel(disp.shape[0], info, cfg, state)]))

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break
        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break
        if key in LABEL_KEYS:
            state["label"] = LABEL_KEYS[key]
        elif key in (ord("d"), ord("D")):
            state["depth_on"] = not state["depth_on"]
        elif key in (ord("s"), ord("S")):
            d = OUT_DIR / "frames"
            d.mkdir(parents=True, exist_ok=True)
            path = d / f"{datetime.now():%Y%m%d_%H%M%S_%f}.png"
            cv2.imwrite(str(path), frame)
            print(f"Saved {path.relative_to(ROOT)}")
        elif key in (ord("e"), ord("E")):
            run_attempt(cap, models, cfg, args, state, enroll=True)
            live.history.clear()
        elif key == ord(" "):
            run_attempt(cap, models, cfg, args, state, enroll=False)
            live.history.clear()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
