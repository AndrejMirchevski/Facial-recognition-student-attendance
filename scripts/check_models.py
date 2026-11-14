"""Smoke test: prove every model loads and runs on this machine (used by setup.bat).

Usage:
    .venv\\Scripts\\python scripts\\check_models.py                       # synthetic inputs only
    .venv\\Scripts\\python scripts\\check_models.py face1.jpg face2.jpg   # also run on real photos

With real photos it runs each step of the pipeline on each image (landmarks -> alignment ->
ArcFace embedding, depth scores, MiniFASNet P(real)) using the real code in app/vision/, and,
if 2+ images are given, prints the pairwise cosine similarity of the embeddings.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import mediapipe as mp  # noqa: E402
import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402

from app.config import VisionConfig  # noqa: E402
from app.vision import VisionModels  # noqa: E402
from app.vision.align import align_arcface  # noqa: E402
from app.vision.detector import Face, select_main_face  # noqa: E402
from app.vision.liveness_depth import depth_frame_scores  # noqa: E402
from app.vision.quality import trim_black_borders  # noqa: E402


def timed(label, fn, *a, **kw):
    t = time.perf_counter()
    out = fn(*a, **kw)
    print(f"  {label:<38} {1000 * (time.perf_counter() - t):7.1f} ms")
    return out


def synthetic_check(m: VisionModels, cfg: VisionConfig):
    print("\nSynthetic-input run (proves each model executes)")
    rng = np.random.default_rng(0)
    noise = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
    faces = timed("Face Landmarker on noise", m.detector.detect, noise)
    print(f"    faces found on noise: {len(faces)} (expected 0)")
    e = timed("ArcFace on noise", m.embedder.embed, noise[:112, :112].copy())
    print(f"    embedding dim {e.shape[0]}, norm {np.linalg.norm(e):.3f}")
    d = timed(f"Depth Anything ({cfg.depth_input_size}px)", m.depth.predict, noise)
    print(f"    depth map {d.shape}")
    # a fake "face" in the middle of the noise image, just to run MiniFASNet
    pts = np.zeros((478, 2), np.float32)
    pts[:, 0] = rng.uniform(270, 370, 478)
    pts[:, 1] = rng.uniform(190, 290, 478)
    p = timed("MiniFASNet V2 + V1SE", m.texture.real_probs, noise, Face(pts))
    print(f"    P(real) on noise = {p[0]:.3f} / {p[1]:.3f}")


def image_check(m: VisionModels, cfg: VisionConfig, path: Path):
    print(f"\nImage: {path.name}")
    bgr = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        print("  could not decode image")
        return None
    bgr, trimmed = trim_black_borders(bgr)
    if any(trimmed):
        print(f"    letterbox removed (top, bottom, left, right): {trimmed}")
    faces = timed("Face Landmarker", m.detector.detect, bgr)
    sel = select_main_face(faces, cfg.duplicate_overlap, cfg.secondary_face_min_area_ratio)
    print(f"    {bgr.shape[1]}x{bgr.shape[0]}, faces: {len(faces)} "
          f"(main {'yes' if sel.main else 'no'}, other {len(sel.others)}, ignored {len(sel.ignored)})")
    if sel.main is None:
        return None
    face = sel.main
    aligned = align_arcface(bgr, face)
    emb = timed("ArcFace embedding", m.embedder.embed, aligned)
    dr = timed("Depth Anything on face crop", depth_frame_scores, bgr, face, m.depth, cfg.depth_crop_scale)
    print(f"    depth relief {dr.relief:.3f}, nose {dr.nose:+.3f}")
    p = timed("MiniFASNet V2 + V1SE", m.texture.real_probs, bgr, face)
    print(f"    P(real): V2 {p[0]:.3f}, V1SE {p[1]:.3f}, mean {np.mean(p):.3f}")
    return emb


def main(argv):
    print(f"onnxruntime {ort.__version__}, mediapipe {mp.__version__}, opencv {cv2.__version__}")
    cfg = VisionConfig()
    print("Loading models")
    m = timed("all models", VisionModels.load, cfg)
    synthetic_check(m, cfg)
    embs = {}
    for a in argv:
        e = image_check(m, cfg, Path(a))
        if e is not None:
            embs[Path(a).name] = e
    if len(embs) >= 2:
        print("\nCosine similarity between images (same person ~0.5-0.9, different people ~0-0.3):")
        names = list(embs)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                print(f"  {names[i]} vs {names[j]}: {float(embs[names[i]] @ embs[names[j]]):.3f}")
    print("\nAll models loaded and ran successfully.")


if __name__ == "__main__":
    main(sys.argv[1:])
