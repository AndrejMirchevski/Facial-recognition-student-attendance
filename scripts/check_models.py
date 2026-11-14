"""Stage-1 smoke test: prove every model loads and runs on this machine.

Usage:
    .venv\\Scripts\\python scripts\\check_models.py                       # synthetic inputs only
    .venv\\Scripts\\python scripts\\check_models.py face1.jpg face2.jpg   # also run on real photos

With real photos it runs the full chain per image (MediaPipe landmarks -> 5-point alignment
-> ArcFace embedding, Depth Anything on the face crop, MiniFASNet real/spoof probability)
and, if 2+ images are given, prints the pairwise cosine similarity of the embeddings.

This script is deliberately self-contained. The real implementation lives in app/vision/
(stage 2); this file only proves the models and their pre-processing work.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import onnxruntime as ort
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

MODELS = Path(__file__).resolve().parent.parent / "models"

# Standard ArcFace 112x112 reference positions: image-left eye, image-right eye, nose tip,
# image-left mouth corner, image-right mouth corner (as used by InsightFace).
ARCFACE_TEMPLATE = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                             [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)
# Matching MediaPipe Face Landmarker indices: iris centres 468/473, nose tip 1, mouth corners 61/291.
MP_FIVE_POINTS = [468, 473, 1, 61, 291]


def timed(label, fn, *a, **kw):
    t = time.perf_counter()
    out = fn(*a, **kw)
    print(f"  {label:<38} {1000 * (time.perf_counter() - t):7.1f} ms")
    return out


def ort_session(name: str) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.log_severity_level = 3
    return ort.InferenceSession(str(MODELS / name), so, providers=["CPUExecutionProvider"])


def load_all():
    print("Loading models")
    opts = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_buffer=(MODELS / "face_landmarker.task").read_bytes()),
        running_mode=mp_vision.RunningMode.IMAGE, num_faces=2)
    return {
        "landmarker": timed("MediaPipe Face Landmarker", mp_vision.FaceLandmarker.create_from_options, opts),
        "arcface": timed("ArcFace w600k_r50", ort_session, "w600k_r50.onnx"),
        "depth": timed("Depth Anything V2 Small", ort_session, "depth_anything_v2_small.onnx"),
        "fas_v2": timed("MiniFASNetV2", ort_session, "MiniFASNetV2.onnx"),
        "fas_v1se": timed("MiniFASNetV1SE", ort_session, "MiniFASNetV1SE.onnx"),
    }


# ---------------------------------------------------------------- per-model helpers
def embed(sess, face112_bgr):
    x = cv2.cvtColor(face112_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    x = ((x - 127.5) / 127.5).transpose(2, 0, 1)[None]
    e = sess.run(None, {sess.get_inputs()[0].name: x})[0][0]
    return e / np.linalg.norm(e)


def depth_map(sess, bgr, size=266):  # size must be a multiple of 14
    x = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (size, size)).astype(np.float32) / 255.0
    x = ((x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]).astype(np.float32).transpose(2, 0, 1)[None]
    return sess.run(None, {"pixel_values": x})[0][0]  # relative inverse depth: larger = closer


def fas_real_prob(sess, bgr, bbox_xywh, scale):
    """MiniFASNet as in Silent-Face-Anti-Spoofing: enlarged square-ish crop, 80x80, raw BGR 0-255."""
    h, w = bgr.shape[:2]
    x, y, bw, bh = bbox_xywh
    scale = min((h - 1) / bh, (w - 1) / bw, scale)
    cx, cy, nw, nh = x + bw / 2, y + bh / 2, bw * scale, bh * scale
    x1, y1 = max(0, int(cx - nw / 2)), max(0, int(cy - nh / 2))
    x2, y2 = min(w - 1, int(cx + nw / 2)), min(h - 1, int(cy + nh / 2))
    crop = cv2.resize(bgr[y1:y2 + 1, x1:x2 + 1], (80, 80)).astype(np.float32).transpose(2, 0, 1)[None]
    logits = sess.run(None, {"input": crop})[0][0]
    p = np.exp(logits - logits.max())
    p /= p.sum()
    return float(p[1])  # class 1 = real (0 and 2 = spoof types)


# ---------------------------------------------------------------- checks
def synthetic_check(m):
    print("\nSynthetic-input run (proves each model executes)")
    rng = np.random.default_rng(0)
    noise = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
    res = timed("Face Landmarker on noise", m["landmarker"].detect,
                mp.Image(image_format=mp.ImageFormat.SRGB, data=noise))
    print(f"    faces found on noise: {len(res.face_landmarks)} (expected 0)")
    e = timed("ArcFace on noise", embed, m["arcface"], noise[:112, :112].copy())
    print(f"    embedding dim {e.shape[0]}, norm {np.linalg.norm(e):.3f}")
    d = timed("Depth Anything (266x266)", depth_map, m["depth"], noise)
    print(f"    depth map {d.shape}")
    p = timed("MiniFASNetV2", fas_real_prob, m["fas_v2"], noise, (200, 150, 200, 200), 2.7)
    print(f"    P(real) on noise = {p:.3f}")


def image_check(m, path: Path):
    print(f"\nImage: {path.name}")
    bgr = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        print("  could not decode image")
        return None
    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    res = timed("Face Landmarker", m["landmarker"].detect, mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    print(f"    {w}x{h}, faces: {len(res.face_landmarks)}")
    if len(res.face_landmarks) != 1:
        return None
    lm = np.array([[p.x * w, p.y * h] for p in res.face_landmarks[0]], dtype=np.float32)
    print(f"    landmarks: {len(lm)}")

    five = lm[MP_FIVE_POINTS]
    assert five[0, 0] < five[1, 0] and five[3, 0] < five[4, 0], "landmark left/right order unexpected"
    M, _ = cv2.estimateAffinePartial2D(five, ARCFACE_TEMPLATE, method=cv2.LMEDS)
    aligned = cv2.warpAffine(bgr, M, (112, 112), borderValue=0)
    resid = np.linalg.norm((five @ M[:, :2].T + M[:, 2]) - ARCFACE_TEMPLATE, axis=1).mean()
    print(f"    alignment: mean residual to ArcFace template {resid:.2f} px (of 112)")
    emb = timed("ArcFace embedding", embed, m["arcface"], aligned)

    x0, y0 = lm.min(0)
    x1, y1 = lm.max(0)
    bbox = (int(x0), int(y0), int(x1 - x0), int(y1 - y0))
    pad = int(0.15 * max(bbox[2], bbox[3]))
    crop = bgr[max(0, bbox[1] - pad):bbox[1] + bbox[3] + pad, max(0, bbox[0] - pad):bbox[0] + bbox[2] + pad]
    d = timed("Depth Anything on face crop", depth_map, m["depth"], crop)
    # nose tip vs cheeks in the depth map (larger = closer), just to show the signal exists
    sx, sy = d.shape[1] / crop.shape[1], d.shape[0] / crop.shape[0]
    ox, oy = max(0, bbox[0] - pad), max(0, bbox[1] - pad)
    val = lambda i: float(d[int((lm[i, 1] - oy) * sy), int((lm[i, 0] - ox) * sx)])
    print(f"    depth: nose {val(1):.2f}, cheeks {val(234):.2f}/{val(454):.2f}, "
          f"range {d.min():.2f}..{d.max():.2f}")

    p2 = timed("MiniFASNetV2 (scale 2.7)", fas_real_prob, m["fas_v2"], bgr, bbox, 2.7)
    p1 = timed("MiniFASNetV1SE (scale 4.0)", fas_real_prob, m["fas_v1se"], bgr, bbox, 4.0)
    print(f"    P(real): V2 {p2:.3f}, V1SE {p1:.3f}, mean {(p1 + p2) / 2:.3f}")
    return emb


def main(argv):
    print(f"onnxruntime {ort.__version__}, mediapipe {mp.__version__}, opencv {cv2.__version__}")
    m = load_all()
    synthetic_check(m)
    embs = {}
    for a in argv:
        e = image_check(m, Path(a))
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
