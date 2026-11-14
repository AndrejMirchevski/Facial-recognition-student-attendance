"""Liveness check (a): parallax / planarity of the landmarks across frames. No model.

Idea (in plain words): when a FLAT object - a printed photo or a phone screen - moves or tilts
in front of a camera, every point on it moves according to one 3x3 "homography" transform. So
if we fit a homography between the landmark positions of two frames, a flat face is explained
almost perfectly. A real head is 3D: when it turns, the nose tip shifts relative to the cheeks
and the face outline (parallax), and no single homography can explain that motion.

For every pair of the 5 frames we measure, in units of the inter-ocular distance (IOD):
  * movement - how much the landmarks moved in a way that a simple shift/rotate/zoom cannot
    explain. If this is tiny the head did not turn and the test says nothing either way, so we
    ask the student to turn their head and try again.
  * planarity error - median distance between the landmarks and their homography prediction.
    Large => not flat => live. Small => flat => photo/screen.
The score is the median planarity error over the frame pairs that moved enough.

Two ways of getting the points are implemented and both are logged; config.parallax_method
chooses which one decides:
  * "landmarks" (this file): the 468 MediaPipe mesh points. Caveat: MediaPipe fits a learned 3D
    face model, so its landmarks may look slightly "3D" even on a flat photo seen at an angle.
  * "flow" (liveness_flow.py): real corner points of the image texture inside the face, tracked
    with optical flow - they follow the actual pixels, with no face prior.

Note: this defeats static photos and still images on a screen. A replayed VIDEO of the real
student turning their head contains real parallax; that attack is for the depth and texture
checks to catch.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import cv2
import numpy as np

from .detector import NUM_MESH_POINTS


@dataclass
class ParallaxResult:
    status: str                  # "pass" | "fail" | "insufficient_movement" | "insufficient_points"
    score: float | None          # median planarity error over informative pairs (IOD units)
    movement: float              # largest non-similarity movement among the pairs (IOD units)
    informative_pairs: int
    pairs: list[dict] = field(default_factory=list)  # per pair: i, j, points, movement, error
    points: int = 0              # points available (landmarks: 468; flow: fewest tracked in a frame)

    @property
    def passed(self) -> bool | None:
        return {"pass": True, "fail": False}.get(self.status)


def similarity_residuals(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Per-point error after the best least-squares similarity transform (shift+rotate+scale)."""
    s = src - src.mean(0)
    d = dst - dst.mean(0)
    sc = s[:, 0] + 1j * s[:, 1]
    dc = d[:, 0] + 1j * d[:, 1]
    a = np.vdot(sc, dc) / max(np.vdot(sc, sc).real, 1e-12)  # complex number = rotation * scale
    return np.abs(dc - a * sc)


def homography_residuals(src: np.ndarray, dst: np.ndarray, ransac_px: float) -> np.ndarray | None:
    """Per-point error after a RANSAC homography fit (None if no homography could be fitted)."""
    H, _ = cv2.findHomography(src.astype(np.float64), dst.astype(np.float64), cv2.RANSAC, ransac_px)
    if H is None:
        return None
    pred = cv2.perspectiveTransform(src.reshape(-1, 1, 2).astype(np.float64), H).reshape(-1, 2)
    return np.linalg.norm(pred - dst, axis=1)


def score_tracks(tracks: list[np.ndarray], valid: list[np.ndarray] | None, interoculars: list[float], *,
                 min_movement: float, threshold: float, ransac_reproj: float,
                 min_points: int = 8) -> ParallaxResult:
    """Score point tracks: per frame an (N, 2) array of the SAME N points (row k = point k).

    `valid[f][k]` says whether point k was found in frame f (None = all valid). Used by both the
    landmark method (points = MediaPipe mesh) and the optical-flow method (points = tracked corners).
    For every frame pair: movement = median error after a similarity fit, error = median error after
    a RANSAC homography fit, both divided by the inter-ocular distance.
    """
    n = len(tracks)
    if valid is None:
        valid = [np.ones(len(t), bool) for t in tracks]
    pairs = []
    for i, j in combinations(range(n), 2):
        common = valid[i] & valid[j]
        npts = int(common.sum())
        if npts < max(min_points, 4):
            pairs.append({"i": i, "j": j, "points": npts, "movement": float("nan"), "error": float("nan")})
            continue
        src = tracks[i][common].astype(np.float64)
        dst = tracks[j][common].astype(np.float64)
        iod = (interoculars[i] + interoculars[j]) / 2
        movement = float(np.median(similarity_residuals(src, dst))) / iod
        res = homography_residuals(src, dst, ransac_reproj * iod)
        error = float(np.median(res)) / iod if res is not None else float("nan")
        pairs.append({"i": i, "j": j, "points": npts, "movement": movement, "error": error})

    points = int(min(v.sum() for v in valid)) if valid else 0
    scored = [p for p in pairs if np.isfinite(p["movement"])]
    max_movement = max((p["movement"] for p in scored), default=0.0)
    if not scored:
        return ParallaxResult("insufficient_points", None, 0.0, 0, pairs, points)
    informative = [p for p in scored if p["movement"] >= min_movement and np.isfinite(p["error"])]
    if not informative:
        return ParallaxResult("insufficient_movement", None, max_movement, 0, pairs, points)
    score = float(np.median([p["error"] for p in informative]))
    status = "pass" if score >= threshold else "fail"
    return ParallaxResult(status, score, max_movement, len(informative), pairs, points)


def parallax_check(landmark_sets: list[np.ndarray], interoculars: list[float], *,
                   min_movement: float, threshold: float, ransac_reproj: float) -> ParallaxResult:
    """Landmark method. landmark_sets: per frame an (N, 2) array (MediaPipe 478 points; only the
    468 mesh points are used). interoculars: per-frame inter-ocular distance in pixels."""
    return score_tracks([lm[:NUM_MESH_POINTS] for lm in landmark_sets], None, interoculars,
                        min_movement=min_movement, threshold=threshold, ransac_reproj=ransac_reproj)
