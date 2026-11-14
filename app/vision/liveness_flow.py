"""Liveness check (a, second method): parallax from REAL image points tracked with optical flow.

Why a second method: the landmark method uses MediaPipe's 468 mesh points, and MediaPipe fits a
learned 3D face model. So even a flat photo seen at an angle can get landmarks that move a little
"like a 3D face", which narrows the gap between real faces and photos (stage-2 test: real face
0.039-0.051 vs a PC-screen attack 0.033). Here the points are corners of the actual image texture
(skin pores, eyebrows, nostrils, lip edges ...), followed pixel by pixel - no face prior involved.

Steps:
  1. In frame 1, find up to N corner points (cv2.goodFeaturesToTrack) inside the face: the convex
     hull of the mesh, shrunk a little so background and hair edges are left out.
  2. Track them into each other frame with pyramidal Lucas-Kanade optical flow
     (cv2.calcOpticalFlowPyrLK). Each frame is first warped by the face's overall motion (a
     similarity transform fitted to the MediaPipe landmarks), so LK only follows small residual
     movements; it then locks on to the real pixels.
  3. Forward-backward check: track each point back to frame 1; if it does not land where it started
     (within a fraction of the inter-ocular distance), the track is unreliable and is dropped.
  4. Score exactly like the landmark method (liveness_parallax.score_tracks): per frame pair a RANSAC
     homography, median reprojection error in IOD units; a flat object is explained by it, a real
     head is not.

All images are first scaled so the face has about the same size (inter-ocular distance ~120 px),
which keeps the behaviour independent of how far the student sits and keeps it fast (~15 ms).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .detector import NUM_MESH_POINTS, Face
from .liveness_parallax import ParallaxResult, score_tracks


@dataclass(frozen=True)
class FlowParams:
    max_corners: int = 300
    quality_level: float = 0.005     # goodFeaturesToTrack: relative corner strength
    min_distance: float = 0.04       # between corners, IOD units
    mask_erode: float = 0.08         # shrink the face hull by this much (IOD units)
    win_size: int = 21               # LK window (pixels, after scaling)
    max_level: int = 3               # LK pyramid levels
    fb_max: float = 0.01             # forward-backward error limit, IOD units
    target_iod_px: float = 120.0     # images are downscaled so the IOD is about this
    blur_sigma: float = 0.0          # optional Gaussian blur (suppresses sensor noise / screen moire)
    min_points: int = 20             # fewer tracked points in a frame pair -> pair not scored


@dataclass
class FlowTracks:
    points: list[np.ndarray]         # per frame (N, 2), in ORIGINAL image pixels
    valid: list[np.ndarray]          # per frame (N,) bool


_LK_CRITERIA = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 30, 0.01)


def _gray(img: np.ndarray) -> np.ndarray:
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def track_face_points(images: list[np.ndarray], faces: list[Face], p: FlowParams = FlowParams()) -> FlowTracks:
    """Detect corners inside the face in frame 0 and track them into every other frame.

    Each frame k is first warped into frame 0's coordinates with the similarity transform
    (shift + rotation + zoom) that best maps its MediaPipe landmarks onto frame 0's. This removes
    the large overall motion (e.g. a phone moved twice as close), so optical flow only has to follow
    small residual movements. Tracked points are then mapped back into frame k's own coordinates.
    A similarity transform keeps a flat object flat and a 3D object 3D, so the parallax measurement
    is not affected by this step.
    """
    f0 = faces[0]
    scale = min(1.0, p.target_iod_px / max(f0.interocular, 1e-6))
    S = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]], np.float64)
    g0 = _prep(images[0], scale, p.blur_sigma)
    h, w = g0.shape
    lm0 = f0.landmarks[:NUM_MESH_POINTS].astype(np.float32) * scale
    iod0 = f0.interocular * scale

    mask = np.zeros(g0.shape, np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull(lm0.astype(np.int32)), 255)
    k = int(round(p.mask_erode * iod0))
    if k > 1:
        mask = cv2.erode(mask, np.ones((k, k), np.uint8))
    corners = cv2.goodFeaturesToTrack(g0, maxCorners=p.max_corners, qualityLevel=p.quality_level,
                                      minDistance=max(2.0, p.min_distance * iod0), mask=mask, blockSize=7)
    if corners is None:
        empty = np.zeros((0, 2), np.float32)
        return FlowTracks([empty] * len(images), [np.zeros(0, bool)] * len(images))
    p0 = corners.reshape(-1, 2).astype(np.float32)

    points, valid = [p0 / scale], [np.ones(len(p0), bool)]
    lk = dict(winSize=(p.win_size, p.win_size), maxLevel=p.max_level, criteria=_LK_CRITERIA,
              flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
    fb_limit = max(0.5, p.fb_max * iod0)
    for k in range(1, len(images)):
        lmk = faces[k].landmarks[:NUM_MESH_POINTS].astype(np.float32)
        # A: frame-k pixels -> frame-0 (scaled) pixels
        A, _ = cv2.estimateAffinePartial2D(lmk, lm0, method=cv2.LMEDS)
        if A is None:
            points.append(np.full_like(p0, np.nan))
            valid.append(np.zeros(len(p0), bool))
            continue
        gk = cv2.warpAffine(_prep(images[k], 1.0, 0.0), A, (w, h), flags=cv2.INTER_LINEAR)
        if p.blur_sigma > 0:
            gk = cv2.GaussianBlur(gk, (0, 0), p.blur_sigma)
        start = p0.reshape(-1, 1, 2)
        p1, st1, _ = cv2.calcOpticalFlowPyrLK(g0, gk, start, start.copy(), **lk)
        pb, st2, _ = cv2.calcOpticalFlowPyrLK(gk, g0, p1, start.copy(), **lk)
        p1, pb = p1.reshape(-1, 2), pb.reshape(-1, 2)
        ok = (st1.ravel() == 1) & (st2.ravel() == 1) & (np.linalg.norm(pb - p0, axis=1) < fb_limit)
        ok &= (p1[:, 0] >= 0) & (p1[:, 0] < w) & (p1[:, 1] >= 0) & (p1[:, 1] < h)
        # back to frame k's own pixel coordinates
        Ainv = cv2.invertAffineTransform(A)
        pk = p1 @ Ainv[:, :2].T + Ainv[:, 2]
        points.append(pk.astype(np.float32))
        valid.append(ok)
    return FlowTracks(points, valid)


def _prep(img: np.ndarray, scale: float, blur_sigma: float) -> np.ndarray:
    g = _gray(img)
    if scale < 1.0:
        g = cv2.resize(g, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    if blur_sigma > 0:
        g = cv2.GaussianBlur(g, (0, 0), blur_sigma)
    return g


def flow_parallax_check(images: list[np.ndarray], faces: list[Face], *, min_movement: float, threshold: float,
                        ransac_reproj: float, params: FlowParams = FlowParams()) -> ParallaxResult:
    """images: the frames (BGR or grey, same coordinates as the faces' landmarks)."""
    tracks = track_face_points(images, faces, params)
    return score_tracks(tracks.points, tracks.valid, [f.interocular for f in faces],
                        min_movement=min_movement, threshold=threshold, ransac_reproj=ransac_reproj,
                        min_points=params.min_points)
