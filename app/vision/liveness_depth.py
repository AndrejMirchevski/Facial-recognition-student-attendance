"""Liveness check (b): monocular depth with Depth Anything V2 Small (ONNX).

The model estimates, from a single RGB image, how near each pixel is (relative inverse depth:
larger = closer; the scale is arbitrary, so values are normalised per crop). On a real face the
nose is clearly closer than the cheeks and the face surface is curved. A printed photo or a
phone screen is a plane, so - if the model sees it as such - the face region fits a flat plane.

Per frame we compute, on a crop ~2x the face size (so the model also sees context such as the
edges of a photo or phone):
  * relief - RMS distance of the face's depth values from the best-fitting plane
  * nose   - depth at the nose tip minus depth at the cheeks (positive = nose closer)
Both are in units of the crop's depth range (2nd..98th percentile -> 0..1).

Caveat for the paper: monocular depth models infer depth from appearance, so they can
"see" relief in a good photo of a face. This check is one of three for that reason.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .align import crop_around_face
from .detector import CHEEK_LEFT, CHEEK_RIGHT, NOSE_TIP, NUM_MESH_POINTS, Face
from .embedder import ort_session

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class DepthEstimator:
    def __init__(self, model_path: Path, input_size: int = 266):
        if input_size % 14:
            raise ValueError("Depth Anything input size must be a multiple of 14")
        self.session = ort_session(model_path)
        self._input = self.session.get_inputs()[0].name
        self.size = input_size

    def predict(self, bgr: np.ndarray) -> np.ndarray:
        """BGR image -> (size, size) relative inverse depth map (larger = closer)."""
        x = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (self.size, self.size),
                       interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255.0
        x = ((x - _MEAN) / _STD).transpose(2, 0, 1)[None].astype(np.float32)
        return self.session.run(None, {self._input: x})[0][0]


@dataclass
class DepthFrameResult:
    relief: float
    nose: float
    depth_map: np.ndarray  # normalised 0..1 (for display/debugging only)
    mask: np.ndarray       # face mask used (uint8, same size as depth_map)


def _disk_mean(img: np.ndarray, centre: np.ndarray, radius: float) -> float:
    m = np.zeros(img.shape, np.uint8)
    cv2.circle(m, (int(round(centre[0])), int(round(centre[1]))), max(1, int(round(radius))), 1, -1)
    return float(img[m > 0].mean()) if m.any() else float("nan")


def depth_frame_scores(bgr: np.ndarray, face: Face, estimator: DepthEstimator,
                       crop_scale: float) -> DepthFrameResult:
    crop, (x0, y0) = crop_around_face(bgr, face, crop_scale)
    d = estimator.predict(crop)
    size = d.shape[0]
    sx, sy = size / crop.shape[1], size / crop.shape[0]
    lm = (face.landmarks - np.array([x0, y0], np.float32)) * np.array([sx, sy], np.float32)

    lo, hi = np.percentile(d, [2, 98])
    dn = np.clip((d - lo) / max(hi - lo, 1e-6), -0.5, 1.5).astype(np.float32)

    # Face mask: convex hull of the mesh, shrunk a little to keep hair/background edges out.
    mask = np.zeros((size, size), np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull(lm[:NUM_MESH_POINTS].astype(np.int32)), 1)
    iod = float(np.linalg.norm(lm[468] - lm[473]))
    k = max(1, int(round(0.15 * iod)))
    mask = cv2.erode(mask, np.ones((k, k), np.uint8))

    ys, xs = np.nonzero(mask)
    if len(xs) < 30:
        return DepthFrameResult(float("nan"), float("nan"), dn, mask)
    z = dn[ys, xs]
    A = np.column_stack([xs / size, ys / size, np.ones_like(xs, dtype=np.float64)])
    coef, *_ = np.linalg.lstsq(A, z, rcond=None)
    relief = float(np.sqrt(np.mean((z - A @ coef) ** 2)))

    r = 0.12 * iod
    nose = _disk_mean(dn, lm[NOSE_TIP], r) - np.mean([_disk_mean(dn, lm[CHEEK_LEFT], r),
                                                      _disk_mean(dn, lm[CHEEK_RIGHT], r)])
    return DepthFrameResult(relief, float(nose), dn, mask)
