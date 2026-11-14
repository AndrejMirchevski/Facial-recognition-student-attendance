"""Face detection + 478 landmarks with MediaPipe Face Landmarker (Tasks API).

Runs in IMAGE mode: every frame is processed independently, with no tracking/smoothing
between frames. That matters for the parallax check, which compares raw landmark positions
across frames.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# MediaPipe landmark indices used across the pipeline.
IRIS_LEFT = 468    # centre of the iris on the image-left side
IRIS_RIGHT = 473   # centre of the iris on the image-right side
NOSE_TIP = 1
MOUTH_LEFT = 61
MOUTH_RIGHT = 291
CHEEK_LEFT = 50    # cheek centres (on the cheekbone, not on the face outline)
CHEEK_RIGHT = 280
NUM_MESH_POINTS = 468  # 0..467 are the face mesh; 468..477 are the irises


@dataclass
class Face:
    landmarks: np.ndarray  # (478, 2) float32, pixel coordinates (x, y)

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        """x0, y0, x1, y1 of the face mesh."""
        x0, y0 = self.landmarks[:NUM_MESH_POINTS].min(0)
        x1, y1 = self.landmarks[:NUM_MESH_POINTS].max(0)
        return float(x0), float(y0), float(x1), float(y1)

    @property
    def width(self) -> float:
        x0, _, x1, _ = self.bbox
        return x1 - x0

    @property
    def height(self) -> float:
        _, y0, _, y1 = self.bbox
        return y1 - y0

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def interocular(self) -> float:
        """Distance between the iris centres in pixels (the face-size unit used everywhere)."""
        return float(np.linalg.norm(self.landmarks[IRIS_LEFT] - self.landmarks[IRIS_RIGHT]))

    @property
    def yaw_proxy(self) -> float:
        """How far the nose tip sits sideways from the midpoint between the eyes, along the eye
        line, in inter-ocular units. ~0 looking straight at the camera; changes by ~0.12 for a
        ~10 degree head turn. Positive = nose towards the image's right = the student's LEFT
        (in the raw, un-mirrored camera image). A tilted flat photo barely changes it."""
        left, right = self.landmarks[IRIS_LEFT], self.landmarks[IRIS_RIGHT]
        axis = right - left
        iod = float(np.linalg.norm(axis))
        if iod < 1e-6:
            return 0.0
        return float(np.dot(self.landmarks[NOSE_TIP] - (left + right) / 2, axis / iod) / iod)

    def square_box(self, scale: float = 1.0) -> tuple[float, float, float]:
        """Centre x, centre y, side of a square around the face mesh box (side = longer box edge)."""
        x0, y0, x1, y1 = self.bbox
        return (x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, y1 - y0) * scale


def _overlap_fraction(inner: Face, outer: Face) -> float:
    """Fraction of `inner`'s box area that lies inside `outer`'s box."""
    ax0, ay0, ax1, ay1 = inner.bbox
    bx0, by0, bx1, by1 = outer.bbox
    iw = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    ih = max(0.0, min(ay1, by1) - max(ay0, by0))
    return iw * ih / max(inner.area, 1e-6)


@dataclass
class FaceSelection:
    main: Face | None
    others: list[Face]       # real additional faces (-> "more than one face" retry)
    ignored: list[Face]      # duplicate detections of the main face, or tiny background faces


def select_main_face(faces: list[Face], duplicate_overlap: float, min_area_ratio: float) -> FaceSelection:
    """Pick the largest face; classify the rest as ignorable or as genuine extra faces.

    MediaPipe occasionally reports a second, smaller "face" overlapping the real one (seen in
    stage-1 tests). Such detections, and faces far smaller than the main one, are ignored so that
    a genuine student is not rejected for "more than one face".
    """
    if not faces:
        return FaceSelection(None, [], [])
    ordered = sorted(faces, key=lambda f: f.area, reverse=True)
    main, others, ignored = ordered[0], [], []
    for f in ordered[1:]:
        if _overlap_fraction(f, main) >= duplicate_overlap or f.area < min_area_ratio * main.area:
            ignored.append(f)
        else:
            others.append(f)
    return FaceSelection(main, others, ignored)


class FaceDetector:
    def __init__(self, model_path: Path, max_faces: int = 3):
        opts = mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_buffer=Path(model_path).read_bytes()),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=max_faces,
        )
        self._landmarker = mp_vision.FaceLandmarker.create_from_options(opts)
        self._lock = threading.Lock()  # MediaPipe graphs are not documented as thread-safe

    def detect(self, bgr: np.ndarray) -> list[Face]:
        h, w = bgr.shape[:2]
        rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        with self._lock:
            res = self._landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        scale = np.array([w, h], dtype=np.float32)
        return [Face(np.array([[p.x, p.y] for p in lms], dtype=np.float32) * scale)
                for lms in res.face_landmarks]
