"""Geometric helpers: ArcFace 5-point alignment and square crops around a face."""
from __future__ import annotations

import cv2
import numpy as np

from .detector import IRIS_LEFT, IRIS_RIGHT, MOUTH_LEFT, MOUTH_RIGHT, NOSE_TIP, Face

# Standard ArcFace 112x112 reference positions (as used by InsightFace): image-left eye,
# image-right eye, nose tip, image-left mouth corner, image-right mouth corner.
ARCFACE_TEMPLATE = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                             [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)
FIVE_POINTS = [IRIS_LEFT, IRIS_RIGHT, NOSE_TIP, MOUTH_LEFT, MOUTH_RIGHT]


def align_arcface(bgr: np.ndarray, face: Face) -> np.ndarray:
    """Warp the face to the 112x112 ArcFace template with a similarity transform.

    This is the ONLY alignment used for embeddings, for both enrollment and check-in.
    """
    five = face.landmarks[FIVE_POINTS].astype(np.float32)
    M, _ = cv2.estimateAffinePartial2D(five, ARCFACE_TEMPLATE, method=cv2.LMEDS)
    if M is None:  # degenerate points; fall back to a plain least-squares fit
        M, _ = cv2.estimateAffinePartial2D(five, ARCFACE_TEMPLATE, method=0)
    return cv2.warpAffine(bgr, M, (112, 112), flags=cv2.INTER_LINEAR, borderValue=0)


def square_crop_box(img_w: int, img_h: int, cx: float, cy: float, side_w: float, side_h: float):
    """Box of the requested size around (cx, cy), shifted (not cut) to stay inside the image.

    Same behaviour as CropImage in Silent-Face-Anti-Spoofing: if the enlarged box would leave
    the image it is shrunk to fit, then moved back inside, so the crop keeps its aspect ratio
    instead of being clipped at the border.
    Returns integer (x0, y0, x1, y1), inclusive.
    """
    scale = min(1.0, (img_h - 1) / side_h, (img_w - 1) / side_w)
    side_w, side_h = side_w * scale, side_h * scale
    x0, y0, x1, y1 = cx - side_w / 2, cy - side_h / 2, cx + side_w / 2, cy + side_h / 2
    if x0 < 0:
        x1 -= x0
        x0 = 0
    if y0 < 0:
        y1 -= y0
        y0 = 0
    if x1 > img_w - 1:
        x0 -= x1 - img_w + 1
        x1 = img_w - 1
    if y1 > img_h - 1:
        y0 -= y1 - img_h + 1
        y1 = img_h - 1
    return int(x0), int(y0), int(x1), int(y1)


def crop_around_face(bgr: np.ndarray, face: Face, scale: float, out_size: int | None = None):
    """Square crop of `scale` x the face size, centred on the face (shift-inside, see above).

    Returns (crop, (x0, y0)) where (x0, y0) is the crop's top-left corner in the frame.
    """
    h, w = bgr.shape[:2]
    cx, cy, side = face.square_box()
    x0, y0, x1, y1 = square_crop_box(w, h, cx, cy, side * scale, side * scale)
    crop = bgr[y0:y1 + 1, x0:x1 + 1]
    if out_size is not None:
        crop = cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
    return crop, (x0, y0)
