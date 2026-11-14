"""Frame and face quality measurements."""
from __future__ import annotations

import cv2
import numpy as np


def trim_black_borders(bgr: np.ndarray, level: int = 16, max_trim_frac: float = 0.3):
    """Remove letterbox / pillarbox bars (uniform black rows/columns at the frame edges).

    Some webcams and phone-as-webcam apps (e.g. Iriun) send a 16:9 picture inside a 4:3 frame
    with black bars. The bars look like a screen border to the texture anti-spoof model and
    pushed a real face's P(real) from ~0.5 down to ~0.00 in testing, so they are cut off before
    any processing. Only whole rows/columns at the edges whose brightest pixels are near-black
    are removed, and at most `max_trim_frac` of the frame per side.

    Returns (trimmed_image, (top, bottom, left, right)) with the number of pixels removed.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    row_bright = np.percentile(gray, 99, axis=1)
    col_bright = np.percentile(gray, 99, axis=0)

    def run_length(values: np.ndarray, limit: int) -> int:
        n = 0
        while n < limit and values[n] < level:
            n += 1
        return n if n >= 4 else 0  # ignore 1-3 px dark edges

    top = run_length(row_bright, int(h * max_trim_frac))
    bottom = run_length(row_bright[::-1], int(h * max_trim_frac))
    left = run_length(col_bright, int(w * max_trim_frac))
    right = run_length(col_bright[::-1], int(w * max_trim_frac))
    if top or bottom or left or right:
        bgr = bgr[top:h - bottom, left:w - right]
    return np.ascontiguousarray(bgr), (top, bottom, left, right)


def brightness(face112_bgr: np.ndarray) -> float:
    """Mean grey level (0-255) of the central part of the aligned face."""
    gray = cv2.cvtColor(face112_bgr, cv2.COLOR_BGR2GRAY)
    return float(gray[24:100, 24:88].mean())


def sharpness(face112_bgr: np.ndarray) -> float:
    """Variance of the Laplacian of the aligned face (higher = sharper)."""
    gray = cv2.cvtColor(face112_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())
