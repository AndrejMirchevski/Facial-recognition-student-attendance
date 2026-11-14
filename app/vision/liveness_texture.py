"""Liveness check (c): texture anti-spoofing with MiniFASNet (Silent-Face-Anti-Spoofing, ONNX).

Two small CNNs look at the face plus its surroundings at two zoom levels and output, after
softmax, P(real) vs two spoof classes (print / replay). Like the original project we average
the two models' P(real).

Pre-processing reproduces the original project exactly (it matters - the model is sensitive):
  * face box: the original uses a RetinaFace detector whose box is roughly square. We
    approximate it with a square around the MediaPipe mesh (side = longer edge of the mesh box);
    on test images this matches the original detector's box to within a few percent.
  * the box is enlarged by the model's scale (2.7 for V2, 4.0 for V1SE); if that does not fit in
    the frame it is shrunk to fit and SHIFTED inside the frame (not clipped).
  * resize to 80x80, BGR channel order, raw 0-255 values (no /255, no mean/std).
Letterbox bars must be removed from the frame beforehand (see quality.trim_black_borders).
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .align import square_crop_box
from .detector import Face
from .embedder import ort_session

# (file name, crop scale) - the scale is part of the original model names (2.7_80x80_..., 4_0_0_80x80_...)
FAS_MODELS = (("MiniFASNetV2.onnx", 2.7), ("MiniFASNetV1SE.onnx", 4.0))
REAL_CLASS = 1


class TextureAntiSpoof:
    def __init__(self, models_dir: Path):
        self.models = [(ort_session(Path(models_dir) / name), scale) for name, scale in FAS_MODELS]

    @staticmethod
    def crop(bgr: np.ndarray, face: Face, scale: float) -> np.ndarray:
        h, w = bgr.shape[:2]
        cx, cy, side = face.square_box()
        x0, y0, x1, y1 = square_crop_box(w, h, cx, cy, side * scale, side * scale)
        return cv2.resize(bgr[y0:y1 + 1, x0:x1 + 1], (80, 80), interpolation=cv2.INTER_LINEAR)

    def real_probs(self, bgr: np.ndarray, face: Face) -> list[float]:
        """P(real) from each model, in FAS_MODELS order."""
        out = []
        for sess, scale in self.models:
            x = self.crop(bgr, face, scale).astype(np.float32).transpose(2, 0, 1)[None]
            logits = sess.run(None, {sess.get_inputs()[0].name: x})[0][0]
            p = np.exp(logits - logits.max())
            out.append(float(p[REAL_CLASS] / p.sum()))
        return out
