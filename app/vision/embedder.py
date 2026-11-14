"""Face embeddings with ArcFace (InsightFace buffalo_l w600k_r50.onnx) on onnxruntime."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

MODEL_NAME = "arcface-w600k_r50"  # stored with each template so a model change is detectable


def ort_session(path: Path) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.log_severity_level = 3
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def l2_normalise(v: np.ndarray) -> np.ndarray:
    return v / max(float(np.linalg.norm(v)), 1e-12)


def aggregate(embeddings: list[np.ndarray]) -> np.ndarray:
    """Mean of L2-normalised embeddings, re-normalised (used for templates and check-ins)."""
    return l2_normalise(np.mean([l2_normalise(e) for e in embeddings], axis=0)).astype(np.float32)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(l2_normalise(a), l2_normalise(b)))


class ArcFaceEmbedder:
    def __init__(self, model_path: Path):
        self.session = ort_session(model_path)
        self._input = self.session.get_inputs()[0].name

    def embed(self, face112_bgr: np.ndarray) -> np.ndarray:
        """112x112 aligned BGR face -> L2-normalised 512-d float32 embedding."""
        x = cv2.cvtColor(face112_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        x = ((x - 127.5) / 127.5).transpose(2, 0, 1)[None]
        e = self.session.run(None, {self._input: x})[0][0]
        return l2_normalise(e).astype(np.float32)
