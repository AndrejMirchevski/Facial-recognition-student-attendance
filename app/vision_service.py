"""Bridge between the web app and the vision pipeline.

* Models are loaded ONCE per process (at startup when VISION_PRELOAD is on, else on first use).
* At most VISION_MAX_PARALLEL attempts are analysed at the same time (CPU protection).
* `analyse()` is the single function the web app calls - tests replace it with a mock.
"""
from __future__ import annotations

import dataclasses
import threading

import numpy as np

from .config import VisionConfig
from .vision import PipelineResult, VisionModels, run_pipeline
from .vision.guidance import HINTS, INSTRUCTIONS, STEPS

_cfg = VisionConfig()
_models: VisionModels | None = None
_load_lock = threading.Lock()
_slots = threading.BoundedSemaphore(2)


def vision_config() -> VisionConfig:
    return _cfg


def set_max_parallel(n: int) -> None:
    global _slots
    _slots = threading.BoundedSemaphore(max(1, n))


def get_models() -> VisionModels:
    global _models
    if _models is None:
        with _load_lock:
            if _models is None:
                _models = VisionModels.load(_cfg)
    return _models


def analyse(frames: list[np.ndarray], template: np.ndarray | None) -> PipelineResult:
    """Run the pipeline: enrollment if template is None, otherwise 1:1 check-in."""
    models = get_models()
    with _slots:
        return run_pipeline(frames, models, _cfg, template=template)


def thresholds_snapshot() -> dict:
    """All VisionConfig values (stored with every attempt so old attempts stay interpretable)."""
    d = dataclasses.asdict(_cfg)
    d.pop("models_dir", None)
    return d


def guidance_settings() -> dict:
    """Everything the browser's guided capture needs - the SAME values and texts as
    app/vision/guidance.py, so the browser follows exactly the same rules."""
    c = _cfg
    return {
        "num_frames": c.num_frames,
        "steps": list(STEPS),
        "instructions": INSTRUCTIONS,
        "hints": HINTS,
        "guide_turn_target": c.guide_turn_target,
        "guide_centre_tolerance": c.guide_centre_tolerance,
        "guide_straight_max": c.guide_straight_max,
        "guide_hold_s": c.guide_hold_s,
        "guide_alt_movement": c.guide_alt_movement,
        "guide_first_timeout_s": c.guide_first_timeout_s,
        "guide_step_timeout_s": c.guide_step_timeout_s,
        "max_faces_detected": c.max_faces_detected,
        "duplicate_overlap": c.duplicate_overlap,
        "secondary_face_min_area_ratio": c.secondary_face_min_area_ratio,
        "min_interocular_px": c.min_interocular_px,
        "max_face_height_frac": c.max_face_height_frac,
        "min_face_brightness": c.min_face_brightness,
        "min_sharpness": c.min_sharpness,
    }


def template_to_bytes(v: np.ndarray) -> bytes:
    return np.asarray(v, dtype=np.float32).tobytes()


def template_from_bytes(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype=np.float32).copy()
