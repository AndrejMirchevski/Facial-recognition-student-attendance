"""The vision pipeline: ONE function, `run_pipeline`, used by both enrollment and check-in.

    frames (5 BGR images) --> per frame: trim letterbox bars, detect face + landmarks,
                              quality checks (one face, size, brightness, sharpness)
                          --> ArcFace embedding per frame (same alignment + model always)
                          --> consistency (all frames same person), aggregate embedding
                          --> identity: cosine similarity to the student's template (check-in only)
                          --> liveness: parallax (landmarks AND optical flow; config picks the decider),
                              depth (Depth Anything), texture (MiniFASNet)
                          --> decision: "pass" / "fail" / "retry" + reason + every score

Independent of Flask so it can be used from scripts and tests. Models are loaded once
(`VisionModels.load`) and passed in.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from itertools import combinations

import numpy as np

from ..config import VisionConfig
from .align import align_arcface
from .detector import Face, FaceDetector, select_main_face
from .embedder import MODEL_NAME, ArcFaceEmbedder, aggregate, cosine
from .liveness_depth import DepthEstimator, depth_frame_scores
from .liveness_flow import FlowParams, flow_parallax_check
from .liveness_parallax import parallax_check
from .liveness_texture import TextureAntiSpoof
from .quality import brightness, sharpness, trim_black_borders

PASS, FAIL, RETRY = "pass", "fail", "retry"

# reason code -> message shown to the student
MESSAGES = {
    "ok": "Verified.",
    "wrong_frame_count": "The capture was incomplete. Please try again.",
    "no_face": "No face was found in one of the pictures. Make sure your whole face is visible and well lit.",
    "multiple_faces": "More than one face is visible. Make sure only you are in front of the camera.",
    "face_too_small": "Your face is too small in the picture. Move closer to the camera.",
    "face_too_close": "You are too close to the camera. Move back a little so your whole head and some background are visible.",
    "too_dark": "The picture is too dark. Turn on a light or face a window, then try again.",
    "too_blurry": "The picture is blurry. Move slowly, make sure the camera is in focus, and try again.",
    "no_movement": "Please turn your head slowly a little to the left and right while the pictures are taken, and try again.",
    "tracking_failed": "Your face could not be followed between the pictures. Make sure your face is well lit and in focus, and try again.",
    "inconsistent_frames": "The pictures do not all show the same person. Please try again.",
    "identity_mismatch": "Your face does not match the face enrolled for this account.",
    "liveness_failed": "The liveness check failed. Make sure you are in front of the camera yourself, in good light, and try again.",
}
# If several frames have problems, report the most useful one first.
_QUALITY_PRIORITY = ["no_face", "multiple_faces", "face_too_small", "face_too_close", "too_dark", "too_blurry"]
LIVENESS_CHECKS = ("parallax", "depth", "texture")


@dataclass
class VisionModels:
    detector: FaceDetector
    embedder: ArcFaceEmbedder
    depth: DepthEstimator
    texture: TextureAntiSpoof

    @classmethod
    def load(cls, cfg: VisionConfig | None = None) -> "VisionModels":
        cfg = cfg or VisionConfig()
        d = cfg.models_dir
        return cls(
            detector=FaceDetector(d / "face_landmarker.task", max_faces=cfg.max_faces_detected),
            embedder=ArcFaceEmbedder(d / "w600k_r50.onnx"),
            depth=DepthEstimator(d / "depth_anything_v2_small.onnx", cfg.depth_input_size),
            texture=TextureAntiSpoof(d),
        )


@dataclass
class FrameReport:
    index: int
    problem: str | None = None
    faces: int = 0
    ignored_faces: int = 0
    trimmed: tuple[int, int, int, int] = (0, 0, 0, 0)  # letterbox px removed: top, bottom, left, right
    interocular_px: float | None = None
    face_height_frac: float | None = None
    yaw_proxy: float | None = None           # head turn indicator (see Face.yaw_proxy)
    brightness: float | None = None
    sharpness: float | None = None
    similarity: float | None = None          # this frame vs the template (check-in only)
    texture: list[float] | None = None       # P(real) per MiniFASNet model
    depth_relief: float | None = None
    depth_nose: float | None = None


@dataclass
class PipelineResult:
    decision: str                       # pass | fail | retry
    reason: str                         # key of MESSAGES
    message: str
    mode: str                           # "checkin" (template given) | "enroll"
    checks: dict[str, bool | None] = field(default_factory=dict)  # None = not evaluated
    failed_checks: list[str] = field(default_factory=list)
    similarity: float | None = None     # aggregate embedding vs template
    consistency: float | None = None    # lowest similarity between two frames of this attempt
    parallax_method: str = ""           # which method decided the parallax check
    parallax_status: str | None = None  # the deciding method's status (pass/fail/insufficient_...)
    parallax_landmark_score: float | None = None
    parallax_landmark_movement: float | None = None
    parallax_flow_score: float | None = None
    parallax_flow_movement: float | None = None
    parallax_flow_points: int | None = None   # fewest points tracked in any frame
    depth_relief: float | None = None
    depth_nose: float | None = None
    texture_score: float | None = None
    model_name: str = MODEL_NAME
    frames: list[FrameReport] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)
    processing_ms: float = 0.0
    # Biometric data - never logged, never shown to professors:
    embedding: np.ndarray | None = field(default=None, repr=False)
    frame_embeddings: list[np.ndarray] = field(default_factory=list, repr=False)
    # Kept only for debugging displays (webcam test script); not stored:
    debug: dict = field(default_factory=dict, repr=False)

    @property
    def passed(self) -> bool:
        return self.decision == PASS

    def summary(self) -> dict:
        """JSON-serialisable scores and decision - no embeddings, no images."""
        d = {k: v for k, v in asdict(self).items()
             if k not in ("embedding", "frame_embeddings", "debug")}
        return d


def assess_frame(frame: np.ndarray, detector: FaceDetector, cfg: VisionConfig, index: int = 0):
    """Detection + quality checks for one frame (shared by the pipeline and the guided capture).

    Returns (trimmed_image, main_face_or_None, aligned_112_or_None, FrameReport).
    FrameReport.problem is None when the frame is usable.
    """
    rep = FrameReport(index=index)
    img, rep.trimmed = trim_black_borders(frame)
    sel = select_main_face(detector.detect(img), cfg.duplicate_overlap, cfg.secondary_face_min_area_ratio)
    rep.faces = (sel.main is not None) + len(sel.others)
    rep.ignored_faces = len(sel.ignored)
    if sel.main is None:
        rep.problem = "no_face"
        return img, None, None, rep
    face = sel.main
    if sel.others:
        rep.problem = "multiple_faces"
        return img, face, None, rep
    rep.interocular_px = round(face.interocular, 1)
    rep.face_height_frac = round(face.height / img.shape[0], 3)
    rep.yaw_proxy = round(face.yaw_proxy, 3)
    aligned = align_arcface(img, face)
    rep.brightness = round(brightness(aligned), 1)
    rep.sharpness = round(sharpness(aligned), 1)
    if face.interocular < cfg.min_interocular_px:
        rep.problem = "face_too_small"
    elif rep.face_height_frac > cfg.max_face_height_frac:
        rep.problem = "face_too_close"
    elif rep.brightness < cfg.min_face_brightness:
        rep.problem = "too_dark"
    elif rep.sharpness < cfg.min_sharpness:
        rep.problem = "too_blurry"
    return img, face, aligned, rep


def flow_params(cfg: VisionConfig) -> FlowParams:
    return FlowParams(max_corners=cfg.parallax_flow_max_corners, fb_max=cfg.parallax_flow_fb_max,
                      min_points=cfg.parallax_flow_min_points)


def _r(v: float | None, nd: int) -> float | None:
    return None if v is None else round(v, nd)


def _finish(result: PipelineResult, t_start: float) -> PipelineResult:
    result.processing_ms = round(1000 * (time.perf_counter() - t_start), 1)
    result.message = MESSAGES[result.reason]
    return result


def run_pipeline(frames: list[np.ndarray], models: VisionModels, cfg: VisionConfig | None = None,
                 template: np.ndarray | None = None, keep_debug: bool = False) -> PipelineResult:
    """Analyse one capture (enrollment if `template` is None, otherwise check-in).

    frames: BGR uint8 images (as decoded by cv2), in capture order.
    template: the logged-in student's enrolled embedding (1:1 verification only).
    """
    cfg = cfg or VisionConfig()
    t0 = time.perf_counter()
    mode = "enroll" if template is None else "checkin"
    res = PipelineResult(decision=RETRY, reason="wrong_frame_count", message="", mode=mode)
    if len(frames) != cfg.num_frames:
        return _finish(res, t0)

    # ---- 1. per-frame detection + quality ------------------------------------------------
    t = time.perf_counter()
    imgs, faces, aligned = [], [], []
    for i, frame in enumerate(frames):
        img, face, al, rep = assess_frame(frame, models.detector, cfg, i)
        res.frames.append(rep)
        imgs.append(img)
        faces.append(face)
        aligned.append(al)
    res.timings_ms["detect"] = round(1000 * (time.perf_counter() - t), 1)

    problems = {r.problem for r in res.frames if r.problem}
    if problems:
        res.reason = next(p for p in _QUALITY_PRIORITY if p in problems)
        return _finish(res, t0)

    # ---- 2. identity -----------------------------------------------------------------------
    t = time.perf_counter()
    res.frame_embeddings = [models.embedder.embed(a) for a in aligned]
    res.consistency = round(min(cosine(a, b) for a, b in combinations(res.frame_embeddings, 2)), 4)
    res.embedding = aggregate(res.frame_embeddings)
    res.checks["consistency"] = res.consistency >= cfg.consistency_threshold
    if template is not None:
        res.similarity = round(cosine(res.embedding, template), 4)
        res.checks["identity"] = res.similarity >= cfg.identity_threshold
        for rep, e in zip(res.frames, res.frame_embeddings):
            rep.similarity = round(cosine(e, template), 4)
    else:
        res.checks["identity"] = None
    res.timings_ms["embed"] = round(1000 * (time.perf_counter() - t), 1)

    # ---- 3a. parallax: two methods, both logged; cfg.parallax_method decides ------------------
    t = time.perf_counter()
    lm = parallax_check([f.landmarks for f in faces], [f.interocular for f in faces],
                        min_movement=cfg.parallax_min_movement, threshold=cfg.parallax_threshold,
                        ransac_reproj=cfg.parallax_ransac_reproj)
    res.timings_ms["parallax_landmarks"] = round(1000 * (time.perf_counter() - t), 1)
    t = time.perf_counter()
    fl = flow_parallax_check(imgs, faces, min_movement=cfg.parallax_flow_min_movement,
                             threshold=cfg.parallax_flow_threshold, ransac_reproj=cfg.parallax_ransac_reproj,
                             params=flow_params(cfg))
    res.timings_ms["parallax_flow"] = round(1000 * (time.perf_counter() - t), 1)
    res.parallax_landmark_score = _r(lm.score, 5)
    res.parallax_landmark_movement = round(lm.movement, 5)
    res.parallax_flow_score = _r(fl.score, 5)
    res.parallax_flow_movement = round(fl.movement, 5)
    res.parallax_flow_points = fl.points
    if cfg.parallax_method not in ("landmarks", "flow"):
        raise ValueError(f"unknown parallax_method {cfg.parallax_method!r}")
    decider = fl if cfg.parallax_method == "flow" else lm
    res.parallax_method = cfg.parallax_method
    res.parallax_status = decider.status
    res.checks["parallax"] = decider.passed
    res.checks["parallax_landmarks"] = lm.passed   # informational (not used by decide())
    res.checks["parallax_flow"] = fl.passed        # informational (not used by decide())
    if keep_debug:
        res.debug["parallax"] = {"landmarks": lm, "flow": fl}

    # ---- 3b. depth -------------------------------------------------------------------------
    t = time.perf_counter()
    reliefs, noses = [], []
    for img, face, rep in zip(imgs, faces, res.frames):
        dr = depth_frame_scores(img, face, models.depth, cfg.depth_crop_scale)
        rep.depth_relief, rep.depth_nose = round(dr.relief, 4), round(dr.nose, 4)
        reliefs.append(dr.relief)
        noses.append(dr.nose)
        if keep_debug:
            res.debug.setdefault("depth_maps", []).append(dr.depth_map)
    res.depth_relief = round(float(np.nanmedian(reliefs)), 4)
    res.depth_nose = round(float(np.nanmedian(noses)), 4)
    res.checks["depth"] = bool(res.depth_relief >= cfg.depth_relief_threshold
                               and res.depth_nose >= cfg.depth_nose_threshold)
    res.timings_ms["depth"] = round(1000 * (time.perf_counter() - t), 1)

    # ---- 3c. texture -----------------------------------------------------------------------
    t = time.perf_counter()
    per_frame = []
    for img, face, rep in zip(imgs, faces, res.frames):
        probs = models.texture.real_probs(img, face)
        rep.texture = [round(p, 4) for p in probs]
        per_frame.append(float(np.mean(probs)))
    res.texture_score = round(float(np.mean(per_frame)), 4)
    res.checks["texture"] = res.texture_score >= cfg.texture_threshold
    res.timings_ms["texture"] = round(1000 * (time.perf_counter() - t), 1)

    # ---- 4. decision -------------------------------------------------------------------------
    return _finish(decide(res, cfg), t0)


def decide(res: PipelineResult, cfg: VisionConfig) -> PipelineResult:
    """Turn the individual check results into pass / fail / retry. Pure logic (unit-tested).

    Rules: attendance/enrollment only if every required check passed.
      * frames not all the same person            -> fail
      * identity below threshold (check-in)       -> fail
      * any required liveness check failed        -> fail
      * head did not move enough for parallax     -> retry ("turn your head")
        (or too few points could be tracked)       -> retry ("tracking failed")
      * otherwise                                 -> pass
    """
    required = {"consistency": True, "identity": res.mode == "checkin",
                "parallax": cfg.require_parallax, "depth": cfg.require_depth,
                "texture": cfg.require_texture}
    res.failed_checks = [k for k, req in required.items() if req and res.checks.get(k) is False]
    if "consistency" in res.failed_checks:
        res.decision, res.reason = FAIL, "inconsistent_frames"
    elif "identity" in res.failed_checks:
        res.decision, res.reason = FAIL, "identity_mismatch"
    elif any(k in res.failed_checks for k in LIVENESS_CHECKS):
        res.decision, res.reason = FAIL, "liveness_failed"
    elif required["parallax"] and res.checks.get("parallax") is None:
        res.decision = RETRY
        res.reason = "tracking_failed" if res.parallax_status == "insufficient_points" else "no_movement"
    else:
        res.decision, res.reason = PASS, "ok"
    return res
