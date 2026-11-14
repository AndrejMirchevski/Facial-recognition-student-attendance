"""Pipeline logic that does not need the models: face selection, letterbox trimming, decisions."""
import numpy as np
import pytest

from app.config import VisionConfig
from app.vision.detector import Face, select_main_face
from app.vision.pipeline import FAIL, PASS, RETRY, PipelineResult, decide
from app.vision.quality import trim_black_borders

CFG = VisionConfig()


def box_face(x0, y0, x1, y1) -> Face:
    """A Face whose mesh spans the given box (points spread over the box)."""
    rng = np.random.default_rng(0)
    pts = np.column_stack([rng.uniform(x0, x1, 478), rng.uniform(y0, y1, 478)]).astype(np.float32)
    pts[:4] = [[x0, y0], [x1, y0], [x0, y1], [x1, y1]]
    return Face(pts)


def select(faces):
    return select_main_face(faces, CFG.duplicate_overlap, CFG.secondary_face_min_area_ratio)


# ---------------------------------------------------------------- face selection
def test_single_face():
    s = select([box_face(100, 100, 300, 350)])
    assert s.main is not None and not s.others and not s.ignored


def test_small_overlapping_duplicate_is_ignored():
    main = box_face(100, 100, 300, 350)
    dup = box_face(150, 160, 250, 280)  # smaller, lying on top of the main face (MediaPipe glitch)
    s = select([dup, main])            # order of detections must not matter
    assert s.main is main and not s.others and s.ignored == [dup]


def test_tiny_background_face_is_ignored():
    s = select([box_face(100, 100, 300, 350), box_face(500, 50, 540, 100)])
    assert not s.others and len(s.ignored) == 1


def test_second_real_face_is_reported():
    s = select([box_face(50, 100, 250, 350), box_face(350, 120, 530, 340)])
    assert len(s.others) == 1


def test_no_face():
    assert select([]).main is None


# ---------------------------------------------------------------- letterbox trimming
def test_letterbox_bars_are_trimmed():
    img = np.zeros((480, 640, 3), np.uint8)
    img[60:420] = np.random.default_rng(0).integers(40, 200, (360, 640, 3))
    out, (top, bottom, left, right) = trim_black_borders(img)
    assert (top, bottom, left, right) == (60, 60, 0, 0)
    assert out.shape == (360, 640, 3)


def test_normal_image_is_not_trimmed():
    img = np.random.default_rng(0).integers(40, 200, (480, 640, 3)).astype(np.uint8)
    out, trimmed = trim_black_borders(img)
    assert trimmed == (0, 0, 0, 0) and out.shape == img.shape


# ---------------------------------------------------------------- decisions
def result(mode="checkin", **checks):
    base = {"consistency": True, "identity": True if mode == "checkin" else None,
            "parallax": True, "depth": True, "texture": True}
    base.update(checks)
    return PipelineResult(decision="", reason="", message="", mode=mode, checks=base)


@pytest.mark.parametrize("checks, decision, reason", [
    ({}, PASS, "ok"),
    ({"identity": False}, FAIL, "identity_mismatch"),
    ({"consistency": False}, FAIL, "inconsistent_frames"),
    ({"texture": False}, FAIL, "liveness_failed"),
    ({"depth": False}, FAIL, "liveness_failed"),
    ({"parallax": False}, FAIL, "liveness_failed"),
    ({"parallax": None}, RETRY, "no_movement"),
    ({"parallax": None, "texture": False}, FAIL, "liveness_failed"),
    ({"identity": False, "texture": False}, FAIL, "identity_mismatch"),
])
def test_checkin_decisions(checks, decision, reason):
    r = decide(result(**checks), CFG)
    assert (r.decision, r.reason) == (decision, reason)


def test_failed_checks_lists_every_failure():
    r = decide(result(identity=False, texture=False, depth=False), CFG)
    assert set(r.failed_checks) == {"identity", "texture", "depth"}


def test_enrollment_does_not_need_identity_but_needs_liveness():
    assert decide(result(mode="enroll"), CFG).decision == PASS
    assert decide(result(mode="enroll", texture=False), CFG).decision == FAIL


def test_disabled_check_is_not_required():
    cfg = VisionConfig(require_depth=False)
    assert decide(result(depth=False), cfg).decision == PASS
