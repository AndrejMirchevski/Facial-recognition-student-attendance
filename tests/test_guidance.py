"""Guided capture state machine (no models needed)."""
import numpy as np

from app.config import VisionConfig
from app.vision.detector import IRIS_LEFT, IRIS_RIGHT, NOSE_TIP, Face
from app.vision.guidance import HINTS, GuidedCapture

CFG = VisionConfig()


def face_with_yaw(yaw: float, iod: float = 100.0) -> Face:
    rng = np.random.default_rng(0)
    pts = np.column_stack([rng.uniform(-120, 120, 478), rng.uniform(-150, 150, 478)]) + 320
    pts[IRIS_LEFT] = (320 - iod / 2, 300)
    pts[IRIS_RIGHT] = (320 + iod / 2, 300)
    pts[NOSE_TIP] = (320 + yaw * iod, 350)
    return Face(pts.astype(np.float32))


def feed(guide, t, yaw, seconds, problem=None, dt=0.05):
    """Feed preview frames for `seconds`; returns (new time, list of states that captured)."""
    caps = []
    end = t + seconds
    while t < end:
        st = guide.update(t, None if yaw is None else face_with_yaw(yaw), problem)
        if st.capture:
            caps.append(st)
        if st.done:
            break
        t += dt
    return t, caps


def test_yaw_proxy_sign_and_scale():
    assert abs(face_with_yaw(0.0).yaw_proxy) < 1e-6
    assert abs(face_with_yaw(0.15).yaw_proxy - 0.15) < 1e-6


def test_real_head_follows_the_steps():
    g = GuidedCapture(CFG)
    t, caps = feed(g, 0.0, 0.0, 1.0)            # straight, held -> picture 1
    assert len(caps) == 1 and g.step == 1
    t, caps = feed(g, t, 0.05, 1.0)             # small turn: not enough yet
    assert not caps
    t, caps = feed(g, t, 0.15, 0.2)             # turned left -> picture 2
    assert len(caps) == 1 and g.step == 2
    t, caps = feed(g, t, 0.0, 0.2)              # back to centre -> picture 3
    assert len(caps) == 1 and g.step == 3
    t, caps = feed(g, t, 0.15, 1.0)             # same side again does NOT count
    assert not caps
    t, caps = feed(g, t, -0.15, 0.2)            # other side -> picture 4
    assert len(caps) == 1
    t, caps = feed(g, t, 0.0, 0.2)              # centre -> picture 5, done
    assert len(caps) == 1 and caps[0].done


def test_turning_right_first_then_asks_for_left():
    g = GuidedCapture(CFG)
    t, _ = feed(g, 0.0, 0.0, 1.0)
    t, caps = feed(g, t, -0.15, 0.2)            # student turned RIGHT first: accepted
    assert len(caps) == 1
    t, _ = feed(g, t, 0.0, 0.2)
    st = g.update(t, face_with_yaw(0.0), None)
    assert "LEFT" in st.instruction and st.direction == "left"


def test_photo_that_cannot_turn_is_still_captured_after_timeouts():
    g = GuidedCapture(CFG)
    t, caps = feed(g, 0.0, 0.0, 60.0)           # yaw never changes
    total = 5
    assert g.step == total
    assert t < 1 + 2 * CFG.guide_step_timeout_s + 2  # two turn steps time out; centre steps are instant


def test_too_close_gives_hint_and_waits():
    g = GuidedCapture(CFG)
    st = g.update(0.0, face_with_yaw(0.0), "face_too_close")
    assert st.hint == HINTS["face_too_close"] and not st.capture
    t, caps = feed(g, 0.0, 0.0, 5.0, problem="face_too_close")
    assert not caps                             # keeps waiting (until the long first timeout)


def test_nobody_in_front_of_camera_aborts():
    g = GuidedCapture(CFG)
    t, _ = feed(g, 0.0, None, CFG.guide_first_timeout_s + 1, problem="no_face")
    st = g.update(t, None, "no_face")
    assert st.done
