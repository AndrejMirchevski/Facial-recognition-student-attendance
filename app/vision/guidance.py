"""Guided capture: tells the student what to do and decides WHEN each of the 5 frames is taken.

Instead of taking frames on a fixed timer, each frame is taken when the student has done the
step:   1 look straight  ->  2 turn LEFT  ->  3 back to centre  ->  4 turn RIGHT  ->  5 centre.

The head turn is measured with Face.yaw_proxy relative to the first frame. The first turn may go
to either side (the second must then go to the other side), so the capture works whatever the
camera mirroring. Every step after the first also has a timeout: a printed photo cannot turn its
head, so its frames are taken anyway after a few seconds and the liveness checks then judge it.

This only decides when to take pictures. It is NOT a security check: the server re-checks
everything in run_pipeline, whatever the client did.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import VisionConfig
from .detector import NUM_MESH_POINTS, Face
from .liveness_parallax import similarity_residuals

STEPS = ("straight", "left", "centre", "right", "centre")
INSTRUCTIONS = {
    "straight": "Look straight at the camera",
    "left": "Slowly turn your head LEFT",
    "centre": "Back to the centre",
    "right": "Slowly turn your head RIGHT",
}
HINTS = {
    "no_face": "Face not found - look at the camera",
    "multiple_faces": "Only one person in front of the camera, please",
    "face_too_small": "Move a little closer to the camera",
    "face_too_close": "Move back a little - your whole head should be visible",
    "too_dark": "Too dark - turn on a light or face a window",
    "too_blurry": "Hold still for a moment",
    "lost_face": "Not so far - keep your face in view",
    "turn_more": "A little more...",
    "not_straight": "Turn your face towards the camera",
}


@dataclass
class GuideState:
    step: int                  # index of the frame being waited for (0..4); 5 when done
    instruction: str
    hint: str | None = None
    capture: bool = False      # True -> take THIS frame
    done: bool = False         # all frames taken
    aborted: bool = False      # gave up (no face at all on the first step)
    turn: float = 0.0          # head turn relative to the first frame (+ = student's left)
    target: float = 0.0        # turn needed for the current step (0 for centre steps)
    direction: str | None = None  # "left" / "right" for turn steps
    time_left: float = 0.0


@dataclass
class GuidedCapture:
    cfg: VisionConfig = field(default_factory=VisionConfig)
    step: int = 0
    _step_started: float | None = None
    _ok_since: float | None = None
    _base_yaw: float = 0.0
    _base_landmarks: np.ndarray | None = None
    _first_side: float = 0.0   # sign of the first turn

    def __post_init__(self):
        if self.cfg.num_frames != len(STEPS):
            raise ValueError(f"guided capture is defined for {len(STEPS)} frames")

    def update(self, now: float, face: Face | None, problem: str | None) -> GuideState:
        """Feed every preview frame. `problem` is FrameReport.problem from assess_frame."""
        cfg = self.cfg
        if self.step >= len(STEPS):
            return GuideState(self.step, "Done", done=True)
        if self._step_started is None:
            self._step_started = now
        name = STEPS[self.step]
        if name in ("left", "right") and self._first_side != 0.0:
            # second turn: the side opposite to the first one (+ = student's left)
            name = "right" if self._first_side > 0 else "left"
        timeout = cfg.guide_first_timeout_s if self.step == 0 else cfg.guide_step_timeout_s
        elapsed = now - self._step_started
        st = GuideState(self.step, INSTRUCTIONS[name], time_left=max(0.0, timeout - elapsed))
        if name in ("left", "right"):
            st.direction = name
            st.target = cfg.guide_turn_target

        if face is not None and self.step > 0:
            st.turn = face.yaw_proxy - self._base_yaw

        # ---- step 0: wait for good conditions, facing the camera, held briefly
        if self.step == 0:
            ok = problem is None and face is not None and abs(face.yaw_proxy) <= cfg.guide_straight_max
            if problem is not None:
                st.hint = HINTS.get(problem)
            elif not ok:
                st.hint = HINTS["not_straight"]
            if ok:
                self._ok_since = self._ok_since if self._ok_since is not None else now
                if now - self._ok_since >= cfg.guide_hold_s:
                    return self._take(st, now, face)
            else:
                self._ok_since = None
            if elapsed >= timeout:
                if face is None:  # nobody there at all
                    st.aborted, st.done = True, True
                    return st
                return self._take(st, now, face)
            return st

        # ---- later steps
        if face is None or problem in ("no_face", "multiple_faces"):
            st.hint = HINTS["lost_face"] if face is None and name != "centre" else HINTS.get(problem or "no_face")
            return self._take(st, now, face) if elapsed >= timeout else st
        if problem is not None:
            st.hint = HINTS.get(problem)

        if name == "centre":
            done = abs(st.turn) <= cfg.guide_centre_tolerance
        else:
            if self._first_side == 0.0:          # first turn: either side counts
                reached = abs(st.turn) >= cfg.guide_turn_target
            else:                                 # second turn: must be the other side
                reached = -self._first_side * st.turn >= cfg.guide_turn_target
            done = (reached and problem is None) or self._moved(face) >= cfg.guide_alt_movement
            if not done and st.hint is None and abs(st.turn) > 0.03:
                st.hint = HINTS["turn_more"]
            if done and self._first_side == 0.0:
                self._first_side = float(np.sign(st.turn)) or 1.0
        if done or elapsed >= timeout:
            return self._take(st, now, face)
        return st

    def _moved(self, face: Face) -> float:
        """Landmark motion since the first frame that a flat shift/rotate/zoom cannot explain."""
        if self._base_landmarks is None:
            return 0.0
        res = similarity_residuals(self._base_landmarks[:NUM_MESH_POINTS].astype(np.float64),
                                   face.landmarks[:NUM_MESH_POINTS].astype(np.float64))
        return float(np.median(res)) / max(face.interocular, 1e-6)

    def _take(self, st: GuideState, now: float, face: Face | None) -> GuideState:
        if self.step == 0 and face is not None:
            self._base_yaw = face.yaw_proxy
            self._base_landmarks = face.landmarks.copy()
        st.capture = True
        self.step += 1
        self._step_started = now
        self._ok_since = None
        st.done = self.step >= len(STEPS)
        return st
