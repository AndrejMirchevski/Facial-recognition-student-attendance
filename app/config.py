"""Configuration.

`VisionConfig` holds every threshold used by the vision pipeline (app/vision). It is plain
Python (no Flask) so the pipeline can be used from scripts and tests.

IMPORTANT: the thresholds below are INITIAL GUESSES. They come from a few test images and
synthetic experiments, not from real attempts on this system. They must be calibrated with
real attempts (genuine / printed photo / phone screen) using scripts/webcam_test.py and,
once the web app exists, the attempts log + scripts/threshold_report.py.

Flask settings (secret key, database, cookies) are added in stage 3.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class VisionConfig:
    models_dir: Path = BASE_DIR / "models"

    # ---------------------------------------------------------------- capture
    # Frames per attempt (check-in and enrollment). The browser captures them ~0.5 s apart
    # while the student slowly turns their head.
    num_frames: int = 5

    # ---------------------------------------------------------------- guided capture (app/vision/guidance.py)
    # The 5 frames are taken at: straight, LEFT, centre, RIGHT, centre. Each frame is taken when
    # the student has done the step (not on a timer). Head turn is measured with Face.yaw_proxy.
    guide_turn_target: float = 0.12        # ~10 degrees; the turn frames are taken at this point
    guide_centre_tolerance: float = 0.06   # "back to centre" = within this of the first frame
    guide_straight_max: float = 0.15       # first frame: roughly facing the camera
    guide_hold_s: float = 0.3              # first frame: hold still this long before it is taken
    # A turn step also completes if the landmarks moved this much in a way a flat move/zoom cannot
    # explain (IOD units) - so a tilted photo is still captured and then judged by the checks.
    guide_alt_movement: float = 0.06
    guide_first_timeout_s: float = 15.0    # first frame: give up waiting for good conditions
    guide_step_timeout_s: float = 5.0      # other steps: take the frame anyway after this

    # ---------------------------------------------------------------- face detection
    # MediaPipe may return up to this many faces per frame (we need >1 to notice a second person).
    max_faces_detected: int = 3
    # MediaPipe sometimes reports the SAME face twice: a second, smaller detection lying on top
    # of the real one. A secondary detection with at least this fraction of its box inside the
    # main face's box is treated as such a duplicate and ignored.
    duplicate_overlap: float = 0.30
    # Secondary faces much smaller than the main face (e.g. someone far behind the student) are
    # ignored if their box area is below this fraction of the main face's box area.
    secondary_face_min_area_ratio: float = 0.25

    # ---------------------------------------------------------------- frame quality
    # Distance between the eye (iris) centres, in pixels. Too small = face too far / too few pixels.
    min_interocular_px: float = 40.0
    # Face box height / frame height. If the face fills the frame, the anti-spoof model loses the
    # surrounding context it was trained with -> ask the student to move back.
    max_face_height_frac: float = 0.70
    # Mean grey level (0-255) of the face region. Dark, noisy frames fool the texture model.
    min_face_brightness: float = 50.0
    # Variance of the Laplacian of the aligned 112x112 face (higher = sharper). Sharp test
    # images score 600-1500; sensor noise also raises it, so it only catches strong blur.
    min_sharpness: float = 40.0

    # ---------------------------------------------------------------- identity (ArcFace)
    # Cosine similarity between the check-in embedding (mean of the 5 frames) and the student's
    # enrolled template. Same person on a webcam is typically 0.5-0.8; different people < 0.3.
    identity_threshold: float = 0.40
    # Every pair of frames in one attempt must look like the same person (catches a face being
    # swapped mid-capture, and bad captures during enrollment).
    consistency_threshold: float = 0.50

    # ---------------------------------------------------------------- liveness a: parallax (no model)
    require_parallax: bool = True
    # Landmark motion between two frames that a simple move/rotate/zoom (similarity transform)
    # cannot explain, as a median over landmarks, in units of the inter-ocular distance (IOD).
    # Below this, the head did not turn enough for the check to be meaningful -> "turn your head".
    # Only frame pairs that moved at least this much are scored.
    # (Synthetic test: a head turning +-5..15 deg gives 0.04-0.14.)
    parallax_min_movement: float = 0.04
    # Median reprojection error after fitting a homography (the transform a FLAT object
    # undergoes), in IOD units. A photo/screen is explained almost perfectly (small error);
    # a real turning head is not. Below this threshold the face is judged flat.
    # (Synthetic tests: tilted photos through MediaPipe 0.004-0.022 even at +-35 deg tilt;
    #  simulated real head turns 0.034-0.05.)
    parallax_threshold: float = 0.025
    # RANSAC inlier threshold for cv2.findHomography, in IOD units.
    parallax_ransac_reproj: float = 0.03
    # Which point source DECIDES the parallax check. Both are always computed and logged.
    #   "landmarks": MediaPipe's 468 mesh points (liveness_parallax.py)
    #   "flow":      real texture corners tracked with optical flow (liveness_flow.py)
    parallax_method: str = "landmarks"
    # Thresholds for the flow method (same meaning and units as the two above). PLACEHOLDERS until
    # calibrated with replay_captures.py --suggest; they only decide if parallax_method = "flow".
    parallax_flow_min_movement: float = 0.04
    parallax_flow_threshold: float = 0.025
    # Flow tracking settings: max corners to track, forward-backward error limit (IOD units),
    # and the minimum number of points tracked in both frames for a frame pair to be scored.
    parallax_flow_max_corners: int = 300
    parallax_flow_fb_max: float = 0.01
    parallax_flow_min_points: int = 20

    # ---------------------------------------------------------------- liveness b: monocular depth
    require_depth: bool = True
    # Depth Anything V2 input side (multiple of 14). 266 ~ 80 ms per frame on CPU.
    depth_input_size: int = 266
    # The depth model sees a square crop this many times the face size, so it also sees some
    # background (and the edges of a photo/phone, if any).
    depth_crop_scale: float = 2.0
    # Depth values are normalised to 0..1 over the crop (2nd..98th percentile). "Relief" is the
    # RMS distance of the face's depth values from the best-fitting plane. Flat -> small.
    # (Sample images: real face 0.021; photo-of-screen 0.029; phone held in hand 0.012 -
    #  this score alone is weak, hence the nose check too. NEEDS calibration on real attempts.)
    depth_relief_threshold: float = 0.015
    # Nose tip depth minus mean cheek depth (same normalised units; positive = nose closer).
    # (Sample images: real face +0.054; photo-of-screen -0.019; phone in hand +0.045.)
    depth_nose_threshold: float = 0.02

    # ---------------------------------------------------------------- liveness c: texture (MiniFASNet)
    require_texture: bool = True
    # Mean P(real) of MiniFASNetV2 + MiniFASNetV1SE, averaged over the frames.
    texture_threshold: float = 0.50
