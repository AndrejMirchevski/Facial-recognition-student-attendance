"""Configuration.

`VisionConfig` holds every threshold used by the vision pipeline (app/vision). It is plain
Python (no Flask) so the pipeline can be used from scripts and tests.

IMPORTANT: the thresholds below are INITIAL GUESSES. They come from a few test images and
synthetic experiments, not from real attempts on this system. They must be calibrated with
real attempts (genuine / printed photo / phone screen) using scripts/webcam_test.py and,
once the web app exists, the attempts log + scripts/threshold_report.py.

Flask settings (secret key, database, cookies, login lockout) are the `*Config` classes at the
bottom. Secrets and deployment settings never live in this file: they come from environment
variables or from the git-ignored instance/ folder (see `BaseConfig` and app/__init__.py).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import timedelta
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


# ===================================================================================== Flask
def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    return default if v is None else v.strip().lower() in ("1", "true", "yes", "on")


class BaseConfig:
    """Settings shared by every environment. Override any of them with an environment variable of
    the same name (where read below) or in instance/config.py (loaded after this class)."""
    # SECRET_KEY: environment variable SECRET_KEY, else instance/secret_key (created on first run).
    SECRET_KEY: str | None = os.environ.get("SECRET_KEY")
    # Database: SQLite file in instance/ by default; set DATABASE_URL to use e.g. PostgreSQL later
    # (postgresql+psycopg://user:pass@host/dbname) - only this setting changes.
    SQLALCHEMY_DATABASE_URI: str | None = os.environ.get("DATABASE_URL")
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Calendar day used for "one check-in per day" (server time is the source of truth).
    TIMEZONE = os.environ.get("APP_TIMEZONE", "Europe/Skopje")

    # ---- cookies / sessions
    SESSION_COOKIE_HTTPONLY = True          # JavaScript cannot read the session cookie
    SESSION_COOKIE_SAMESITE = "Lax"         # not sent on cross-site POSTs
    SESSION_COOKIE_SECURE = _env_bool("SESSION_COOKIE_SECURE", False)  # True in production (HTTPS)
    SESSION_COOKIE_NAME = "attendance_session"
    REMEMBER_COOKIE_HTTPONLY = True
    REMEMBER_COOKIE_SAMESITE = "Lax"
    REMEMBER_COOKIE_SECURE = SESSION_COOKIE_SECURE
    PERMANENT_SESSION_LIFETIME = timedelta(hours=8)

    # ---- CSRF (Flask-WTF): every POST/PUT/PATCH/DELETE needs a token - forms send a hidden field,
    # JavaScript sends the X-CSRFToken header (token from <meta name="csrf-token">).
    WTF_CSRF_ENABLED = True
    WTF_CSRF_TIME_LIMIT = 8 * 3600

    # ---- uploads (check-in frames, stage 4): whole request size limit
    MAX_CONTENT_LENGTH = 12 * 1024 * 1024

    # ---- login lockout (brute-force protection)
    LOGIN_MAX_FAILURES = 5            # failed logins for one username within the window -> locked
    LOGIN_MAX_FAILURES_PER_IP = 20    # failed logins from one IP address within the window -> locked
    LOGIN_LOCKOUT_MINUTES = 15        # window length and lockout duration

    # ---- passwords
    PASSWORD_MIN_LENGTH = 8
    # argon2id cost parameters (argon2-cffi defaults: time 3, memory 64 MiB, parallelism 4)
    ARGON2_TIME_COST = 3
    ARGON2_MEMORY_COST = 64 * 1024
    ARGON2_PARALLELISM = 4

    # Behind a reverse proxy (internet deployment), set to the number of proxies in front of the
    # app so the client IP (for the login lockout) and https scheme are taken from X-Forwarded-*.
    TRUST_PROXY_HOPS = int(os.environ.get("TRUST_PROXY_HOPS", "0"))


class DevelopmentConfig(BaseConfig):
    """Local use on http://localhost (browsers allow the camera on localhost without HTTPS)."""
    DEBUG = False


class ProductionConfig(BaseConfig):
    """Internet deployment behind HTTPS: cookies are only ever sent over HTTPS."""
    SESSION_COOKIE_SECURE = True
    REMEMBER_COOKIE_SECURE = True
    PREFERRED_URL_SCHEME = "https"


class TestingConfig(BaseConfig):
    TESTING = True
    SECRET_KEY = "test-secret-key"
    SQLALCHEMY_DATABASE_URI = "sqlite://"   # in-memory
    WTF_CSRF_ENABLED = False                 # individual tests switch it back on
    ARGON2_TIME_COST = 1                     # fast hashing in tests only
    ARGON2_MEMORY_COST = 8 * 1024
    ARGON2_PARALLELISM = 1


CONFIGS = {"development": DevelopmentConfig, "production": ProductionConfig, "testing": TestingConfig}
