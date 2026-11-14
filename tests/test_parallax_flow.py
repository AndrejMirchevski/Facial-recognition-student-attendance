"""Optical-flow parallax method on synthetic images (no models needed).

* Flat: a textured image warped by the homographies of a plane tilting left/right (what a
  printed photo or a screen does) -> every tracked point obeys one homography -> low score.
* Non-planar: the same texture, but the middle of the "face" moves further than its edges as the
  view turns (like a nose in front of the cheeks) -> no single homography fits -> high score.
"""
import cv2
import numpy as np
import pytest

from app.config import VisionConfig
from app.vision.detector import IRIS_LEFT, IRIS_RIGHT, NUM_MESH_POINTS, Face
from app.vision.liveness_flow import FlowParams, flow_parallax_check, track_face_points

CFG = VisionConfig()
W, H = 640, 480
CX, CY = 320.0, 240.0
YAWS = [-12, -6, 0, 6, 12]


def texture(seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = cv2.GaussianBlur(rng.normal(0, 1, (H, W)).astype(np.float32), (0, 0), 2.0)
    t = cv2.normalize(t, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(t.astype(np.uint8), cv2.COLOR_GRAY2BGR)


def base_landmarks(seed=0) -> np.ndarray:
    """468 points inside an ellipse ("face") + iris points 80 px apart."""
    rng = np.random.default_rng(seed + 100)
    pts = []
    while len(pts) < NUM_MESH_POINTS:
        x, y = rng.uniform(-1, 1, 2)
        if x * x + y * y < 1:
            pts.append((CX + 100 * x, CY + 130 * y))
    lm = np.zeros((478, 2), np.float32)
    lm[:NUM_MESH_POINTS] = pts
    lm[NUM_MESH_POINTS:] = (CX, CY - 25)
    lm[IRIS_LEFT] = (CX - 40, CY - 25)
    lm[IRIS_RIGHT] = (CX + 40, CY - 25)
    return lm


def plane_homography(yaw_deg: float, shift=(0.0, 0.0)) -> np.ndarray:
    """Homography of a flat image rotating about its vertical axis in front of a pinhole camera."""
    f, dist = float(W), 2.0
    y = np.radians(yaw_deg)
    R = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    corners = np.array([[0, 0], [W, 0], [W, H], [0, H]], np.float64)
    P = np.c_[(corners - [CX, CY]) / f, np.zeros(4)] @ R.T + [0, 0, dist]
    uv = P[:, :2] / P[:, 2:] * f * dist + [CX + shift[0], CY + shift[1]]
    return cv2.getPerspectiveTransform(corners.astype(np.float32), uv.astype(np.float32))


def apply_h(H_: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(pts.reshape(-1, 1, 2).astype(np.float64), H_).reshape(-1, 2)


def bump(x, y):
    """Smooth 'depth' of the face: 1 in the middle, ~0 at the edges."""
    return np.exp(-((x - CX) ** 2 / (2 * 45.0 ** 2) + (y - CY) ** 2 / (2 * 60.0 ** 2)))


def render(img, H_, parallax_px):
    """Forward map F(x) = H(x + parallax_px * bump(x) * e_x), rendered with cv2.remap."""
    uu, vv = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    y = apply_h(np.linalg.inv(H_), np.column_stack([uu.ravel(), vv.ravel()]))
    x = y.copy()
    for _ in range(4):  # invert x + d(x) = y by fixed-point iteration
        x[:, 0] = y[:, 0] - parallax_px * bump(x[:, 0], x[:, 1])
    mx = x[:, 0].reshape(H, W).astype(np.float32)
    my = x[:, 1].reshape(H, W).astype(np.float32)
    return cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def forward(lm, H_, parallax_px):
    moved = lm.astype(np.float64).copy()
    moved[:, 0] += parallax_px * bump(lm[:, 0], lm[:, 1])
    return apply_h(H_, moved).astype(np.float32)


def scene(nonplanar: bool, seed=0, noise=2.0):
    rng = np.random.default_rng(seed + 7)
    img0, lm0 = texture(seed), base_landmarks(seed)
    images, faces = [], []
    for yaw in YAWS:
        if nonplanar:   # small rotation + parallax: the middle moves up to ~0.18 IOD further
            H_, par = plane_homography(yaw * 0.5, (yaw * 0.5, 0)), yaw * 1.2
        else:           # flat: a strong tilt of the whole plane (up to 30 deg), no parallax
            H_, par = plane_homography(yaw * 2.5, (yaw, 0)), 0.0
        im = render(img0, H_, par).astype(np.float32) + rng.normal(0, noise, (H, W, 3))
        images.append(np.clip(im, 0, 255).astype(np.uint8))
        faces.append(Face(forward(lm0, H_, par)))
    return images, faces


def check(images, faces):
    return flow_parallax_check(images, faces, min_movement=CFG.parallax_flow_min_movement,
                               threshold=CFG.parallax_flow_threshold,
                               ransac_reproj=CFG.parallax_ransac_reproj,
                               params=FlowParams(min_points=CFG.parallax_flow_min_points))


@pytest.mark.parametrize("seed", range(3))
def test_flat_textured_image_scores_low(seed):
    r = check(*scene(nonplanar=False, seed=seed))
    assert r.score is not None and r.score < 0.01, r
    assert r.status == "fail"


@pytest.mark.parametrize("seed", range(3))
def test_nonplanar_scene_scores_high(seed):
    r = check(*scene(nonplanar=True, seed=seed))
    assert r.score is not None and r.score > 0.02, r      # flat scenes score ~0.002
    assert r.status == "pass"


def test_many_points_are_tracked_on_texture():
    images, faces = scene(nonplanar=True)
    tracks = track_face_points(images, faces)
    assert len(tracks.points[0]) >= 100
    assert all(v.sum() >= 0.8 * len(tracks.points[0]) for v in tracks.valid)


def test_textureless_face_gives_insufficient_points():
    images, faces = scene(nonplanar=False)
    flat_grey = [np.full_like(im, 128) for im in images]
    r = check(flat_grey, faces)
    assert r.status == "insufficient_points" and r.passed is None
