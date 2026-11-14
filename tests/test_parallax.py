"""Parallax / planarity check on synthetic landmark sets (no models needed).

A synthetic 3D "face" (curved surface + nose bump) is rotated and projected with a pinhole
camera. A real head turning must pass; the same landmarks flattened onto a plane (a photo)
moving the same way must fail; a head that does not move must give "insufficient_movement".
"""
import numpy as np
import pytest

from app.config import VisionConfig
from app.vision.liveness_parallax import parallax_check

CFG = VisionConfig()
IOD_PX = 80.0


def synthetic_face(seed=0) -> np.ndarray:
    """478 x 3 points in inter-ocular units: 468 'mesh' points + 10 'iris' points (468..477)."""
    rng = np.random.default_rng(seed)
    pts = []
    while len(pts) < 468:
        x, y = rng.uniform(-1.3, 1.3), rng.uniform(-1.7, 1.7)
        r = (x / 1.3) ** 2 + (y / 1.7) ** 2
        if r < 1:
            z = 0.9 * np.sqrt(1 - r) + 0.6 * np.exp(-(x ** 2 + (y - 0.3) ** 2) / 0.08)  # face + nose
            pts.append((x, y, z))
    iris = [(-0.5, 0.0, 0.75)] * 5 + [(0.5, 0.0, 0.75)] * 5  # 468 = left iris, 473 = right iris
    return np.array(pts + iris, dtype=np.float64)


def rotation(yaw, pitch=0.0, roll=0.0):
    y, p, r = np.radians([yaw, pitch, roll])
    ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    rz = np.array([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1]])
    return rz @ rx @ ry


def capture(points3d, yaws, noise=0.005, seed=1, distance=9.0):
    """Project the points for each yaw angle; returns landmark sets in pixels + IODs."""
    rng = np.random.default_rng(seed)
    sets, iods = [], []
    for yaw in yaws:
        X = points3d @ rotation(yaw, rng.normal(0, 1), rng.normal(0, 1)).T
        X = X + np.array([rng.normal(0, 0.2), rng.normal(0, 0.2), distance])
        uv = X[:, :2] / X[:, 2:3]
        uv = uv / np.linalg.norm(uv[468] - uv[473]) * IOD_PX + 320
        uv += rng.normal(0, noise * IOD_PX, uv.shape)
        sets.append(uv.astype(np.float32))
        iods.append(float(np.linalg.norm(uv[468] - uv[473])))
    return sets, iods


def check(sets, iods):
    return parallax_check(sets, iods, min_movement=CFG.parallax_min_movement,
                          threshold=CFG.parallax_threshold, ransac_reproj=CFG.parallax_ransac_reproj)


@pytest.mark.parametrize("seed", range(5))
def test_real_head_turning_passes(seed):
    r = check(*capture(synthetic_face(seed), [-12, -6, 0, 6, 12], seed=seed))
    assert r.status == "pass", r
    assert r.score > CFG.parallax_threshold


@pytest.mark.parametrize("seed", range(5))
def test_flat_photo_moving_fails(seed):
    flat = synthetic_face(seed)
    flat[:, 2] = 0.0  # a photo: every point on one plane
    # tilt the photo strongly so it moves enough to be scored
    r = check(*capture(flat, [-40, -20, 0, 20, 40], seed=seed))
    assert r.status == "fail", r
    assert r.score < CFG.parallax_threshold


def test_no_head_movement_asks_to_retry():
    r = check(*capture(synthetic_face(), [0, 0, 0, 0, 0], noise=0.002))
    assert r.status == "insufficient_movement"
    assert r.passed is None and r.score is None


def test_flat_photo_barely_moving_is_not_a_pass():
    flat = synthetic_face()
    flat[:, 2] = 0.0
    r = check(*capture(flat, [-3, -1, 0, 1, 3]))
    assert r.status in ("fail", "insufficient_movement")
