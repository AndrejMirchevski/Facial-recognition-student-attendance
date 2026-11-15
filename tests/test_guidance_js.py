"""Runs the Node.js tests of the browser guidance port (tests/js/guidance.test.mjs) with the real
settings from config.py. Skipped if Node.js is not installed (it is not needed to run the app)."""
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from app import vision_service
from app.vision.quality import brightness, sharpness

ROOT = Path(__file__).resolve().parent.parent
NODE = shutil.which("node")


@pytest.mark.skipif(NODE is None, reason="Node.js not installed")
def test_browser_guidance_port(tmp_path):
    cfg = tmp_path / "guide.json"
    cfg.write_text(json.dumps(vision_service.guidance_settings()))
    proc = subprocess.run([NODE, "--test", str(ROOT / "tests" / "js" / "guidance.test.mjs")],
                          capture_output=True, text=True, env={**os.environ, "GUIDE_CONFIG": str(cfg)}, timeout=120)
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_python_quality_on_js_reference_image():
    """The same alternating 0/200 image the JS test uses: brightness 100 in Python too."""
    img = np.zeros((112, 112, 3), np.uint8)
    img.reshape(-1, 3)[1::2] = 200
    assert abs(brightness(img) - 100) < 1e-9
    assert sharpness(img) > 1000
