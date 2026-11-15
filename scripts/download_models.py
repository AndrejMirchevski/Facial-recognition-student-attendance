"""Download the model weight files into ./models (and the MediaPipe browser runtime into
app/static/vendor/) and verify their SHA-256 hashes.

Usage:  .venv\\Scripts\\python scripts\\download_models.py [--force]

Only uses the Python standard library, so it can run before anything else is set up.
Every file is checked against a pinned SHA-256 hash; a file with the wrong hash is
rejected (and deleted), so a changed or corrupted upstream file is never used silently.
Sources and licences are documented in models/README.md.

The browser runtime (@mediapipe/tasks-vision, used for the guided capture on the enrollment and
check-in pages) is served by our own server, never from a CDN; its files are extracted from the
official npm package and are not committed to git either.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
TASKS_VISION_VERSION = "1.2.0"
WEB_VENDOR_DIR = ROOT / "app" / "static" / "vendor" / "mediapipe" / f"tasks-vision-{TASKS_VISION_VERSION}"
_NPM_TGZ = (f"https://registry.npmjs.org/@mediapipe/tasks-vision/-/"
            f"tasks-vision-{TASKS_VISION_VERSION}.tgz")


@dataclass
class Source:
    url: str
    zip_member: str | None = None  # if set, the URL is a zip and we extract this one file
    tar_member: str | None = None  # if set, the URL is a .tgz and we extract this one file


@dataclass
class ModelFile:
    filename: str
    sha256: str
    purpose: str
    sources: list[Source] = field(default_factory=list)  # tried in order
    dest_dir: Path = MODELS_DIR                            # filename may contain a sub-folder


MODEL_FILES: list[ModelFile] = [
    ModelFile(
        filename="face_landmarker.task",
        sha256="64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff",
        purpose="MediaPipe Face Landmarker (face detection + 478 landmarks)",
        sources=[Source("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                        "face_landmarker/float16/1/face_landmarker.task")],
    ),
    ModelFile(
        filename="w600k_r50.onnx",
        sha256="4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43",
        purpose="ArcFace ResNet-50 (InsightFace buffalo_l recognition model)",
        sources=[
            # Mirror of the single file (174 MB); byte-identical to the file in the official zip.
            Source("https://huggingface.co/public-data/insightface/resolve/main/models/buffalo_l/w600k_r50.onnx"),
            # Official InsightFace release (289 MB zip containing 5 models; we keep only this one).
            Source("https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
                   zip_member="w600k_r50.onnx"),
        ],
    ),
    ModelFile(
        filename="depth_anything_v2_small.onnx",
        sha256="afb6a5c28f3b6bf1618c6e43f02073ef9dfdc70e937502d51603e57b0a1df10c",
        purpose="Depth Anything V2 Small (monocular relative depth)",
        sources=[Source("https://huggingface.co/onnx-community/depth-anything-v2-small/resolve/main/onnx/model.onnx")],
    ),
    ModelFile(
        filename="MiniFASNetV2.onnx",
        sha256="b32929adc2d9c34b9486f8c4c7bc97c1b69bc0ea9befefc380e4faae4e463907",
        purpose="MiniFASNetV2 texture anti-spoofing (crop scale 2.7)",
        sources=[Source("https://github.com/yakhyo/face-anti-spoofing/releases/download/weights/MiniFASNetV2.onnx")],
    ),
    ModelFile(
        filename="MiniFASNetV1SE.onnx",
        sha256="ebab7f90c7833fbccd46d3a555410e78d969db5438e169b6524be444862b3676",
        purpose="MiniFASNetV1SE texture anti-spoofing (crop scale 4.0)",
        sources=[Source("https://github.com/yakhyo/face-anti-spoofing/releases/download/weights/MiniFASNetV1SE.onnx")],
    ),
]


# MediaPipe Tasks for JavaScript (Apache-2.0): ES module + WebAssembly (SIMD and non-SIMD builds).
WEB_FILES: list[ModelFile] = [
    ModelFile(filename=name, sha256=digest, dest_dir=WEB_VENDOR_DIR,
              purpose=f"@mediapipe/tasks-vision {TASKS_VISION_VERSION} browser runtime",
              sources=[Source(_NPM_TGZ, tar_member=f"package/{name}")])
    for name, digest in [
        ("vision_bundle.mjs", "870da5cfc43bfcf2a8b0f14e8152dae8c7e701722e1191b0e592b94b81b4127a"),
        ("wasm/vision_wasm_internal.js", "85c1a342be70427987f900969a5f26e17557351111364e6585e97fb03e93e108"),
        ("wasm/vision_wasm_internal.wasm", "bcc13df22d3d69ec8cbe8e280989356d844ae00f4523eb82072d2484f44134dc"),
        ("wasm/vision_wasm_nosimd_internal.js", "e9b2b799e0f1078f5538b300a57835b823742c37e2646308a03f0f25b39fdcdc"),
        ("wasm/vision_wasm_nosimd_internal.wasm", "ed5e5e5812490bd90d6a440290845ff9e4c516f696847a28a0c0e0573470d469"),
    ]
]


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "face-attendance-model-downloader"})
    with urllib.request.urlopen(req, timeout=60) as resp, dest.open("wb") as out:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r    {done / 1e6:7.1f} / {total / 1e6:.1f} MB", end="", flush=True)
        print()


def fetch(model: ModelFile, force: bool, cache: dict[str, Path]) -> bool:
    target = model.dest_dir / model.filename
    if target.exists() and not force:
        if sha256_of(target) == model.sha256:
            print(f"[ok]   {model.filename} already present")
            return True
        print(f"[warn] {model.filename} exists but hash differs - re-downloading")
    target.parent.mkdir(parents=True, exist_ok=True)

    for src in model.sources:
        print(f"[get]  {model.filename}  <-  {src.url}" + (" (cached)" if src.url in cache else ""))
        with tempfile.TemporaryDirectory(dir=target.parent) as tmp:
            tmp_path = Path(tmp) / "download"
            try:
                if src.url in cache:          # one archive, several files: download it only once
                    shutil.copyfile(cache[src.url], tmp_path)
                else:
                    _download(src.url, tmp_path)
                    if src.zip_member or src.tar_member:
                        cache[src.url] = Path(cache["__dir__"]) / f"{len(cache)}.bin"
                        shutil.copyfile(tmp_path, cache[src.url])
                if src.zip_member:
                    with zipfile.ZipFile(tmp_path) as zf, zf.open(src.zip_member) as zsrc, \
                            (Path(tmp) / "member").open("wb") as zdst:
                        shutil.copyfileobj(zsrc, zdst)
                    tmp_path = Path(tmp) / "member"
                elif src.tar_member:
                    with tarfile.open(tmp_path, "r:gz") as tf:
                        member = tf.extractfile(src.tar_member)   # read one file; never extract paths
                        if member is None:
                            raise FileNotFoundError(src.tar_member)
                        with (Path(tmp) / "member").open("wb") as out:
                            shutil.copyfileobj(member, out)
                    tmp_path = Path(tmp) / "member"
            except Exception as exc:  # network error, bad archive, ...
                print(f"[fail] {exc}")
                continue
            digest = sha256_of(tmp_path)
            if digest != model.sha256:
                print(f"[fail] SHA-256 mismatch (got {digest}) - file rejected")
                continue
            shutil.move(str(tmp_path), target)
            print(f"[ok]   {model.filename} verified")
            return True
    print(f"[ERROR] could not obtain {model.filename} from any source")
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="re-download even if files are present")
    args = parser.parse_args()
    MODELS_DIR.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as cache_dir:
        cache: dict[str, Path] = {"__dir__": Path(cache_dir)}
        ok = all([fetch(m, args.force, cache) for m in MODEL_FILES + WEB_FILES])
    print("All models and browser files ready." if ok else "Some files are missing - see errors above.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
