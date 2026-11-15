"""Validation and decoding of the uploaded camera frames. Nothing from the browser is trusted.

Checks, in order: number of files, size per file, file type by its first bytes (JPEG / PNG only),
image dimensions read from the file HEADER before decoding (so a small file claiming a gigantic
resolution - a "decompression bomb" - is refused without being decoded), then the decode itself,
and that all frames have the same size.
"""
from __future__ import annotations

import struct

import cv2
import numpy as np
from werkzeug.datastructures import FileStorage

MAX_FRAME_BYTES = 4 * 1024 * 1024
MIN_SIDE = 240
MAX_SIDE = 2560
MAX_PIXELS = 2560 * 1600


class FrameError(ValueError):
    """Invalid upload; .message is safe to show to the user."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


def image_size(data: bytes) -> tuple[str, int, int] | None:
    """(kind, width, height) from a JPEG or PNG header, or None if it is neither."""
    if data[:8] == b"\x89PNG\r\n\x1a\n" and len(data) >= 24 and data[12:16] == b"IHDR":
        w, h = struct.unpack(">II", data[16:24])
        return "png", w, h
    if data[:3] == b"\xff\xd8\xff":
        i = 2
        while i + 9 < len(data):
            if data[i] != 0xFF:
                return None
            marker = data[i + 1]
            if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:   # markers without a length
                i += 2
                continue
            seg_len = struct.unpack(">H", data[i + 2:i + 4])[0]
            # SOF0..SOF15 except DHT (C4), JPG (C8), DAC (CC) carry the frame size
            if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
                h, w = struct.unpack(">HH", data[i + 5:i + 9])
                return "jpeg", w, h
            i += 2 + seg_len
        return None
    return None


def read_frames(files: list[FileStorage], expected: int) -> list[np.ndarray]:
    if len(files) != expected:
        raise FrameError(f"Expected {expected} pictures, got {len(files)}.")
    frames = []
    for n, f in enumerate(files, start=1):
        data = f.read(MAX_FRAME_BYTES + 1)
        if len(data) > MAX_FRAME_BYTES:
            raise FrameError(f"Picture {n} is too large.")
        info = image_size(data)
        if info is None:
            raise FrameError(f"Picture {n} is not a JPEG or PNG image.")
        _, w, h = info
        if min(w, h) < MIN_SIDE or max(w, h) > MAX_SIDE or w * h > MAX_PIXELS:
            raise FrameError(f"Picture {n} has an unsupported size ({w}x{h}).")
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None or img.shape[1] != w or img.shape[0] != h:
            raise FrameError(f"Picture {n} could not be read.")
        frames.append(img)
    if len({im.shape for im in frames}) != 1:
        raise FrameError("The pictures do not all have the same size.")
    return frames
