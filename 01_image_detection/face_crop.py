"""
face_crop.py
-------------
Detect the most confident face in an image and return a square crop around it.

WHY: the default image model (Deep-Fake-Detector-v2) was trained on FACE
images. Feeding it a full scene or a screenshot - where the face is a small
part of the frame, or there is no face at all - pushes its output toward
50/50. Classifying the face crop instead gives it the kind of input it
knows.

Uses OpenCV's bundled Haar cascade: no download, works offline, CPU only.
Haar is fast but modest - it finds mostly frontal faces. If OpenCV isn't
installed or no face is found, callers fall back to the full image.
"""

from __future__ import annotations
from typing import Optional, Tuple

from PIL import Image

_cascade = None
_cascade_failed = False
_failure_reason = ""

MAX_DETECT_SIDE = 1024        # detect on a downscaled copy for speed
CROP_SCALE = 1.6              # crop side = CROP_SCALE * face size (adds forehead/chin/ears context)
SKIP_IF_CROP_COVERS = 0.85    # if the crop is already ~the whole image, don't bother
MIN_FACE_CONFIDENCE = 1.0     # Haar level-weight; true faces score ~5, textured-background false positives score < 0


def _cascade_paths(cv2):
    """Where OpenCV keeps its face model. Different installs (opencv-python, -headless, mixed installs
    in Colab) put it in different places, so try several instead of trusting cv2.data alone."""
    name = "haarcascade_frontalface_default.xml"
    import glob
    import os
    paths = []
    try:
        paths.append(os.path.join(cv2.data.haarcascades, name))
    except Exception:
        pass
    here = os.path.dirname(getattr(cv2, "__file__", "") or "")
    if here:
        paths.append(os.path.join(here, "data", name))
        paths += glob.glob(os.path.join(os.path.dirname(here), "**", name), recursive=True)
    return paths


def _get_cascade():
    global _cascade, _cascade_failed, _failure_reason
    if _cascade is not None or _cascade_failed:
        return _cascade
    try:
        import cv2
    except Exception as exc:                      # not installed, or broken (e.g. numpy/OpenCV version clash)
        _cascade_failed = True
        _failure_reason = f"OpenCV could not be imported ({type(exc).__name__}: {exc})"
        return None
    try:
        for p in _cascade_paths(cv2):
            c = cv2.CascadeClassifier(p)
            if not c.empty():
                _cascade = c
                return _cascade
        raise RuntimeError("face-model file (haarcascade_frontalface_default.xml) not found or unreadable")
    except Exception as exc:
        _cascade_failed = True
        _failure_reason = f"OpenCV is installed but its face model failed to load ({exc})"
    return None


def face_detection_available() -> bool:
    return _get_cascade() is not None


def face_detection_status() -> Tuple[bool, str]:
    """(available, reason_if_not) - used by the warning text and by the API's /api/health."""
    ok = face_detection_available()
    return ok, "" if ok else _failure_reason


def crop_largest_face(image: Image.Image) -> Tuple[Optional[Image.Image], dict]:
    """Returns (crop_or_None, info).

    info keys: available (bool), n_faces (int, confident detections), face_box (x,y,w,h in original
    pixels) and crop_box (l,t,r,b) when a face was found, skipped_reason when
    a face was found but cropping would change nothing.
    """
    info = {"available": False, "n_faces": 0}
    cascade = _get_cascade()
    if cascade is None:
        info["reason"] = _failure_reason
        return None, info
    info["available"] = True

    import cv2
    import numpy as np

    rgb = image.convert("RGB")
    W, H = rgb.size
    scale = min(1.0, MAX_DETECT_SIDE / max(W, H))
    small = rgb if scale == 1.0 else rgb.resize((max(1, int(W * scale)), max(1, int(H * scale))))
    gray = cv2.cvtColor(np.asarray(small), cv2.COLOR_RGB2GRAY)

    min_side = max(30, int(0.06 * min(gray.shape[:2])))
    # detectMultiScale3 also returns a confidence per box. Picking the most
    # CONFIDENT box (not the biggest) avoids Haar's classic false positives on
    # textured backgrounds, which are often larger than the real face.
    rects, _, weights = cascade.detectMultiScale3(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(min_side, min_side), outputRejectLevels=True
    )
    kept = [(tuple(r), float(w)) for r, w in zip(rects, weights) if float(w) >= MIN_FACE_CONFIDENCE]
    info["n_faces"] = len(kept)
    if not kept:
        return None, info

    (x, y, w, h), conf = max(kept, key=lambda kw: kw[1])
    info["face_confidence"] = round(conf, 2)
    x, y, w, h = [int(round(v / scale)) for v in (x, y, w, h)]
    info["face_box"] = (x, y, w, h)

    side = int(round(CROP_SCALE * max(w, h)))
    cx, cy = x + w / 2.0, y + h / 2.0
    left, top = int(round(cx - side / 2)), int(round(cy - side / 2))
    left, top = max(0, min(left, W - 1)), max(0, min(top, H - 1))
    right, bottom = min(W, left + side), min(H, top + side)
    info["crop_box"] = (left, top, right, bottom)

    if (right - left) * (bottom - top) >= SKIP_IF_CROP_COVERS * W * H:
        info["skipped_reason"] = "face already fills the frame"
        return None, info

    return rgb.crop((left, top, right, bottom)), info
