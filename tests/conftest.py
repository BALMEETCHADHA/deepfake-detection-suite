"""Shared fixtures. Run from the suite root:  pytest tests -v

The real Hugging Face models need internet + torch, so most tests inject a
STUB classifier with the same call signature as a transformers pipeline
(PIL image in -> [{label, score}, ...] out). That exercises everything AROUND
the model (face crop, fusion, calibration, verdict, API, evaluation) with
controllable scores. It says nothing about how accurate the real model is -
only batch_evaluate.py on real labelled data can tell you that.
"""
import os
import sys
import wave

import numpy as np
import pytest
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for sub in ["", "common", "01_image_detection", "02_document_detection", "03_audio_detection",
            "04_video_detection", "05_signature_detection", "06_dataset_and_evaluation", "07_api_backend"]:
    p = os.path.join(ROOT, sub) if sub else ROOT
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ.setdefault("DEEPFAKE_CALIBRATION", "off")     # tests opt in explicitly


class StubPipeline:
    """Mimics transformers' image-classification pipeline."""

    def __init__(self, fake_pct=50.0, fn=None):
        self.fake_pct = fake_pct
        self.fn = fn
        self.seen_sizes = []

    def __call__(self, image):
        self.seen_sizes.append(image.size)
        f = self.fn(image) if self.fn else self.fake_pct
        return [{"label": "Deepfake", "score": f / 100.0}, {"label": "Realism", "score": 1 - f / 100.0}]


def make_stub_detector(fake_pct=50.0, fn=None, **kw):
    from image_detector import ImageDeepfakeDetector
    det = ImageDeepfakeDetector(**kw)          # model load fails offline -> we replace it
    det._pipe = StubPipeline(fake_pct, fn)
    det._ml_available = True
    return det


@pytest.fixture
def stub_detector():
    return make_stub_detector


@pytest.fixture
def natural_image(tmp_path):
    from skimage import data
    p = tmp_path / "natural.png"
    Image.fromarray(data.coffee()).save(p)
    return str(p)


@pytest.fixture
def noise_image(tmp_path):
    rng = np.random.default_rng(0)
    p = tmp_path / "noise.png"
    Image.fromarray(rng.integers(0, 255, (300, 300, 3), dtype=np.uint8)).save(p)
    return str(p)


@pytest.fixture
def face_image(tmp_path):
    from skimage import data
    p = tmp_path / "face.png"
    bg = Image.fromarray(data.coffee()).resize((1000, 750))
    bg.paste(Image.fromarray(data.astronaut()), (400, 150))
    bg.save(p)
    return str(p)


@pytest.fixture
def wav_file(tmp_path):
    p = tmp_path / "tone.wav"
    sr = 16000
    t = np.arange(sr * 2) / sr
    y = (0.3 * np.sin(2 * np.pi * 220 * t) * (1 + 0.1 * np.sin(2 * np.pi * 3 * t)) +
         0.01 * np.random.default_rng(0).standard_normal(len(t)))
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes((np.clip(y, -1, 1) * 32767).astype("<i2").tobytes())
    return str(p)


@pytest.fixture
def video_file(tmp_path):
    import cv2
    p = str(tmp_path / "clip.mp4")
    vw = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*"mp4v"), 10, (160, 120))
    rng = np.random.default_rng(0)
    for i in range(30):
        frame = np.full((120, 160, 3), 90, np.uint8)
        cv2.circle(frame, (20 + i * 4, 60), 15, (200, 120, 40), -1)
        frame = np.clip(frame + rng.normal(0, 3, frame.shape), 0, 255).astype(np.uint8)
        vw.write(frame)
    vw.release()
    return p


@pytest.fixture
def pdf_file(tmp_path):
    import pymupdf
    from skimage import data
    img = tmp_path / "i.png"
    Image.fromarray(data.coffee()).save(img)
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_image(pymupdf.Rect(50, 50, 350, 280), filename=str(img))
    p = str(tmp_path / "doc.pdf")
    doc.save(p)
    return p


@pytest.fixture
def signature_files(tmp_path):
    from PIL import ImageDraw
    def make(name, jitter):
        im = Image.new("L", (300, 120), 255)
        d = ImageDraw.Draw(im)
        pts = [(20 + i * 20, 60 + int(30 * np.sin(i / 1.5)) + jitter * (i % 3)) for i in range(13)]
        d.line(pts, fill=0, width=3)
        p = str(tmp_path / name)
        im.save(p)
        return p
    return make("ref.png", 0), make("q.png", 2)
