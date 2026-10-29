"""
generate_smoke_test_dataset.py
--------------------------------
Generates a LABELED synthetic test dataset at whatever scale you ask for,
with a real/fake (or genuine/forged) folder split so you can batch-run
every module and measure accuracy against KNOWN ground truth.

IMPORTANT — read this before you trust any accuracy number from it:
This data is synthetic (procedurally generated patterns), not real deepfakes
or real forged signatures. It is useful for:
  - proving the pipeline runs correctly end-to-end at scale (no crashes,
    correct file counts, correct folder routing)
  - catching integration bugs before you touch real data
  - giving your batch_evaluate.py script something to run against immediately

It is NOT useful for:
  - claiming a real accuracy percentage in your project report
  - representing what actual AI-generated faces/voices/videos look like

For a real accuracy number, download real datasets (FaceForensics++, Celeb-DF,
ASVspoof2019, CEDAR — see README.md) and point generate_documents_from_pool()
or batch_evaluate.py at those folders instead.

Usage:
    python generate_smoke_test_dataset.py --out ./smoke_test_data \
        --n_images 300 --n_audio 300 --n_signatures 300 --n_videos 50 \
        --video_duration 42 --n_documents 10 --images_per_document 30
"""

from __future__ import annotations
import argparse
import os
import random
import shutil

import numpy as np
from PIL import Image, ImageDraw


# --------------------------------------------------------------------------- #
# Images: "real-like" = smooth natural gradients + mild noise (low high-freq
# energy). "fake-like" = checkerboard/high-frequency patterns, mimicking the
# GAN upsampling-artifact signature described in your own survey papers.
# --------------------------------------------------------------------------- #
def _make_real_like_image(size=(256, 256)) -> Image.Image:
    w, h = size
    yy, xx = np.mgrid[0:h, 0:w]
    base = (
        128
        + 60 * np.sin(xx / (20 + random.random() * 30))
        + 60 * np.cos(yy / (20 + random.random() * 30))
    )
    noise = np.random.normal(0, 6, size=(h, w))
    channel = np.clip(base + noise, 0, 255)
    rgb = np.stack([channel] * 3, axis=-1).astype(np.uint8)
    # slight per-channel tint so it's not perfectly grayscale
    rgb = rgb.astype(np.int16)
    rgb[..., 0] = np.clip(rgb[..., 0] + random.randint(-15, 15), 0, 255)
    rgb[..., 2] = np.clip(rgb[..., 2] + random.randint(-15, 15), 0, 255)
    return Image.fromarray(rgb.astype(np.uint8))


def _make_fake_like_image(size=(256, 256)) -> Image.Image:
    w, h = size
    checker = np.indices((h, w)).sum(axis=0) % (2 + random.randint(0, 3))
    base = 128 + 90 * (checker % 2)
    noise = np.random.normal(0, 25, size=(h, w))  # elevated high-freq noise
    channel = np.clip(base + noise, 0, 255)
    rgb = np.stack([channel] * 3, axis=-1).astype(np.uint8)
    return Image.fromarray(rgb)


def generate_images(out_dir: str, n: int) -> None:
    real_dir = os.path.join(out_dir, "images", "real")
    fake_dir = os.path.join(out_dir, "images", "fake")
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(fake_dir, exist_ok=True)

    n_real = n // 2
    n_fake = n - n_real
    for i in range(n_real):
        _make_real_like_image().save(os.path.join(real_dir, f"real_{i:04d}.jpg"), quality=90)
    for i in range(n_fake):
        _make_fake_like_image().save(os.path.join(fake_dir, f"fake_{i:04d}.jpg"), quality=90)
    print(f"[images] wrote {n_real} real-like + {n_fake} fake-like to {out_dir}/images/")


# --------------------------------------------------------------------------- #
# Audio: "real-like" = jittered pitch + broadband noise (mimics natural voice
# micro-variation). "fake-like" = perfectly smooth pure tone (mimics an
# overly-clean synthetic/vocoder signal, matching explain_audio.py's heuristic).
# --------------------------------------------------------------------------- #
def _make_real_like_audio(sr=16000, duration=2.5) -> np.ndarray:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    f0_base = random.uniform(120, 260)
    jitter = f0_base + 10 * np.sin(2 * np.pi * random.uniform(2, 5) * t) + np.random.normal(0, 3, len(t))
    phase = 2 * np.pi * np.cumsum(jitter) / sr
    signal = 0.3 * np.sin(phase) + np.random.normal(0, 0.03, len(t))
    return signal.astype(np.float32)


def _make_fake_like_audio(sr=16000, duration=2.5) -> np.ndarray:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    freq = random.uniform(120, 260)
    signal = 0.3 * np.sin(2 * np.pi * freq * t)  # perfectly smooth, no jitter
    return signal.astype(np.float32)


def generate_audio(out_dir: str, n: int) -> None:
    import soundfile as sf

    real_dir = os.path.join(out_dir, "audio", "real")
    fake_dir = os.path.join(out_dir, "audio", "fake")
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(fake_dir, exist_ok=True)

    n_real = n // 2
    n_fake = n - n_real
    for i in range(n_real):
        sf.write(os.path.join(real_dir, f"real_{i:04d}.wav"), _make_real_like_audio(), 16000)
    for i in range(n_fake):
        sf.write(os.path.join(fake_dir, f"fake_{i:04d}.wav"), _make_fake_like_audio(), 16000)
    print(f"[audio] wrote {n_real} real-like + {n_fake} fake-like to {out_dir}/audio/")


# --------------------------------------------------------------------------- #
# Signatures: one "signer" identity = a fixed random seed producing a
# consistent stroke pattern; genuine samples reuse that seed with small
# variation, forged samples use a different seed (different "hand").
# --------------------------------------------------------------------------- #
def _draw_signature(seed: int, wobble: float, size=(400, 150)) -> Image.Image:
    rng = np.random.RandomState(seed)
    img = Image.new("L", size, 255)
    draw = ImageDraw.Draw(img)
    pts, x = [], 30
    for _ in range(60):
        x += rng.uniform(4, 7)
        y = size[1] / 2 + 30 * np.sin(x / 25) + rng.uniform(-wobble, wobble)
        pts.append((x, y))
    draw.line(pts, fill=0, width=3)
    return img


def generate_signatures(out_dir: str, n: int, n_signers: int = 30) -> None:
    genuine_dir = os.path.join(out_dir, "signatures", "genuine")
    forged_dir = os.path.join(out_dir, "signatures", "forged")
    ref_dir = os.path.join(out_dir, "signatures", "references")
    for d in (genuine_dir, forged_dir, ref_dir):
        os.makedirs(d, exist_ok=True)

    per_signer = max(1, n // n_signers)
    count_genuine, count_forged = 0, 0

    for signer_id in range(n_signers):
        signer_seed = 1000 + signer_id
        # reference exemplar for this signer
        _draw_signature(signer_seed, wobble=3).save(
            os.path.join(ref_dir, f"signer{signer_id:03d}_reference.png")
        )
        for k in range(per_signer):
            if count_genuine < n // 2:
                # genuine: same signer identity, small natural variation
                img = _draw_signature(signer_seed + k, wobble=3)
                img.save(os.path.join(genuine_dir, f"signer{signer_id:03d}_genuine_{k:02d}.png"))
                count_genuine += 1
            if count_forged < n - n // 2:
                # forged: different "hand" entirely (unrelated seed, more wobble)
                forged_seed = 9000 + signer_id * 37 + k
                img = _draw_signature(forged_seed, wobble=14)
                img.save(os.path.join(forged_dir, f"signer{signer_id:03d}_forged_{k:02d}.png"))
                count_forged += 1

    print(
        f"[signatures] wrote {count_genuine} genuine + {count_forged} forged "
        f"(+{n_signers} reference exemplars) to {out_dir}/signatures/"
    )


# --------------------------------------------------------------------------- #
# Video: "real-like" = smoothly evolving frames (temporally consistent).
# "fake-like" = independent random noise per frame (high flicker), matching
# the temporal-coherence artifact explain_video.py specifically checks for.
# --------------------------------------------------------------------------- #
def generate_videos(out_dir: str, n: int, duration_sec: float, fps: int = 10, size=(320, 240)) -> None:
    import cv2

    real_dir = os.path.join(out_dir, "videos", "real")
    fake_dir = os.path.join(out_dir, "videos", "fake")
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(fake_dir, exist_ok=True)
    w, h = size
    n_frames = int(duration_sec * fps)

    n_real = n // 2
    n_fake = n - n_real

    for i in range(n_real):
        path = os.path.join(real_dir, f"real_{i:03d}.mp4")
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        phase = random.uniform(0, 6.28)
        for f in range(n_frames):
            yy, xx = np.mgrid[0:h, 0:w]
            frame = (128 + 100 * np.sin((xx + f * 2) / 40 + phase) + 20 * np.sin((yy) / 30)).astype(np.uint8)
            bgr = cv2.cvtColor(np.stack([frame] * 3, axis=-1), cv2.COLOR_RGB2BGR)
            writer.write(bgr)
        writer.release()

    for i in range(n_fake):
        path = os.path.join(fake_dir, f"fake_{i:03d}.mp4")
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for f in range(n_frames):
            frame = (np.random.rand(h, w, 3) * 255).astype(np.uint8)  # independent noise per frame = flicker
            writer.write(frame)
        writer.release()

    print(f"[videos] wrote {n_real} real-like + {n_fake} fake-like ({duration_sec}s each) to {out_dir}/videos/")


# --------------------------------------------------------------------------- #
# Documents: assemble PDFs, each embedding `images_per_document` images
# randomly sampled from the images/ pool already generated above, with a
# ground-truth manifest recording which images (real/fake) went where.
# --------------------------------------------------------------------------- #
def generate_documents(out_dir: str, n_documents: int, images_per_document: int) -> None:
    import json
    import glob
    import fitz  # pymupdf

    doc_dir = os.path.join(out_dir, "documents")
    os.makedirs(doc_dir, exist_ok=True)

    real_pool = glob.glob(os.path.join(out_dir, "images", "real", "*.jpg"))
    fake_pool = glob.glob(os.path.join(out_dir, "images", "fake", "*.jpg"))
    if not real_pool or not fake_pool:
        raise RuntimeError("Run generate_images() first — documents are built from the image pool.")

    manifest = []
    for d in range(n_documents):
        # Each document gets a random mix; roughly half the documents contain
        # at least one fake image (to give batch_evaluate.py real positives).
        contains_fake = d % 2 == 0
        n_fake_in_doc = random.randint(1, 5) if contains_fake else 0
        n_real_in_doc = images_per_document - n_fake_in_doc

        chosen = (
            random.sample(real_pool, min(n_real_in_doc, len(real_pool)))
            + random.sample(fake_pool, min(n_fake_in_doc, len(fake_pool)))
        )
        random.shuffle(chosen)

        pdf_path = os.path.join(doc_dir, f"document_{d:02d}.pdf")
        doc = fitz.open()
        for img_path in chosen:
            page = doc.new_page()
            page.insert_image(fitz.Rect(50, 50, 300, 300), filename=img_path)
        doc.set_metadata({"producer": "smoke-test-generator", "creator": "smoke-test-generator"})
        doc.save(pdf_path)
        doc.close()

        manifest.append({
            "document": os.path.basename(pdf_path),
            "n_images": len(chosen),
            "n_fake_images": n_fake_in_doc,
            "expected_doc_verdict": "FAKE" if n_fake_in_doc > 0 else "REAL",
        })

    with open(os.path.join(doc_dir, "ground_truth_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[documents] wrote {n_documents} PDFs (~{images_per_document} images each) to {out_dir}/documents/")


# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Generate a labeled synthetic smoke-test dataset")
    parser.add_argument("--out", type=str, default="./smoke_test_data")
    parser.add_argument("--n_images", type=int, default=300)
    parser.add_argument("--n_audio", type=int, default=300)
    parser.add_argument("--n_signatures", type=int, default=300)
    parser.add_argument("--n_videos", type=int, default=50)
    parser.add_argument("--video_duration", type=float, default=42.0, help="seconds per video")
    parser.add_argument("--n_documents", type=int, default=10)
    parser.add_argument("--images_per_document", type=int, default=30)
    parser.add_argument("--skip", nargs="*", default=[], choices=["images", "audio", "signatures", "videos", "documents"])
    parser.add_argument("--fresh", action="store_true", help="delete --out first if it already exists")
    args = parser.parse_args()

    if args.fresh and os.path.exists(args.out):
        shutil.rmtree(args.out)
    os.makedirs(args.out, exist_ok=True)

    if "images" not in args.skip:
        generate_images(args.out, args.n_images)
    if "audio" not in args.skip:
        generate_audio(args.out, args.n_audio)
    if "signatures" not in args.skip:
        generate_signatures(args.out, args.n_signatures)
    if "videos" not in args.skip:
        generate_videos(args.out, args.n_videos, args.video_duration)
    if "documents" not in args.skip:
        generate_documents(args.out, args.n_documents, args.images_per_document)

    print(f"\nDone. Dataset written to: {os.path.abspath(args.out)}")
    print("Reminder: this is SYNTHETIC pipeline-testing data, not a real accuracy benchmark. See README.md.")


if __name__ == "__main__":
    main()
