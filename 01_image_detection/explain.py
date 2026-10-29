"""
explain.py
----------
Classical, model-independent evidence generators for image manipulation.
These do NOT decide real/fake by themselves — they give supporting *evidence*
that is combined with the ML model's verdict in image_detector.py, so the
final report can say *why* something looks fake, not just *that* it does.

Two techniques, both standard in image forensics literature:

1. Error Level Analysis (ELA): re-save the image at a known JPEG quality and
   diff it against the original. Regions that were edited/generated tend to
   have a different recompression error signature than untouched regions.

2. High-frequency energy ratio: GAN/diffusion-generated images often have
   unnatural high-frequency spectral patterns (checkerboard/upsampling
   artifacts) compared to camera photos. We approximate this with a simple
   FFT-based ratio, no extra heavy dependencies required.
"""

from __future__ import annotations
import io
import numpy as np
from PIL import Image, ImageChops


def error_level_analysis(image: Image.Image, quality: int = 90) -> tuple[Image.Image, float]:
    """Returns (ela_image, ela_score in [0,1]).

    ela_score is the mean normalized pixel difference between the original
    and its JPEG-requantized copy. Higher = more inconsistent compression
    history = more suspicious.
    """
    rgb = image.convert("RGB")
    buffer = io.BytesIO()
    rgb.save(buffer, "JPEG", quality=quality)
    buffer.seek(0)
    recompressed = Image.open(buffer)

    diff = ImageChops.difference(rgb, recompressed)
    diff_arr = np.asarray(diff, dtype=np.float32)

    max_diff = diff_arr.max() if diff_arr.max() > 0 else 1.0
    scaled = (diff_arr / max_diff * 255).astype(np.uint8)
    ela_image = Image.fromarray(scaled)

    # Normalize score 0..1 using the raw (unscaled) mean difference so it is
    # comparable across images instead of always re-scaled to full contrast.
    ela_score = float(diff_arr.mean() / 255.0)
    return ela_image, ela_score


def high_frequency_ratio(image: Image.Image) -> float:
    """Fraction of spectral energy sitting in the outer (high-frequency) ring
    of the 2D FFT magnitude spectrum, computed on a grayscale, resized copy.
    """
    gray = image.convert("L").resize((256, 256))
    arr = np.asarray(gray, dtype=np.float32)

    f = np.fft.fftshift(np.fft.fft2(arr))
    magnitude = np.abs(f)

    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    max_radius = np.sqrt(cy ** 2 + cx ** 2)

    low_mask = radius <= 0.25 * max_radius
    high_mask = radius > 0.6 * max_radius

    low_energy = magnitude[low_mask].sum()
    high_energy = magnitude[high_mask].sum()
    total = low_energy + high_energy
    if total == 0:
        return 0.0
    return float(high_energy / total)


def build_forensic_evidence(image: Image.Image) -> tuple[list[str], dict]:
    """Runs both heuristics and returns (evidence_strings, raw_scores)."""
    _, ela_score = error_level_analysis(image)
    hf_ratio = high_frequency_ratio(image)

    evidence = []
    # Thresholds below are heuristic starting points from image-forensics
    # literature, not a trained classifier — treat as supporting signals only.
    if ela_score > 0.12:
        evidence.append(
            f"Error-Level-Analysis score is elevated ({ela_score:.3f}); "
            "suggests inconsistent recompression history typical of edited/"
            "composited/generated regions."
        )
    else:
        evidence.append(
            f"Error-Level-Analysis score is low ({ela_score:.3f}); consistent "
            "with a single, unedited compression history."
        )

    if hf_ratio > 0.30:
        evidence.append(
            f"High-frequency spectral ratio is elevated ({hf_ratio:.3f}); "
            "can indicate GAN/diffusion upsampling artifacts."
        )
    else:
        evidence.append(
            f"High-frequency spectral ratio is within the natural-photo range "
            f"({hf_ratio:.3f})."
        )

    return evidence, {"ela_score": ela_score, "high_freq_ratio": hf_ratio}


# --------------------------------------------------------------------------- #
# Turning the two heuristics into one suspicion number (used by fusion and by
# the forensic-only fallback, so both agree).
# --------------------------------------------------------------------------- #
FORENSIC_BASELINE = 0.30   # natural photos measured 0.16-0.23; below this, forensics say nothing


def forensic_suspicion(scores: dict) -> float:
    """Combine ELA + high-frequency ratio into one 0..1 suspicion score."""
    ela = float(scores["ela_score"])
    hf = float(scores["high_freq_ratio"])
    return max(0.0, min(1.0, 0.6 * (ela / 0.25) + 0.4 * (hf / 0.45)))


def forensic_push(suspicion: float) -> float:
    """How strongly forensics argue for FAKE: 0 at/below the natural-photo
    baseline, rising linearly to 1.0 at maximum suspicion. Never negative:
    clean forensics are NOT evidence of 'real' (a good fake passes them)."""
    if suspicion <= FORENSIC_BASELINE:
        return 0.0
    return float(min(1.0, (suspicion - FORENSIC_BASELINE) / (1.0 - FORENSIC_BASELINE)))
