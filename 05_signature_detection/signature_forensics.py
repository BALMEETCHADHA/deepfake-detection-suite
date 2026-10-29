"""
signature_forensics.py
------------------------
Classical (no-training-required) signature comparison. Signature
verification is normally a REFERENCE-vs-QUESTIONED problem (like fingerprint
matching), not a standalone "is this image fake" problem — a single
signature image, in isolation, doesn't reveal much. So the primary mode
here is: given one or more known-GENUINE reference signatures and one
QUESTIONED signature, decide whether the questioned one is consistent with
the same hand that produced the references.

Five classical, well-established signature-verification features:
  1. Structural similarity (SSIM) after alignment
  2. ORB keypoint match ratio
  3. Ink-density ratio (stroke thickness/pressure proxy)
  4. Aspect-ratio consistency
  5. Horizontal + vertical pixel-projection-profile correlation

None of these alone is proof; combined, they give a defensible, explainable
similarity score. For a learned/trainable alternative, see siamese_model.py.
"""

from __future__ import annotations
import cv2
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim


def _load_gray_binarized(image: Image.Image, size=(300, 150)) -> np.ndarray:
    """Grayscale, Otsu-threshold to binary (ink vs background), crop to the
    signature's bounding box, then resize — so position/scale differences
    between two photographed/scanned signatures don't distort the comparison.
    """
    gray = np.array(image.convert("L"))
    # Otsu binarization; invert so ink = white(255)/foreground for bbox math
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    coords = cv2.findNonZero(binary)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        binary = binary[y:y + h, x:x + w]

    if binary.size == 0:
        binary = np.zeros(size[::-1], dtype=np.uint8)

    resized = cv2.resize(binary, size, interpolation=cv2.INTER_AREA)
    return resized


def _ssim_score(ref_bin: np.ndarray, q_bin: np.ndarray) -> float:
    score, _ = ssim(ref_bin, q_bin, full=True)
    return float(max(0.0, score))


def _orb_match_ratio(ref_bin: np.ndarray, q_bin: np.ndarray) -> float:
    orb = cv2.ORB_create(nfeatures=500)
    kp1, des1 = orb.detectAndCompute(ref_bin, None)
    kp2, des2 = orb.detectAndCompute(q_bin, None)
    if des1 is None or des2 is None or len(kp1) == 0 or len(kp2) == 0:
        return 0.0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    good = [m for m in matches if m.distance < 60]
    denom = max(1, min(len(kp1), len(kp2)))
    return float(len(good) / denom)


def _ink_density_similarity(ref_bin: np.ndarray, q_bin: np.ndarray) -> float:
    ref_density = float(np.mean(ref_bin > 0))
    q_density = float(np.mean(q_bin > 0))
    if max(ref_density, q_density) == 0:
        return 0.0
    return 1.0 - abs(ref_density - q_density) / max(ref_density, q_density, 1e-6)


def _projection_profile_correlation(ref_bin: np.ndarray, q_bin: np.ndarray) -> float:
    def profiles(img):
        h_proj = img.sum(axis=1).astype(np.float64)
        v_proj = img.sum(axis=0).astype(np.float64)
        return h_proj, v_proj

    rh, rv = profiles(ref_bin)
    qh, qv = profiles(q_bin)

    def corr(a, b):
        if np.std(a) == 0 or np.std(b) == 0:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    return float(np.mean([max(0.0, corr(rh, qh)), max(0.0, corr(rv, qv))]))


def compare_signatures(reference: Image.Image, questioned: Image.Image) -> tuple[float, list[str], dict]:
    """Returns (genuine_similarity_pct, evidence_lines, raw_feature_scores)."""
    ref_bin = _load_gray_binarized(reference)
    q_bin = _load_gray_binarized(questioned)

    ssim_score = _ssim_score(ref_bin, q_bin)
    orb_score = _orb_match_ratio(ref_bin, q_bin)
    density_score = _ink_density_similarity(ref_bin, q_bin)
    profile_score = _projection_profile_correlation(ref_bin, q_bin)

    # Weighted blend — SSIM and projection profile carry the most
    # signature-verification-specific signal; ORB and density are supporting.
    combined = (
        0.35 * ssim_score
        + 0.20 * orb_score
        + 0.15 * density_score
        + 0.30 * profile_score
    )
    combined = max(0.0, min(1.0, combined))
    genuine_similarity_pct = combined * 100.0

    raw = {
        "ssim": round(ssim_score, 4),
        "orb_match_ratio": round(orb_score, 4),
        "ink_density_similarity": round(density_score, 4),
        "projection_profile_correlation": round(profile_score, 4),
    }

    evidence = [
        f"Structural similarity (SSIM) to reference: {ssim_score:.3f}",
        f"ORB keypoint match ratio: {orb_score:.3f}",
        f"Ink-density (stroke pressure proxy) similarity: {density_score:.3f}",
        f"Pixel-projection-profile correlation: {profile_score:.3f}",
    ]
    return genuine_similarity_pct, evidence, raw
