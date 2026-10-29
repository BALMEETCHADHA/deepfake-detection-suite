"""
signature_detector.py
-----------------------
STEP 5 of 5 — Signature detection: genuine vs forged/AI-generated signature.

Primary mode (recommended, most reliable): REFERENCE + QUESTIONED comparison.
  You supply one known-genuine signature and the signature you want checked;
  the detector compares them using classical signature-verification features
  (signature_forensics.py), and — if you've trained it (siamese_model.py) —
  a learned embedding-distance score too.

Fallback mode: QUESTIONED only, no reference available.
  Signature authenticity genuinely can't be judged from one image with any
  real confidence (that's true for human forensic examiners too — they
  always compare against known exemplars). In this mode the detector says
  so explicitly and only runs a weak, clearly-labeled auxiliary check
  (Step 1's general image-manipulation detector, reused as-is) to catch
  digital editing / obvious AI-image artifacts around the signature — NOT
  a substitute for reference comparison.
"""

from __future__ import annotations
import os
import sys
import time
from typing import Optional

from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "01_image_detection"))

from common.schema import DetectionResult, verdict_from_scores  # noqa: E402
from signature_forensics import compare_signatures  # noqa: E402
from siamese_model import SiameseSignatureVerifier  # noqa: E402


class SignatureDetector:
    def __init__(self, device: Optional[str] = None, use_general_image_check: bool = True):
        self.siamese = SiameseSignatureVerifier(device=device)
        self._image_detector = None
        self._use_general_image_check = use_general_image_check
        if use_general_image_check:
            try:
                from image_detector import ImageDeepfakeDetector
                self._image_detector = ImageDeepfakeDetector(device=device)
            except Exception:
                self._image_detector = None

    # ------------------------------------------------------------------ #
    def compare(self, reference_path: str, questioned_path: str) -> DetectionResult:
        start = time.time()
        warnings = []
        reference = Image.open(reference_path)
        questioned = Image.open(questioned_path)

        classical_pct, classical_evidence, raw = compare_signatures(reference, questioned)
        evidence = ["Reference-vs-questioned comparison mode (most reliable)."] + classical_evidence

        if self.siamese.available:
            siamese_pct = self.siamese.similarity(reference, questioned)
            genuine_pct = 0.5 * classical_pct + 0.5 * siamese_pct
            evidence.append(
                f"Trained Siamese-network similarity: {siamese_pct:.2f}% "
                f"(blended 50/50 with classical score)."
            )
            model_used = "classical signature-forensics + trained Siamese CNN"
        else:
            genuine_pct = classical_pct
            model_used = "classical signature-forensics only (no trained Siamese weights found)"
            warnings.append(
                "No trained Siamese model weights found (signature_siamese.pt) — using the "
                "classical comparator only. See siamese_model.py / module README to train "
                "one on a labeled dataset (e.g. CEDAR) for a learned accuracy figure."
            )

        forged_pct = 100.0 - genuine_pct
        verdict_raw = verdict_from_scores(forged_pct, genuine_pct)
        verdict = {"FAKE": "FORGED", "REAL": "GENUINE", "UNCERTAIN": "UNCERTAIN"}[verdict_raw]

        return DetectionResult(
            file=f"{os.path.basename(reference_path)} vs {os.path.basename(questioned_path)}",
            modality="signature",
            verdict=verdict,
            confidence_real_pct=round(genuine_pct, 2),
            confidence_fake_pct=round(forged_pct, 2),
            model_used=model_used,
            model_reported_accuracy_pct=None,  # only meaningful once you've trained+measured on a real dataset
            evidence=evidence,
            processing_time_sec=round(time.time() - start, 3),
            warnings=warnings,
        )

    # ------------------------------------------------------------------ #
    def single_image_check(self, questioned_path: str) -> DetectionResult:
        """Fallback when no reference signature is available. Deliberately
        low-confidence — see module docstring."""
        start = time.time()
        warnings = [
            "No reference signature was provided. Signature authenticity cannot be reliably "
            "judged from a single image — this result only checks for generic digital-editing "
            "/ AI-image artifacts, and should be treated as a weak signal, not a verdict. "
            "For a real answer, provide a known-genuine reference with --reference."
        ]
        evidence = []

        if self._image_detector is not None:
            img_result = self._image_detector.predict(questioned_path)
            fake_pct = img_result.confidence_fake_pct
            real_pct = img_result.confidence_real_pct
            evidence.append(
                "Ran general image-manipulation detector (not signature-specific) as an "
                f"auxiliary signal: FAKE {fake_pct:.2f}%, REAL {real_pct:.2f}%."
            )
            model_used = f"{self._image_detector.model_id} (general-purpose, NOT signature-trained)"
        else:
            fake_pct, real_pct = 0.0, 0.0
            evidence.append("General image-manipulation model unavailable — no auxiliary signal computed.")
            model_used = "none available"

        verdict_raw = verdict_from_scores(fake_pct, real_pct, uncertain_band=100.0)  # force UNCERTAIN
        verdict = "UNCERTAIN"

        return DetectionResult(
            file=os.path.basename(questioned_path),
            modality="signature",
            verdict=verdict,
            confidence_real_pct=round(real_pct, 2),
            confidence_fake_pct=round(fake_pct, 2),
            model_used=model_used,
            model_reported_accuracy_pct=None,
            evidence=evidence,
            processing_time_sec=round(time.time() - start, 3),
            warnings=warnings,
        )


if __name__ == "__main__":
    print("This module is a library. Run `python demo.py --questioned <path> --reference <path>` instead.")
