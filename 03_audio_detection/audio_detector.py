"""
audio_detector.py
------------------
STEP 3 of 5 — Audio detection: real human voice vs AI-generated/cloned voice,
for ANY audio clip (voice, or general audio).

Two layers, same pattern as Steps 1-2:
  1. ML classifier: Wav2Vec2-XLSR fine-tuned specifically for deepfake/spoof
     audio detection (Gustking/wav2vec2-large-xlsr-deepfake-audio-classification),
     with published accuracy 92.86% on the ASVspoof2019 evaluation subset.
  2. Classical forensic evidence (explain_audio.py): pitch-jitter naturalness
     + spectral flatness — works even with no internet/ML model.

Bonus feature (as requested): `compare_speakers()` — given a REFERENCE clip of
someone's real voice and a QUESTIONED clip, estimates whether they sound like
the same speaker using an MFCC-embedding cosine similarity. This is a
lightweight heuristic, NOT a production speaker-verification system (that
would need something like SpeechBrain's ECAPA-TDNN) — the report says so
explicitly so you don't oversell it.
"""

from __future__ import annotations
import os
import sys
import time
from typing import Optional

import numpy as np
import librosa

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from common.schema import DetectionResult, verdict_from_scores  # noqa: E402
from explain_audio import build_audio_forensic_evidence  # noqa: E402

DEFAULT_MODEL_ID = "Gustking/wav2vec2-large-xlsr-deepfake-audio-classification"
DEFAULT_MODEL_REPORTED_ACCURACY = 92.86  # published accuracy on ASVspoof2019 eval subset
TARGET_SR = 16000


class AudioDeepfakeDetector:
    def __init__(self, model_id: str = DEFAULT_MODEL_ID, device: Optional[str] = None):
        self.model_id = model_id
        self.device = device
        self._pipe = None
        self._ml_available = self._try_load_model()

    def _try_load_model(self) -> bool:
        try:
            import torch  # noqa: F401
            from transformers import pipeline

            device_arg = 0 if (self.device == "cuda") else -1
            self._pipe = pipeline(
                "audio-classification",
                model=self.model_id,
                device=device_arg,
            )
            return True
        except Exception as exc:  # pragma: no cover
            print(
                f"[audio_detector] Could not load ML model '{self.model_id}' "
                f"({exc.__class__.__name__}: {exc}). Falling back to "
                "forensic-heuristics-only mode."
            )
            return False

    def _ml_scores(self, path: str) -> Optional[dict]:
        if not self._ml_available:
            return None
        outputs = self._pipe(path)
        fake_pct, real_pct = 0.0, 0.0
        for o in outputs:
            label = o["label"].lower()
            pct = float(o["score"]) * 100.0
            if "fake" in label or "spoof" in label or "synthetic" in label:
                fake_pct += pct
            else:
                real_pct += pct
        total = fake_pct + real_pct
        if total > 0:
            fake_pct = fake_pct / total * 100.0
            real_pct = real_pct / total * 100.0
        return {"fake_pct": fake_pct, "real_pct": real_pct, "raw": outputs}

    # ------------------------------------------------------------------ #
    def predict(self, audio_path: str) -> DetectionResult:
        start = time.time()
        warnings = []

        y, sr = librosa.load(audio_path, sr=TARGET_SR, mono=True)
        if len(y) < sr * 0.5:
            warnings.append("Audio clip is very short (<0.5s) — results may be unreliable.")

        forensic_evidence, forensic_scores = build_audio_forensic_evidence(y, sr)
        ml_result = self._ml_scores(audio_path)

        if ml_result is not None:
            fake_pct = ml_result["fake_pct"]
            real_pct = ml_result["real_pct"]
            model_used = self.model_id
            model_acc = DEFAULT_MODEL_REPORTED_ACCURACY
            evidence = [
                f"ML classifier confidence — FAKE: {fake_pct:.2f}%, REAL: {real_pct:.2f}%"
            ] + forensic_evidence
        else:
            warnings.append(
                "ML model unavailable in this environment — verdict is based on classical "
                "forensic heuristics only (lower reliability). Install `torch` + "
                "`transformers` with internet access for the full ML-backed verdict."
            )
            jitter = forensic_scores["pitch_jitter"]
            flatness = forensic_scores["spectral_flatness"]
            jitter_suspicion = 0.0 if jitter < 0 else max(0.0, min(1.0, (0.006 - jitter) / 0.006))
            flatness_suspicion = max(0.0, min(1.0, (flatness - 0.2) / 0.3))
            suspicion = 0.6 * jitter_suspicion + 0.4 * flatness_suspicion
            fake_pct = suspicion * 100.0
            real_pct = 100.0 - fake_pct
            model_used = "forensic-heuristics-only (no ML model loaded)"
            model_acc = None
            evidence = forensic_evidence

        verdict = verdict_from_scores(fake_pct, real_pct)

        return DetectionResult(
            file=os.path.basename(audio_path),
            modality="audio",
            verdict=verdict,
            confidence_fake_pct=round(fake_pct, 2),
            confidence_real_pct=round(real_pct, 2),
            model_used=model_used,
            model_reported_accuracy_pct=model_acc,
            evidence=evidence,
            processing_time_sec=round(time.time() - start, 3),
            warnings=warnings,
        )

    # ------------------------------------------------------------------ #
    def compare_speakers(self, reference_path: str, questioned_path: str) -> dict:
        """Lightweight MFCC-embedding cosine-similarity speaker comparison.

        Returns a dict (not a DetectionResult, since this answers a different
        question — "same speaker?" not "AI-generated?"). Combine with
        `predict(questioned_path)` for the full picture: is it a real human
        voice at all, AND does it sound like the reference speaker.
        """
        def embed(path: str) -> np.ndarray:
            y, sr = librosa.load(path, sr=TARGET_SR, mono=True)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
            return np.mean(mfcc, axis=1)

        ref_emb = embed(reference_path)
        q_emb = embed(questioned_path)
        cos_sim = float(
            np.dot(ref_emb, q_emb) / (np.linalg.norm(ref_emb) * np.linalg.norm(q_emb) + 1e-9)
        )
        # Rough banding — heuristic only, not a calibrated verification system.
        if cos_sim > 0.95:
            verdict = "LIKELY SAME SPEAKER"
        elif cos_sim > 0.85:
            verdict = "UNCERTAIN"
        else:
            verdict = "LIKELY DIFFERENT SPEAKER"

        return {
            "reference_file": os.path.basename(reference_path),
            "questioned_file": os.path.basename(questioned_path),
            "mfcc_cosine_similarity": round(cos_sim, 4),
            "verdict": verdict,
            "note": (
                "Heuristic MFCC-similarity comparison, not a calibrated speaker-"
                "verification system. For production-grade speaker verification, "
                "use a dedicated embedding model (e.g. SpeechBrain ECAPA-TDNN)."
            ),
        }


if __name__ == "__main__":
    print("This module is a library. Run `python demo.py --audio <path>` instead.")
