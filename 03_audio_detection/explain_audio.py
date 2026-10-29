"""
explain_audio.py
-----------------
Classical, model-independent evidence for audio: pitch (F0) naturalness and
spectral flatness. Same philosophy as Step 1's explain.py — these are
supporting signals shown alongside the ML verdict, not a standalone judge.

Why these two:
- Natural human speech has micro-variation in pitch frame-to-frame ("jitter").
  Many TTS/voice-clone systems produce unnaturally smooth pitch contours.
- Spectral flatness (how "noise-like" vs "tonal" a spectrum is) tends to sit
  in a different range for vocoder-synthesized audio than for a microphone
  recording, because vocoders reconstruct phase/harmonics differently.
"""

from __future__ import annotations
import numpy as np
import librosa


def pitch_jitter_score(y: np.ndarray, sr: int) -> float:
    """Returns average relative frame-to-frame pitch change (0..1-ish).
    Very low values (near-perfectly smooth pitch) are the suspicious end.
    """
    f0, voiced_flag, _ = librosa.pyin(
        y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), sr=sr
    )
    f0 = f0[~np.isnan(f0)]
    if len(f0) < 5:
        return -1.0  # not enough voiced frames to judge
    diffs = np.abs(np.diff(f0)) / (f0[:-1] + 1e-6)
    return float(np.mean(diffs))


def spectral_flatness_score(y: np.ndarray) -> float:
    flatness = librosa.feature.spectral_flatness(y=y)
    return float(np.mean(flatness))


def build_audio_forensic_evidence(y: np.ndarray, sr: int) -> tuple[list[str], dict]:
    evidence = []
    jitter = pitch_jitter_score(y, sr)
    flatness = spectral_flatness_score(y)

    if jitter < 0:
        evidence.append(
            "Not enough clearly voiced (pitched) audio to compute a pitch-jitter score "
            "— this check was skipped."
        )
    elif jitter < 0.004:
        evidence.append(
            f"Pitch contour is unusually smooth (jitter={jitter:.5f}); natural human "
            "speech typically shows more frame-to-frame pitch variation. This can "
            "indicate synthetic/cloned speech."
        )
    else:
        evidence.append(
            f"Pitch contour shows natural jitter ({jitter:.5f}), consistent with "
            "live human speech."
        )

    if flatness > 0.35:
        evidence.append(
            f"Spectral flatness is elevated ({flatness:.4f}) — the spectrum is more "
            "noise-like than typical clean speech, which can appear in some "
            "vocoder-synthesized audio."
        )
    else:
        evidence.append(f"Spectral flatness is in a typical speech range ({flatness:.4f}).")

    return evidence, {"pitch_jitter": jitter, "spectral_flatness": flatness}
