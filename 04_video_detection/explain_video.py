"""
explain_video.py
-----------------
Video-specific forensic evidence: temporal (frame-to-frame) consistency.

Your own literature survey (DeepFake Detection for Human Face Images and
Videos: A Survey) flags "temporal coherence" — flickering/jittering between
frames — as a known deepfake generation artifact, because most generators
process frames independently without enforcing consistency across time.
This module turns that into a measurable, reusable signal by reusing Step 1's
frequency-artifact heuristic on every sampled frame and looking at how much
it swings from frame to frame.
"""

from __future__ import annotations
import os
import sys
import numpy as np
from typing import List, Tuple
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "01_image_detection"))
from explain import high_frequency_ratio  # noqa: E402


def temporal_consistency_evidence(frames: List[Tuple[str, Image.Image]]) -> Tuple[List[str], dict]:
    if len(frames) < 3:
        return (
            ["Too few sampled frames to assess temporal consistency (need >= 3)."],
            {"flicker_std": None},
        )

    hf_ratios = [high_frequency_ratio(img) for _, img in frames]
    flicker_std = float(np.std(hf_ratios))
    flicker_mean = float(np.mean(hf_ratios))

    evidence = []
    # Threshold is a heuristic starting point, not a trained/calibrated value.
    if flicker_std > 0.06:
        evidence.append(
            f"High-frequency artifact ratio swings noticeably between sampled frames "
            f"(std={flicker_std:.4f}, mean={flicker_mean:.4f}) — consistent with the "
            "'temporal coherence' flickering artifact commonly reported in deepfake "
            "generation literature."
        )
    else:
        evidence.append(
            f"High-frequency artifact ratio is stable across sampled frames "
            f"(std={flicker_std:.4f}) — consistent with a naturally, continuously "
            "filmed video."
        )

    return evidence, {"flicker_std": flicker_std, "flicker_mean": flicker_mean}
