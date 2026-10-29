"""
video_detector.py
------------------
STEP 4 of 5 — Video detection: real vs AI-generated/deepfake video, for ANY
video (a person talking, a screen recording, anything).

Reuses everything already built instead of duplicating model logic:
  1. Samples frames across the whole video (video_scanner.py) and runs EACH
     one through Step 1's ImageDeepfakeDetector.
  2. Extracts the audio track (if any) and runs it through Step 3's
     AudioDeepfakeDetector — catches AI voice dubbed onto footage.
  3. Computes frame-to-frame "flicker" temporal-consistency evidence
     (explain_video.py) — a classic deepfake tell your own literature
     survey specifically calls out.

All three signals are combined into one DetectionResult, with every
individual frame's (and the audio's) own result kept in `sub_results`.
"""

from __future__ import annotations
import os
import sys
import tempfile
import time
from typing import Optional

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "01_image_detection"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "03_audio_detection"))

from common.schema import DetectionResult, verdict_from_scores  # noqa: E402
from image_detector import ImageDeepfakeDetector  # noqa: E402
from audio_detector import AudioDeepfakeDetector  # noqa: E402
from video_scanner import extract_frames, extract_audio_track  # noqa: E402
from explain_video import temporal_consistency_evidence  # noqa: E402


class VideoDeepfakeDetector:
    def __init__(
        self,
        image_detector: ImageDeepfakeDetector = None,
        audio_detector: AudioDeepfakeDetector = None,
        device: Optional[str] = None,
        max_frames: int = 12,
    ):
        self.image_detector = image_detector or ImageDeepfakeDetector(device=device)
        self.audio_detector = audio_detector or AudioDeepfakeDetector(device=device)
        self.max_frames = max_frames

    def predict(self, video_path: str) -> DetectionResult:
        start = time.time()
        warnings = []
        evidence = []
        sub_results = []

        frames, video_info = extract_frames(video_path, max_frames=self.max_frames)
        evidence.append(f"Video info: {video_info}")

        if not frames:
            raise ValueError(f"Could not extract any frames from {video_path}")

        # ---- 1. Per-frame image analysis (reuses Step 1) ----
        per_frame_fake_scores = []
        for label, pil_img in frames:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp_path = tmp.name
            try:
                pil_img.convert("RGB").save(tmp_path)
                frame_result = self.image_detector.predict(tmp_path)
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

            frame_result.file = f"{os.path.basename(video_path)} :: {label}"
            sub_results.append(frame_result.to_dict())
            per_frame_fake_scores.append(frame_result.confidence_fake_pct)

        mean_frame_fake = sum(per_frame_fake_scores) / len(per_frame_fake_scores)
        worst_idx = max(range(len(per_frame_fake_scores)), key=lambda i: per_frame_fake_scores[i])
        worst_label = frames[worst_idx][0]
        worst_score = per_frame_fake_scores[worst_idx]
        evidence.append(
            f"Analyzed {len(frames)} sampled frames. Mean FAKE confidence: "
            f"{mean_frame_fake:.2f}%. Most suspicious frame: '{worst_label}' "
            f"({worst_score:.2f}% FAKE)."
        )

        # ---- 2. Temporal consistency (flicker) ----
        temporal_evidence, temporal_scores = temporal_consistency_evidence(frames)
        evidence.extend(temporal_evidence)
        flicker_std = temporal_scores.get("flicker_std") or 0.0
        flicker_suspicion = max(0.0, min(1.0, flicker_std / 0.10))  # 0..1

        # ---- 3. Audio track (reuses Step 3) ----
        audio_fake_pct = None
        with tempfile.TemporaryDirectory() as tmp_dir:
            wav_path = os.path.join(tmp_dir, "audio_track.wav")
            extracted = extract_audio_track(video_path, wav_path)
            if extracted is None:
                evidence.append("No usable audio track found (silent video, or ffmpeg unavailable) — audio check skipped.")
            else:
                audio_result = self.audio_detector.predict(extracted)
                audio_result.file = f"{os.path.basename(video_path)} :: audio_track"
                sub_results.append(audio_result.to_dict())
                audio_fake_pct = audio_result.confidence_fake_pct
                evidence.append(
                    f"Audio track analysis — FAKE confidence: {audio_fake_pct:.2f}% "
                    f"(verdict: {audio_result.verdict})."
                )

        # ---- Combine into one document-level score ----
        # Weighting: worst frame matters most (one clearly fake frame is
        # damning), mean frame score for overall consistency, flicker as a
        # smaller nudge, audio folded in when available.
        visual_score = 0.55 * worst_score + 0.30 * mean_frame_fake + 0.15 * (flicker_suspicion * 100.0)
        if audio_fake_pct is not None:
            fake_pct = 0.75 * visual_score + 0.25 * audio_fake_pct
        else:
            fake_pct = visual_score
        fake_pct = max(0.0, min(100.0, fake_pct))
        real_pct = 100.0 - fake_pct

        verdict = verdict_from_scores(fake_pct, real_pct)

        model_used = (
            f"{self.image_detector.model_id} (per sampled frame) + "
            f"{self.audio_detector.model_id} (audio track) + temporal-flicker heuristic"
        )

        return DetectionResult(
            file=os.path.basename(video_path),
            modality="video",
            verdict=verdict,
            confidence_fake_pct=round(fake_pct, 2),
            confidence_real_pct=round(real_pct, 2),
            model_used=model_used,
            model_reported_accuracy_pct=None,  # composite score, not one model's benchmark figure
            evidence=evidence,
            sub_results=sub_results,
            processing_time_sec=round(time.time() - start, 3),
            warnings=warnings,
        )


if __name__ == "__main__":
    print("This module is a library. Run `python demo.py --video <path>` instead.")
