"""
video_scanner.py
-----------------
Low-level extraction for video: sampled frames (as PIL Images) + the audio
track (as a .wav file), so the video detector can reuse Step 1's image
detector on frames and Step 3's audio detector on the soundtrack, instead
of duplicating any model logic.
"""

from __future__ import annotations
import os
import shutil
import subprocess
from typing import List, Optional, Tuple

import cv2
from PIL import Image


def extract_frames(
    video_path: str, max_frames: int = 12
) -> Tuple[List[Tuple[str, Image.Image]], dict]:
    """Uniformly samples up to `max_frames` frames across the whole video
    (not just the first few seconds) so a fake segment anywhere gets seen.

    Returns (frames, video_info) where frames = [(label, PIL.Image), ...].
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    duration_sec = (total_frames / fps) if fps > 0 else 0.0

    if total_frames <= 0:
        cap.release()
        raise ValueError(f"Video appears to have no frames: {video_path}")

    n = min(max_frames, total_frames)
    # evenly spaced frame indices across the whole video
    indices = sorted(set(int(i * (total_frames - 1) / max(1, n - 1)) for i in range(n)))

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame_bgr = cap.read()
        if not ok:
            continue
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        timestamp = (idx / fps) if fps > 0 else 0.0
        label = f"frame_{idx:06d}_t{timestamp:.2f}s.png"
        frames.append((label, pil_img))

    cap.release()

    video_info = {
        "total_frames": total_frames,
        "fps": round(fps, 2),
        "duration_sec": round(duration_sec, 2),
        "frames_sampled": len(frames),
    }
    return frames, video_info


def extract_audio_track(video_path: str, out_wav_path: str) -> Optional[str]:
    """Extracts the audio track to a 16kHz mono wav using ffmpeg.
    Returns the output path, or None if there's no audio track / ffmpeg
    isn't installed (caller should treat that as 'skip audio check', not
    an error — plenty of valid videos are silent or muted).
    """
    if shutil.which("ffmpeg") is None:
        return None

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", video_path,
        "-vn", "-ac", "1", "-ar", "16000",
        out_wav_path,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError:
        return None

    if not os.path.exists(out_wav_path) or os.path.getsize(out_wav_path) < 1000:
        # ffmpeg "succeeded" but produced ~empty file => no audio stream
        if os.path.exists(out_wav_path):
            os.remove(out_wav_path)
        return None
    return out_wav_path
