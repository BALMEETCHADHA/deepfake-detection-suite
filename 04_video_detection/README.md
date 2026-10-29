# Step 4 — Video Detection

Works on any video (a person talking, a screen recording, anything): tells you real vs
AI-generated/deepfake, by combining three signals into one report.

## Files
- `video_scanner.py` — samples frames evenly across the whole video + extracts the audio track
- `explain_video.py` — frame-to-frame "flicker" temporal-consistency evidence
- `video_detector.py` — `VideoDeepfakeDetector`, reuses Step 1's image detector per frame
  and Step 3's audio detector on the soundtrack
- `demo.py` — command-line interface

## How it decides + how it proves it
1. **Frames**: samples up to N frames (default 12) evenly across the *entire* video duration
   (not just the start), and runs each one through Step 1's exact image detector.
2. **Temporal consistency**: measures how much a frequency-artifact score swings frame-to-frame.
   Your own literature survey specifically names "temporal coherence" flicker/jitter as a
   generation artifact most deepfake pipelines leave behind (they process frames independently) —
   this heuristic directly targets that.
3. **Audio track**: if the video has sound, extracts it (via `ffmpeg`) and runs it through
   Step 3's audio detector — catches AI-dubbed voice over real or fake footage.
4. All three combine into one score; `sub_results` breaks it down per frame and for the audio
   track, so you can see exactly which frame(s) or the audio drove the verdict.

## Requires `ffmpeg` on PATH for audio extraction
- Most Colab runtimes and Linux/Mac systems already have it.
- Windows: install from ffmpeg.org and add to PATH, or `winget install ffmpeg`.
- If `ffmpeg` isn't found, the module doesn't crash — it just skips the audio check and says so.

## Run in VS Code
```bash
cd 04_video_detection
pip install -r requirements.txt
python demo.py --video clip.mp4
python demo.py --video clip.mov --frames 20 --save report.json
```

## Run in Google Colab
In Colab, open `../colab_notebooks/1_run_demo_and_backend.ipynb`, run the install and load cells, set `MODALITY = "video"` in the "Try a detector" cell, then upload a video.
(Colab already has `ffmpeg` preinstalled.)

## Performance note
This calls the image model once per sampled frame — 12 frames ≈ 12x the cost of a single
image check. On CPU that's still fine for a handful of clips; for batch processing many
videos, use `--device cuda` on a GPU runtime (Colab's free GPU works well here).
