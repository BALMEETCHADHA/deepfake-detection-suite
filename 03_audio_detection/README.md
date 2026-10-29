# Step 3 — Audio Detection (real voice vs AI/cloned voice)

Works on any audio clip (`.wav`, `.mp3`, `.flac`, `.m4a` — anything `librosa`/`soundfile`
can decode): tells you whether it's a genuine human recording or AI-generated/cloned/
synthetic speech, and optionally whether it sounds like the same speaker as a reference clip.

## Files
- `audio_detector.py` — `AudioDeepfakeDetector`: `.predict()` for real-vs-fake, plus
  `.compare_speakers()` for the "is this the same person's voice as my reference clip" check
- `explain_audio.py` — pitch-jitter naturalness + spectral flatness forensic evidence
- `demo.py` — command-line interface

## How it decides + how it proves it
1. ML classifier: a Wav2Vec2-XLSR model fine-tuned specifically for deepfake/spoofed-speech
   detection, published accuracy **92.86%** on the ASVspoof2019 evaluation subset (a standard
   academic benchmark for this exact task).
2. Classical forensic evidence, always computed regardless of ML availability:
   - **Pitch jitter**: natural speech has small frame-to-frame pitch fluctuations; many
     TTS/voice-clone systems sound unnaturally smooth.
   - **Spectral flatness**: vocoder-synthesized audio often has a different noise-vs-tonal
     balance than a microphone recording.
3. `--reference` flag: give it a clip of the real person's voice alongside the clip you're
   checking, and it will also estimate whether they're the same speaker (MFCC-cosine-similarity
   heuristic — clearly labeled as approximate, not a calibrated verification system).

## Run in VS Code
```bash
cd 03_audio_detection
pip install -r requirements.txt
python demo.py --audio clip.wav
python demo.py --audio questioned.wav --reference my_real_voice.wav --save report.json
```

## Run in Google Colab
In Colab, open `../colab_notebooks/1_run_demo_and_backend.ipynb`, run the install and load cells, set `MODALITY = "audio"` in the "Try a detector" cell, then upload audio when prompted.

## Accuracy honesty note
92.86% is the model's *own* published number on its *own* benchmark (ASVspoof2019), not a
guarantee on arbitrary real-world audio you throw at it — voice-cloning technology keeps
improving, and a model trained on one set of TTS engines may not catch every future one.
The report always shows this figure next to the prediction so you present it accurately.
