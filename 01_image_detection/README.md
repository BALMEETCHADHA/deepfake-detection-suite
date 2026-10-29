# Step 1 — Image Detection

Detects whether ANY image (a person's photo, a street scene, a screenshot, a graph — anything)
is a real camera photo or AI-generated / manipulated.

## Files
- `image_detector.py` — the detector class (`ImageDeepfakeDetector`), importable as a library
- `explain.py` — classical forensic evidence generators (ELA + frequency analysis) used for "proof"
- `demo.py` — command-line interface
- `requirements.txt` — dependencies for this module only

## How it decides + how it proves it
1. Runs the image through a pretrained Vision Transformer fine-tuned for real-vs-fake
   classification (`prithivMLmods/Deep-Fake-Detector-v2-Model`, ~92% published precision).
2. Independently computes Error-Level-Analysis and a high-frequency spectral ratio —
   classic forensic techniques that don't need any model, and get printed as evidence lines.
3. Combines both into one `DetectionResult` (see `common/schema.py`) with a verdict of
   `REAL`, `FAKE`, or `UNCERTAIN` (used when the two confidences are within 10 points of each other).

## Run in VS Code
```bash
cd 01_image_detection
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
python demo.py --image path/to/photo.jpg
```

## Run in Google Colab
In Colab, open `../colab_notebooks/1_run_demo_and_backend.ipynb`, run the install and load cells, set `MODALITY = "image"` in the "Try a detector" cell, then upload an image when
prompted. No local setup needed — Colab gives you a free GPU which also speeds this up.

## Batch mode (whole folder)
```bash
python demo.py --folder path/to/folder --save results.json
```

## No internet / no GPU available?
The detector automatically drops into a forensic-heuristics-only mode (ELA + frequency
analysis, no ML model) instead of crashing — the report will say so explicitly in its
`warnings` field, and accuracy will be lower than the ML-backed mode. This is meant as a
safety net, not the primary way to run it — for the real ~92% accuracy figure you need
`torch`/`transformers` installed and one-time internet access to download the model weights.

## Pipeline options
Constructor / env: `face_crop` (`IMAGE_FACE_CROP=0` to disable), `fusion` (`IMAGE_FUSION=off`), calibration (`DEEPFAKE_CALIBRATION=off`).
See the main README for what each does, and `06_dataset_and_evaluation` to measure whether they help on your data.
`result.scores_breakdown` records exactly what was applied.
