# Multi-Modal Deepfake / Forgery Detection Suite

Detects AI-generated or manipulated **images, documents (PDF/DOCX/PPTX), audio, video and signatures**. Each result comes with a verdict
(REAL / FAKE / UNCERTAIN), the two scores, a confidence label and the evidence behind it, through a Python API (FastAPI) and a React web page.
Everything runs on free, open models. Trained and evaluated on Google Colab.

## Results (measured on labelled images, not claimed)

| Image model | Test set | Result |
|---|---|---|
| Pretrained model (Hugging Face) | 200 labelled face images | accuracy about 53%, ROC-AUC 0.58 (near chance on this data) |
| **Fine-tuned on 14,000 images** (notebook 3) | 3,000 held-out images never used for training | **accuracy 99.1%, ROC-AUC 1.000**; full pipeline 98.9% |

The 99.1% is on images from the **same dataset** the model was trained on (real faces vs one kind of AI-generated face). It is not a general
deepfake-detection accuracy: measure images from other sources and other generators separately (notebook 2) before claiming more.
Video was only checked with the pretrained model and was weak; see "Important honesty note on accuracy" below.

## Colab notebooks (three, in `colab_notebooks/`)

| Notebook | What it does |
|---|---|
| [1_run_demo_and_backend](https://colab.research.google.com/github/YOUR_USERNAME/YOUR_REPO/blob/main/colab_notebooks/1_run_demo_and_backend.ipynb) | Try any detector, or start the API with a public ngrok link for the web page |
| [2_evaluate_accuracy](https://colab.research.google.com/github/YOUR_USERNAME/YOUR_REPO/blob/main/colab_notebooks/2_evaluate_accuracy.ipynb) | Measure accuracy on your labelled images, then videos |
| [3_finetune_image_model](https://colab.research.google.com/github/YOUR_USERNAME/YOUR_REPO/blob/main/colab_notebooks/3_finetune_image_model.ipynb) | Fine-tune the image model on your own real/fake images, test on held-out images |

Replace `YOUR_USERNAME/YOUR_REPO` in the links with your GitHub path (see `GIT_SETUP.md`). Use a **T4 GPU** runtime.

## Quick start (local)
```bash
pip install -r requirements.txt
cd 01_image_detection && python demo.py --image photo.jpg           # any single module
cd 07_api_backend && uvicorn main:app --port 8000                  # API for the web page
cd 08_frontend && cp env.example .env && npm install && npm run dev   # web page at http://localhost:5173
pip install -r requirements-dev.txt && pytest tests -q            # 98 tests
cd 08_frontend && npm run check                                    # verdict-colour contract between API and page
```

A step-wise backend for detecting AI-generated / manipulated content across five modalities.
Each modality is a **separate, independently runnable module** so you can build, test, and
hand off pieces to teammates without touching the rest.

## Build plan (delivered step by step, as requested)

| Step | Module | Status |
|---|---|---|
| 1 | `01_image_detection/` — JPEG/PNG images, incl. faces & non-face images | ✅ done |
| 2 | `02_document_detection/` — PDF / DOCX / PPTX (scans embedded images + metadata) | ✅ done |
| 3 | `03_audio_detection/` — real voice vs AI/cloned voice, + speaker comparison | ✅ done |
| 4 | `04_video_detection/` — real vs AI-generated/deepfake video (frames + audio + temporal) | ✅ done |
| 5 | `05_signature_detection/` — genuine vs forged/AI-generated signature | ✅ done |
| 6 | `06_dataset_and_evaluation/` — synthetic test-data generator + batch accuracy evaluator | ✅ done |

**All 5 modules are complete.** See each module's own README for details; the summary below
covers how they fit together.

## Quick reference — how to run each one

```bash
# 1. Images (any jpeg/png)
cd 01_image_detection && python demo.py --image photo.jpg

# 2. Documents (pdf/docx/pptx) — scans every embedded image
cd 02_document_detection && python demo.py --doc report.pdf

# 3. Audio (any voice/audio clip) — plus optional speaker comparison
cd 03_audio_detection && python demo.py --audio clip.wav --reference my_real_voice.wav

# 4. Video (any video) — frames + audio track + temporal-flicker check
cd 04_video_detection && python demo.py --video clip.mp4

# 5. Signatures — reference-vs-questioned comparison (recommended mode)
cd 05_signature_detection && python demo.py --questioned sig.png --reference genuine.png
```

Every module writes the same standardized JSON (`common/schema.py`) so your frontend can
render all five with one code path.

## Dependency notes for teammates
- Steps 2, 4, and 5 **import from Step 1** (and Step 4 also imports Step 3) — always keep
  the whole `deepfake_detection_suite/` folder together; don't split modules into separate
  repos.
- Install each module's own `requirements.txt` (or just install the union of all five up
  front if you're running everything).
- First run of Steps 1/3/4/5's ML paths needs internet access once, to download pretrained
  model weights from Hugging Face (cached locally afterward).

**Step 2 reuses Step 1's exact image detector on every embedded image** in a document —
keep both folders together; `02_document_detection` imports from `01_image_detection`.

Every module returns the **same standardized JSON report** (see `common/schema.py`) so your
frontend only needs to learn one response format for all five modalities:

```json
{
  "file": "photo.jpg",
  "modality": "image",
  "verdict": "FAKE",
  "confidence_fake_pct": 91.4,
  "confidence_real_pct": 8.6,
  "model_used": "prithivMLmods/Deep-Fake-Detector-v2-Model",
  "model_reported_accuracy_pct": 92.1,
  "evidence": [
    "Model confidence for FAKE class: 91.4%",
    "Error-Level-Analysis: elevated recompression noise around face boundary (score 0.71)",
    "High-frequency artifact ratio: 0.38 (above natural-photo threshold 0.25)"
  ],
  "processing_time_sec": 0.84,
  "timestamp": "2026-09-09T12:00:00"
}
```

## How a verdict is produced (images) and how to trust it

```
image -> [face crop] -> ML model -> [calibration] -> [forensic fusion] -> verdict -> confidence label
```
* **Face crop** (`01_image_detection/face_crop.py`): the model was trained on faces, so the largest confident face is cropped
  first. No face found -> the full image is used and a warning is shown. Off: `IMAGE_FACE_CROP=0`.
* **Calibration** (`common/calibration.py`): if `calibration/calibration_image.json` exists, raw scores are corrected so
  "70% fake" means ~70%. Created by `batch_evaluate.py --fit-calibration`. Off: `DEEPFAKE_CALIBRATION=off`.
* **Forensic fusion**: ELA/frequency evidence may *raise* the fake score, only when the model is unsure (lead < 30 pts),
  by at most half the remaining distance to 100%. It never lowers a score and is ignored on clean photos. Off: `IMAGE_FUSION=off`.
* **Verdict**: `UNCERTAIN` only when the two scores are within 6 points (`common/schema.py`). Tune it with the band sweep
  printed by `batch_evaluate.py`.
* **Confidence label** (API): every response has `decision_margin_pct` and `confidence_level` (low < 15, medium < 40, high >= 40).
  A "REAL 53%" is therefore visibly *low* confidence. `scores_breakdown` shows exactly how the number was built.

**None of these make the model more accurate by themselves.** Only measuring on labelled data tells you whether each helps:

```bash
cd 06_dataset_and_evaluation
python batch_evaluate.py --real_dir /path/real --fake_dir /path/fake --compare-variants --fit-calibration
```
(or run `colab_notebooks/2_evaluate_accuracy.ipynb`, which does images and then videos). Reports accuracy, coverage, ROC-AUC, the best cut-off, and
whether face crop / fusion / calibration help on *your* data. Audio, video and signature modules share the verdict rule and the
API labels but have no face crop or calibration yet; the signature module has no trained model, only the classical comparison.

## Fine-tuning on your own data (when the pretrained model doesn't fit)

Measured on 200 labelled face images, the pretrained model scored ROC-AUC 0.58 (near chance), so the project ships a fix:
`colab_notebooks/3_finetune_image_model.ipynb` trains the image model further on your real/fake images (70/15/15 train/val/test
split, duplicates removed, best epoch chosen on validation only) and reports accuracy on the untouched test images. It saves
`image_finetuned.zip`; unzip it into `models/` (or use section 3 of notebook 1) and the detector uses it automatically
(env `IMAGE_MODEL_ID` overrides). The page then shows that model's **held-out test accuracy**, not the model-card claim.
The model is only as general as its training data: expect lower accuracy on other generators or on phone photos.
Large model files are git-ignored (`models/`); keep them in Google Drive.

## Troubleshooting: real photos are shown as DEEP FAKE
1. Look at the menu screen of the web page (or the output of the "Check that the live link works" cell in notebook 1). If it says
   **ORIGINAL PRETRAINED model**, that is the cause: this model is close to a coin flip on this project's data (ROC-AUC about 0.58).
   Fix: notebook 1, section 3, `USE_FINETUNED = True`, then re-run the server and tunnel cells.
2. If it says face detection is off, the message gives the real reason (OpenCV missing, or a broken install). Re-run the install cell and
   restart the Colab session.
3. A fine-tuned model is only reliable on the kind of images it was trained on. For phone photos, measure with notebook 2 and, if needed,
   add phone photos (real) and other generators' faces (fake) to the training folders and re-run notebook 3.
4. `UNCERTAIN_BAND` (env, default 6) controls when two scores count as a coin flip. `0` removes UNCERTAIN answers, but it only relabels
   near-50/50 results; it does not make any answer more accurate.

## Tests
```bash
pip install pytest httpx python-docx python-pptx
pytest tests -q      # 89 tests; the real Hugging Face model is replaced by a stub, so they check the code AROUND the model
```

## Important honesty note on accuracy

You asked for ≥85% accuracy on every modality. Here's the real picture, so you don't submit
a claim you can't defend to your guide:

- **Image detection (Step 1)**: uses a pretrained Vision Transformer fine-tuned specifically
  for real-vs-fake face/image classification. Its published test accuracy is in the low-to-high
  90s on its own benchmark — comparable numbers appear in your own survey papers (XceptionNet:
  0.91 AUC-ROC on FF++; EfficientNet: 0.9974 AUC-ROC on DFDC). That's a legitimate, citable
  number *for that model on that benchmark* — it is **not a guarantee** on totally different,
  real-world images you feed it.
- For modalities where no strong pretrained model exists off-the-shelf (signatures especially),
  the code ships with a **trainable architecture** plus a classical fallback, and you'll need to
  fine-tune on a public dataset (links given in each module's README) to responsibly claim a
  number. I will not print a fake accuracy figure that wasn't actually measured.
- Every report includes the **model's own published/measured accuracy** alongside the
  prediction, so your frontend can show "this call is X% confident, and this detector is Y%
  accurate on its benchmark" — which is the scientifically honest way to present this.

## Running anywhere (VS Code / Colab / other)

Each module has its own `requirements.txt`. From the module folder:

```bash
pip install -r requirements.txt
python demo.py --image path/to/file.jpg
```

The three Colab notebooks (`colab_notebooks/`) cover the same things with `!pip install` cells and file
upload widgets; start with notebook 1.

No API keys or paid services are required — every model used is free and open on Hugging Face.
