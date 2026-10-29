# Step 5 — Signature Detection (genuine vs forged/AI-generated)

The odd one out of the five modules, on purpose — read this before you present it to your guide.

## Why this module is different
Signature verification isn't a "is this one image fake" problem the way image/audio/video
are — it's a **comparison** problem: is this signature consistent with a known example of
the real person's handwriting? Forensic document examiners always work this way (reference
vs. questioned), and so does this module.

There also isn't a strong, freely-available pretrained "genuine vs forged signature" model
the way there is for faces (Step 1) or voices (Step 3) — so instead of pointing you at a
model I can't verify, this module ships:
1. A **classical comparator** (`signature_forensics.py`) that works immediately, no training,
   no internet needed, using four standard signature-verification features.
2. A **trainable Siamese CNN** (`siamese_model.py`) you can fine-tune yourself on a labeled
   dataset (recommended: the CEDAR Signature Database, 55 signers × 24 genuine + 24 forged
   each — a standard academic benchmark) to get a real, measured accuracy figure.

## Files
- `signature_forensics.py` — SSIM, ORB keypoint matching, ink-density, and projection-profile
  comparison between a reference and questioned signature
- `siamese_model.py` — PyTorch Siamese embedding network + loader (auto-skips if you haven't
  trained it yet) + a training-skeleton function with instructions
- `signature_detector.py` — `SignatureDetector` with `.compare()` (reference mode, recommended)
  and `.single_image_check()` (weak fallback, no reference)
- `demo.py` — command-line interface

## Run in VS Code
```bash
cd 05_signature_detection
pip install -r requirements.txt

# Recommended: compare against a known-genuine reference
python demo.py --questioned signature_to_check.png --reference known_genuine.png

# Fallback: no reference available (explicitly low-confidence, says so in the report)
python demo.py --questioned signature_to_check.png
```

## Run in Google Colab
In Colab, open `../colab_notebooks/1_run_demo_and_backend.ipynb`, run the install and load cells, set `MODALITY = "signature"` in the "Try a detector" cell, then upload your reference
and questioned signature images when prompted.

## Training the Siamese model for a real accuracy number
1. Download the CEDAR Signature Database (or another labeled genuine/forged signature set).
2. Organize it as `genuine/` and `forged/` folders per the docstring in `siamese_model.py`.
3. Implement the pair-sampling + training loop in `train_on_cedar()` — it's left as a
   documented skeleton, not fabricated, because I haven't run it on real data.
4. Save the resulting weights as `signature_siamese.pt` in this folder — `signature_detector.py`
   will automatically pick it up and blend it into the score on the next run.
5. **Report your actual measured validation accuracy** in your project writeup — that's the
   honest number to cite, not a number pulled from thin air.

## Accuracy honesty note
Until you train it, this module reports a **similarity score**, not a calibrated accuracy
percentage — there's an important difference. The classical comparator is a legitimate,
explainable forensic tool (these are real techniques used in document-forensics literature),
but its "genuine cutoff" hasn't been tuned against a large labeled dataset, so don't quote it
as "X% accurate" in your report. Once trained, the Siamese model gives you that real number.
