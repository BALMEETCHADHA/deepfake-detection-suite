# Step 6 — Measuring accuracy (start here)

## Quickest real measurement: your own labelled images
```bash
python batch_evaluate.py --real_dir path/to/real --fake_dir path/to/fake \
       --compare-variants --fit-calibration --limit 300
```
Subfolders are searched. `--limit` takes a reproducible random sample per class. Needs the ML model to load
(torch + transformers + internet); otherwise it prints a warning that the numbers describe only the heuristic fallback and
skips variant comparison and calibration. Outputs `evaluation_results.csv`, `evaluation_report.json`, and (with
`--fit-calibration`) `calibration/calibration_image.json`.

How to read it
* **accuracy (strict)** counts UNCERTAIN as wrong; **acc (decided)** only looks at answered files; read with **coverage**.
* **ROC-AUC**: 0.5 = cannot separate your real from fake; 0.9+ = separates well. If AUC is high but accuracy low, the cut-offs are wrong, not the model.
* **band sweep**: accuracy/coverage for UNCERTAIN bands 0..30. Pick a row you can defend.
* **best cut-off**: far from 50 means the model is biased on your data.
* **variants**: model only / + face crop / + fusion / both. Keep only what helps on your data.
* **calibration holdout**: ECE before/after on files NOT used for fitting. Use hundreds of images per class; with ~20 every number is noise,
  and keep a separate set for any final number you present.

# Synthetic smoke-test dataset (plumbing check only)

## Read this first
`generate_smoke_test_dataset.py` produces **synthetic, procedurally-generated** test files —
not real deepfakes, not real forged signatures. It exists to let you:
- run all 5 modules against hundreds of files with a single command
- catch crashes/bugs at scale before touching real data
- get a working batch-accuracy harness (`batch_evaluate.py`) ready to point at real data later

**Do not quote numbers from this synthetic run as your project's real accuracy.** For that,
swap in a real dataset (see below) — the folder layout is identical, so `batch_evaluate.py`
works on either without changes.

## 1. Generate the synthetic dataset (matches the volumes you asked for)
```bash
cd 06_dataset_and_evaluation
pip install -r requirements.txt

python generate_smoke_test_dataset.py \
  --out ./smoke_test_data \
  --n_images 300 \
  --n_audio 300 \
  --n_signatures 300 \
  --n_videos 50 --video_duration 42 \
  --n_documents 10 --images_per_document 30
```
This takes a few minutes (video generation is the slow part). Add `--skip videos` while
you're iterating on the other modalities to go faster, then generate videos separately once.

Resulting layout:
```
smoke_test_data/
  images/{real,fake}/*.jpg              (300 total)
  audio/{real,fake}/*.wav               (300 total)
  signatures/{genuine,forged}/*.png     (300 total) + signatures/references/*.png (30, one per signer)
  videos/{real,fake}/*.mp4              (50 total, ~42s each)
  documents/*.pdf (10, ~30 images each) + documents/ground_truth_manifest.json
```

## 2. Run the batch evaluator
```bash
python batch_evaluate.py --data ./smoke_test_data
# or just one modality while iterating:
python batch_evaluate.py --data ./smoke_test_data --modalities images
```
Prints per-modality accuracy/precision/recall/F1 against the known synthetic ground truth,
and saves a per-file CSV (`evaluation_results.csv`) you can open in Excel/Sheets.

## 3. Swapping in REAL data for a real accuracy number
Download and arrange real data into the exact same folders:

| Folder | Real dataset to use | Source |
|---|---|---|
| `images/real/`, `images/fake/` | Frames extracted from FaceForensics++ or Celeb-DF real/fake videos | GitHub `ondyari/FaceForensics` (request form) or Kaggle mirror `xdxd003/ff-c23` |
| `audio/real/`, `audio/fake/` | ASVspoof2019 bonafide/spoof clips | `datashare.ed.ac.uk` (free direct download) or Kaggle mirror |
| `signatures/genuine/`, `signatures/forged/`, `signatures/references/` | CEDAR Signature Database | Kaggle, search "CEDAR signature dataset" |
| `videos/real/`, `videos/fake/` | FaceForensics++ or Celeb-DF original videos | same as images row |
| `documents/*.pdf` + `ground_truth_manifest.json` | Build with `generate_documents(...)` from `generate_smoke_test_dataset.py`, pointed at your real image pool instead of the synthetic one — see the function, it just needs `images/real/` and `images/fake/` populated with real files first |

Once real files are in those folders, re-run `batch_evaluate.py` unchanged — the numbers it
prints then ARE real, citable accuracy figures for your report.

## Notes on scale / performance
- 300 images/audio/signatures run quickly even in heuristics-only (no-GPU) mode.
- 50 videos × 42s is the heaviest part — video detection samples multiple frames per video
  and (if ML models are loaded) runs the image model on each, so this step benefits most from
  a GPU (`--device cuda` is wired through the individual detector classes if you want to modify
  `batch_evaluate.py` to pass it through).
- If you only need to sanity-check the pipeline, generate a smaller batch first:
  `--n_images 20 --n_audio 20 --n_signatures 20 --n_videos 4 --n_documents 2`
