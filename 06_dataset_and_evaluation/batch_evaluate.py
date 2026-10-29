"""
batch_evaluate.py
-------------------
Measure how accurate the detectors REALLY are on files whose true label you know.

Two ways to point it at data
  A) Your own image folders (the quickest way to get a real number):
        python batch_evaluate.py --real_dir path/to/real --fake_dir path/to/fake
        python batch_evaluate.py --real_dir R --fake_dir F --compare_variants --fit_calibration
  B) The standard layout (also what generate_smoke_test_dataset.py writes):
        <data>/images/{real,fake}/  audio/{real,fake}/  signatures/{genuine,forged}/ + references/
        videos/{real,fake}/  documents/*.pdf + ground_truth_manifest.json
        python batch_evaluate.py --data ./smoke_test_data --modalities images audio

What it reports (per modality)
  * accuracy (strict: UNCERTAIN counts as a miss), accuracy on decided files, coverage
  * precision / recall / F1, ROC-AUC (threshold-free: does the model rank fakes above reals?)
  * band_sweep: accuracy vs coverage for different UNCERTAIN bands (tune the "6")
  * best_threshold: the fake-% cut-off that best separates YOUR data (Youden's J)
  * --compare_variants (images): model alone vs +face crop vs +forensic fusion vs both
  * --fit_calibration: fits Platt scaling on raw model scores, checks it on held-out files,
    writes calibration/calibration_<modality>.json which the detectors then apply automatically.

SYNTHETIC data only tests the plumbing; quote accuracy only from real labelled data.
"""

from __future__ import annotations
import argparse
import contextlib
import glob
import json
import os
import re
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
for _sub in ["..", "../01_image_detection", "../02_document_detection", "../03_audio_detection",
             "../04_video_detection", "../05_signature_detection"]:
    _p = os.path.abspath(os.path.join(_HERE, _sub))
    if _p not in sys.path:
        sys.path.append(_p)

from common import calibration as cal  # noqa: E402

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
MODALITY_KEY = {"images": "image", "audio": "audio", "videos": "video",
                "signatures": "signature", "documents": "document"}


@contextlib.contextmanager
def _env(**kv):
    """Temporarily set env vars (and drop the calibration cache so it re-reads them)."""
    old = {k: os.environ.get(k) for k in kv}
    os.environ.update({k: str(v) for k, v in kv.items()})
    cal.clear_cache()
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        cal.clear_cache()


def _decide(verdict: str):
    """FAKE/FORGED -> True, REAL/GENUINE -> False, anything else (UNCERTAIN) -> None."""
    if verdict in ("FAKE", "FORGED"):
        return True
    if verdict in ("REAL", "GENUINE"):
        return False
    return None


def _metrics(y_true, y_pred) -> dict:
    """y_true: bool (True = fake/forged). y_pred: True / False / None (None = UNCERTAIN).

    Strict accuracy counts UNCERTAIN as wrong, so a model cannot look good by refusing
    to answer. accuracy_decided_pct only looks at files it committed to; read it together
    with coverage_pct.
    """
    n = len(y_true)
    pairs = list(zip(y_true, y_pred))
    tp = sum(1 for t, p in pairs if t and p is True)
    tn = sum(1 for t, p in pairs if (not t) and p is False)
    fp = sum(1 for t, p in pairs if (not t) and p is True)
    fn = sum(1 for t, p in pairs if t and p is False)
    unc = sum(1 for _, p in pairs if p is None)
    decided = n - unc
    pct = lambda a, b: (a / b * 100.0) if b else 0.0  # noqa: E731
    precision, recall = pct(tp, tp + fp), pct(tp, tp + fn)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "n": n, "tp": tp, "tn": tn, "fp": fp, "fn": fn, "uncertain": unc,
        "coverage_pct": round(pct(decided, n), 2),
        "accuracy_pct": round(pct(tp + tn, n), 2),
        "accuracy_decided_pct": round(pct(tp + tn, decided), 2),
        "precision_pct": round(precision, 2), "recall_pct": round(recall, 2), "f1_pct": round(f1, 2),
    }


def _with_auc(metrics: dict, truth, fake_pcts) -> dict:
    auc = cal.roc_auc(fake_pcts, truth)
    metrics["roc_auc"] = None if auc is None else round(auc, 4)
    return metrics


def band_sweep(truth, fake_pcts, bands=(0, 2, 4, 6, 8, 10, 15, 20, 30)) -> list:
    """For each UNCERTAIN band (lead |fake-real| <= band -> UNCERTAIN) report coverage/accuracy."""
    out = []
    for b in bands:
        preds = []
        for f in fake_pcts:
            lead = abs(2.0 * f - 100.0)
            preds.append(None if lead <= b else (f > 50.0))
        m = _metrics(truth, preds)
        out.append({"band": b, "coverage_pct": m["coverage_pct"], "accuracy_pct": m["accuracy_pct"],
                    "accuracy_decided_pct": m["accuracy_decided_pct"], "uncertain": m["uncertain"]})
    return out


def best_threshold(truth, fake_pcts) -> dict:
    """Cut-off on fake-% (predict FAKE if score >= cut) maximising Youden's J = TPR - FPR.
    If it differs a lot from 50, the model is biased on your data."""
    truth = [bool(t) for t in truth]
    pos, neg = sum(truth), len(truth) - sum(truth)
    if not pos or not neg:
        return {"threshold_fake_pct": None, "youden_j": None, "tpr_pct": None, "fpr_pct": None}
    best = None
    for t in sorted(set(float(f) for f in fake_pcts)):
        tpr = sum(1 for y, f in zip(truth, fake_pcts) if y and f >= t) / pos
        fpr = sum(1 for y, f in zip(truth, fake_pcts) if (not y) and f >= t) / neg
        j = tpr - fpr
        key = (j, -abs(t - 50.0))
        if best is None or key > best[0]:
            best = (key, t, tpr, fpr, j)
    _, t, tpr, fpr, j = best
    return {"threshold_fake_pct": t, "youden_j": round(j, 4),
            "tpr_pct": round(tpr * 100, 2), "fpr_pct": round(fpr * 100, 2)}


def _list_images(d: str, limit: int = None, seed: int = 0) -> list:
    """Images in d (subfolders included). limit = random sample of that size (reproducible)."""
    if not os.path.isdir(d):
        raise FileNotFoundError(f"folder not found: {d}")
    files = sorted(os.path.join(root, f) for root, _, fs in os.walk(d) for f in fs
                   if f.lower().endswith(IMAGE_EXTS))
    if not files:
        raise FileNotFoundError(f"no images ({', '.join(IMAGE_EXTS)}) in {d}")
    if limit and len(files) > limit:
        import random
        files = sorted(random.Random(seed).sample(files, limit))
    return files


VARIANTS = [  # (name, face_crop, fusion)
    ("model only (full image)", False, False),
    ("model + face crop", True, False),
    ("model + forensic fusion", False, True),
    ("model + face crop + fusion", True, True),
]


def eval_images(data_dir: str, real_dir: str = None, fake_dir: str = None,
                detector=None, compare_variants: bool = False, limit: int = None) -> dict:
    """Evaluate the image detector. Result keys: metrics, rows, is_fake, fake_pcts,
    raw_ml_fake_pcts (None when no ML model is loaded), and for compare_variants:
    variants, faces_found, n_images."""
    real_dir = real_dir or os.path.join(data_dir, "images", "real")
    fake_dir = fake_dir or os.path.join(data_dir, "images", "fake")
    files = [(p, False) for p in _list_images(real_dir, limit)] + [(p, True) for p in _list_images(fake_dir, limit)]
    truth = [t for _, t in files]

    if detector is None:
        from image_detector import ImageDeepfakeDetector
        detector = ImageDeepfakeDetector()
    ml = bool(detector._ml_available)
    saved = (detector.face_crop, detector.fusion)
    skipped = 0

    def run(face_crop, fusion):
        nonlocal skipped
        detector.face_crop, detector.fusion = face_crop, fusion
        res = []
        skipped = 0
        for p, _ in files:
            try:
                res.append(detector.predict(p))
            except Exception as exc:  # unreadable file: record, keep going
                skipped += 1
                print(f"  ! skipped {os.path.basename(p)}: {exc.__class__.__name__}: {exc}")
                res.append(None)
        return res

    try:
        variants, faces_found = None, None
        if compare_variants and ml:
            variants, results_by_variant = {}, {}
            for name, fc, fu in VARIANTS:
                print(f"  variant: {name}")
                results_by_variant[name] = run(fc, fu)
            for name, rs in results_by_variant.items():
                ok = [(t, r) for t, r in zip(truth, rs) if r is not None]
                m = _metrics([t for t, _ in ok], [_decide(r.verdict) for _, r in ok])
                variants[name] = _with_auc(m, [t for t, _ in ok], [r.confidence_fake_pct for _, r in ok])
            crop_rs = results_by_variant["model + face crop"]
            faces_found = sum(1 for r in crop_rs if r is not None and r.scores_breakdown.get("face_cropped"))
            current = next(n for n, fc, fu in VARIANTS if (fc, fu) == (saved[0], saved[1]))
            results = results_by_variant[current]
        else:
            if compare_variants and not ml:
                print("  (variant comparison skipped: no ML model loaded, only forensic heuristics)")
            results = run(*saved)
    finally:
        detector.face_crop, detector.fusion = saved

    keep = [i for i, r in enumerate(results) if r is not None]
    truth_k = [truth[i] for i in keep]
    rs = [results[i] for i in keep]
    fake_pcts = [r.confidence_fake_pct for r in rs]
    metrics = _with_auc(_metrics(truth_k, [_decide(r.verdict) for r in rs]), truth_k, fake_pcts)
    metrics["skipped_unreadable"] = skipped
    raw = [r.scores_breakdown.get("ml_fake_pct_raw") for r in rs] if ml else None
    if raw is not None and any(v is None for v in raw):
        raw = None
    out = {
        "metrics": metrics,
        "rows": [(os.path.basename(files[i][0]), "fake" if truth[i] else "real",
                  results[i].verdict, results[i].confidence_fake_pct) for i in keep],
        "is_fake": truth_k, "fake_pcts": fake_pcts, "raw_ml_fake_pcts": raw,
        "ml_model_loaded": ml, "n_images": len(files),
    }
    if variants is not None:
        out["variants"], out["faces_found"] = variants, faces_found
    return out


def _generic_eval(items, predict):
    """items: [(path, is_fake, true_label_str)]; predict(path) -> DetectionResult."""
    truth, preds, rows, scores = [], [], [], []
    for path, is_fake, label in items:
        r = predict(path)
        truth.append(is_fake)
        preds.append(_decide(r.verdict))
        scores.append(r.confidence_fake_pct)
        rows.append((os.path.basename(path), label, r.verdict, r.confidence_fake_pct))
    m = _with_auc(_metrics(truth, preds), truth, scores)
    raw = None
    return {"metrics": m, "rows": rows, "is_fake": truth, "fake_pcts": scores, "raw_ml_fake_pcts": raw}


def _folder_items(data_dir, sub, neg, pos, pattern="*"):
    items = []
    for label, is_fake in [(neg, False), (pos, True)]:
        for p in sorted(glob.glob(os.path.join(data_dir, sub, label, pattern))):
            items.append((p, is_fake, label))
    if not items:
        raise FileNotFoundError(os.path.join(data_dir, sub))
    return items


def eval_audio(data_dir: str, **_) -> dict:
    from audio_detector import AudioDeepfakeDetector
    det = AudioDeepfakeDetector()
    return _generic_eval(_folder_items(data_dir, "audio", "real", "fake", "*.wav"), det.predict)


def eval_videos(data_dir: str, max_frames: int = 6, **_) -> dict:
    from video_detector import VideoDeepfakeDetector
    det = VideoDeepfakeDetector(max_frames=max_frames)
    return _generic_eval(_folder_items(data_dir, "videos", "real", "fake", "*.mp4"), det.predict)


def eval_signatures(data_dir: str, **_) -> dict:
    from signature_detector import SignatureDetector
    det = SignatureDetector(use_general_image_check=False)
    sid = lambda f: (re.match(r"(signer\d+)_", os.path.basename(f)) or [None, None])[1]  # noqa: E731
    refs = {sid(p): p for p in glob.glob(os.path.join(data_dir, "signatures", "references", "*.png")) if sid(p)}
    items = [it for it in _folder_items(data_dir, "signatures", "genuine", "forged", "*.png") if sid(it[0]) in refs]
    return _generic_eval(items, lambda p: det.compare(refs[sid(p)], p))


def eval_documents(data_dir: str, **_) -> dict:
    from doc_detector import DocumentDeepfakeDetector
    det = DocumentDeepfakeDetector()
    with open(os.path.join(data_dir, "documents", "ground_truth_manifest.json")) as f:
        manifest = json.load(f)
    items = [(os.path.join(data_dir, "documents", e["document"]), e["expected_doc_verdict"] == "FAKE",
              e["expected_doc_verdict"]) for e in manifest]
    return _generic_eval(items, det.predict)


EVALUATORS = {"images": eval_images, "audio": eval_audio, "signatures": eval_signatures,
              "videos": eval_videos, "documents": eval_documents}


# --------------------------------------------------------------------------- #
# Calibration fitting with an honest held-out check
# --------------------------------------------------------------------------- #
def fit_and_save_calibration(modality: str, out: dict, cal_dir: str, holdout_frac: float = 0.3, seed: int = 0) -> dict:
    """Fit Platt scaling on RAW model scores. Never runs on heuristic fallback scores
    (raw_ml_fake_pcts is None then) because calibrating a heuristic would be meaningless.
    The reported holdout numbers come from a model fitted WITHOUT those files."""
    raw, truth = out.get("raw_ml_fake_pcts"), out.get("is_fake")
    if not raw or not truth:
        return {"skipped": "no raw ML scores (model not loaded or modality has no calibratable ML score)"}
    import numpy as np
    raw, truth = np.asarray(raw, float), np.asarray(truth, bool)
    if truth.all() or (~truth).all() or len(raw) < 8:
        return {"skipped": "need at least 8 files with both real and fake labels"}

    rng = np.random.default_rng(seed)
    hold = np.zeros(len(raw), bool)
    for cls in (True, False):                       # stratified split
        idx = np.where(truth == cls)[0]
        rng.shuffle(idx)
        hold[idx[: int(round(len(idx) * holdout_frac))]] = True
    holdout = "skipped: too few files for a held-out check (need >= 10 held out with both classes)"
    if hold.sum() >= 10 and truth[hold].any() and (~truth[hold]).any() and (~hold).sum() >= 4:
        p = cal.fit_platt(raw[~hold], truth[~hold])
        after = [cal.calibrate_pct(s, p["a"], p["b"]) for s in raw[hold]]
        holdout = {
            "n_holdout": int(hold.sum()),
            "ece_before_pts": round(cal.expected_calibration_error(raw[hold], truth[hold]), 2),
            "ece_after_pts": round(cal.expected_calibration_error(after, truth[hold]), 2),
            "auc_holdout": cal.roc_auc(raw[hold], truth[hold]),
        }
    params = cal.fit_platt(raw, truth)             # final parameters use every file
    path = os.path.join(cal_dir, f"calibration_{MODALITY_KEY.get(modality, modality)}.json")
    cal.save_calibration(path, params, meta={"modality": modality, "n": int(len(raw)),
                                              "holdout": holdout, "fitted_on": time.strftime("%Y-%m-%d")})
    return {"file": path, "params": params, "holdout": holdout}


# --------------------------------------------------------------------------- #
def _json_safe(o):
    if isinstance(o, dict):
        return {str(k): _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if hasattr(o, "item"):
        return o.item()
    return o


def _print_metrics(m):
    print(f"  n={m['n']}  accuracy={m['accuracy_pct']}% (strict)  decided-only={m['accuracy_decided_pct']}%  "
          f"coverage={m['coverage_pct']}%  uncertain={m['uncertain']}")
    print(f"  precision={m['precision_pct']}%  recall={m['recall_pct']}%  f1={m['f1_pct']}%  "
          f"ROC-AUC={m.get('roc_auc')}   [tp={m['tp']} tn={m['tn']} fp={m['fp']} fn={m['fn']}]")


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description="Measure detector accuracy on labelled data")
    ap.add_argument("--data", type=str, help="Dataset folder (standard layout, see module docstring)")
    ap.add_argument("--real_dir", type=str, help="Folder of REAL images (use with --fake_dir)")
    ap.add_argument("--fake_dir", type=str, help="Folder of FAKE images (use with --real_dir)")
    ap.add_argument("--modalities", nargs="*", choices=list(EVALUATORS))
    ap.add_argument("--compare_variants", "--compare-variants", dest="compare_variants", action="store_true",
                    help="images: compare face-crop / fusion options")
    ap.add_argument("--fit_calibration", "--fit-calibration", dest="fit_calibration", action="store_true",
                    help="fit + save Platt calibration (images)")
    ap.add_argument("--limit", type=int, default=None, help="images: max files per class (random, reproducible sample)")
    ap.add_argument("--calibration_dir", type=str, default=os.path.join(_HERE, "..", "calibration"))
    ap.add_argument("--save_csv", type=str, default="evaluation_results.csv")
    ap.add_argument("--save_json", type=str, default="evaluation_report.json")
    args = ap.parse_args(argv)

    if bool(args.real_dir) != bool(args.fake_dir):
        ap.error("--real_dir and --fake_dir must be given together")
    if not (args.data or args.real_dir):
        ap.error("give --data, or --real_dir and --fake_dir")
    modalities = args.modalities or (["images"] if args.real_dir and not args.data else list(EVALUATORS))

    report, all_rows = {}, []
    for modality in modalities:
        print(f"\n=== Evaluating: {modality} ===")
        t0 = time.time()
        kw = {}
        if modality == "images":
            kw = dict(real_dir=args.real_dir, fake_dir=args.fake_dir, compare_variants=args.compare_variants, limit=args.limit)
        try:
            out = EVALUATORS[modality](args.data or "", **kw)
        except FileNotFoundError as e:
            print(f"  Skipped ({e})")
            continue
        m = out["metrics"]
        if out.get("ml_model_loaded") is False:
            print("  !! WARNING: the ML model did NOT load (no torch/transformers or no internet).\n"
                  "  !! These numbers describe the classical-heuristics fallback, NOT the deep-learning model.\n"
                  "  !! Do not quote them as the model's accuracy; run this in Colab / with internet.")
        _print_metrics(m)
        entry = {"metrics": m,
                 "band_sweep": band_sweep(out["is_fake"], out["fake_pcts"]),
                 "best_threshold": best_threshold(out["is_fake"], out["fake_pcts"]),
                 "seconds": round(time.time() - t0, 1)}
        bt = entry["best_threshold"]
        if bt["threshold_fake_pct"] is not None:
            print(f"  best cut-off on your data: FAKE if fake% >= {bt['threshold_fake_pct']} "
                  f"(Youden J={bt['youden_j']}, TPR={bt['tpr_pct']}%, FPR={bt['fpr_pct']}%)  [default is 50]")
        print("  UNCERTAIN-band sweep (wider band = fewer answers, but the answers given are more often right):")
        print("    band  coverage  acc(decided)  acc(strict)")
        for r in entry["band_sweep"]:
            print(f"    {r['band']:>4}  {r['coverage_pct']:7.1f}%  {r['accuracy_decided_pct']:11.1f}%  {r['accuracy_pct']:10.1f}%")
        if out.get("variants"):
            entry["variants"], entry["faces_found"] = out["variants"], out["faces_found"]
            print(f"  faces found in {out['faces_found']}/{out['n_images']} images")
            for name, vm in out["variants"].items():
                print(f"    {name:32s} acc={vm['accuracy_pct']:6.2f}%  decided={vm['accuracy_decided_pct']:6.2f}%  "
                      f"cov={vm['coverage_pct']:6.2f}%  AUC={vm['roc_auc']}")
        if args.fit_calibration:
            entry["calibration"] = fit_and_save_calibration(modality, out, args.calibration_dir)
            print(f"  calibration: {entry['calibration']}")
        report[modality] = entry
        all_rows += [[modality, *r] for r in out["rows"]]

    if args.save_csv and all_rows:
        import csv
        with open(args.save_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["modality", "file", "ground_truth_label", "predicted_verdict", "confidence_fake_pct"])
            w.writerows(all_rows)
        print(f"\nPer-file results: {args.save_csv}")
    if args.save_json and report:
        with open(args.save_json, "w") as f:
            json.dump(_json_safe(report), f, indent=2)
        print(f"Full report:      {args.save_json}")
    print("\nReminder: synthetic data tests the plumbing only. Quote accuracy only from real labelled data.")
    return report


if __name__ == "__main__":
    main()
