import json
import os

import numpy as np
import pytest
from PIL import Image

import batch_evaluate as be
from common import calibration as cal
from conftest import make_stub_detector


# ---------------- metrics ----------------
def test_metrics_counts_uncertain_as_a_miss_not_as_real():
    truth = [False, False, True, True]
    pred = [False, None, True, None]          # one real correct, one real UNCERTAIN, one fake correct, one fake UNCERTAIN
    m = be._metrics(truth, pred)
    assert m["uncertain"] == 2 and m["coverage_pct"] == 50.0
    assert m["accuracy_pct"] == 50.0          # strict
    assert m["accuracy_decided_pct"] == 100.0
    assert (m["tp"], m["tn"], m["fp"], m["fn"]) == (1, 1, 0, 0)

def test_metrics_precision_recall_f1_hand_computed():
    truth = [True, True, True, False, False, False]
    pred = [True, True, False, True, False, False]       # tp2 fn1 fp1 tn2
    m = be._metrics(truth, pred)
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (2, 1, 1, 2)
    assert m["precision_pct"] == pytest.approx(66.67, abs=0.01) and m["recall_pct"] == pytest.approx(66.67, abs=0.01)
    assert m["accuracy_pct"] == pytest.approx(66.67, abs=0.01)

def test_decide_maps_all_verdict_vocabularies():
    assert be._decide("FAKE") is True and be._decide("FORGED") is True
    assert be._decide("REAL") is False and be._decide("GENUINE") is False
    assert be._decide("UNCERTAIN") is None

def test_band_sweep_monotone_coverage_and_matches_manual():
    truth = [False, False, True, True]
    scores = [47.0, 20.0, 53.0, 90.0]                   # leads: 6, 60, 6, 80
    sweep = {r["band"]: r for r in be.band_sweep(truth, scores)}
    assert sweep[6]["coverage_pct"] == 50.0             # lead<=6 -> UNCERTAIN for the two borderline ones
    assert sweep[4]["coverage_pct"] == 100.0
    cov = [sweep[b]["coverage_pct"] for b in sorted(sweep)]
    assert all(a >= b for a, b in zip(cov, cov[1:]))

def test_best_threshold_finds_shifted_boundary():
    truth = [False] * 5 + [True] * 5
    scores = [10, 20, 30, 40, 45, 55, 60, 70, 80, 90]    # perfect split at 50
    assert be.best_threshold(truth, scores)["youden_j"] == 1.0
    shifted = [30, 35, 38, 40, 42, 44, 50, 60, 70, 80]   # model biased toward REAL; best cut ~44, not 50
    bt = be.best_threshold(truth, shifted)
    assert bt["threshold_fake_pct"] == 44.0 and bt["tpr_pct"] == 100.0 and bt["fpr_pct"] == 0.0


# ---------------- end to end with a stub model on a labelled folder ----------------
@pytest.fixture
def labelled_folder(tmp_path):
    """REAL images: calm scenes, half containing a face. FAKE images: noisy copies. 12 each."""
    from skimage import data
    rng = np.random.default_rng(0)
    for lab in ("real", "fake"):
        os.makedirs(tmp_path / lab)
    for i in range(12):
        bg = Image.fromarray(data.coffee()).resize((600, 450))
        if i % 2 == 0:       # a ~55 px face: above the detector's minimum face size
            bg.paste(Image.fromarray(data.astronaut()).resize((300, 300)), (150, 60))
        bg.save(tmp_path / "real" / f"r{i}.png")
        arr = np.clip(np.asarray(bg, dtype=float) + rng.normal(0, 40 + 2 * i, (450, 600, 3)), 0, 255).astype(np.uint8)
        Image.fromarray(arr).save(tmp_path / "fake" / f"f{i}.png")
    return str(tmp_path)

def _stub_that_is_weak_but_informative(img):
    """Fake score: 50 + a little signal from pixel noise -> weak, near-coin-flip outputs (like the user's 53/47)."""
    a = np.asarray(img.convert("L"), dtype=float)
    hp = np.abs(np.diff(a, axis=1)).mean()
    return float(np.clip(40 + hp * 0.9, 1, 99))

def test_eval_images_on_user_style_folders(labelled_folder):
    det = make_stub_detector(fn=_stub_that_is_weak_but_informative)
    out = be.eval_images("", real_dir=os.path.join(labelled_folder, "real"),
                         fake_dir=os.path.join(labelled_folder, "fake"), detector=det)
    m = out["metrics"]
    assert m["n"] == 24 and m["roc_auc"] > 0.9
    assert len(out["raw_ml_fake_pcts"]) == 24

def test_compare_variants_runs_and_fusion_helps_when_ml_is_weak(labelled_folder):
    det = make_stub_detector(fn=lambda im: 50.0)            # ML knows nothing at all
    out = be.eval_images("", real_dir=os.path.join(labelled_folder, "real"),
                         fake_dir=os.path.join(labelled_folder, "fake"), detector=det, compare_variants=True)
    v = out["variants"]
    assert set(v) == {"model only (full image)", "model + face crop", "model + forensic fusion", "model + face crop + fusion"}
    assert out["faces_found"] >= 6 and out["n_images"] == 24      # the 6 real face images (+ noisy copies)
    # With a clueless model, only the forensic fusion has any signal on these noisy fakes:
    assert v["model only (full image)"]["roc_auc"] == 0.5
    assert v["model + forensic fusion"]["roc_auc"] > 0.9
    assert v["model + forensic fusion"]["accuracy_pct"] > v["model only (full image)"]["accuracy_pct"]
    assert det.face_crop is True and det.fusion is True    # evaluator restored the detector's settings

def test_face_crop_variant_changes_scores_only_for_images_with_faces(labelled_folder):
    # stub is fooled by big images: gives a different score if it sees a small crop. Verifies variant B really crops.
    det = make_stub_detector(fn=lambda im: 80.0 if im.size[0] >= 600 else 20.0)
    out = be.eval_images("", real_dir=os.path.join(labelled_folder, "real"),
                         fake_dir=os.path.join(labelled_folder, "fake"), detector=det, compare_variants=True)
    a = out["variants"]["model only (full image)"]; b = out["variants"]["model + face crop"]
    assert a != b

def test_fit_calibration_writes_file_and_improves_heldout_ece(labelled_folder, tmp_path):
    # Deliberately miscalibrated model: says 85% on fakes and 70% on reals, i.e. overconfident toward FAKE.
    det = make_stub_detector(fn=lambda im: 85.0 if np.abs(np.diff(np.asarray(im.convert("L"), float), axis=1)).mean() > 12 else 70.0,
                             face_crop=False)
    # replicate dataset bigger for a meaningful hold-out
    import shutil
    for lab in ("real", "fake"):
        d = os.path.join(labelled_folder, lab)
        for f in list(os.listdir(d)):
            for k in range(3):
                shutil.copy(os.path.join(d, f), os.path.join(d, f"copy{k}_{f}"))
    with be._env(DEEPFAKE_CALIBRATION="off"):
        out = be.eval_images("", real_dir=os.path.join(labelled_folder, "real"),
                             fake_dir=os.path.join(labelled_folder, "fake"), detector=det)
    res = be.fit_and_save_calibration("images", out, str(tmp_path / "cal"))
    assert os.path.isfile(res["file"])
    h = res["holdout"]
    assert isinstance(h, dict) and h["n_holdout"] >= 10
    assert h["ece_after_pts"] < h["ece_before_pts"]          # calibration did its job on data it never saw
    saved = json.load(open(res["file"]))
    assert saved["meta"]["modality"] == "images" and np.isfinite(saved["a"])

def test_fit_calibration_skips_cleanly_without_raw_ml_scores():
    assert "skipped" in be.fit_and_save_calibration("images", {"raw_ml_fake_pcts": None, "is_fake": [True]}, "/tmp/x")

def test_cli_main_on_images_folder_writes_reports(labelled_folder, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rep = be.main(["--real_dir", os.path.join(labelled_folder, "real"), "--fake_dir", os.path.join(labelled_folder, "fake"),
                   "--save_csv", str(tmp_path / "o.csv"), "--save_json", str(tmp_path / "o.json")])
    assert "images" in rep and os.path.isfile(tmp_path / "o.csv")
    j = json.load(open(tmp_path / "o.json"))
    assert "band_sweep" in j["images"] and "best_threshold" in j["images"]


# ---------------- regression: never calibrate heuristic scores ----------------
def test_no_calibration_file_is_written_when_ml_model_is_missing(labelled_folder, tmp_path):
    from image_detector import ImageDeepfakeDetector
    det = ImageDeepfakeDetector()                      # offline -> forensic-only fallback
    assert det._ml_available is False
    for variants in (False, True):
        out = be.eval_images("", real_dir=os.path.join(labelled_folder, "real"),
                             fake_dir=os.path.join(labelled_folder, "fake"), detector=det, compare_variants=variants)
        assert out["raw_ml_fake_pcts"] is None
        res = be.fit_and_save_calibration("images", out, str(tmp_path / "cal"))
        assert "skipped" in res and not os.path.exists(tmp_path / "cal")

def test_variant_mode_csv_rows_carry_real_verdicts(labelled_folder):
    det = make_stub_detector(fn=lambda im: 50.0)
    out = be.eval_images("", real_dir=os.path.join(labelled_folder, "real"),
                         fake_dir=os.path.join(labelled_folder, "fake"), detector=det, compare_variants=True)
    assert {r[2] for r in out["rows"]} <= {"REAL", "FAKE", "UNCERTAIN"}
    assert out["raw_ml_fake_pcts"] == [50.0] * 24      # the stub's raw ML score, untouched by fusion
