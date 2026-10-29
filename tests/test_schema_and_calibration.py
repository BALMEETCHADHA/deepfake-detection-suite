import json
import numpy as np
import pytest

from common.schema import verdict_from_scores, DetectionResult
from common import calibration as cal


# ---------------- the "UNCERTAIN everywhere" fix ----------------
def test_users_exact_screenshot_numbers_are_real():
    # REAL 53.04 / FAKE 46.96 used to be UNCERTAIN (10-pt band)
    assert verdict_from_scores(fake_pct=46.96, real_pct=53.04) == "REAL"

def test_mirror_case_is_fake():
    assert verdict_from_scores(fake_pct=53.04, real_pct=46.96) == "FAKE"

@pytest.mark.parametrize("fake,real", [(49, 51), (50, 50), (48, 52), (47, 53)])
def test_true_coin_flips_stay_uncertain(fake, real):
    assert verdict_from_scores(fake, real) == "UNCERTAIN"       # lead <= 6

def test_band_boundary_is_inclusive_at_6():
    assert verdict_from_scores(47.0, 53.0) == "UNCERTAIN"       # lead exactly 6
    assert verdict_from_scores(46.9, 53.1) == "REAL"            # lead 6.2

def test_custom_band_still_works():
    assert verdict_from_scores(10, 90, uncertain_band=100.0) == "UNCERTAIN"   # signature fallback relies on this
    assert verdict_from_scores(45, 55, uncertain_band=0) == "REAL"

def test_result_has_breakdown_field_and_serialises():
    r = DetectionResult(file="x", modality="image", verdict="REAL", confidence_real_pct=60, confidence_fake_pct=40,
                        model_used="m", model_reported_accuracy_pct=None, scores_breakdown={"a": 1})
    assert json.loads(r.to_json())["scores_breakdown"] == {"a": 1}


# ---------------- calibration ----------------
def _synthetic(n=20000, a=0.5, b=-0.8, seed=0):
    """Scores from a model that is overconfident & biased toward FAKE; labels drawn from the TRUE calibrated prob."""
    rng = np.random.default_rng(seed)
    raw = rng.uniform(0.02, 0.98, n)
    true_p = cal._sigmoid(a * cal._logit(raw) + b)
    y = rng.uniform(size=n) < true_p
    return raw * 100, y

def test_platt_recovers_known_parameters():
    raw, y = _synthetic()
    p = cal.fit_platt(raw, y)
    assert abs(p["a"] - 0.5) < 0.05 and abs(p["b"] - (-0.8)) < 0.05

def test_calibration_reduces_ece_on_heldout_data():
    raw, y = _synthetic(seed=1)
    tr, te = slice(0, 10000), slice(10000, None)
    p = cal.fit_platt(raw[tr], y[tr])
    before = cal.expected_calibration_error(raw[te], y[te])
    after = cal.expected_calibration_error([cal.calibrate_pct(s, p["a"], p["b"]) for s in raw[te]], y[te])
    assert before > 10 and after < 3 and after < before

def test_fit_needs_both_classes():
    with pytest.raises(ValueError):
        cal.fit_platt([10, 20, 30, 40, 50], [True] * 5)

def test_perfectly_separable_data_does_not_blow_up():
    p = cal.fit_platt([5, 10, 15, 85, 90, 95], [False, False, False, True, True, True])
    assert np.isfinite(p["a"]) and np.isfinite(p["b"])

def test_roc_auc_known_values():
    assert cal.roc_auc([10, 20, 80, 90], [False, False, True, True]) == 1.0
    assert cal.roc_auc([90, 80, 20, 10], [False, False, True, True]) == 0.0
    assert cal.roc_auc([50, 50, 50, 50], [False, False, True, True]) == 0.5
    # hand-computed: pos={0.8,0.4}, neg={0.6,0.2}: pairs (0.8>0.6)(0.8>0.2)(0.4<0.6)(0.4>0.2) = 3/4
    assert cal.roc_auc([80, 60, 40, 20], [True, False, True, False]) == 0.75
    assert cal.roc_auc([1, 2], [True, True]) is None

def test_ece_zero_for_perfectly_calibrated_bins():
    # 10 items at 80% fake, 8 of them fake -> gap 0
    assert cal.expected_calibration_error([80] * 10, [True] * 8 + [False] * 2) == pytest.approx(0, abs=1e-9)
    assert cal.expected_calibration_error([90] * 10, [True] * 5 + [False] * 5) == pytest.approx(40, abs=1e-6)

def test_apply_calibration_uses_file_and_can_be_disabled(tmp_path, monkeypatch):
    cal.save_calibration(str(tmp_path / "calibration_image.json"), {"a": 1.0, "b": 1.0})
    monkeypatch.setenv("DEEPFAKE_CALIBRATION_DIR", str(tmp_path))
    monkeypatch.setenv("DEEPFAKE_CALIBRATION", "on")
    cal.clear_cache()
    out, used = cal.apply_calibration("image", 50.0)
    assert used and out == pytest.approx(73.1, abs=0.1)         # sigmoid(0*1+1)
    monkeypatch.setenv("DEEPFAKE_CALIBRATION", "off"); cal.clear_cache()
    assert cal.apply_calibration("image", 50.0) == (50.0, False)
    monkeypatch.setenv("DEEPFAKE_CALIBRATION", "on"); cal.clear_cache()
    assert cal.apply_calibration("audio", 50.0) == (50.0, False)   # no file for audio -> passthrough
    cal.clear_cache()

def test_calibration_is_monotonic():
    xs = np.linspace(1, 99, 50)
    ys = [cal.calibrate_pct(x, 0.7, 0.3) for x in xs]
    assert all(b > a for a, b in zip(ys, ys[1:]))
