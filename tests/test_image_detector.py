import numpy as np
import pytest
from PIL import Image

import image_detector as imd
from explain import forensic_push, forensic_suspicion
from face_crop import crop_largest_face
from common import calibration as cal


# ---------------- forensic helpers ----------------
def test_natural_photo_has_low_suspicion_and_zero_push(natural_image):
    from explain import build_forensic_evidence
    _, sc = build_forensic_evidence(Image.open(natural_image))
    assert forensic_suspicion(sc) < 0.30
    assert forensic_push(forensic_suspicion(sc)) == 0.0

def test_noise_image_has_max_push(noise_image):
    from explain import build_forensic_evidence
    _, sc = build_forensic_evidence(Image.open(noise_image))
    assert forensic_push(forensic_suspicion(sc)) > 0.9

def test_push_is_zero_at_baseline_and_bounded():
    assert forensic_push(0.30) == 0.0 and forensic_push(0.0) == 0.0
    assert forensic_push(1.0) == 1.0 and 0 < forensic_push(0.65) < 1


# ---------------- fuse(): the design guarantees ----------------
def test_fusion_never_lowers_fake_score():
    rng = np.random.default_rng(0)
    for _ in range(2000):
        ml, sus = rng.uniform(0, 100), rng.uniform(0, 1)
        out, w = imd.fuse(ml, sus)
        assert out >= ml - 1e-9 and 0 <= w <= imd.FUSION_MAX_WEIGHT and out <= 100

def test_fusion_ignored_when_ml_is_confident():
    assert imd.fuse(95.0, 1.0) == (95.0, 0.0)
    assert imd.fuse(5.0, 1.0) == (5.0, 0.0)
    assert imd.fuse(70.0, 1.0) == (70.0, 0.0)      # margin 40 >= cutoff 30

def test_fusion_ignored_when_no_artifacts():
    assert imd.fuse(50.0, 0.25) == (50.0, 0.0)

def test_fusion_at_coin_flip_with_extreme_artifacts():
    out, w = imd.fuse(50.0, 1.0)
    assert out == pytest.approx(75.0) and w == pytest.approx(0.5)


# ---------------- the user's scenario through the real detector code path ----------------
def test_user_screenshot_numbers_give_real_not_uncertain(stub_detector, natural_image):
    det = stub_detector(fake_pct=46.96)
    r = det.predict(natural_image)
    assert r.verdict == "REAL"
    assert r.confidence_real_pct == pytest.approx(53.04, abs=0.01)
    assert r.scores_breakdown["fusion_weight"] == 0.0         # natural image -> forensics must not tip it

def test_true_coin_flip_still_uncertain_on_natural_image(stub_detector, natural_image):
    r = stub_detector(fake_pct=49.0).predict(natural_image)
    assert r.verdict == "UNCERTAIN"

def test_fusion_does_not_manufacture_REAL_verdicts(stub_detector, natural_image):
    # 50/50 + clean natural photo must stay UNCERTAIN; if fusion were a symmetric blend it would tilt to REAL
    r = stub_detector(fake_pct=50.0).predict(natural_image)
    assert r.verdict == "UNCERTAIN" and r.confidence_fake_pct == 50.0

def test_noise_image_with_weak_ml_is_pulled_to_fake(stub_detector, noise_image):
    r = stub_detector(fake_pct=50.0, face_crop=False).predict(noise_image)
    assert r.verdict == "FAKE" and r.confidence_fake_pct > 70
    assert any("forensic evidence raised" in e for e in r.evidence)

def test_fusion_can_be_switched_off(stub_detector, noise_image):
    r = stub_detector(fake_pct=50.0, fusion=False).predict(noise_image)
    assert r.confidence_fake_pct == 50.0 and r.verdict == "UNCERTAIN"

def test_env_var_switches_fusion_off(stub_detector, noise_image, monkeypatch):
    monkeypatch.setenv("IMAGE_FUSION", "off")
    assert stub_detector(fake_pct=50.0).predict(noise_image).confidence_fake_pct == 50.0

def test_confident_ml_not_overridden_by_forensics(stub_detector, noise_image):
    r = stub_detector(fake_pct=3.0).predict(noise_image)      # model very sure it's REAL
    assert r.verdict == "REAL" and r.confidence_fake_pct == 3.0


# ---------------- face crop ----------------
def test_face_crop_finds_face_and_ignores_textured_false_positive(face_image):
    crop, info = crop_largest_face(Image.open(face_image))
    assert crop is not None and info["n_faces"] == 1
    x, y, w, h = info["face_box"]
    assert abs(x + w / 2 - (400 + 177 + 47)) < 40            # the pasted face, not the background

def test_no_face_returns_none(natural_image):
    crop, info = crop_largest_face(Image.open(natural_image))
    assert crop is None and info["n_faces"] == 0

def test_detector_feeds_model_the_crop(stub_detector, face_image):
    det = stub_detector(fake_pct=30.0)
    r = det.predict(face_image)
    assert det._pipe.seen_sizes[0][0] < 300                    # model saw ~face-sized crop, not 1000px frame
    assert r.scores_breakdown["face_cropped"] is True
    assert any("face crop" in e.lower() for e in r.evidence)

def test_face_crop_off_feeds_full_frame(stub_detector, face_image):
    det = stub_detector(fake_pct=30.0, face_crop=False)
    det.predict(face_image)
    assert det._pipe.seen_sizes[0] == (1000, 750)

def test_no_face_warns_and_uses_full_frame(stub_detector, natural_image):
    det = stub_detector(fake_pct=30.0)
    r = det.predict(natural_image)
    assert det._pipe.seen_sizes[0] == Image.open(natural_image).size
    assert r.scores_breakdown["face_cropped"] is False
    assert any("No face found" in w for w in r.warnings)

def test_face_crop_env_off(stub_detector, face_image, monkeypatch):
    monkeypatch.setenv("IMAGE_FACE_CROP", "0")
    det = stub_detector(fake_pct=30.0)
    det.predict(face_image)
    assert det._pipe.seen_sizes[0] == (1000, 750)


# ---------------- calibration through the detector ----------------
def test_detector_applies_calibration_file(stub_detector, natural_image, tmp_path, monkeypatch):
    # a=1, b=-1.0986 (= logit(1/3)): maps raw 50 -> 25% fake, i.e. "model over-calls FAKE on my data"
    cal.save_calibration(str(tmp_path / "calibration_image.json"), {"a": 1.0, "b": float(-np.log(2))})
    monkeypatch.setenv("DEEPFAKE_CALIBRATION_DIR", str(tmp_path)); monkeypatch.setenv("DEEPFAKE_CALIBRATION", "on")
    cal.clear_cache()
    r = stub_detector(fake_pct=50.0).predict(natural_image)
    assert r.scores_breakdown["calibrated"] is True
    assert r.confidence_fake_pct == pytest.approx(33.33, abs=0.05)
    assert r.verdict == "REAL"
    assert r.scores_breakdown["ml_fake_pct_raw"] == 50.0
    cal.clear_cache()

def test_forensic_only_fallback_still_works(natural_image):
    det = imd.ImageDeepfakeDetector()                         # offline: ML load fails
    assert det._ml_available is False
    r = det.predict(natural_image)
    assert "forensic-heuristics-only" in r.model_used and r.warnings
