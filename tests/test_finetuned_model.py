"""Tests for picking up a fine-tuned model folder (notebook 3) and reporting its HELD-OUT accuracy."""
import json
import pytest
import image_detector as imd
from conftest import make_stub_detector


def _model_dir(tmp_path, report=True):
    d = tmp_path / "image_finetuned"
    d.mkdir()
    (d / "config.json").write_text("{}")
    if report:
        (d / "training_report.json").write_text(json.dumps({"test_accuracy_pct": 96.5, "n_test": 600}))
    return str(d)


# ---------------- which model gets used ----------------
def test_default_is_hub_model_when_nothing_else(monkeypatch, tmp_path):
    monkeypatch.delenv("IMAGE_MODEL_ID", raising=False)
    monkeypatch.setattr(imd, "FINETUNED_DIR", str(tmp_path / "nothing_here"))
    assert imd.resolve_model_id() == imd.DEFAULT_MODEL_ID

def test_finetuned_folder_is_picked_up_automatically(monkeypatch, tmp_path):
    monkeypatch.delenv("IMAGE_MODEL_ID", raising=False)
    d = _model_dir(tmp_path)
    monkeypatch.setattr(imd, "FINETUNED_DIR", d)
    assert imd.resolve_model_id() == d

def test_folder_without_config_is_ignored(monkeypatch, tmp_path):
    monkeypatch.delenv("IMAGE_MODEL_ID", raising=False)
    empty = tmp_path / "image_finetuned"; empty.mkdir()
    monkeypatch.setattr(imd, "FINETUNED_DIR", str(empty))
    assert imd.resolve_model_id() == imd.DEFAULT_MODEL_ID

def test_env_overrides_folder_and_explicit_overrides_env(monkeypatch, tmp_path):
    monkeypatch.setattr(imd, "FINETUNED_DIR", _model_dir(tmp_path))
    monkeypatch.setenv("IMAGE_MODEL_ID", "someone/else-model")
    assert imd.resolve_model_id() == "someone/else-model"
    assert imd.resolve_model_id("explicit/id") == "explicit/id"


# ---------------- what the result says about accuracy ----------------
def test_hub_model_reports_model_card_number_unchanged():
    name, acc, report = imd._local_model_info(imd.DEFAULT_MODEL_ID)
    assert name == imd.DEFAULT_MODEL_ID and acc == imd.DEFAULT_MODEL_REPORTED_ACCURACY and report is None

def test_local_model_reports_heldout_accuracy_from_training_report(tmp_path):
    name, acc, report = imd._local_model_info(_model_dir(tmp_path))
    assert "fine-tuned" in name and acc == 96.5 and report["n_test"] == 600

def test_local_model_without_report_claims_no_accuracy(tmp_path):
    name, acc, report = imd._local_model_info(_model_dir(tmp_path, report=False))
    assert acc is None and report is None          # never invent a number

def test_corrupt_report_does_not_crash(tmp_path):
    d = _model_dir(tmp_path, report=False)
    open(d + "/training_report.json", "w").write("{not json")
    assert imd._local_model_info(d)[1] is None


def test_prediction_shows_heldout_accuracy_and_caveat(tmp_path, natural_image, monkeypatch):
    monkeypatch.setenv("DEEPFAKE_CALIBRATION", "off")
    det = make_stub_detector(fake_pct=97.0, face_crop=False, fusion=False)
    det.model_id = _model_dir(tmp_path)
    r = det.predict(natural_image)
    assert r.verdict == "FAKE"
    assert r.model_reported_accuracy_pct == 96.5
    assert "fine-tuned" in r.model_used
    text = " ".join(r.evidence)
    assert "held-out" in text and "600" in text and "other kinds of images" in text

def test_hub_model_prediction_has_no_finetune_claim(natural_image, monkeypatch):
    monkeypatch.setenv("DEEPFAKE_CALIBRATION", "off")
    det = make_stub_detector(fake_pct=30.0, face_crop=False, fusion=False)
    r = det.predict(natural_image)
    assert not any("held-out" in e for e in r.evidence)


# ---------------- label names written by notebook 3 ("Real"/"Fake") must be read correctly ----------------
class _LabelPipe:
    def __init__(self, labels): self.labels = labels
    def __call__(self, image): return self.labels

@pytest.mark.parametrize("labels,expected_fake", [
    ([{"label": "Fake", "score": 0.9}, {"label": "Real", "score": 0.1}], 90.0),      # notebook-10 labels
    ([{"label": "Real", "score": 0.8}, {"label": "Fake", "score": 0.2}], 20.0),      # order must not matter
    ([{"label": "Deepfake", "score": 0.7}, {"label": "Realism", "score": 0.3}], 70.0),  # original hub model
])
def test_fake_probability_read_correctly_for_both_label_schemes(natural_image, labels, expected_fake):
    from PIL import Image
    det = make_stub_detector(face_crop=False, fusion=False)
    det._pipe = _LabelPipe(labels)
    assert det._ml_scores(Image.open(natural_image))["fake_pct"] == pytest.approx(expected_fake)
