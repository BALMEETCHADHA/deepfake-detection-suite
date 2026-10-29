"""
image_detector.py
------------------
STEP 1 of 5 — Image deepfake / AI-generated-image detection.

Works on ANY jpeg/png image: a face, a street photo, a graph screenshot,
anything. Two layers of analysis are combined:

  1. ML classifier: a pretrained Vision Transformer fine-tuned specifically
     to separate real photos from AI-generated/deepfake images
     (prithivMLmods/Deep-Fake-Detector-v2-Model — published test precision
     ~92%, weights hosted free on Hugging Face, no API key needed).

  2. Classical forensic evidence (explain.py): Error-Level-Analysis and a
     frequency-domain check. These don't override the model, they add
     human-readable "proof" to the report, and they still work even if you
     have no internet access to download the ML model.

If `transformers`/`torch` or an internet connection isn't available, the
detector automatically falls back to a forensic-heuristics-only verdict so
the script never crashes — it just tells you which mode it ran in.

Usage (see demo.py for the CLI):

    from image_detector import ImageDeepfakeDetector
    det = ImageDeepfakeDetector()
    result = det.predict("photo.jpg")
    result.print_report()
"""

from __future__ import annotations
import os
import sys
import time
from typing import Optional

from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from common.schema import DetectionResult, verdict_from_scores  # noqa: E402
from common.calibration import apply_calibration  # noqa: E402
from explain import build_forensic_evidence, forensic_suspicion, forensic_push  # noqa: E402
import face_crop as _face_crop  # noqa: E402

DEFAULT_MODEL_ID = "prithivMLmods/Deep-Fake-Detector-v2-Model"
DEFAULT_MODEL_REPORTED_ACCURACY = 92.1  # the MODEL CARD's own claim, not measured on your data

# A model you fine-tuned (notebook 3) is used automatically when this folder exists.
# Override with env IMAGE_MODEL_ID (a Hugging Face id or a local folder).
_SUITE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FINETUNED_DIR = os.path.join(_SUITE_ROOT, "models", "image_finetuned")


def resolve_model_id(model_id: Optional[str] = None) -> str:
    if model_id:
        return model_id
    env = os.environ.get("IMAGE_MODEL_ID", "").strip()
    if env:
        return env
    if os.path.isfile(os.path.join(FINETUNED_DIR, "config.json")):
        return FINETUNED_DIR
    return DEFAULT_MODEL_ID


def _local_model_info(model_id: str):
    """For a local fine-tuned folder: (display name, held-out test accuracy %, report dict)."""
    if not os.path.isdir(model_id):
        return model_id, DEFAULT_MODEL_REPORTED_ACCURACY, None
    report = None
    rp = os.path.join(model_id, "training_report.json")
    if os.path.isfile(rp):
        try:
            import json
            with open(rp) as f:
                report = json.load(f)
        except Exception:
            report = None
    acc = report.get("test_accuracy_pct") if report else None
    return f"fine-tuned model ({os.path.basename(model_id.rstrip('/'))})", acc, report

# --- forensic fusion design -------------------------------------------------
# Forensics may only help when the ML model is unsure, and may only push toward
# FAKE (clean ELA/FFT is not proof of authenticity - good fakes pass them).
FUSION_MAX_WEIGHT = 0.5      # at most half the remaining distance to 100% fake
FUSION_MARGIN_CUTOFF = 30.0  # ML lead (|fake-real|, in points) at/above which forensics are ignored


def fuse(ml_fake_pct: float, suspicion: float):
    """Returns (fused_fake_pct, weight). Guarantees: never lowers the fake score,
    weight is 0 when the model is confident or forensics are clean."""
    margin = abs(2.0 * ml_fake_pct - 100.0)
    uncertainty = max(0.0, 1.0 - margin / FUSION_MARGIN_CUTOFF)
    weight = float(FUSION_MAX_WEIGHT * forensic_push(suspicion) * uncertainty)
    if weight <= 0.0:
        return float(ml_fake_pct), 0.0
    return float(ml_fake_pct + weight * (100.0 - ml_fake_pct)), weight


def _env_on(name: str, default: bool = True) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() not in ("0", "off", "false", "no")


class ImageDeepfakeDetector:
    def __init__(self, model_id: Optional[str] = None, device: Optional[str] = None,
                 face_crop: Optional[bool] = None, fusion: Optional[bool] = None):
        """face_crop / fusion default to ON; env IMAGE_FACE_CROP=0 / IMAGE_FUSION=off disable them."""
        self.model_id = resolve_model_id(model_id)
        self.device = device
        self.face_crop = _env_on("IMAGE_FACE_CROP") if face_crop is None else bool(face_crop)
        self.fusion = _env_on("IMAGE_FUSION") if fusion is None else bool(fusion)
        self._pipe = None
        self._ml_available = self._try_load_model()

    # ------------------------------------------------------------------ #
    def _try_load_model(self) -> bool:
        try:
            import torch  # noqa: F401
            from transformers import pipeline

            use_gpu = self.device == "cuda" or (self.device is None and torch.cuda.is_available())
            device_arg = 0 if use_gpu else -1
            self._pipe = pipeline(
                "image-classification",
                model=self.model_id,
                device=device_arg,
            )
            return True
        except Exception as exc:  # pragma: no cover - environment dependent
            print(
                f"[image_detector] Could not load ML model '{self.model_id}' "
                f"({exc.__class__.__name__}: {exc}). Falling back to "
                "forensic-heuristics-only mode."
            )
            return False

    # ------------------------------------------------------------------ #
    def _ml_scores(self, image: Image.Image) -> Optional[dict]:
        """Returns {'fake_pct': float, 'real_pct': float, 'raw': [...]}"""
        if not self._ml_available:
            return None
        outputs = self._pipe(image)  # list of {label, score}
        fake_pct, real_pct = 0.0, 0.0
        for o in outputs:
            label = o["label"].lower()
            pct = float(o["score"]) * 100.0
            if "fake" in label or "deepfake" in label or "ai" in label:
                fake_pct += pct
            else:
                real_pct += pct
        # normalize in case labels don't perfectly cover 100%
        total = fake_pct + real_pct
        if total > 0:
            fake_pct = fake_pct / total * 100.0
            real_pct = real_pct / total * 100.0
        return {"fake_pct": fake_pct, "real_pct": real_pct, "raw": outputs}

    # ------------------------------------------------------------------ #
    def predict(self, image_path: str) -> DetectionResult:
        start = time.time()
        warnings = []
        image = Image.open(image_path)
        image.load()

        forensic_evidence, forensic_scores = build_forensic_evidence(image)
        suspicion = forensic_suspicion(forensic_scores)
        breakdown = {
            "forensic_suspicion": round(suspicion, 3),
            "forensic_push": round(forensic_push(suspicion), 3),
        }

        if self._ml_available:
            # 1) face crop: the model was trained on faces
            model_input, face_cropped, crop_note = image, False, None
            if self.face_crop:
                crop, info = _face_crop.crop_largest_face(image)
                if crop is not None:
                    model_input, face_cropped = crop, True
                    crop_note = (f"Classified a face crop ({crop.size[0]}x{crop.size[1]}px) taken from the "
                                 f"{image.size[0]}x{image.size[1]}px image, the kind of input this model was trained on.")
                elif not info.get("available"):
                    warnings.append("Face detection unavailable: " + (info.get("reason") or "OpenCV problem") + ". The full image was classified instead of a face crop.")
                elif info.get("n_faces", 0) == 0:
                    warnings.append("No face found: this model is trained on faces, so on a full "
                                    "scene or screenshot its output is less reliable.")
            breakdown["face_cropped"] = face_cropped

            # 2) model, 3) calibration, 4) forensic fusion
            ml = self._ml_scores(model_input)
            raw_fake = ml["fake_pct"]
            fake_pct, calibrated = apply_calibration("image", raw_fake)
            breakdown.update({"ml_fake_pct_raw": round(raw_fake, 2), "calibrated": calibrated})

            weight = 0.0
            if self.fusion:
                fused, weight = fuse(fake_pct, suspicion)
            else:
                fused = fake_pct
            breakdown["fusion_weight"] = round(weight, 3)
            fake_pct = fused
            real_pct = 100.0 - fake_pct

            model_used, model_acc, train_report = _local_model_info(self.model_id)
            evidence = [f"ML classifier confidence - FAKE: {raw_fake:.2f}%, REAL: {100 - raw_fake:.2f}%"]
            if train_report:
                evidence.append(
                    f"Model was fine-tuned on a labelled dataset; accuracy on held-out test images it never "
                    f"trained on: {train_report.get('test_accuracy_pct')}% (n={train_report.get('n_test')}). "
                    f"Expect lower accuracy on other kinds of images.")
            if crop_note:
                evidence.append(crop_note)
            if calibrated:
                evidence.append(f"Scores were calibrated on labelled data: raw FAKE {raw_fake:.2f}% -> {ml_cal(fake_pct, weight):.2f}%.")
            if weight > 0:
                evidence.append(f"Model was unsure, so forensic evidence raised FAKE by weight {weight:.2f} "
                                f"(forensic suspicion {suspicion:.2f}); final FAKE {fake_pct:.2f}%.")
            evidence += forensic_evidence
        else:
            warnings.append(
                "ML model unavailable in this environment - verdict is based on classical forensic "
                "heuristics only (lower reliability). Install `torch` + `transformers` and run with "
                "internet access for the full ML-backed verdict."
            )
            fake_pct = suspicion * 100.0
            real_pct = 100.0 - fake_pct
            model_used = "forensic-heuristics-only (no ML model loaded)"
            model_acc = None
            evidence = forensic_evidence
            breakdown.update({"face_cropped": False, "calibrated": False, "fusion_weight": 0.0})

        verdict = verdict_from_scores(fake_pct, real_pct)
        return DetectionResult(
            file=os.path.basename(image_path),
            modality="image",
            verdict=verdict,
            confidence_fake_pct=round(fake_pct, 2),
            confidence_real_pct=round(real_pct, 2),
            model_used=model_used,
            model_reported_accuracy_pct=model_acc,
            evidence=evidence,
            processing_time_sec=round(time.time() - start, 3),
            warnings=warnings,
            scores_breakdown=breakdown,
        )


def ml_cal(fake_pct: float, weight: float) -> float:
    """Calibrated score before fusion (fusion only ever raises it)."""
    if weight <= 0:
        return fake_pct
    # invert fuse(): fused = c + w*(100-c)  ->  c = (fused - 100*w) / (1 - w)
    return (fake_pct - 100.0 * weight) / (1.0 - weight)


if __name__ == "__main__":
    print("This module is a library. Run `python demo.py --image <path>` instead.")
