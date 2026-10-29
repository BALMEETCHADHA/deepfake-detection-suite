"""
main.py — Unified REST API for the Multi-Modal Deepfake Detection Suite
--------------------------------------------------------------------------
Wraps all 5 detector modules (image, document, audio, video, signature)
behind one FastAPI app so the React frontend has a single backend to talk to.

Run from the deepfake_detection_suite root (one level above this file's
parent), or set DEEPFAKE_SUITE_ROOT to point at it:

    cd deepfake_detection_suite/07_api_backend
    pip install -r requirements.txt
    uvicorn main:app --reload --port 8000

Then open http://localhost:8000/docs for interactive API docs, or point the
React frontend's VITE_API_URL at http://localhost:8000.
"""

from __future__ import annotations
import os
import sys
import shutil
import tempfile
import traceback
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# --------------------------------------------------------------------------- #
# Make every module folder importable, exactly like the CLI demos do.
# --------------------------------------------------------------------------- #
SUITE_ROOT = os.environ.get(
    "DEEPFAKE_SUITE_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
)
for sub in [
    "", "common",
    "01_image_detection", "02_document_detection", "03_audio_detection",
    "04_video_detection", "05_signature_detection",
]:
    p = os.path.join(SUITE_ROOT, sub) if sub else SUITE_ROOT
    if p not in sys.path:
        sys.path.append(p)

app = FastAPI(
    title="Deepfake Detection Suite API",
    description="Image, document, audio, video, and signature forgery detection.",
    version="1.0.0",
)

# React dev servers: Vite dev 5173, Vite preview 4173, CRA 3000. Add your
# deployed frontend origin here too if you host this somewhere later.
# --- CORS -------------------------------------------------------------------
# Default: the local React dev/preview servers. Extend without editing code:
#   CORS_ALLOW_ORIGINS="https://my-frontend.example.com,http://192.168.1.5:5173"
#   CORS_ALLOW_ALL=1     (any origin - convenient for demos/Colab, not for production)
ALLOWED_ORIGINS = [
    "http://localhost:5173", "http://127.0.0.1:5173",
    "http://localhost:4173", "http://127.0.0.1:4173",
    "http://localhost:3000", "http://127.0.0.1:3000",
]
ALLOWED_ORIGINS += [o.strip() for o in os.environ.get("CORS_ALLOW_ORIGINS", "").split(",") if o.strip()]
if os.environ.get("CORS_ALLOW_ALL") == "1":
    ALLOWED_ORIGINS = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------------------------------- #
# Lazy-loaded detector singletons. Each one downloads/loads its ML model on
# first use, not at import time — so the API server starts instantly and only
# pays the (one-time, per-process) model-load cost when a modality is first hit.
# --------------------------------------------------------------------------- #
_detectors: dict = {}


def get_detector(name: str):
    if name in _detectors:
        return _detectors[name]
    try:
        if name == "image":
            from image_detector import ImageDeepfakeDetector
            _detectors[name] = ImageDeepfakeDetector()
        elif name == "document":
            from doc_detector import DocumentDeepfakeDetector
            _detectors[name] = DocumentDeepfakeDetector()
        elif name == "audio":
            from audio_detector import AudioDeepfakeDetector
            _detectors[name] = AudioDeepfakeDetector()
        elif name == "video":
            from video_detector import VideoDeepfakeDetector
            _detectors[name] = VideoDeepfakeDetector()
        elif name == "signature":
            from signature_detector import SignatureDetector
            _detectors[name] = SignatureDetector()
        else:
            raise ValueError(f"Unknown detector: {name}")
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load '{name}' detector: {exc.__class__.__name__}: {exc}",
        )
    return _detectors[name]


def _save_upload(upload: UploadFile, dest_dir: str) -> str:
    dest_path = os.path.join(dest_dir, upload.filename or "upload.bin")
    with open(dest_path, "wb") as f:
        shutil.copyfileobj(upload.file, f)
    return dest_path


def _annotate(d: dict) -> dict:
    """Adds an honest strength-of-evidence label to every result.

    A verdict alone hides how close the call was: "REAL" at 53/47 looks the same
    as "REAL" at 97/3. The margin (gap between the two scores) and a coarse
    confidence level make weak calls visibly weak.
    """
    try:
        margin = abs(float(d.get("confidence_fake_pct", 0)) - float(d.get("confidence_real_pct", 0)))
    except (TypeError, ValueError):
        margin = 0.0
    if d.get("verdict") == "UNCERTAIN" or margin < 15:
        level = "low"
    elif margin < 40:
        level = "medium"
    else:
        level = "high"
    d["decision_margin_pct"] = round(margin, 2)
    d["confidence_level"] = level
    return d


def _error_response(exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content={
            "error": exc.__class__.__name__,
            "message": str(exc),
            "trace": traceback.format_exc(limit=3),
        },
    )


@app.get("/api/health")
def health():
    out = {"status": "ok", "suite_root": SUITE_ROOT}
    try:
        from image_detector import resolve_model_id, DEFAULT_MODEL_ID
        from face_crop import face_detection_status
        model = resolve_model_id()
        finetuned = os.path.isdir(model)
        ok, why = face_detection_status()
        out.update({
            "image_model": os.path.basename(model.rstrip("/")) if finetuned else model,
            "image_model_is_finetuned": finetuned,
            "image_model_note": ("Fine-tuned on your labelled data." if finetuned else
                                 "Original pretrained model. Near chance on the project's own test data (ROC-AUC about 0.58); "
                                 "load the fine-tuned model (notebook 1, section 3)." if model == DEFAULT_MODEL_ID else "Custom model."),
            "face_detection": ok,
            "face_detection_reason": why,
        })
    except Exception as exc:                       # health must never fail because of diagnostics
        out["diagnostics_error"] = f"{type(exc).__name__}: {exc}"
    return out


# --------------------------------------------------------------------------- #
# Step 1: Images
# --------------------------------------------------------------------------- #
@app.post("/api/image")
async def check_image(file: UploadFile = File(...)):
    with tempfile.TemporaryDirectory() as tmp:
        try:
            path = _save_upload(file, tmp)
            result = get_detector("image").predict(path)
            return _annotate(result.to_dict())
        except HTTPException:
            raise
        except Exception as exc:
            return _error_response(exc)


# --------------------------------------------------------------------------- #
# Step 2: Documents
# --------------------------------------------------------------------------- #
@app.post("/api/document")
async def check_document(file: UploadFile = File(...)):
    with tempfile.TemporaryDirectory() as tmp:
        try:
            path = _save_upload(file, tmp)
            result = get_detector("document").predict(path)
            return _annotate(result.to_dict())
        except HTTPException:
            raise
        except Exception as exc:
            return _error_response(exc)


# --------------------------------------------------------------------------- #
# Step 3: Audio (+ optional speaker-comparison reference clip)
# --------------------------------------------------------------------------- #
@app.post("/api/audio")
async def check_audio(
    file: UploadFile = File(...),
    reference: Optional[UploadFile] = File(None),
):
    with tempfile.TemporaryDirectory() as tmp:
        try:
            path = _save_upload(file, tmp)
            detector = get_detector("audio")
            result = detector.predict(path)
            output = result.to_dict()

            if reference is not None:
                ref_path = _save_upload(reference, tmp)
                output["speaker_comparison"] = detector.compare_speakers(ref_path, path)

            return _annotate(output)
        except HTTPException:
            raise
        except Exception as exc:
            return _error_response(exc)


# --------------------------------------------------------------------------- #
# Step 4: Video
# --------------------------------------------------------------------------- #
@app.post("/api/video")
async def check_video(file: UploadFile = File(...), max_frames: int = Form(12)):
    max_frames = max(1, min(60, max_frames))          # 0 or huge values used to crash with HTTP 500
    with tempfile.TemporaryDirectory() as tmp:
        try:
            path = _save_upload(file, tmp)
            from video_detector import VideoDeepfakeDetector
            key = f"video_{max_frames}"
            if key not in _detectors:
                _detectors[key] = VideoDeepfakeDetector(max_frames=max_frames)
            result = _detectors[key].predict(path)
            return _annotate(result.to_dict())
        except HTTPException:
            raise
        except Exception as exc:
            return _error_response(exc)


# --------------------------------------------------------------------------- #
# Step 5: Signatures — reference (recommended) or single-image fallback
# --------------------------------------------------------------------------- #
@app.post("/api/signature")
async def check_signature(
    questioned: UploadFile = File(...),
    reference: Optional[UploadFile] = File(None),
):
    with tempfile.TemporaryDirectory() as tmp:
        try:
            q_path = _save_upload(questioned, tmp)
            detector = get_detector("signature")

            if reference is not None:
                ref_path = _save_upload(reference, tmp)
                result = detector.compare(ref_path, q_path)
            else:
                result = detector.single_image_check(q_path)

            return _annotate(result.to_dict())
        except HTTPException:
            raise
        except Exception as exc:
            return _error_response(exc)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
