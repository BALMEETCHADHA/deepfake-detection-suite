# Step 7 — API Backend (FastAPI)

One REST API in front of all 5 detectors, for the React frontend (`08_frontend/`) to call.

## Setup
Drop this folder (`07_api_backend/`) into your existing `deepfake_detection_suite/` project,
alongside `01_image_detection/`, `02_document_detection/`, etc. — it imports from them directly.

```bash
cd deepfake_detection_suite/07_api_backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Visit **http://localhost:8000/docs** for interactive Swagger docs — you can upload files and
test every endpoint directly in the browser before touching the frontend at all.

## Endpoints

| Method | Path | Body | Notes |
|---|---|---|---|
| GET | `/api/health` | — | Sanity check |
| POST | `/api/image` | `file` | Any jpeg/png |
| POST | `/api/document` | `file` | .pdf/.docx/.pptx |
| POST | `/api/audio` | `file`, optional `reference` | `reference` triggers speaker comparison |
| POST | `/api/video` | `file`, optional `max_frames` (default 12) | |
| POST | `/api/signature` | `questioned`, optional `reference` | Omit `reference` for the weak fallback mode |

Every endpoint returns the same JSON shape as `DetectionResult.to_dict()` from `common/schema.py`
(`verdict`, `confidence_real_pct`, `confidence_fake_pct`, `evidence`, `model_used`, etc.) — plus
`speaker_comparison` on `/api/audio` when a reference clip is supplied.

## Notes
- **First request per modality is slow** — that's when the ML model actually downloads/loads
  (needs internet, one-time per server restart). Subsequent requests are fast.
- Models are loaded lazily and cached in memory (`_detectors` dict) — the server doesn't load
  anything until the first request hits that modality.
- CORS is pre-configured for Vite's default port (5173) and CRA's (3000). If you deploy the
  frontend somewhere else, add that origin to the `allow_origins` list in `main.py`.
- Errors come back as JSON (`error`, `message`, `trace`) with HTTP 500 instead of crashing the
  server — the frontend should display `message` to the user.

## Response additions
Every endpoint adds `decision_margin_pct` and `confidence_level` (`low`/`medium`/`high`); `/api/image` also returns `scores_breakdown`
(raw model score, face crop used, calibration applied, forensic-fusion weight).

## CORS
Default: local dev origins (localhost 5173/4173/3000). The origins list is about where your *browser* runs, not where the backend runs,
so a Colab + ngrok backend needs no change for a local frontend. Extra origins: `CORS_ALLOW_ORIGINS="https://a.com,http://192.168.1.5:5173"`.
Any origin (demos only): `CORS_ALLOW_ALL=1` (already set in Colab notebook 1).

`GET /api/health` also reports `image_model`, `image_model_is_finetuned`, `image_model_note`, `face_detection` and `face_detection_reason`, so a client can show which model is live.
