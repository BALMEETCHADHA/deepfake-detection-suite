# Forensic Suite — Frontend (dark forensic-lab UI)

Screens: Landing (animated scan, TUTORIAL / DETECTION buttons) -> Tutorial (Previous / Next / Skip)
-> Menu (Image, Video, Signature, Audio, Document) -> Upload page -> Result page
(huge green REAL / red DEEP FAKE, percentage, reasons, PDF / Word / Image download,
then Previous, Exit, Restart stacked at the bottom).

## Run
```bash
cp env.example .env      # set VITE_API_URL to your backend (default http://localhost:8000)
npm install
npm run dev              # http://localhost:5173
```
Backend: `cd ../07_api_backend && uvicorn main:app --port 8000`

## Try the UI without the backend
Set `VITE_DEMO=1` in `.env`. A file name containing "fake" or "forg" gives a DEEP FAKE result, anything else gives REAL.
The numbers are invented to show the interface, and the app says "DEMO MODE" in its status line. Never leave this on for real use or a submission demo.

## Button behaviour on the result page
- **Previous**: back to that detector's upload page (your files are kept)
- **Exit**: clears everything and returns to the landing page
- **Restart**: clears everything and returns to the five-option menu

## Change the face photos
Replace `src/assets/real.jpg` and `src/assets/fake.jpg` (5:6 portrait works best, 500x600 or larger).
