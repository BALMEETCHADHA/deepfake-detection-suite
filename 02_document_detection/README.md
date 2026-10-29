# Step 2 — Document Detection (PDF / Word / PowerPoint)

Scans a document for embedded images (photos, graphs, screenshots — anything pasted or
inserted into the file) and runs each one through the Step 1 image detector, plus checks
the document's own metadata for signs of tampering or AI-tool authorship.

## Files
- `document_scanner.py` — extracts images + metadata from `.pdf` / `.docx` / `.pptx`
- `metadata_forensics.py` — heuristic evidence from authoring metadata
- `doc_detector.py` — `DocumentDeepfakeDetector`, reuses Step 1's `ImageDeepfakeDetector`
- `demo.py` — command-line interface

## Supported formats
`.pdf`, `.docx`, `.pptx`. Legacy `.doc` / `.ppt` (old binary Office formats) aren't
directly supported — convert first:
```bash
soffice --headless --convert-to docx yourfile.doc
soffice --headless --convert-to pptx yourfile.ppt
```

## How it decides + how it proves it
1. Extracts every embedded image. For PDFs with no embedded images at all (e.g. a scanned
   page saved as one big background), it falls back to rendering each page as an image so
   nothing gets missed.
2. Runs **every** extracted image through the exact same ML model + forensic checks from
   Step 1 — one fake image anywhere in the document is enough to flag the whole file.
3. Separately reads the document's metadata (author, last-modified-by, creating application,
   dates) and flags inconsistencies or references to known generative-AI tools.
4. Combines both into one report; `sub_results` in the JSON output lists every individual
   image's own verdict so you can see exactly *which* image/page triggered the flag.

## Run in VS Code
```bash
cd 02_document_detection
pip install -r requirements.txt
python demo.py --doc report.pdf
python demo.py --doc slides.pptx --save report.json
```

## Run in Google Colab
In Colab, open `../colab_notebooks/1_run_demo_and_backend.ipynb`, run the install and load cells, set `MODALITY = "document"` in the "Try a detector" cell, then upload a document.

## Notes
- This module depends on Step 1's files (`../01_image_detection/image_detector.py`,
  `../common/schema.py`) — keep the folder structure intact, don't move this folder out
  on its own.
- Metadata evidence is a *supporting* signal, not proof — a document edited in a normal
  tool by two different people isn't necessarily fake. Read the `evidence` list, not just
  the verdict.
