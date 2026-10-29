"""
doc_detector.py
----------------
STEP 2 of 5 — Document detection: PDF / DOCX / PPTX.

Strategy: a document itself isn't "real or fake" the way a photo is — what
matters is (a) whether any image embedded in it is AI-generated/deepfake,
and (b) whether the document's own authoring metadata looks tampered with.
So this module:

  1. Extracts every embedded image (document_scanner.py) and runs EACH one
     through the exact same ImageDeepfakeDetector built in Step 1 — full
     reuse, no duplicated model logic.
  2. Extracts document metadata and runs it through metadata_forensics.py.
  3. Rolls everything up into ONE DetectionResult for the document, with
     every individual image's result kept in `sub_results` so the frontend
     can show "which page / which image" triggered the verdict.

Usage:
    from doc_detector import DocumentDeepfakeDetector
    det = DocumentDeepfakeDetector()
    result = det.predict("report.pdf")
    result.print_report()
"""

from __future__ import annotations
import os
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "01_image_detection"))

from common.schema import DetectionResult, verdict_from_scores  # noqa: E402
from image_detector import ImageDeepfakeDetector  # noqa: E402
from document_scanner import extract  # noqa: E402
from metadata_forensics import (  # noqa: E402
    build_pdf_metadata_evidence,
    build_office_metadata_evidence,
)


class DocumentDeepfakeDetector:
    def __init__(self, image_detector: ImageDeepfakeDetector = None, device: str = None):
        # Reuse a single loaded image model across every image in the document
        # instead of reloading it per image (much faster on multi-image PDFs).
        self.image_detector = image_detector or ImageDeepfakeDetector(device=device)

    def predict(self, doc_path: str) -> DetectionResult:
        start = time.time()
        warnings = []
        ext = os.path.splitext(doc_path)[1].lower()

        images, meta = extract(doc_path)

        if ext == ".pdf":
            meta_evidence, meta_suspicion = build_pdf_metadata_evidence(meta)
        else:
            meta_evidence, meta_suspicion = build_office_metadata_evidence(meta)

        evidence = [f"Document metadata: {meta}"] + meta_evidence
        sub_results = []

        if not images:
            warnings.append(
                "No embedded/renderable images found in this document — verdict is based "
                "on metadata forensics only, no visual content was analyzed."
            )
            fake_pct = meta_suspicion * 100.0
            real_pct = 100.0 - fake_pct
            model_used = "metadata-forensics-only (no images in document)"
            model_acc = None
        else:
            per_image_fake_scores = []
            for label, pil_img in images:
                tmp_path = f"__tmp_{label}"
                try:
                    pil_img.convert("RGB").save(tmp_path)
                    img_result = self.image_detector.predict(tmp_path)
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)

                img_result.file = f"{os.path.basename(doc_path)} :: {label}"
                sub_results.append(img_result.to_dict())
                per_image_fake_scores.append(img_result.confidence_fake_pct)

            worst_idx = max(range(len(per_image_fake_scores)), key=lambda i: per_image_fake_scores[i])
            worst_label = images[worst_idx][0]
            worst_score = per_image_fake_scores[worst_idx]

            # Document-level score: driven mainly by the single most suspicious
            # embedded image (one fake image is enough to flag the whole doc),
            # nudged by the metadata suspicion score.
            fake_pct = min(100.0, worst_score * 0.85 + meta_suspicion * 100 * 0.15)
            real_pct = 100.0 - fake_pct
            model_used = f"{self.image_detector.model_id} (applied per embedded image) + metadata forensics"
            model_acc = None  # document-level score is a composite, not one model's benchmark figure

            evidence.append(
                f"{len(images)} image(s) extracted/rendered from document. "
                f"Most suspicious: '{worst_label}' with {worst_score:.2f}% FAKE confidence."
            )

        verdict = verdict_from_scores(fake_pct, real_pct)

        result = DetectionResult(
            file=os.path.basename(doc_path),
            modality="document",
            verdict=verdict,
            confidence_fake_pct=round(fake_pct, 2),
            confidence_real_pct=round(real_pct, 2),
            model_used=model_used,
            model_reported_accuracy_pct=model_acc,
            evidence=evidence,
            sub_results=sub_results,
            processing_time_sec=round(time.time() - start, 3),
            warnings=warnings,
        )
        return result


if __name__ == "__main__":
    print("This module is a library. Run `python demo.py --doc <path>` instead.")
