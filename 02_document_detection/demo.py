"""
demo.py — Step 2: Document Detection CLI

Examples:
    python demo.py --doc report.pdf
    python demo.py --doc slides.pptx --save report.json
    python demo.py --doc essay.docx
"""

import argparse
import os
import sys

from doc_detector import DocumentDeepfakeDetector

SUPPORTED = (".pdf", ".docx", ".pptx")


def main():
    parser = argparse.ArgumentParser(description="Document (PDF/DOCX/PPTX) deepfake scanner")
    parser.add_argument("--doc", type=str, required=True, help="Path to a .pdf/.docx/.pptx file")
    parser.add_argument("--save", type=str, help="Path to save the JSON report")
    parser.add_argument("--device", type=str, default=None, choices=[None, "cpu", "cuda"])
    args = parser.parse_args()

    if not os.path.isfile(args.doc):
        print(f"File not found: {args.doc}")
        sys.exit(1)

    ext = os.path.splitext(args.doc)[1].lower()
    if ext not in SUPPORTED:
        print(f"Unsupported file type '{ext}'. Supported: {SUPPORTED}")
        print("Tip: convert legacy .doc/.ppt with LibreOffice first:")
        print("  soffice --headless --convert-to docx yourfile.doc")
        sys.exit(1)

    detector = DocumentDeepfakeDetector(device=args.device)
    result = detector.predict(args.doc)
    result.print_report()

    if result.sub_results:
        print(f"\nPer-image breakdown ({len(result.sub_results)} image(s)):")
        for sr in result.sub_results:
            print(f"  - {sr['file']}: {sr['verdict']} (FAKE {sr['confidence_fake_pct']}%)")

    if args.save:
        result.save(args.save)
        print(f"\nSaved report to {args.save}")


if __name__ == "__main__":
    main()
