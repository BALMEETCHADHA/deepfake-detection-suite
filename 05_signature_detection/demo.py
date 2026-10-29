"""
demo.py — Step 5: Signature Detection CLI

Recommended usage (reference comparison — reliable):
    python demo.py --questioned signature_to_check.png --reference known_genuine.png

Fallback usage (no reference available — weak signal only):
    python demo.py --questioned signature_to_check.png
"""

import argparse
import os
import sys

from signature_detector import SignatureDetector


def main():
    parser = argparse.ArgumentParser(description="Signature genuine-vs-forged/AI detector")
    parser.add_argument("--questioned", type=str, required=True, help="Signature image to check")
    parser.add_argument(
        "--reference", type=str, default=None,
        help="Known-genuine reference signature image (strongly recommended)"
    )
    parser.add_argument("--save", type=str, help="Path to save the JSON report")
    parser.add_argument("--device", type=str, default=None, choices=[None, "cpu", "cuda"])
    args = parser.parse_args()

    if not os.path.isfile(args.questioned):
        print(f"File not found: {args.questioned}")
        sys.exit(1)

    detector = SignatureDetector(device=args.device)

    if args.reference:
        if not os.path.isfile(args.reference):
            print(f"Reference file not found: {args.reference}")
            sys.exit(1)
        result = detector.compare(args.reference, args.questioned)
    else:
        print(
            "\n[!] No --reference signature provided. Running weak fallback check only.\n"
            "    For a reliable verdict, re-run with: --reference known_genuine.png\n"
        )
        result = detector.single_image_check(args.questioned)

    result.print_report()

    if args.save:
        result.save(args.save)
        print(f"\nSaved report to {args.save}")


if __name__ == "__main__":
    main()
