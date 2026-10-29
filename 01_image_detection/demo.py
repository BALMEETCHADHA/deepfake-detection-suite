"""
demo.py — Step 1: Image Detection CLI

Examples:
    python demo.py --image samples/photo.jpg
    python demo.py --image samples/photo.jpg --save report.json
    python demo.py --folder samples/                 # batch mode
"""

import argparse
import glob
import json
import os
import sys

from image_detector import ImageDeepfakeDetector

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def main():
    parser = argparse.ArgumentParser(description="Image deepfake / AI-image detector")
    parser.add_argument("--image", type=str, help="Path to a single image file")
    parser.add_argument("--folder", type=str, help="Path to a folder of images (batch mode)")
    parser.add_argument("--save", type=str, help="Path to save JSON report(s)")
    parser.add_argument("--device", type=str, default=None, choices=[None, "cpu", "cuda"])
    args = parser.parse_args()

    if not args.image and not args.folder:
        print("Provide --image <file> or --folder <dir>. Use -h for help.")
        sys.exit(1)

    detector = ImageDeepfakeDetector(device=args.device)

    if args.image:
        if not os.path.isfile(args.image):
            print(f"File not found: {args.image}")
            sys.exit(1)
        result = detector.predict(args.image)
        result.print_report()
        if args.save:
            result.save(args.save)
            print(f"\nSaved report to {args.save}")

    elif args.folder:
        paths = [
            p for p in glob.glob(os.path.join(args.folder, "*"))
            if p.lower().endswith(IMG_EXTS)
        ]
        if not paths:
            print(f"No images found in {args.folder}")
            sys.exit(1)

        all_reports = []
        for p in paths:
            result = detector.predict(p)
            result.print_report()
            all_reports.append(result.to_dict())

        if args.save:
            with open(args.save, "w", encoding="utf-8") as f:
                json.dump(all_reports, f, indent=2)
            print(f"\nSaved {len(all_reports)} reports to {args.save}")


if __name__ == "__main__":
    main()
