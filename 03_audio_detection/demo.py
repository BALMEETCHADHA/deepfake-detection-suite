"""
demo.py — Step 3: Audio Detection CLI

Examples:
    python demo.py --audio clip.wav
    python demo.py --audio clip.mp3 --save report.json
    python demo.py --audio questioned.wav --reference my_real_voice.wav   # speaker comparison
"""

import argparse
import os
import sys

from audio_detector import AudioDeepfakeDetector


def main():
    parser = argparse.ArgumentParser(description="Audio deepfake / AI-voice detector")
    parser.add_argument("--audio", type=str, required=True, help="Path to the audio clip to check")
    parser.add_argument(
        "--reference", type=str, default=None,
        help="Optional: path to a known-real reference voice clip, to also check "
             "whether --audio sounds like the same speaker."
    )
    parser.add_argument("--save", type=str, help="Path to save the JSON report")
    parser.add_argument("--device", type=str, default=None, choices=[None, "cpu", "cuda"])
    args = parser.parse_args()

    if not os.path.isfile(args.audio):
        print(f"File not found: {args.audio}")
        sys.exit(1)

    detector = AudioDeepfakeDetector(device=args.device)
    result = detector.predict(args.audio)
    result.print_report()

    output = result.to_dict()

    if args.reference:
        if not os.path.isfile(args.reference):
            print(f"Reference file not found: {args.reference}")
            sys.exit(1)
        comparison = detector.compare_speakers(args.reference, args.audio)
        print("\n---- Speaker comparison ----")
        for k, v in comparison.items():
            print(f"  {k}: {v}")
        output["speaker_comparison"] = comparison

    if args.save:
        import json
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
        print(f"\nSaved report to {args.save}")


if __name__ == "__main__":
    main()
