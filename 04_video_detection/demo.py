"""
demo.py — Step 4: Video Detection CLI

Examples:
    python demo.py --video clip.mp4
    python demo.py --video clip.mov --frames 20 --save report.json
"""

import argparse
import os
import sys

from video_detector import VideoDeepfakeDetector


def main():
    parser = argparse.ArgumentParser(description="Video deepfake detector (frames + audio + temporal)")
    parser.add_argument("--video", type=str, required=True, help="Path to a video file")
    parser.add_argument("--frames", type=int, default=12, help="Number of frames to sample (default 12)")
    parser.add_argument("--save", type=str, help="Path to save the JSON report")
    parser.add_argument("--device", type=str, default=None, choices=[None, "cpu", "cuda"])
    args = parser.parse_args()

    if not os.path.isfile(args.video):
        print(f"File not found: {args.video}")
        sys.exit(1)

    detector = VideoDeepfakeDetector(device=args.device, max_frames=args.frames)
    result = detector.predict(args.video)
    result.print_report()

    frame_results = [sr for sr in result.sub_results if sr["modality"] == "image"]
    audio_results = [sr for sr in result.sub_results if sr["modality"] == "audio"]

    if frame_results:
        print(f"\nPer-frame breakdown ({len(frame_results)} frame(s)):")
        for sr in frame_results:
            print(f"  - {sr['file']}: {sr['verdict']} (FAKE {sr['confidence_fake_pct']}%)")
    if audio_results:
        print("\nAudio track:")
        for sr in audio_results:
            print(f"  - {sr['file']}: {sr['verdict']} (FAKE {sr['confidence_fake_pct']}%)")

    if args.save:
        result.save(args.save)
        print(f"\nSaved report to {args.save}")


if __name__ == "__main__":
    main()
