"""
prepare_real_dataset.py
-------------------------
Converts a downloaded FaceForensics++ (or Celeb-DF) folder into the exact
images/{real,fake}/ and videos/{real,fake}/ layout that batch_evaluate.py
expects — so you can go from "just downloaded it" to "running real accuracy
numbers" without manually sorting thousands of files.

FaceForensics++ (Kaggle: xdxd003/ff-c23) layout after unzip:
    ffpp_raw/
      original/*.mp4            <- REAL
      Deepfakes/*.mp4           <- FAKE
      Face2Face/*.mp4           <- FAKE
      FaceSwap/*.mp4            <- FAKE
      NeuralTextures/*.mp4      <- FAKE
      FaceShifter/*.mp4         <- FAKE
      DeepFakeDetection/*.mp4   <- FAKE

Celeb-DF layout after unzip (folder names vary slightly by mirror, check
yours and adjust --real_dirs / --fake_dirs accordingly):
    Celeb-real/*.mp4            <- REAL
    Celeb-synthesis/*.mp4       <- FAKE

Usage:
    # Copy/symlink whole videos into videos/real, videos/fake (for Step 4 testing)
    python prepare_real_dataset.py --source ./ffpp_raw \
        --real_dirs original --fake_dirs Deepfakes Face2Face FaceSwap NeuralTextures \
        --out ./real_data --mode videos --max_per_class 500

    # ALSO extract N frames per video into images/real, images/fake (for Step 1 testing)
    python prepare_real_dataset.py --source ./ffpp_raw \
        --real_dirs original --fake_dirs Deepfakes Face2Face FaceSwap NeuralTextures \
        --out ./real_data --mode images --frames_per_video 3 --max_per_class 500
"""

from __future__ import annotations
import argparse
import os
import glob
import shutil
import random

import cv2


def collect_videos(source: str, subdirs: list[str], max_count: int) -> list[str]:
    all_videos = []
    for sub in subdirs:
        all_videos.extend(glob.glob(os.path.join(source, sub, "*.mp4")))
        all_videos.extend(glob.glob(os.path.join(source, sub, "**", "*.mp4"), recursive=True))
    all_videos = sorted(set(all_videos))
    random.shuffle(all_videos)
    return all_videos[:max_count] if max_count else all_videos


def copy_videos(video_paths: list[str], out_dir: str) -> int:
    os.makedirs(out_dir, exist_ok=True)
    n = 0
    for src in video_paths:
        dst = os.path.join(out_dir, f"{n:05d}_{os.path.basename(src)}")
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
        n += 1
    return n


def extract_frames(video_paths: list[str], out_dir: str, frames_per_video: int) -> int:
    os.makedirs(out_dir, exist_ok=True)
    n = 0
    for vid_path in video_paths:
        cap = cv2.VideoCapture(vid_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            continue
        indices = sorted(set(
            int(i * (total - 1) / max(1, frames_per_video - 1)) for i in range(frames_per_video)
        ))
        base = os.path.splitext(os.path.basename(vid_path))[0]
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                continue
            out_path = os.path.join(out_dir, f"{base}_f{idx:06d}.jpg")
            cv2.imwrite(out_path, frame)
            n += 1
        cap.release()
    return n


def main():
    parser = argparse.ArgumentParser(description="Prepare a downloaded real deepfake dataset for batch_evaluate.py")
    parser.add_argument("--source", type=str, required=True, help="Root folder of the downloaded/unzipped dataset")
    parser.add_argument("--real_dirs", nargs="+", required=True, help="Subfolder name(s) containing REAL videos")
    parser.add_argument("--fake_dirs", nargs="+", required=True, help="Subfolder name(s) containing FAKE videos")
    parser.add_argument("--out", type=str, default="./real_data")
    parser.add_argument("--mode", choices=["videos", "images", "both"], default="both")
    parser.add_argument("--frames_per_video", type=int, default=3)
    parser.add_argument("--max_per_class", type=int, default=300, help="Cap videos per class (0 = no cap)")
    args = parser.parse_args()

    real_videos = collect_videos(args.source, args.real_dirs, args.max_per_class)
    fake_videos = collect_videos(args.source, args.fake_dirs, args.max_per_class)
    print(f"Found {len(real_videos)} real videos, {len(fake_videos)} fake videos to process.")

    if args.mode in ("videos", "both"):
        n_real = copy_videos(real_videos, os.path.join(args.out, "videos", "real"))
        n_fake = copy_videos(fake_videos, os.path.join(args.out, "videos", "fake"))
        print(f"[videos] copied {n_real} real, {n_fake} fake to {args.out}/videos/")

    if args.mode in ("images", "both"):
        n_real = extract_frames(real_videos, os.path.join(args.out, "images", "real"), args.frames_per_video)
        n_fake = extract_frames(fake_videos, os.path.join(args.out, "images", "fake"), args.frames_per_video)
        print(f"[images] extracted {n_real} real frames, {n_fake} fake frames to {args.out}/images/")

    print(f"\nDone. Now run:\n  python batch_evaluate.py --data {args.out} --modalities images videos")


if __name__ == "__main__":
    main()
