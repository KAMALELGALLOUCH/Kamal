"""
Extract Visual Features from MELD MP4 Videos
=============================================
Extracts ResNet-50 visual features from MELD MP4 files.

Pipeline per video:
  1. Sample frames (1 per second)
  2. Resize to 224×224
  3. ResNet-50 (pretrained ImageNet) → 2048-dim per frame
  4. Average pool over frames → 1 × 2048-dim per utterance
  5. Save all as dict → video_cache.pt

Run: CUDA_VISIBLE_DEVICES=3 python scripts/extract_meld_video.py

Output:
  /data8/luoyan/Kamal/thesis/data/meld_video_cache.pt
  Keys: 'dia{X}_utt{Y}' → tensor [2048]
"""

import os
import sys
import cv2
import torch
import numpy as np
from tqdm import tqdm
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights

# ── Config ───────────────────────────────────────────────────────────────────
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MELD_ROOT   = '/data8/luoyan/MELD/MELD.Raw'
CACHE_PATH  = '/data8/luoyan/Kamal/thesis/data/meld_video_cache.pt'
BATCH_SIZE  = 32   # frames per batch through ResNet
FPS_SAMPLE  = 1    # sample 1 frame per second
MAX_FRAMES  = 30   # cap at 30 frames per video (~30 seconds)

SPLITS = {
    'train': 'train_splits',
    'dev':   'dev_splits_complete',
    'test':  'output_repeated_splits_test',
}

# ── ResNet-50 feature extractor ───────────────────────────────────────────────
print(f"Loading ResNet-50 on {DEVICE}...")
resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
# Remove final FC layer → output 2048-dim avg pool features
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
resnet = resnet.to(DEVICE).eval()

# ImageNet normalization
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


def extract_frames(mp4_path):
    """Sample frames from a video at FPS_SAMPLE rate."""
    cap = cv2.VideoCapture(mp4_path)
    if not cap.isOpened():
        return None

    fps     = cap.get(cv2.CAP_PROP_FPS)
    fps     = fps if fps > 0 else 25
    step    = max(1, int(fps / FPS_SAMPLE))
    frames  = []
    idx     = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            # BGR → RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            if len(frames) >= MAX_FRAMES:
                break
        idx += 1

    cap.release()
    return frames if frames else None


def frames_to_feature(frames):
    """Run frames through ResNet-50 → average pool → 2048-dim feature."""
    tensors = []
    for frame in frames:
        try:
            t = transform(frame)
            tensors.append(t)
        except Exception:
            continue

    if not tensors:
        return torch.zeros(2048)

    # Process in batches
    feats = []
    for i in range(0, len(tensors), BATCH_SIZE):
        batch = torch.stack(tensors[i:i+BATCH_SIZE]).to(DEVICE)
        with torch.no_grad():
            out = resnet(batch)          # [B, 2048, 1, 1]
            out = out.squeeze(-1).squeeze(-1)  # [B, 2048]
        feats.append(out.cpu())

    feats  = torch.cat(feats, dim=0)     # [N_frames, 2048]
    return feats.mean(dim=0)             # [2048] — average over frames


def get_key(filename):
    """dia0_utt1.mp4 → dia0_utt1"""
    return os.path.splitext(filename)[0]


# ── Main extraction loop ──────────────────────────────────────────────────────
def main():
    # Load existing cache if present (resume support)
    if os.path.exists(CACHE_PATH):
        print(f"Loading existing cache: {CACHE_PATH}")
        cache = torch.load(CACHE_PATH, weights_only=True)
        print(f"  Already cached: {len(cache)} videos")
    else:
        cache = {}

    total_videos = 0
    skipped      = 0
    errors       = 0

    for split_name, split_dir in SPLITS.items():
        split_path = os.path.join(MELD_ROOT, split_dir)
        mp4_files  = sorted([
            f for f in os.listdir(split_path)
            if f.endswith('.mp4')
        ])

        print(f"\n[{split_name}] {len(mp4_files)} videos → {split_path}")

        for fname in tqdm(mp4_files, desc=f'  {split_name}'):
            key = get_key(fname)

            # Skip if already cached
            if key in cache:
                skipped += 1
                continue

            mp4_path = os.path.join(split_path, fname)

            try:
                frames = extract_frames(mp4_path)
                if frames is None:
                    # Use zero vector for unreadable videos
                    cache[key] = torch.zeros(2048)
                    errors += 1
                else:
                    feat = frames_to_feature(frames)
                    cache[key] = feat
                    total_videos += 1
            except Exception as e:
                cache[key] = torch.zeros(2048)
                errors += 1

        # Save after each split (resume safety)
        torch.save(cache, CACHE_PATH)
        print(f"  ✅ Cache saved: {len(cache)} total entries")

    print(f"\n{'='*55}")
    print(f"  Extraction complete!")
    print(f"  Total extracted : {total_videos}")
    print(f"  Already cached  : {skipped}")
    print(f"  Errors (zeroed) : {errors}")
    print(f"  Cache size      : {len(cache)} videos")
    print(f"  Saved to        : {CACHE_PATH}")
    print(f"{'='*55}")


if __name__ == '__main__':
    main()