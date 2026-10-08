"""
Evaluation Script: MELD with Video Features
============================================
Generates confusion matrix, per-class F1, and classification report.
Works with both checkpoints:
  - best_multimodal_model.pt     (Text + Audio + Speaker + Context)
  - best_meld_video_model.pt     (Text + Audio + Video + Speaker + Context)

Usage:
  # Evaluate base model
  CUDA_VISIBLE_DEVICES=3 python scripts/evaluate_meld_video.py

  # Evaluate video model
  CUDA_VISIBLE_DEVICES=3 python scripts/evaluate_meld_video.py --use_video

  # Compare both side by side
  CUDA_VISIBLE_DEVICES=3 python scripts/evaluate_meld_video.py --compare
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torchaudio.transforms as T
from transformers import AutoModel
from sklearn.metrics import (
    f1_score, classification_report, confusion_matrix
)

sys.path.append('/data8/luoyan/Kamal/thesis')
from data.meld_dataset import (
    get_dataloaders, EMOTION2ID, SENTIMENT2ID,
    SPEAKER2ID, NUM_SPEAKERS
)

NUM_EMOTIONS = len(EMOTION2ID)
ID2EMOTION   = {v: k for k, v in EMOTION2ID.items()}

# ── Config ────────────────────────────────────────────────────────────────────
DEVICE          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE      = 32
MAX_LEN         = 128
CONTEXT_WIN     = 3
TEXT_MODEL_PATH = '/data8/zhangxin/comp5423/gte-modernbert-base'
VIDEO_CACHE     = '/data8/luoyan/Kamal/thesis/data/meld_video_cache.pt'
BASE_CKPT       = '/data8/luoyan/Kamal/thesis/outputs/best_multimodal_model.pt'
VIDEO_CKPT      = '/data8/luoyan/Kamal/thesis/outputs/meld/best_meld_video_model.pt'
FIG_DIR         = '/data8/luoyan/Kamal/thesis/outputs/figures'
os.makedirs(FIG_DIR, exist_ok=True)

EMOTION_LABELS = [ID2EMOTION[i] for i in range(NUM_EMOTIONS)]


# ══════════════════════════════════════════════════════════════════════════════
# Model Components
# ══════════════════════════════════════════════════════════════════════════════

class AudioEncoder(nn.Module):
    def __init__(self, hidden_size=768, dropout=0.3):
        super().__init__()
        self.mel_transform   = T.MelSpectrogram(
            sample_rate=16000, n_fft=512, hop_length=160, n_mels=80
        )
        self.amplitude_to_db = T.AmplitudeToDB()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32),
            nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64),
            nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128),
            nn.ReLU(), nn.AdaptiveAvgPool2d((4, 16))
        )
        self.projection = nn.Sequential(
            nn.Linear(128 * 4 * 16, hidden_size),
            nn.ReLU(), nn.Dropout(dropout)
        )

    def forward(self, waveform):
        mel = self.amplitude_to_db(self.mel_transform(waveform))
        f   = self.cnn(mel).view(mel.size(0), -1)
        return self.projection(f)


class VideoEncoder(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(), nn.Dropout(dropout),
        )

    def forward(self, video_feat):
        return self.projection(video_feat)


class SpeakerEmbedding(nn.Module):
    def __init__(self, num_speakers=7, embedding_dim=64, dropout=0.3):
        super().__init__()
        self.embedding  = nn.Embedding(num_speakers, embedding_dim)
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(), nn.Dropout(dropout)
        )

    def forward(self, speaker_ids):
        return self.projection(self.embedding(speaker_ids))


class CrossModalAttention(nn.Module):
    def __init__(self, hidden_size=768, num_heads=8, dropout=0.3):
        super().__init__()
        self.text_to_audio = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        self.audio_to_text = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1   = nn.LayerNorm(hidden_size)
        self.norm2   = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text_feat, audio_feat):
        t = text_feat.unsqueeze(1)
        a = audio_feat.unsqueeze(1)
        ta, _ = self.text_to_audio(t, a, a)
        at, _ = self.audio_to_text(a, t, t)
        return (self.norm1(t + self.dropout(ta)).squeeze(1),
                self.norm2(a + self.dropout(at)).squeeze(1))


class GatedFusion(nn.Module):
    def __init__(self, hidden_size=768, dropout=0.3):
        super().__init__()
        self.gate    = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size), nn.Sigmoid()
        )
        self.fusion  = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text_feat, audio_feat):
        c = torch.cat([text_feat, audio_feat], dim=-1)
        return self.dropout(
            self.gate(c) * text_feat +
            (1 - self.gate(c)) * audio_feat +
            self.fusion(c)
        )


class MELDBaseModel(nn.Module):
    """Text + Audio + Speaker (matches best_multimodal_model.pt)"""
    def __init__(self, num_emotions=7, hidden_size=768,
                 speaker_emb_dim=64, num_speakers=7, dropout=0.3):
        super().__init__()
        self.text_encoder     = AutoModel.from_pretrained(
            TEXT_MODEL_PATH, local_files_only=True, low_cpu_mem_usage=True
        )
        self.audio_encoder    = AudioEncoder(hidden_size, dropout)
        self.speaker_embedding = SpeakerEmbedding(num_speakers, speaker_emb_dim, dropout)
        self.cross_attention  = CrossModalAttention(hidden_size, 8, dropout)
        self.gated_fusion     = GatedFusion(hidden_size, dropout)
        self.dropout          = nn.Dropout(dropout)
        self.emotion_head = nn.Sequential(
            nn.Linear(hidden_size + speaker_emb_dim, 256),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, num_emotions)
        )

    def forward(self, input_ids, attention_mask, waveform,
                speaker_ids, **kwargs):
        text  = self.text_encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state[:, 0, :]
        text  = self.dropout(text)
        audio = self.audio_encoder(waveform)
        ta, at = self.cross_attention(text, audio)
        fused  = self.dropout(self.gated_fusion(ta, at))
        spk    = self.speaker_embedding(speaker_ids)
        emo_logits = self.emotion_head(torch.cat([fused, spk], dim=-1))
        # Base model returns tuple (emo, sent) — return just emo
        return emo_logits, None


class MELDVideoModel(nn.Module):
    """Text + Audio + Video + Speaker"""
    def __init__(self, num_emotions=7, hidden_size=768,
                 video_hidden=256, speaker_emb_dim=64,
                 num_speakers=7, dropout=0.3):
        super().__init__()
        self.text_encoder     = AutoModel.from_pretrained(
            TEXT_MODEL_PATH, local_files_only=True, low_cpu_mem_usage=True
        )
        self.audio_encoder    = AudioEncoder(hidden_size, dropout)
        self.video_encoder    = VideoEncoder(2048, video_hidden, dropout)
        self.speaker_embedding = SpeakerEmbedding(num_speakers, speaker_emb_dim, dropout)
        self.cross_attention  = CrossModalAttention(hidden_size, 8, dropout)
        self.gated_fusion     = GatedFusion(hidden_size, dropout)
        self.dropout          = nn.Dropout(dropout)
        clf_input = hidden_size + video_hidden + speaker_emb_dim
        self.emotion_head = nn.Sequential(
            nn.Linear(clf_input, 256),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, num_emotions)
        )

    def forward(self, input_ids, attention_mask, waveform,
                video, speaker_ids, **kwargs):
        text  = self.text_encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state[:, 0, :]
        text  = self.dropout(text)
        audio = self.audio_encoder(waveform)
        ta, at = self.cross_attention(text, audio)
        fused  = self.dropout(self.gated_fusion(ta, at))
        vid    = self.video_encoder(video)
        spk    = self.speaker_embedding(speaker_ids)
        return self.emotion_head(torch.cat([fused, vid, spk], dim=-1))


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation helpers
# ══════════════════════════════════════════════════════════════════════════════

def run_evaluation(model, loader, use_video=False, video_cache=None):
    model.eval()
    all_preds, all_labels = [], []
    zero_video = torch.zeros(2048).to(DEVICE)

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            waveform       = batch['waveform'].to(DEVICE).float()
            speaker_ids    = batch['speaker_id'].to(DEVICE)
            emo_labels     = batch['emotion_label'].to(DEVICE)

            if use_video and video_cache is not None:
                # Get video features from cache using keys
                keys = batch.get('key', [None] * len(emo_labels))
                vids = []
                for k in keys:
                    if k and k in video_cache:
                        vids.append(video_cache[k])
                    else:
                        vids.append(torch.zeros(2048))
                video = torch.stack(vids).to(DEVICE).float()
                out = model(input_ids, attention_mask, waveform,
                            video, speaker_ids)
            else:
                out = model(input_ids, attention_mask, waveform, speaker_ids)
                if isinstance(out, tuple):
                    out = out[0]

            all_preds.extend(torch.argmax(out, 1).cpu().numpy())
            all_labels.extend(emo_labels.cpu().numpy())

    return np.array(all_labels), np.array(all_preds)


def print_results(labels, preds, model_name):
    wf1 = f1_score(labels, preds, average='weighted')
    print(f"\n{'='*60}")
    print(f"  {model_name}")
    print(f"  Weighted F1: {wf1:.4f}")
    print(f"{'='*60}")
    print(classification_report(
        labels, preds,
        target_names=EMOTION_LABELS,
        zero_division=0
    ))
    return wf1


def plot_confusion_matrix(labels, preds, model_name, filename):
    cm   = confusion_matrix(labels, preds)
    cm_n = cm.astype('float') / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle(f'Confusion Matrix — {model_name}',
                 fontsize=14, fontweight='bold')

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=EMOTION_LABELS, yticklabels=EMOTION_LABELS,
                ax=axes[0], linewidths=0.5)
    axes[0].set_title('Raw Counts')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')

    sns.heatmap(cm_n, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=EMOTION_LABELS, yticklabels=EMOTION_LABELS,
                ax=axes[1], linewidths=0.5, vmin=0, vmax=1)
    axes[1].set_title('Normalized (row %)')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True')

    plt.tight_layout()
    out = os.path.join(FIG_DIR, filename)
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Confusion matrix saved → {out}")


def plot_per_class_f1(labels, preds, model_name, filename):
    f1s    = f1_score(labels, preds, average=None, zero_division=0)
    colors = ['#2EC4B6' if f >= 0.70 else '#1A3A6B' if f >= 0.40
              else '#FF9F1C' for f in f1s]

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(EMOTION_LABELS, f1s, color=colors,
                  edgecolor='white', linewidth=0.8)

    for bar, val in zip(bars, f1s):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom',
                fontsize=10, fontweight='bold', color='#1E293B')

    ax.set_ylim(0, 1.05)
    ax.set_title(f'Per-Class F1 — {model_name}',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Emotion Class')
    ax.set_ylabel('F1 Score')
    ax.axhline(y=0.70, color='#22C55E', linestyle='--',
               linewidth=1, label='F1 = 0.70')
    ax.legend(fontsize=9)
    ax.set_facecolor('#F8FAFC')
    fig.patch.set_facecolor('white')
    plt.tight_layout()

    out = os.path.join(FIG_DIR, filename)
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Per-class F1 saved → {out}")


def plot_comparison(labels_a, preds_a, labels_b, preds_b):
    f1_a = f1_score(labels_a, preds_a, average=None, zero_division=0)
    f1_b = f1_score(labels_b, preds_b, average=None, zero_division=0)

    x = np.arange(len(EMOTION_LABELS))
    w = 0.35
    fig, ax = plt.subplots(figsize=(12, 5))

    b1 = ax.bar(x - w/2, f1_a, w,
                label='T+A+Speaker (base)', color='#1A3A6B')
    b2 = ax.bar(x + w/2, f1_b, w,
                label='T+A+Video+Speaker', color='#2EC4B6')

    for bars in [b1, b2]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.01,
                    f'{bar.get_height():.2f}',
                    ha='center', va='bottom',
                    fontsize=8.5, color='#1E293B')

    ax.set_xticks(x)
    ax.set_xticklabels(EMOTION_LABELS)
    ax.set_ylim(0, 1.1)
    ax.set_title('Per-Class F1: MELD Base vs MELD + Video',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Emotion Class')
    ax.set_ylabel('F1 Score')
    ax.legend(fontsize=10)
    ax.set_facecolor('#F8FAFC')
    fig.patch.set_facecolor('white')
    plt.tight_layout()

    out = os.path.join(FIG_DIR, 'meld_comparison_f1.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Comparison chart saved → {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_model(use_video=False):
    model_name = "T+A+Video+Speaker" if use_video else "T+A+Speaker"
    ckpt_path  = VIDEO_CKPT if use_video else BASE_CKPT
    fig_suffix = "video" if use_video else "base"

    print(f"\n{'='*60}")
    print(f"  Evaluating: {model_name}")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  Device    : {DEVICE}")
    print(f"{'='*60}")

    if not os.path.exists(ckpt_path):
        print(f"  ✗ Checkpoint not found: {ckpt_path}")
        return None, None

    # Load data
    print("\nLoading test data...")
    _, _, test_loader = get_dataloaders(
        batch_size=BATCH_SIZE,
        max_text_len=MAX_LEN,
        context_window=CONTEXT_WIN,
    )

    # Load video cache if needed
    video_cache = None
    if use_video:
        print(f"Loading video cache...")
        if not os.path.exists(VIDEO_CACHE):
            print(f"  ✗ Video cache not found: {VIDEO_CACHE}")
            print(f"  Run scripts/extract_meld_video.py first!")
            return None, None
        video_cache = torch.load(VIDEO_CACHE, weights_only=True)
        print(f"  Video cache: {len(video_cache)} entries")

    # Build model
    print("Loading model...")
    if use_video:
        model = MELDVideoModel(
            num_emotions=NUM_EMOTIONS, hidden_size=768,
            video_hidden=256, speaker_emb_dim=64,
            num_speakers=NUM_SPEAKERS, dropout=0.3
        )
    else:
        model = MELDBaseModel(
            num_emotions=NUM_EMOTIONS, hidden_size=768,
            speaker_emb_dim=64, num_speakers=NUM_SPEAKERS,
            dropout=0.3
        )

    model.load_state_dict(
        torch.load(ckpt_path, map_location=DEVICE, weights_only=True),
        strict=False
    )
    model = model.to(DEVICE).float()

    # Run evaluation
    print("Running inference...")
    labels, preds = run_evaluation(
        model, test_loader, use_video, video_cache
    )

    wf1 = print_results(labels, preds, model_name)
    plot_confusion_matrix(labels, preds, model_name,
                          f'meld_confusion_matrix_{fig_suffix}.png')
    plot_per_class_f1(labels, preds, model_name,
                      f'meld_per_class_f1_{fig_suffix}.png')

    print(f"\n✅ Done — Weighted F1: {wf1:.4f}")
    return labels, preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_video', action='store_true',
                        help='Evaluate the video model')
    parser.add_argument('--compare',   action='store_true',
                        help='Compare base vs video model')
    args = parser.parse_args()

    if args.compare:
        print("\n📊 Comparing: MELD base vs MELD + Video\n")
        labels_a, preds_a = evaluate_model(use_video=False)
        labels_b, preds_b = evaluate_model(use_video=True)
        if labels_a is not None and labels_b is not None:
            plot_comparison(labels_a, preds_a, labels_b, preds_b)
            wf1_a = f1_score(labels_a, preds_a, average='weighted')
            wf1_b = f1_score(labels_b, preds_b, average='weighted')
            print(f"\n{'='*60}")
            print(f"  COMPARISON SUMMARY — MELD")
            print(f"{'='*60}")
            print(f"  T+A+Speaker (base) : {wf1_a:.4f}")
            print(f"  T+A+Video+Speaker  : {wf1_b:.4f}")
            print(f"  Video gain         : {wf1_b - wf1_a:+.4f}")
            print(f"  Figures saved to   : {FIG_DIR}")
    else:
        evaluate_model(use_video=args.use_video)


if __name__ == '__main__':
    main()