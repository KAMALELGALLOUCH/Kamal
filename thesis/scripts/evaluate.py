"""
Evaluate Best Model — Multimodal Emotion Recognition on MELD
=============================================================
Loads the best saved checkpoint and produces:
  - Full classification report
  - Per-class F1 scores
  - Confusion matrix (text + PNG)
  - Per-class F1 bar chart (PNG)
  - Results saved to .txt

Run: CUDA_VISIBLE_DEVICES=3 python scripts/evaluate.py
"""

import os
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    f1_score, classification_report, confusion_matrix
)

sys.path.append('/data8/luoyan/Kamal/thesis')
from data.meld_dataset import get_dataloaders, EMOTION2ID
from models.multimodal_model import MultimodalEmotionModel

# ── Config ─────────────────────────────────────────────────────
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 32
MAX_LEN    = 128
CONTEXT    = 3
MODEL_PATH = '/data8/luoyan/Kamal/thesis/outputs/best_multimodal_model.pt'
FIG_DIR    = '/data8/luoyan/Kamal/thesis/outputs/figures'
os.makedirs(FIG_DIR, exist_ok=True)

ID2EMOTION = {v: k for k, v in EMOTION2ID.items()}
EMOTION_LABELS = [ID2EMOTION[i] for i in range(len(ID2EMOTION))]

EMOTION_EMOJI = {
    'neutral':  '😐',
    'joy':      '😄',
    'surprise': '😲',
    'anger':    '😠',
    'sadness':  '😢',
    'disgust':  '🤢',
    'fear':     '😨',
}


# ══════════════════════════════════════════════════════════════
# Figure helpers
# ══════════════════════════════════════════════════════════════

def plot_confusion_matrix(labels, preds):
    cm   = confusion_matrix(labels, preds)
    cm_n = cm.astype('float') / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle('Confusion Matrix — MELD (Text + Audio + Speaker + Context)',
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
    out = os.path.join(FIG_DIR, 'meld_confusion_matrix.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Confusion matrix saved → {out}")


def plot_per_class_f1(labels, preds):
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
    ax.set_title('Per-Class F1 Score — MELD', fontsize=13, fontweight='bold')
    ax.set_xlabel('Emotion Class')
    ax.set_ylabel('F1 Score')
    ax.axhline(y=0.70, color='#22C55E', linestyle='--',
               linewidth=1, label='F1 = 0.70 threshold')
    ax.legend(fontsize=9)
    ax.set_facecolor('#F8FAFC')
    fig.patch.set_facecolor('white')
    plt.tight_layout()

    out = os.path.join(FIG_DIR, 'meld_per_class_f1.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Per-class F1 chart saved → {out}")


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

def evaluate_model():
    print(f"Device    : {DEVICE}")
    print(f"Checkpoint: {MODEL_PATH}\n")

    # ── load data ──────────────────────────────────────────────
    _, _, test_loader = get_dataloaders(
        batch_size=BATCH_SIZE,
        max_text_len=MAX_LEN,
        context_window=CONTEXT,
    )

    # ── load model ─────────────────────────────────────────────
    model = MultimodalEmotionModel(
        num_emotions=7,
        num_sentiments=3,
        hidden_size=768,
        speaker_emb_dim=64,
        num_speakers=7,
        dropout=0.0,
        freeze_text=False,
    ).to(DEVICE)
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    )
    model = model.float()
    model.eval()

    total = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total:,}\n")

    # ── run inference ──────────────────────────────────────────
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in test_loader:
            input_ids      = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            waveform       = batch['waveform'].to(DEVICE).float()
            speaker_ids    = batch['speaker_id'].to(DEVICE)
            emo_labels     = batch['emotion_label'].to(DEVICE)

            emo_logits, _ = model(
                input_ids, attention_mask, waveform, speaker_ids
            )
            preds = torch.argmax(emo_logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(emo_labels.cpu().numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    labels      = list(ID2EMOTION.keys())
    label_names = [ID2EMOTION[i] for i in labels]

    # ── classification report ──────────────────────────────────
    print("=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    report = classification_report(
        all_labels, all_preds,
        labels=labels, target_names=label_names, zero_division=0
    )
    print(report)

    weighted_f1 = f1_score(all_labels, all_preds, average='weighted')
    macro_f1    = f1_score(all_labels, all_preds, average='macro')
    print(f"Weighted F1 : {weighted_f1:.4f}")
    print(f"Macro F1    : {macro_f1:.4f}")

    # ── per-class summary ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("PER-CLASS F1 SUMMARY")
    print("=" * 60)
    per_class = f1_score(all_labels, all_preds, average=None, zero_division=0)
    for i, f1 in enumerate(per_class):
        emotion = ID2EMOTION[i]
        emoji   = EMOTION_EMOJI.get(emotion, '')
        bar     = '█' * int(f1 * 20)
        print(f"  {emoji} {emotion:<10} {f1:.4f}  {bar}")

    # ── confusion matrix (text) ────────────────────────────────
    print("\n" + "=" * 60)
    print("CONFUSION MATRIX")
    print("=" * 60)
    cm = confusion_matrix(all_labels, all_preds, labels=labels)
    header = f"{'':>10}" + "".join(f"{ID2EMOTION[i]:>10}" for i in labels)
    print(header)
    for i, row in zip(labels, cm):
        print(f"{ID2EMOTION[i]:>10}" + "".join(f"{v:>10}" for v in row))

    # ── PNG figures ────────────────────────────────────────────
    print("\nGenerating figures...")
    plot_confusion_matrix(all_labels, all_preds)
    plot_per_class_f1(all_labels, all_preds)

    # ── save results to txt ────────────────────────────────────
    out_path = '/data8/luoyan/Kamal/thesis/outputs/meld_evaluation_results.txt'
    with open(out_path, 'w') as f:
        f.write("MELD Evaluation Results\n")
        f.write("=" * 60 + "\n")
        f.write(f"Weighted F1 : {weighted_f1:.4f}\n")
        f.write(f"Macro F1    : {macro_f1:.4f}\n\n")
        f.write(report)
    print(f"\n✅ Results saved → {out_path}")
    print(f"✅ Figures saved → {FIG_DIR}/meld_*.png")
    print(f"\n📊 Final Score — Weighted F1: {weighted_f1:.4f}")


if __name__ == '__main__':
    evaluate_model()