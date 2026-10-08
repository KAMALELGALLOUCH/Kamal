"""
Evaluation Script: IEMOCAP Emotion Recognition
================================================
Generates confusion matrix, per-class F1, and classification report.
Works with both checkpoints:
  - best_iemocap_model.pt       (Text + Audio + Speaker + Context)
  - best_iemocap_video_model.pt (Text + Audio + MOCAP + Speaker + Context)

Usage:
  # Evaluate text+audio model
  CUDA_VISIBLE_DEVICES=3 python scripts/evaluate_iemocap.py

  # Evaluate video (MOCAP) model
  CUDA_VISIBLE_DEVICES=3 python scripts/evaluate_iemocap.py --use_mocap

  # Compare both side by side
  CUDA_VISIBLE_DEVICES=3 python scripts/evaluate_iemocap.py --compare
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
from data.iemocap_dataset import (
    get_dataloaders, NUM_EMOTIONS, NUM_SPEAKERS,
    ID2EMOTION, MOCAP_FEATURES, MAX_MOCAP_LEN
)

# ── Config ──────────────────────────────────────────────────────────────────
DEVICE          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE      = 16
MAX_LEN         = 128
CONTEXT_WIN     = 3
TEST_SESSION    = 5
TEXT_MODEL_PATH = '/data8/zhangxin/comp5423/gte-modernbert-base'
SAVE_DIR        = '/data8/luoyan/Kamal/thesis/outputs/iemocap'
FIG_DIR         = '/data8/luoyan/Kamal/thesis/outputs/figures'
os.makedirs(FIG_DIR, exist_ok=True)

EMOTION_LABELS  = [ID2EMOTION[i] for i in range(NUM_EMOTIONS)]


# ══════════════════════════════════════════════════════════════════════════
# Model Components (same as training scripts)
# ══════════════════════════════════════════════════════════════════════════

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


class MOCAPEncoder(nn.Module):
    def __init__(self, input_size=6, hidden_size=128,
                 num_layers=2, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        self.projection = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(), nn.Dropout(dropout)
        )

    def forward(self, mocap):
        _, hidden = self.gru(mocap)
        combined  = torch.cat([hidden[-2], hidden[-1]], dim=-1)
        return self.projection(combined)


class SpeakerEmbedding(nn.Module):
    def __init__(self, num_speakers=10, embedding_dim=64, dropout=0.3):
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


class ContextAggregator(nn.Module):
    def __init__(self, hidden_size=768, dropout=0.3):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            hidden_size, num_heads=8, dropout=dropout, batch_first=True
        )
        self.norm    = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, current_feat, context_feats):
        q = current_feat.unsqueeze(1)
        attended, _ = self.attention(q, context_feats, context_feats)
        return self.norm(current_feat + self.dropout(attended.squeeze(1)))


class IEMOCAPModel(nn.Module):
    """Text + Audio + Speaker + Context
    Architecture exactly matches best_iemocap_model.pt checkpoint keys:
      text_encoder, audio_encoder, speaker_embedding,
      cross_attention, gated_fusion, emotion_head
    Note: context is handled at data level (context_window in dataloader),
    not as a separate nn.Module — so no context_aggregator layer.
    """
    def __init__(self, num_emotions=5, hidden_size=768,
                 speaker_emb_dim=64, num_speakers=10,
                 dropout=0.3):
        super().__init__()
        self.text_encoder    = AutoModel.from_pretrained(
            TEXT_MODEL_PATH, local_files_only=True, low_cpu_mem_usage=True
        )
        self.audio_encoder     = AudioEncoder(hidden_size, dropout)
        self.speaker_embedding = SpeakerEmbedding(num_speakers, speaker_emb_dim, dropout)
        self.cross_attention   = CrossModalAttention(hidden_size, 8, dropout)
        self.gated_fusion      = GatedFusion(hidden_size, dropout)
        self.dropout           = nn.Dropout(dropout)
        # classifier: 768 (fused) + 64 (speaker) = 832
        clf_input = hidden_size + speaker_emb_dim
        self.emotion_head = nn.Sequential(
            nn.Linear(clf_input, 256), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(256, num_emotions)
        )

    def forward(self, input_ids, attention_mask, waveform,
                speaker_ids, ctx_input_ids=None,
                ctx_attention_mask=None, ctx_waveform=None, **kwargs):
        # text
        text  = self.text_encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state[:, 0, :]
        text  = self.dropout(text)
        # audio
        audio = self.audio_encoder(waveform)
        # cross-modal attention + gated fusion
        ta, at = self.cross_attention(text, audio)
        fused  = self.dropout(self.gated_fusion(ta, at))
        # speaker
        spk = self.speaker_embedding(speaker_ids)
        # classify
        return self.emotion_head(torch.cat([fused, spk], dim=-1))


class IEMOCAPVideoModel(nn.Module):
    """Text + Audio + MOCAP + Speaker + Context"""
    def __init__(self, num_emotions=5, hidden_size=768,
                 mocap_hidden=128, speaker_emb_dim=64,
                 num_speakers=10, dropout=0.3):
        super().__init__()
        self.text_encoder    = AutoModel.from_pretrained(
            TEXT_MODEL_PATH, local_files_only=True, low_cpu_mem_usage=True
        )
        self.audio_encoder   = AudioEncoder(hidden_size, dropout)
        self.mocap_encoder   = MOCAPEncoder(MOCAP_FEATURES, mocap_hidden, 2, dropout)
        self.speaker_embedding     = SpeakerEmbedding(num_speakers, speaker_emb_dim, dropout)
        self.cross_attention = CrossModalAttention(hidden_size, 8, dropout)
        self.gated_fusion    = GatedFusion(hidden_size, dropout)
        self.context_aggregator     = ContextAggregator(hidden_size, dropout)
        self.dropout         = nn.Dropout(dropout)
        clf_input = hidden_size + mocap_hidden + speaker_emb_dim  # 960
        self.emotion_head = nn.Sequential(
            nn.Linear(clf_input, 256), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(256, num_emotions)
        )

    def encode_utterance(self, input_ids, attention_mask, waveform):
        text  = self.text_encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state[:, 0, :]
        text  = self.dropout(text)
        audio = self.audio_encoder(waveform)
        ta, at = self.cross_attention(text, audio)
        return self.dropout(self.gated_fusion(ta, at))

    def forward(self, input_ids, attention_mask, waveform,
                mocap, speaker_ids, ctx_input_ids=None,
                ctx_attention_mask=None, ctx_waveform=None):
        fused = self.encode_utterance(input_ids, attention_mask, waveform)
        if ctx_input_ids is not None and ctx_input_ids.size(1) > 0:
            B, C, L = ctx_input_ids.shape
            ctx_f = self.encode_utterance(
                ctx_input_ids.view(B*C, L),
                ctx_attention_mask.view(B*C, L),
                ctx_waveform.view(B*C, ctx_waveform.size(-1))
            ).view(B, C, -1)
            fused = self.context_aggregator(fused, ctx_f)
        mocap_f = self.mocap_encoder(mocap)
        spk     = self.speaker_embedding(speaker_ids)
        return self.emotion_head(torch.cat([fused, mocap_f, spk], dim=-1))


# ══════════════════════════════════════════════════════════════════════════
# Evaluation helpers
# ══════════════════════════════════════════════════════════════════════════

def _to_device(batch, use_mocap):
    ids   = batch['input_ids'].to(DEVICE)
    mask  = batch['attention_mask'].to(DEVICE)
    wav   = batch['waveform'].to(DEVICE).float()
    spk   = batch['speaker_id'].to(DEVICE)
    lbl   = batch['emotion_label'].to(DEVICE)
    mocap = batch['mocap'].to(DEVICE).float() if use_mocap else None

    ctx_ids  = batch.get('ctx_input_ids')
    ctx_mask = batch.get('ctx_attention_mask')
    ctx_wav  = batch.get('ctx_waveform')
    if ctx_ids is not None:
        ctx_ids  = ctx_ids.to(DEVICE)
        ctx_mask = ctx_mask.to(DEVICE)
        ctx_wav  = ctx_wav.to(DEVICE).float()

    return ids, mask, wav, mocap, spk, lbl, ctx_ids, ctx_mask, ctx_wav


def run_evaluation(model, loader, use_mocap=False):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            (ids, mask, wav, mocap, spk, lbl,
             ctx_ids, ctx_mask, ctx_wav) = _to_device(batch, use_mocap)

            if use_mocap:
                logits = model(ids, mask, wav, mocap, spk,
                               ctx_ids, ctx_mask, ctx_wav)
            else:
                logits = model(ids, mask, wav, spk,
                               ctx_ids, ctx_mask, ctx_wav)

            all_preds.extend(torch.argmax(logits, 1).cpu().numpy())
            all_labels.extend(lbl.cpu().numpy())

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
    cm_n = cm.astype('float') / cm.sum(axis=1, keepdims=True)  # normalize

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'Confusion Matrix — {model_name}', fontsize=14, fontweight='bold')

    # raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=EMOTION_LABELS, yticklabels=EMOTION_LABELS,
                ax=axes[0], linewidths=0.5)
    axes[0].set_title('Raw Counts')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')

    # normalized
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
    return out


def plot_per_class_f1(labels, preds, model_name, filename):
    f1s = f1_score(labels, preds, average=None, zero_division=0)

    fig, ax = plt.subplots(figsize=(9, 4))
    colors  = ['#2EC4B6' if f >= 0.70 else '#1A3A6B' if f >= 0.50
               else '#FF9F1C' for f in f1s]
    bars = ax.bar(EMOTION_LABELS, f1s, color=colors, edgecolor='white', linewidth=0.8)

    for bar, val in zip(bars, f1s):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom',
                fontsize=10, fontweight='bold', color='#1E293B')

    ax.set_ylim(0, 1.05)
    ax.set_title(f'Per-Class F1 Score — {model_name}', fontsize=13, fontweight='bold')
    ax.set_xlabel('Emotion Class')
    ax.set_ylabel('F1 Score')
    ax.axhline(y=0.70, color='#22C55E', linestyle='--', linewidth=1,
               label='F1 = 0.70 threshold')
    ax.legend(fontsize=9)
    ax.set_facecolor('#F8FAFC')
    fig.patch.set_facecolor('white')
    plt.tight_layout()

    out = os.path.join(FIG_DIR, filename)
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Per-class F1 chart saved → {out}")
    return out


def plot_comparison(labels_a, preds_a, labels_b, preds_b):
    """Side-by-side per-class F1 comparison (no-MOCAP vs MOCAP)."""
    f1_a = f1_score(labels_a, preds_a, average=None, zero_division=0)
    f1_b = f1_score(labels_b, preds_b, average=None, zero_division=0)

    x    = np.arange(len(EMOTION_LABELS))
    w    = 0.35
    fig, ax = plt.subplots(figsize=(11, 5))

    b1 = ax.bar(x - w/2, f1_a, w, label='T+A+Speaker+Context', color='#1A3A6B')
    b2 = ax.bar(x + w/2, f1_b, w, label='T+A+MOCAP+Speaker+Context', color='#2EC4B6')

    for bars in [b1, b2]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.01,
                    f'{bar.get_height():.2f}',
                    ha='center', va='bottom', fontsize=8.5, color='#1E293B')

    ax.set_xticks(x)
    ax.set_xticklabels(EMOTION_LABELS)
    ax.set_ylim(0, 1.1)
    ax.set_title('Per-Class F1: With vs Without MOCAP', fontsize=13, fontweight='bold')
    ax.set_xlabel('Emotion Class')
    ax.set_ylabel('F1 Score')
    ax.legend()
    ax.set_facecolor('#F8FAFC')
    fig.patch.set_facecolor('white')
    plt.tight_layout()

    out = os.path.join(FIG_DIR, 'iemocap_comparison_f1.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Comparison chart saved → {out}")


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

def evaluate_model(use_mocap=False):
    model_name  = "T+A+MOCAP+Speaker+Context" if use_mocap else "T+A+Speaker+Context"
    ckpt_name   = "best_iemocap_video_model.pt" if use_mocap else "best_iemocap_model.pt"
    ckpt_path   = os.path.join(SAVE_DIR, ckpt_name)
    fig_suffix  = "video" if use_mocap else "base"

    print(f"\n{'='*60}")
    print(f"  Evaluating: {model_name}")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  Device    : {DEVICE}")
    print(f"{'='*60}")

    if not os.path.exists(ckpt_path):
        print(f"  ✗ Checkpoint not found: {ckpt_path}")
        print(f"  Run the training script first.")
        return None, None

    # load data
    print("\nLoading test data...")
    _, _, test_loader = get_dataloaders(
        test_session=TEST_SESSION,
        batch_size=BATCH_SIZE,
        max_text_len=MAX_LEN,
        context_window=CONTEXT_WIN,
        use_mocap=use_mocap,
    )
    print(f"Test batches: {len(test_loader)}")

    # build model
    print("Loading model...")
    if use_mocap:
        model = IEMOCAPVideoModel(
            num_emotions=NUM_EMOTIONS, hidden_size=768,
            mocap_hidden=128, speaker_emb_dim=64,
            num_speakers=NUM_SPEAKERS, dropout=0.3
        )
    else:
        model = IEMOCAPModel(
            num_emotions=NUM_EMOTIONS, hidden_size=768,
            speaker_emb_dim=64, num_speakers=NUM_SPEAKERS,
            dropout=0.3
        )

    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
    model = model.to(DEVICE).float()

    # run evaluation
    print("Running inference on test set...")
    labels, preds = run_evaluation(model, test_loader, use_mocap)

    # print results
    wf1 = print_results(labels, preds, model_name)

    # plots
    plot_confusion_matrix(labels, preds, model_name,
                          f'iemocap_confusion_matrix_{fig_suffix}.png')
    plot_per_class_f1(labels, preds, model_name,
                      f'iemocap_per_class_f1_{fig_suffix}.png')

    print(f"\n✅ Evaluation complete — Weighted F1: {wf1:.4f}")
    return labels, preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_mocap', action='store_true',
                        help='Evaluate the MOCAP video model')
    parser.add_argument('--compare',   action='store_true',
                        help='Compare both models side by side')
    args = parser.parse_args()

    if args.compare:
        print("\n📊 Running comparison: base model vs MOCAP model\n")
        labels_a, preds_a = evaluate_model(use_mocap=False)
        labels_b, preds_b = evaluate_model(use_mocap=True)
        if labels_a is not None and labels_b is not None:
            plot_comparison(labels_a, preds_a, labels_b, preds_b)
            wf1_a = f1_score(labels_a, preds_a, average='weighted')
            wf1_b = f1_score(labels_b, preds_b, average='weighted')
            print(f"\n{'='*60}")
            print(f"  COMPARISON SUMMARY")
            print(f"{'='*60}")
            print(f"  T+A+Speaker+Context      : {wf1_a:.4f}")
            print(f"  T+A+MOCAP+Speaker+Context: {wf1_b:.4f}")
            print(f"  MOCAP gain               : {wf1_b - wf1_a:+.4f}")
            print(f"  Figures saved to         : {FIG_DIR}")
    else:
        evaluate_model(use_mocap=args.use_mocap)


if __name__ == '__main__':
    main()