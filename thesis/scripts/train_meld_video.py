"""
Training Script: MELD with Video Features
==========================================
Text + Audio + Video(ResNet-50) + Speaker + Context
Single GPU training

Run: CUDA_VISIBLE_DEVICES=3 python scripts/train_meld_video.py

Architecture:
  - Text   : ModernBERT → 768-dim
  - Audio  : CNN + MelSpec → 768-dim
  - Video  : ResNet-50 features → FC → 256-dim
  - Fusion : Cross-Modal Attention + Gated Fusion + Context
  - Head   : 768 + 256 + 64 (speaker) = 1088-dim → 7 emotions
"""

import os
import sys
import logging
import torch
import torch.nn as nn
import torchaudio.transforms as T
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup, AutoModel
from sklearn.metrics import f1_score, classification_report
from tqdm import tqdm

sys.path.append('/data8/luoyan/Kamal/thesis')
from data.meld_dataset import (
    EMOTION2ID, SENTIMENT2ID, SPEAKER2ID,
    NUM_SPEAKERS
)

# Derive constants from dicts
NUM_EMOTIONS = len(EMOTION2ID)
ID2EMOTION   = {v: k for k, v in EMOTION2ID.items()}

# ── Config ────────────────────────────────────────────────────────────────────
DEVICE         = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS         = 15
BATCH_SIZE     = 16
MAX_LEN        = 128
CONTEXT_WIN    = 3
UNFREEZE_EPOCH = 4
VIDEO_DIM      = 2048   # ResNet-50 output dim
VIDEO_HIDDEN   = 256    # projected video dim
SAVE_DIR       = '/data8/luoyan/Kamal/thesis/outputs/meld'
LOG_PATH       = '/data8/luoyan/Kamal/thesis/logs/train_meld_video.log'
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

TEXT_MODEL_PATH  = '/data8/zhangxin/comp5423/gte-modernbert-base'
VIDEO_CACHE_PATH = '/data8/luoyan/Kamal/thesis/data/meld_video_cache.pt'

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger(__name__)

# ── Class weights (MELD is imbalanced) ───────────────────────────────────────
EMOTION_COUNTS  = [4710, 1743, 1205, 1109, 683, 268, 268]  # 7 classes
EMOTION_WEIGHTS = torch.tensor(
    [1.0 / c for c in EMOTION_COUNTS], dtype=torch.float32
)
EMOTION_WEIGHTS = EMOTION_WEIGHTS / EMOTION_WEIGHTS.sum()


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
    """Projects ResNet-50 2048-dim features to VIDEO_HIDDEN-dim."""
    def __init__(self, input_dim=2048, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, video_feat):
        return self.projection(video_feat)   # [B, hidden_dim]


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


class MELDVideoModel(nn.Module):
    """
    Full Multimodal Model for MELD:
    Text + Audio + Video + Speaker + Context

    Architecture:
    ┌──────────────┐  ┌─────────────┐  ┌──────────────┐
    │ ModernBERT   │  │ CNN+MelSpec │  │ ResNet-50    │
    │   768-dim    │  │   768-dim   │  │   2048-dim   │
    └──────┬───────┘  └──────┬──────┘  └──────┬───────┘
           └──── Cross-Modal Attention ────────┘
                        │
                  Gated Fusion (768)
                        │
              cat[fused(768) + video(256) + speaker(64)]
                        │
                   = 1088-dim
                        │
                FC → 7 emotions (MELD)
    """

    def __init__(self, num_emotions=7, hidden_size=768,
                 video_hidden=256, speaker_emb_dim=64,
                 num_speakers=7, dropout=0.3, freeze_text=True):
        super().__init__()

        self.text_encoder = AutoModel.from_pretrained(
            TEXT_MODEL_PATH, local_files_only=True, low_cpu_mem_usage=True
        )
        if freeze_text:
            for p in self.text_encoder.parameters():
                p.requires_grad = False

        self.audio_encoder    = AudioEncoder(hidden_size, dropout)
        self.video_encoder    = VideoEncoder(2048, video_hidden, dropout)
        self.speaker_embedding = SpeakerEmbedding(num_speakers, speaker_emb_dim, dropout)
        self.cross_attention  = CrossModalAttention(hidden_size, 8, dropout)
        self.gated_fusion     = GatedFusion(hidden_size, dropout)
        self.dropout          = nn.Dropout(dropout)

        # classifier: 768 (fused) + 256 (video) + 64 (speaker) = 1088
        clf_input = hidden_size + video_hidden + speaker_emb_dim
        log.info(f"[Model] Classifier input size: {clf_input}")

        self.emotion_head = nn.Sequential(
            nn.Linear(clf_input, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_emotions)
        )

    def forward(self, input_ids, attention_mask, waveform,
                video, speaker_ids):
        # ── text ──────────────────────────────────────────────────────────
        text_feat = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).last_hidden_state[:, 0, :]
        text_feat = self.dropout(text_feat)          # [B, 768]

        # ── audio ─────────────────────────────────────────────────────────
        audio_feat = self.audio_encoder(waveform)    # [B, 768]

        # ── cross-modal attention + gated fusion ──────────────────────────
        text_att, audio_att = self.cross_attention(text_feat, audio_feat)
        fused = self.dropout(self.gated_fusion(text_att, audio_att))
        # [B, 768]

        # ── video ─────────────────────────────────────────────────────────
        video_feat = self.video_encoder(video)       # [B, 256]

        # ── speaker ───────────────────────────────────────────────────────
        spk_emb = self.speaker_embedding(speaker_ids)  # [B, 64]

        # ── combine & classify ────────────────────────────────────────────
        combined = torch.cat([fused, video_feat, spk_emb], dim=-1)  # [B, 1088]
        return self.emotion_head(combined)


# ══════════════════════════════════════════════════════════════════════════════
# Training & Evaluation
# ══════════════════════════════════════════════════════════════════════════════

def train_epoch(model, loader, optimizer, scheduler, criterion):
    model.train()
    total_loss, all_preds, all_labels = 0, [], []

    for batch in tqdm(loader, desc='  Training'):
        input_ids      = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        waveform       = batch['waveform'].to(DEVICE).float()
        video          = batch['video'].to(DEVICE).float()
        speaker_ids    = batch['speaker_id'].to(DEVICE)
        emo_labels     = batch['emotion_label'].to(DEVICE)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask, waveform, video, speaker_ids)
        loss   = criterion(logits, emo_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        all_preds.extend(torch.argmax(logits, 1).cpu().numpy())
        all_labels.extend(emo_labels.cpu().numpy())

    return total_loss / len(loader), f1_score(
        all_labels, all_preds, average='weighted'
    )


def evaluate(model, loader, criterion, split='Val'):
    model.eval()
    total_loss, all_preds, all_labels = 0, [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f'  {split}'):
            input_ids      = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            waveform       = batch['waveform'].to(DEVICE).float()
            video          = batch['video'].to(DEVICE).float()
            speaker_ids    = batch['speaker_id'].to(DEVICE)
            emo_labels     = batch['emotion_label'].to(DEVICE)

            logits = model(input_ids, attention_mask, waveform, video, speaker_ids)
            loss   = criterion(logits, emo_labels)

            total_loss += loss.item()
            all_preds.extend(torch.argmax(logits, 1).cpu().numpy())
            all_labels.extend(emo_labels.cpu().numpy())

    f1 = f1_score(all_labels, all_preds, average='weighted')

    if split == 'Test':
        labels      = list(ID2EMOTION.keys())
        label_names = [ID2EMOTION[i] for i in labels]
        report = classification_report(
            all_labels, all_preds,
            labels=labels, target_names=label_names,
            zero_division=0
        )
        log.info("\n" + report)

    return total_loss / len(loader), f1


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # Import video dataset here to avoid circular issues
    from data.meld_video_dataset import get_video_dataloaders

    log.info("=" * 55)
    log.info("  MELD: Text + Audio + Video(ResNet-50) + Speaker + Context")
    log.info(f"  Device      : {DEVICE}")
    log.info(f"  Batch size  : {BATCH_SIZE}")
    log.info(f"  Epochs      : {EPOCHS}")
    log.info(f"  Context win : {CONTEXT_WIN}")
    log.info(f"  Log file    : {LOG_PATH}")
    log.info("=" * 55)

    # ── Load Data ──────────────────────────────────────────────────────────
    log.info("Loading MELD data with video features...")
    train_loader, dev_loader, test_loader = get_video_dataloaders(
        batch_size=BATCH_SIZE,
        max_text_len=MAX_LEN,
        context_window=CONTEXT_WIN,
    )
    log.info(f"Train: {len(train_loader)} batches | "
             f"Dev: {len(dev_loader)} | Test: {len(test_loader)}")

    # ── Build Model ────────────────────────────────────────────────────────
    log.info("\nBuilding MELD video model...")
    model = MELDVideoModel(
        num_emotions=NUM_EMOTIONS,
        hidden_size=768,
        video_hidden=VIDEO_HIDDEN,
        speaker_emb_dim=64,
        num_speakers=NUM_SPEAKERS,
        dropout=0.3,
        freeze_text=True,
    ).to(DEVICE).float()

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Total parameters    : {total:,}")
    log.info(f"Trainable parameters: {trainable:,}")

    # ── Loss & Optimizer ───────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(weight=EMOTION_WEIGHTS.to(DEVICE))
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=2e-5, weight_decay=0.01
    )
    total_steps = len(train_loader) * EPOCHS
    scheduler   = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps
    )

    log.info(f"\nStarting training for {EPOCHS} epochs...")
    log.info(f"ModernBERT unfreezes at epoch {UNFREEZE_EPOCH}\n")

    best_val_f1 = 0
    best_path   = os.path.join(SAVE_DIR, 'best_meld_video_model.pt')

    for epoch in range(1, EPOCHS + 1):
        log.info(f"\n{'='*55}")
        log.info(f"Epoch {epoch}/{EPOCHS}")
        log.info(f"{'='*55}")

        # Gradual ModernBERT unfreeze
        if epoch == UNFREEZE_EPOCH:
            log.info("Unfreezing ModernBERT with lower LR (5e-6)...")
            for p in model.text_encoder.parameters():
                p.requires_grad = True
            optimizer = AdamW([
                {'params': model.text_encoder.parameters(),  'lr': 5e-6},
                {'params': [p for n, p in model.named_parameters()
                            if 'text_encoder' not in n],      'lr': 2e-5},
            ], weight_decay=0.01)
            remaining = len(train_loader) * (EPOCHS - epoch + 1)
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=remaining // 10,
                num_training_steps=remaining
            )
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            log.info(f"   Trainable now: {trainable:,}")

        train_loss, train_f1 = train_epoch(
            model, train_loader, optimizer, scheduler, criterion
        )
        val_loss, val_f1 = evaluate(model, dev_loader, criterion, 'Val')

        log.info(f"\nTrain Loss: {train_loss:.4f} | Train F1: {train_f1:.4f}")
        log.info(f"Val   Loss: {val_loss:.4f} | Val   F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), best_path)
            log.info(f"✅ Best model saved! Val F1: {val_f1:.4f}")

    # ── Final Test ─────────────────────────────────────────────────────────
    log.info(f"\n{'='*55}")
    log.info("Final Test Evaluation")
    log.info(f"{'='*55}")

    model.load_state_dict(torch.load(best_path, weights_only=True))
    test_loss, test_f1 = evaluate(model, test_loader, criterion, 'Test')

    log.info(f"Test Loss : {test_loss:.4f} | Test F1: {test_f1:.4f}")
    log.info(f"\n🎉 Training complete!")
    log.info(f"\n📊 Summary:")
    log.info(f"   Dataset     : MELD")
    log.info(f"   Modalities  : Text + Audio + Video(ResNet-50) + Speaker + Context")
    log.info(f"   Best Val F1 : {best_val_f1:.4f}")
    log.info(f"   Test F1     : {test_f1:.4f}")
    log.info(f"   Log saved to: {LOG_PATH}")


if __name__ == '__main__':
    main()