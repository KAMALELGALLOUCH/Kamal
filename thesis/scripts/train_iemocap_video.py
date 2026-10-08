"""
Training Script: IEMOCAP with MOCAP Video Features
====================================================
Text + Audio + MOCAP(Video) + Speaker + Context


Run: CUDA_VISIBLE_DEVICES=3 python scripts/train_iemocap_video.py

works:
  - Added context window aggregation (+11.45% on IEMOCAP ablation)
  - Fixed EMOTION_COUNTS to match 5 classes
  - Proper logging to file
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
from data.iemocap_dataset import (
    get_dataloaders, NUM_EMOTIONS, NUM_SPEAKERS,
    ID2EMOTION, MOCAP_FEATURES, MAX_MOCAP_LEN
)

# ── Config ─────────────────────────────────────────────────────
DEVICE         = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS         = 15
BATCH_SIZE     = 16
MAX_LEN        = 128
CONTEXT_WIN    = 3
TEST_SESSION   = 5
UNFREEZE_EPOCH = 4
SAVE_DIR       = '/data8/luoyan/Kamal/thesis/outputs/iemocap'
LOG_PATH       = '/data8/luoyan/Kamal/thesis/logs/train_iemocap_video.log'
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

TEXT_MODEL_PATH = '/data8/zhangxin/comp5423/gte-modernbert-base'

# ── Logging setup ───────────────────────────────────────────────
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

# ── Class Weights (5 emotions) ──────────────────────────────────
EMOTION_COUNTS  = [1708, 1636, 1084, 1103, 1849]   # 5 classes
EMOTION_WEIGHTS = torch.tensor(
    [1.0 / c for c in EMOTION_COUNTS], dtype=torch.float32
)
EMOTION_WEIGHTS = EMOTION_WEIGHTS / EMOTION_WEIGHTS.sum()


# ══════════════════════════════════════════════════════════════
# Model Components
# ══════════════════════════════════════════════════════════════

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
    """
    Encodes head motion capture data using Bidirectional GRU.
    Input : [B, MAX_MOCAP_LEN, MOCAP_FEATURES]
    Output: [B, hidden_size]
    """
    def __init__(self, input_size=6, hidden_size=128,
                 num_layers=2, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        self.projection = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, mocap):
        _, hidden = self.gru(mocap)
        # hidden: [num_layers*2, B, hidden_size]
        forward_h  = hidden[-2]   # [B, hidden_size]
        backward_h = hidden[-1]   # [B, hidden_size]
        combined   = torch.cat([forward_h, backward_h], dim=-1)
        return self.projection(combined)   # [B, hidden_size]


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
    """
    Aggregates surrounding utterance features for context-aware emotion.
    This is the component that gave +11.45% F1 in the ablation study.

    Takes the current utterance embedding and a window of context embeddings,
    uses attention to weight their importance, then combines them.
    """
    def __init__(self, hidden_size=768, dropout=0.3):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            hidden_size, num_heads=8, dropout=dropout, batch_first=True
        )
        self.norm    = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, current_feat, context_feats):
        """
        Args:
            current_feat  : [B, hidden_size]  — current utterance
            context_feats : [B, C, hidden_size] — context window (C utterances)
        Returns:
            [B, hidden_size] — context-enriched feature
        """
        q = current_feat.unsqueeze(1)              # [B, 1, H]
        attended, _ = self.attention(q, context_feats, context_feats)
        return self.norm(current_feat + self.dropout(attended.squeeze(1)))


class IEMOCAPVideoModel(nn.Module):
    """
    Full Multimodal Model for IEMOCAP:
    Text + Audio + MOCAP(Video) + Speaker + Context

    Architecture:
    ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
    │ BERT (text) │  │  CNN+Mel    │  │  BiGRU      │
    │  768-dim    │  │  (audio)    │  │  (MOCAP)    │
    │             │  │  768-dim    │  │  128-dim    │
    └──────┬──────┘  └──────┬──────┘  └──────┬──────┘
           │                │                 │
           └────── Cross-Modal Attention ─────┘
                            │
                      Gated Fusion
                            │
                   Context Aggregation  ← +11.45% gain
                            │
              ┌─────────────┴─────────────┐
              │    cat [fused(768)         │
              │       + mocap(128)         │
              │       + speaker(64)]       │
              │    = 960-dim               │
              └─────────────┬─────────────┘
                      FC → 5 emotions
    """

    def __init__(self, num_emotions=5, hidden_size=768,
                 mocap_hidden=128, speaker_emb_dim=64,
                 num_speakers=10, dropout=0.3, freeze_text=True):
        super().__init__()

        # text encoder
        self.text_encoder = AutoModel.from_pretrained(
            TEXT_MODEL_PATH, local_files_only=True, low_cpu_mem_usage=True
        )
        if freeze_text:
            for p in self.text_encoder.parameters():
                p.requires_grad = False

        # audio encoder
        self.audio_encoder = AudioEncoder(hidden_size, dropout)

        # MOCAP encoder
        self.mocap_encoder = MOCAPEncoder(
            input_size=MOCAP_FEATURES,
            hidden_size=mocap_hidden,
            num_layers=2,
            dropout=dropout
        )

        # speaker embedding
        self.speaker_embedding = SpeakerEmbedding(
            num_speakers, speaker_emb_dim, dropout
        )

        # cross-modal attention + gated fusion (text ↔ audio)
        self.cross_attention = CrossModalAttention(hidden_size, 8, dropout)
        self.gated_fusion    = GatedFusion(hidden_size, dropout)

        # ✅ context aggregator — key component for +11.45% gain
        self.context_aggregator = ContextAggregator(hidden_size, dropout)

        self.dropout = nn.Dropout(dropout)

        # classifier: 768 (fused+context) + 128 (mocap) + 64 (speaker) = 960
        clf_input = hidden_size + mocap_hidden + speaker_emb_dim
        log.info(f"[Model] Classifier input size: {clf_input}")

        self.emotion_head = nn.Sequential(
            nn.Linear(clf_input, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_emotions)
        )

    def encode_utterance(self, input_ids, attention_mask, waveform):
        """Encode a single utterance to a fused text+audio feature."""
        text_feat  = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).last_hidden_state[:, 0, :]
        text_feat  = self.dropout(text_feat)

        audio_feat = self.audio_encoder(waveform)

        text_att, audio_att = self.cross_attention(text_feat, audio_feat)
        fused = self.gated_fusion(text_att, audio_att)
        return self.dropout(fused)   # [B, 768]

    def forward(self, input_ids, attention_mask, waveform,
                mocap, speaker_ids,
                ctx_input_ids=None, ctx_attention_mask=None,
                ctx_waveform=None):
        """
        Args:
            input_ids / attention_mask / waveform : current utterance
            mocap                                 : [B, MAX_MOCAP_LEN, MOCAP_FEATURES]
            speaker_ids                           : [B]
            ctx_input_ids / ctx_attention_mask
              / ctx_waveform                      : [B, C, ...] context window
        """
        # ── current utterance ──────────────────────────────────
        fused = self.encode_utterance(input_ids, attention_mask, waveform)
        # [B, 768]

        # ── context aggregation ────────────────────────────────
        if (ctx_input_ids is not None and
                ctx_input_ids.size(1) > 0):
            B, C, L = ctx_input_ids.shape
            # encode each context utterance
            ctx_ids   = ctx_input_ids.view(B * C, L)
            ctx_masks = ctx_attention_mask.view(B * C, L)

            # flatten waveform context: [B, C, T] → [B*C, T]
            ctx_wav = ctx_waveform.view(B * C, ctx_waveform.size(-1))

            ctx_feats = self.encode_utterance(ctx_ids, ctx_masks, ctx_wav)
            ctx_feats = ctx_feats.view(B, C, -1)   # [B, C, 768]

            fused = self.context_aggregator(fused, ctx_feats)
        # [B, 768]

        # ── MOCAP ──────────────────────────────────────────────
        mocap_feat = self.mocap_encoder(mocap)        # [B, 128]

        # ── speaker ────────────────────────────────────────────
        spk_emb    = self.speaker_embedding(speaker_ids)  # [B, 64]

        # ── combine & classify ─────────────────────────────────
        combined = torch.cat([fused, mocap_feat, spk_emb], dim=-1)  # [B, 960]
        return self.emotion_head(combined)


# ══════════════════════════════════════════════════════════════
# Training & Evaluation
# ══════════════════════════════════════════════════════════════

def _batch_to_device(batch):
    """Move all batch tensors to DEVICE."""
    input_ids      = batch['input_ids'].to(DEVICE)
    attention_mask = batch['attention_mask'].to(DEVICE)
    waveform       = batch['waveform'].to(DEVICE).float()
    mocap          = batch['mocap'].to(DEVICE).float()
    speaker_ids    = batch['speaker_id'].to(DEVICE)
    emo_labels     = batch['emotion_label'].to(DEVICE)

    # context (optional — present when context_window > 0)
    ctx_ids  = batch.get('ctx_input_ids')
    ctx_mask = batch.get('ctx_attention_mask')
    ctx_wav  = batch.get('ctx_waveform')

    if ctx_ids is not None:
        ctx_ids  = ctx_ids.to(DEVICE)
        ctx_mask = ctx_mask.to(DEVICE)
        ctx_wav  = ctx_wav.to(DEVICE).float()

    return (input_ids, attention_mask, waveform,
            mocap, speaker_ids, emo_labels,
            ctx_ids, ctx_mask, ctx_wav)


def train_epoch(model, loader, optimizer, scheduler, criterion):
    model.train()
    total_loss, all_preds, all_labels = 0, [], []

    for batch in tqdm(loader, desc='  Training'):
        (input_ids, attention_mask, waveform,
         mocap, speaker_ids, emo_labels,
         ctx_ids, ctx_mask, ctx_wav) = _batch_to_device(batch)

        optimizer.zero_grad()
        logits = model(
            input_ids, attention_mask, waveform,
            mocap, speaker_ids,
            ctx_ids, ctx_mask, ctx_wav
        )
        loss = criterion(logits, emo_labels)
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
            (input_ids, attention_mask, waveform,
             mocap, speaker_ids, emo_labels,
             ctx_ids, ctx_mask, ctx_wav) = _batch_to_device(batch)

            logits = model(
                input_ids, attention_mask, waveform,
                mocap, speaker_ids,
                ctx_ids, ctx_mask, ctx_wav
            )
            loss = criterion(logits, emo_labels)

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


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

def main():
    log.info("=" * 55)
    log.info("  IEMOCAP: Text + Audio + Video(MOCAP) + Speaker + Context")
    log.info(f"  Device      : {DEVICE}")
    log.info(f"  Batch size  : {BATCH_SIZE}")
    log.info(f"  Epochs      : {EPOCHS}")
    log.info(f"  Context win : {CONTEXT_WIN}")
    log.info(f"  Test session: {TEST_SESSION}")
    log.info(f"  Log file    : {LOG_PATH}")
    log.info("=" * 55)

    # ── Load Data ──────────────────────────────────────────────
    log.info("Loading IEMOCAP data with MOCAP + context...")
    train_loader, dev_loader, test_loader = get_dataloaders(
        test_session=TEST_SESSION,
        batch_size=BATCH_SIZE,
        max_text_len=MAX_LEN,
        context_window=CONTEXT_WIN,
        use_mocap=True,
    )
    log.info(f"Train batches: {len(train_loader)} | "
             f"Val batches: {len(dev_loader)} | "
             f"Test batches: {len(test_loader)}")

    # ── Build Model ────────────────────────────────────────────
    log.info("\nBuilding model with MOCAP + Context encoders...")
    model = IEMOCAPVideoModel(
        num_emotions=NUM_EMOTIONS,
        hidden_size=768,
        mocap_hidden=128,
        speaker_emb_dim=64,
        num_speakers=NUM_SPEAKERS,
        dropout=0.3,
        freeze_text=True,
    ).to(DEVICE).float()

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters()
                    if p.requires_grad)
    log.info(f"Total parameters    : {total:,}")
    log.info(f"Trainable parameters: {trainable:,}")

    # ── Loss & Optimizer ───────────────────────────────────────
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
    log.info(f"BERT unfreezes at epoch {UNFREEZE_EPOCH}\n")

    best_val_f1 = 0
    best_path   = os.path.join(SAVE_DIR, 'best_iemocap_video_model.pt')

    for epoch in range(1, EPOCHS + 1):
        log.info(f"\n{'='*55}")
        log.info(f"Epoch {epoch}/{EPOCHS}")
        log.info(f"{'='*55}")

        # ── gradual BERT unfreeze at epoch 4 ──────────────────
        if epoch == UNFREEZE_EPOCH:
            log.info("Unfreezing BERT with lower LR (5e-6)...")
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
            trainable = sum(p.numel() for p in model.parameters()
                            if p.requires_grad)
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

    # ── Final Test ─────────────────────────────────────────────
    log.info(f"\n{'='*55}")
    log.info("Final Test Evaluation")
    log.info(f"{'='*55}")

    model.load_state_dict(torch.load(best_path, weights_only=True))
    test_loss, test_f1 = evaluate(model, test_loader, criterion, 'Test')

    log.info(f"Test Loss : {test_loss:.4f} | Test F1: {test_f1:.4f}")
    log.info(f"\n🎉 Training complete!")
    log.info(f"\n📊 Summary:")
    log.info(f"   Dataset     : IEMOCAP (Session {TEST_SESSION} test)")
    log.info(f"   Modalities  : Text + Audio + MOCAP(Video) + Speaker + Context")
    log.info(f"   Best Val F1 : {best_val_f1:.4f}")
    log.info(f"   Test F1     : {test_f1:.4f}")
    log.info(f"   Log saved to: {LOG_PATH}")


if __name__ == '__main__':
    main()