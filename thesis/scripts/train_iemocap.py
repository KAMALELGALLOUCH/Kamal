"""
Training Script for IEMOCAP — TEXT-DOMINANT v3
================================================
Same architecture as v2 (A+B+C+D) plus anti-overfitting:
  G — Label smoothing (0.1)
  H — Higher dropout (0.4 vs 0.3)
  I — Early stopping (patience=3)
  J — Tuned LRs: text 8e-6, audio 5e-6, other 1.5e-5
  E — Unfreeze at epoch 3 (compromise between v1=4, v2=2)
"""

import os
import sys
import torch
import torch.nn as nn
import torchaudio.transforms as T
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup, AutoModel
from sklearn.metrics import f1_score, classification_report
from tqdm import tqdm

sys.path.append('/data8/luoyan/Kamal/thesis')
from data.iemocap_dataset import (
    get_dataloaders, NUM_EMOTIONS, NUM_SPEAKERS, ID2EMOTION
)

# ── Config ─────────────────────────────────────────────────────
DEVICE         = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS         = 15
BATCH_SIZE     = 8
MAX_LEN        = 128
CONTEXT_WIN    = 3
TEST_SESSION   = 5
SAVE_DIR       = '/data8/luoyan/Kamal/thesis/outputs/iemocap'
os.makedirs(SAVE_DIR, exist_ok=True)

TEXT_MODEL_PATH = '/data8/zhangxin/comp5423/gte-modernbert-base'

# ── [CHANGE E] Unfreeze at epoch 3 (compromise) ───────────────
UNFREEZE_EPOCH = 3

# ── [CHANGE F+J] Tuned differential learning rates ─────────────
LR_TEXT  = 8e-6   # was 1e-5 in v2 → slightly lower to reduce overfit
LR_AUDIO = 5e-6   # unchanged from v2
LR_OTHER = 1.5e-5 # was 2e-5 → slightly lower overall

# ── Class Weights ───────────────────────────────────────────────
EMOTION_COUNTS  = [1708, 1636, 1084, 1103, 1849]
EMOTION_WEIGHTS = torch.tensor(
    [1.0 / c for c in EMOTION_COUNTS], dtype=torch.float32
)
EMOTION_WEIGHTS = EMOTION_WEIGHTS / EMOTION_WEIGHTS.sum()


# ══════════════════════════════════════════════════════════════
# Model Components — Text-Dominant Architecture
# ══════════════════════════════════════════════════════════════

class AudioEncoder(nn.Module):
    """Unchanged from v1."""
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


class SpeakerEmbedding(nn.Module):
    """Unchanged from v1."""
    def __init__(self, num_speakers=10, embedding_dim=64, dropout=0.3):
        super().__init__()
        self.embedding  = nn.Embedding(num_speakers, embedding_dim)
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(), nn.Dropout(dropout)
        )

    def forward(self, speaker_ids):
        return self.projection(self.embedding(speaker_ids))


# ── [CHANGE D] Weighted Layer Pooling ──────────────────────────
class WeightedLayerPooling(nn.Module):
    """
    Learned weighted sum of the last N transformer layers.
    Richer text representation than CLS-only.
    """
    def __init__(self, num_layers=4):
        super().__init__()
        # Learnable weights, initialized uniformly
        self.layer_weights = nn.Parameter(torch.ones(num_layers))

    def forward(self, hidden_states_tuple):
        """
        hidden_states_tuple: tuple of [B, seq_len, H] tensors
                             (all layers from the transformer)
        Returns: [B, H] — weighted CLS pooling
        """
        # Take last N layers
        last_n = hidden_states_tuple[-len(self.layer_weights):]
        # Stack → [num_layers, B, seq_len, H]
        stacked = torch.stack(last_n, dim=0)
        # Softmax over layers so weights sum to 1
        weights = torch.softmax(self.layer_weights, dim=0)
        # Weighted sum → [B, seq_len, H]
        weighted = (stacked * weights.view(-1, 1, 1, 1)).sum(dim=0)
        # Take CLS token → [B, H]
        return weighted[:, 0, :]


# ── [CHANGE B] Asymmetric Cross-Modal Attention ────────────────
class AsymmetricCrossModalAttention(nn.Module):
    """
    Text-attends-to-audio ONLY (one direction).
    Audio enriches text, but text is never diluted by audio queries.
    Audio features pass through unchanged with just a norm.
    """
    def __init__(self, hidden_size=768, num_heads=8, dropout=0.3):
        super().__init__()
        # Only text → audio direction
        self.text_to_audio = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        self.norm_text  = nn.LayerNorm(hidden_size)
        self.norm_audio = nn.LayerNorm(hidden_size)
        self.dropout    = nn.Dropout(dropout)

    def forward(self, text_feat, audio_feat):
        t = text_feat.unsqueeze(1)   # [B, 1, H]
        a = audio_feat.unsqueeze(1)  # [B, 1, H]

        # Text attends to audio (audio informs text)
        ta, _ = self.text_to_audio(query=t, key=a, value=a)
        text_out = self.norm_text(t + self.dropout(ta)).squeeze(1)

        # Audio passes through unchanged (just normalized)
        audio_out = self.norm_audio(audio_feat)

        return text_out, audio_out  # [B, H], [B, H]


# ── [CHANGE C] Text-Biased Gated Fusion ────────────────────────
class TextBiasedGatedFusion(nn.Module):
    """
    Same structure as v1 but gate bias initialized so
    sigmoid(bias) ≈ 0.7, meaning text starts with 70% weight.
    The model can still learn to adjust per-sample.
    """
    def __init__(self, hidden_size=768, dropout=0.3, text_bias=0.85):
        super().__init__()
        self.gate_linear = nn.Linear(hidden_size * 2, hidden_size)
        self.fusion      = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout     = nn.Dropout(dropout)

        # [CHANGE C] Initialize bias so sigmoid(0.85) ≈ 0.70
        nn.init.constant_(self.gate_linear.bias, text_bias)
        print(f"[GatedFusion] Gate bias initialized to {text_bias} "
              f"(sigmoid ≈ {torch.sigmoid(torch.tensor(text_bias)):.2f})")

    def forward(self, text_feat, audio_feat):
        combined = torch.cat([text_feat, audio_feat], dim=-1)
        gate     = torch.sigmoid(self.gate_linear(combined))
        fused    = self.fusion(combined)
        output   = gate * text_feat + (1 - gate) * audio_feat + fused
        return self.dropout(output)


# ── Text-Dominant IEMOCAP Model ────────────────────────────────
class IEMOCAPModelV2(nn.Module):
    """
    Text-dominant multimodal emotion recognition for IEMOCAP.

    Changes from v1:
      A — Text residual: fused = fused + text_feat (skip connection)
      B — Asymmetric attention: text-attends-to-audio only
      C — Text-biased gate: starts at ~70% text weight
      D — Weighted layer pooling: last 4 ModernBERT layers
    """

    def __init__(self, num_emotions=5, hidden_size=768,
                 speaker_emb_dim=64, num_speakers=10,
                 dropout=0.3, freeze_text=True,
                 num_pool_layers=4, gate_text_bias=0.85):
        super().__init__()

        # ── text encoder ───────────────────────────────────────
        self.text_encoder = AutoModel.from_pretrained(
            TEXT_MODEL_PATH, local_files_only=True, low_cpu_mem_usage=True
        )
        if freeze_text:
            for p in self.text_encoder.parameters():
                p.requires_grad = False
        print(f"[Model] Text encoder hidden size : "
              f"{self.text_encoder.config.hidden_size}")
        print(f"[Model] freeze_text              : {freeze_text}")

        # [CHANGE D] Weighted layer pooling
        self.layer_pooling = WeightedLayerPooling(num_layers=num_pool_layers)
        print(f"[Model] Layer pooling            : last {num_pool_layers} layers (learned weights)")

        # ── audio encoder ──────────────────────────────────────
        self.audio_encoder = AudioEncoder(hidden_size, dropout)
        print(f"[Model] Audio encoder            : CNN + MelSpec")

        # ── speaker embedding ──────────────────────────────────
        self.speaker_embedding = SpeakerEmbedding(
            num_speakers, speaker_emb_dim, dropout
        )
        print(f"[Model] Speaker embedding dim    : {speaker_emb_dim}")
        print(f"[Model] Num speakers             : {num_speakers}")

        # [CHANGE B] Asymmetric cross-modal attention
        self.cross_attention = AsymmetricCrossModalAttention(
            hidden_size, 8, dropout
        )
        print(f"[Model] Cross-attention          : ASYMMETRIC (text→audio only)")

        # [CHANGE C] Text-biased gated fusion
        self.gated_fusion = TextBiasedGatedFusion(
            hidden_size, dropout, text_bias=gate_text_bias
        )

        self.dropout = nn.Dropout(dropout)

        # [CHANGE A] Text residual — projection to match dimensions
        # after fusion (in case we need it; here same size so it's identity-like)
        self.text_residual_gate = nn.Parameter(torch.tensor(0.3))
        print(f"[Model] Text residual            : ENABLED (learnable gate, init=0.3)")

        # ── classifier: 768 + 64 = 832 ────────────────────────
        clf_input = hidden_size + speaker_emb_dim
        self.emotion_head = nn.Sequential(
            nn.Linear(clf_input, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_emotions)
        )

    def forward(self, input_ids, attention_mask, waveform, speaker_ids):
        # ── [CHANGE D] text encoding with all hidden states ────
        text_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True     # ← return all layers
        )
        # Weighted pooling of last 4 layers instead of just CLS
        text_feat = self.layer_pooling(text_outputs.hidden_states)
        text_feat = self.dropout(text_feat)        # [B, 768]

        # ── audio encoding ─────────────────────────────────────
        audio_feat = self.audio_encoder(waveform)  # [B, 768]

        # ── [CHANGE B] asymmetric cross-modal attention ────────
        text_att, audio_att = self.cross_attention(text_feat, audio_feat)

        # ── [CHANGE C] text-biased gated fusion ────────────────
        fused = self.gated_fusion(text_att, audio_att)

        # ── [CHANGE A] text residual connection ────────────────
        # Preserve raw text signal so it's never fully lost in fusion
        residual_weight = torch.sigmoid(self.text_residual_gate)
        fused = fused + residual_weight * text_feat

        fused = self.dropout(fused)                # [B, 768]

        # ── speaker embedding ──────────────────────────────────
        spk_emb  = self.speaker_embedding(speaker_ids)  # [B, 64]
        combined = torch.cat([fused, spk_emb], dim=-1)  # [B, 832]

        return self.emotion_head(combined)


# ══════════════════════════════════════════════════════════════
# Training & Evaluation
# ══════════════════════════════════════════════════════════════

def train_epoch(model, loader, optimizer, scheduler, criterion):
    model.train()
    total_loss, all_preds, all_labels = 0, [], []

    for batch in tqdm(loader, desc='  Training'):
        input_ids      = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        waveform       = batch['waveform'].to(DEVICE).float()
        speaker_ids    = batch['speaker_id'].to(DEVICE)
        emo_labels     = batch['emotion_label'].to(DEVICE)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask, waveform, speaker_ids)
        loss   = criterion(logits, emo_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        all_preds.extend(torch.argmax(logits, 1).cpu().numpy())
        all_labels.extend(emo_labels.cpu().numpy())

    f1 = f1_score(all_labels, all_preds, average='weighted')
    return total_loss / len(loader), f1


def evaluate(model, loader, criterion, split='Val'):
    model.eval()
    total_loss, all_preds, all_labels = 0, [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f'  {split}'):
            input_ids      = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            waveform       = batch['waveform'].to(DEVICE).float()
            speaker_ids    = batch['speaker_id'].to(DEVICE)
            emo_labels     = batch['emotion_label'].to(DEVICE)

            logits = model(input_ids, attention_mask, waveform, speaker_ids)
            loss   = criterion(logits, emo_labels)

            total_loss += loss.item()
            all_preds.extend(torch.argmax(logits, 1).cpu().numpy())
            all_labels.extend(emo_labels.cpu().numpy())

    f1 = f1_score(all_labels, all_preds, average='weighted')

    if split == 'Test':
        labels      = list(ID2EMOTION.keys())
        label_names = [ID2EMOTION[i] for i in labels]
        print("\n" + classification_report(
            all_labels, all_preds,
            labels=labels, target_names=label_names,
            zero_division=0
        ))

    return total_loss / len(loader), f1


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

def main():
    print(f"{'='*60}")
    print(f"  IEMOCAP — TEXT-DOMINANT v3 (anti-overfit)")
    print(f"  Arch  : A(residual) B(asym-attn) C(biased-gate) D(layer-pool)")
    print(f"  Train : E(unfreeze@{UNFREEZE_EPOCH}) F(diff-LR) G(label-smooth)")
    print(f"          H(dropout=0.4) I(early-stop=3)")
    print(f"  Device        : {DEVICE}")
    print(f"  Test session  : {TEST_SESSION}")
    print(f"  Epochs        : {EPOCHS} (max)")
    print(f"  Batch size    : {BATCH_SIZE}")
    print(f"  LR text/audio/other : {LR_TEXT}/{LR_AUDIO}/{LR_OTHER}")
    print(f"{'='*60}\n")

    # ── load data ──────────────────────────────────────────────
    print("Loading IEMOCAP data...")
    train_loader, dev_loader, test_loader = get_dataloaders(
        test_session=TEST_SESSION,
        batch_size=BATCH_SIZE,
        max_text_len=MAX_LEN,
        context_window=CONTEXT_WIN,
    )

    # ── build model ────────────────────────────────────────────
    print("\nBuilding text-dominant model (v3)...")
    model = IEMOCAPModelV2(
        num_emotions=NUM_EMOTIONS,
        hidden_size=768,
        speaker_emb_dim=64,
        num_speakers=NUM_SPEAKERS,
        dropout=0.4,          # [CHANGE H] was 0.3 → fight overfitting
        freeze_text=True,
        num_pool_layers=4,
        gate_text_bias=0.85,
    ).to(DEVICE)
    model = model.float()

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters()
                    if p.requires_grad)
    print(f"\nTotal parameters    : {total:,}")
    print(f"Trainable parameters: {trainable:,}")

    # ── loss with label smoothing ─────────────────────────────
    criterion = nn.CrossEntropyLoss(
        weight=EMOTION_WEIGHTS.to(DEVICE),
        label_smoothing=0.1,   # [CHANGE G] softens targets → reduces overfit
    )

    # ── [CHANGE F] Differential learning rates from the start ──
    # Before unfreeze: text is frozen, so only audio+other train
    # We still set up separate groups for audio vs other
    optimizer = AdamW([
        {'params': model.audio_encoder.parameters(),  'lr': LR_AUDIO},
        {'params': [p for n, p in model.named_parameters()
                    if 'text_encoder' not in n
                    and 'audio_encoder' not in n
                    and p.requires_grad],              'lr': LR_OTHER},
    ], weight_decay=0.01)

    total_steps = len(train_loader) * EPOCHS
    scheduler   = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps
    )

    print(f"\nStarting training for {EPOCHS} epochs...")
    print(f"ModernBERT unfreezes at epoch {UNFREEZE_EPOCH}\n")

    best_val_f1 = 0
    best_path   = os.path.join(SAVE_DIR, 'best_iemocap_model_v3.pt')
    patience, patience_counter = 3, 0   # [CHANGE I] early stopping

    for epoch in range(1, EPOCHS + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{EPOCHS}")
        print(f"{'='*60}")

        # ── [CHANGE E] Unfreeze at epoch 2 with differential LR ──
        if epoch == UNFREEZE_EPOCH:
            print(f"🔓 Unfreezing ModernBERT (LR={LR_TEXT})...")
            for p in model.text_encoder.parameters():
                p.requires_grad = True

            # [CHANGE F] Three-group optimizer: text / audio / other
            optimizer = AdamW([
                {'params': model.text_encoder.parameters(),    'lr': LR_TEXT},
                {'params': model.audio_encoder.parameters(),   'lr': LR_AUDIO},
                {'params': [p for n, p in model.named_parameters()
                            if 'text_encoder' not in n
                            and 'audio_encoder' not in n],     'lr': LR_OTHER},
            ], weight_decay=0.01)

            remaining = len(train_loader) * (EPOCHS - epoch + 1)
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=remaining // 10,
                num_training_steps=remaining
            )
            trainable = sum(p.numel() for p in model.parameters()
                            if p.requires_grad)
            print(f"   Trainable now    : {trainable:,}")
            print(f"   LR text/audio/other: {LR_TEXT}/{LR_AUDIO}/{LR_OTHER}")

        train_loss, train_f1 = train_epoch(
            model, train_loader, optimizer, scheduler, criterion
        )
        val_loss, val_f1 = evaluate(
            model, dev_loader, criterion, 'Val'
        )

        print(f"\nTrain Loss: {train_loss:.4f} | Train F1: {train_f1:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Val   F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), best_path)
            print(f"✅ Best model saved! Val F1: {val_f1:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"   No improvement ({patience_counter}/{patience})")
            if patience_counter >= patience and epoch > UNFREEZE_EPOCH:
                print(f"⏹️  Early stopping at epoch {epoch} (patience={patience})")
                break

    # ── final test ─────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Final Test Evaluation — Text-Dominant v3")
    print(f"{'='*60}")
    model.load_state_dict(
        torch.load(best_path, weights_only=True)
    )
    test_loss, test_f1 = evaluate(
        model, test_loader, criterion, 'Test'
    )
    print(f"Test Loss: {test_loss:.4f} | Test F1: {test_f1:.4f}")
    print(f"\n🎉 Training complete! Best Val F1: {best_val_f1:.4f}")
    print(f"\n📊 Summary:")
    print(f"   Dataset  : IEMOCAP (Session {TEST_SESSION} test)")
    print(f"   Model    : Text-Dominant v3 (anti-overfit)")
    print(f"   Changes  : A+B+C+D+E+F+G+H+I")
    print(f"   Test F1  : {test_f1:.4f}")
    print(f"   vs v1    : 0.7085")
    print(f"   vs v2    : 0.7082")


if __name__ == '__main__':
    main()