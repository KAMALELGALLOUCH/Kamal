"""
Ablation Study for Multimodal Emotion Recognition on MELD
==========================================================
Compares 4 configurations:
  Exp 1: Text only
  Exp 2: Text + Audio
  Exp 3: Text + Audio + Speaker
  Exp 4: Text + Audio + Speaker + Context


"""

import os
import sys
import csv
import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup, AutoModel
from sklearn.metrics import f1_score, classification_report
from tqdm import tqdm
import torchaudio.transforms as T

sys.path.append('/data8/luoyan/Kamal/thesis')
from data.meld_dataset import (
    get_dataloaders, EMOTION2ID, SENTIMENT2ID, NUM_SPEAKERS
)

# ── Config ─────────────────────────────────────────────────────
DEVICE         = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS         = 10
BATCH_SIZE     = 16
LR             = 2e-5
MAX_LEN        = 128
UNFREEZE_EPOCH = 4
SAVE_DIR       = '/data8/luoyan/Kamal/thesis/outputs/ablation'
os.makedirs(SAVE_DIR, exist_ok=True)

TEXT_MODEL_PATH = '/data8/zhangxin/comp5423/gte-modernbert-base'

# ── Class Weights ───────────────────────────────────────────────
EMOTION_COUNTS  = [4710, 1743, 1205, 1109, 683, 271, 268]
EMOTION_WEIGHTS = torch.tensor(
    [1.0 / c for c in EMOTION_COUNTS], dtype=torch.float32
)
EMOTION_WEIGHTS = EMOTION_WEIGHTS / EMOTION_WEIGHTS.sum()
ID2EMOTION      = {v: k for k, v in EMOTION2ID.items()}

# ── Experiments Definition ──────────────────────────────────────
EXPERIMENTS = [
    {
        'name':        'Exp1_TextOnly',
        'use_audio':   False,
        'use_speaker': False,
        'context':     0,
        'description': 'Text Only (GTE-ModernBERT)',
    },
    {
        'name':        'Exp2_Text_Audio',
        'use_audio':   True,
        'use_speaker': False,
        'context':     0,
        'description': 'Text + Audio (CNN+MelSpec)',
    },
    {
        'name':        'Exp3_Text_Audio_Speaker',
        'use_audio':   True,
        'use_speaker': True,
        'context':     0,
        'description': 'Text + Audio + Speaker Embedding',
    },
    {
        'name':        'Exp4_Text_Audio_Speaker_Context',
        'use_audio':   True,
        'use_speaker': True,
        'context':     3,
        'description': 'Text + Audio + Speaker + Context (Full Model)',
    },
]


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
        return self.dropout(self.gate(c) * text_feat +
                            (1 - self.gate(c)) * audio_feat +
                            self.fusion(c))


# ── Ablation Model (configurable) ─────────────────────────────
class AblationModel(nn.Module):
    """
    Single model class that supports all 4 ablation configurations.
    use_audio   : include audio encoder + cross-modal attention
    use_speaker : include speaker embedding
    """

    def __init__(self, num_emotions=7, num_sentiments=3,
                 hidden_size=768, dropout=0.3,
                 use_audio=True, use_speaker=True,
                 freeze_text=True):
        super().__init__()
        self.use_audio   = use_audio
        self.use_speaker = use_speaker

        # text encoder (always used)
        self.text_encoder = AutoModel.from_pretrained(
            TEXT_MODEL_PATH, local_files_only=True, low_cpu_mem_usage=True
        )
        if freeze_text:
            for p in self.text_encoder.parameters():
                p.requires_grad = False

        self.dropout = nn.Dropout(dropout)

        # audio components
        if use_audio:
            self.audio_encoder  = AudioEncoder(hidden_size, dropout)
            self.cross_attention = CrossModalAttention(hidden_size, 8, dropout)
            self.gated_fusion    = GatedFusion(hidden_size, dropout)

        # speaker embedding
        speaker_dim = 0
        if use_speaker:
            self.speaker_embedding = SpeakerEmbedding(
                NUM_SPEAKERS, 64, dropout
            )
            speaker_dim = 64

        # classifier heads
        clf_input = hidden_size + speaker_dim
        self.emotion_head = nn.Sequential(
            nn.Linear(clf_input, 256), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(256, num_emotions)
        )
        self.sentiment_head = nn.Sequential(
            nn.Linear(clf_input, 256), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(256, num_sentiments)
        )

    def forward(self, input_ids, attention_mask, waveform, speaker_ids):
        # text
        text_feat = self.text_encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state[:, 0, :]
        text_feat = self.dropout(text_feat)

        # audio + fusion
        if self.use_audio:
            audio_feat = self.audio_encoder(waveform)
            text_att, audio_att = self.cross_attention(text_feat, audio_feat)
            fused = self.gated_fusion(text_att, audio_att)
            fused = self.dropout(fused)
        else:
            fused = text_feat

        # speaker
        if self.use_speaker:
            spk_emb  = self.speaker_embedding(speaker_ids)
            combined = torch.cat([fused, spk_emb], dim=-1)
        else:
            combined = fused

        return self.emotion_head(combined), self.sentiment_head(combined)


# ══════════════════════════════════════════════════════════════
# Training & Evaluation
# ══════════════════════════════════════════════════════════════

def train_epoch(model, loader, optimizer, scheduler,
                emo_crit, sent_crit):
    model.train()
    total_loss, all_preds, all_labels = 0, [], []

    for batch in tqdm(loader, desc='  Training', leave=False):
        input_ids      = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        waveform       = batch['waveform'].to(DEVICE).float()
        speaker_ids    = batch['speaker_id'].to(DEVICE)
        emo_labels     = batch['emotion_label'].to(DEVICE)
        sent_labels    = batch['sentiment_label'].to(DEVICE)

        optimizer.zero_grad()
        emo_logits, sent_logits = model(
            input_ids, attention_mask, waveform, speaker_ids
        )
        loss = emo_crit(emo_logits, emo_labels) + \
               0.3 * sent_crit(sent_logits, sent_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        all_preds.extend(torch.argmax(emo_logits, 1).cpu().numpy())
        all_labels.extend(emo_labels.cpu().numpy())

    return total_loss / len(loader), f1_score(
        all_labels, all_preds, average='weighted'
    )


def evaluate(model, loader, emo_crit, sent_crit, split='Val'):
    model.eval()
    total_loss, all_preds, all_labels = 0, [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f'  {split}', leave=False):
            input_ids      = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            waveform       = batch['waveform'].to(DEVICE).float()
            speaker_ids    = batch['speaker_id'].to(DEVICE)
            emo_labels     = batch['emotion_label'].to(DEVICE)
            sent_labels    = batch['sentiment_label'].to(DEVICE)

            emo_logits, sent_logits = model(
                input_ids, attention_mask, waveform, speaker_ids
            )
            loss = emo_crit(emo_logits, emo_labels) + \
                   0.3 * sent_crit(sent_logits, sent_labels)
            total_loss += loss.item()
            all_preds.extend(torch.argmax(emo_logits, 1).cpu().numpy())
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
# Run One Experiment
# ══════════════════════════════════════════════════════════════

def run_experiment(exp, train_loader, dev_loader, test_loader):
    print(f"\n{'#'*60}")
    print(f"  {exp['name']}: {exp['description']}")
    print(f"{'#'*60}")

    start_time = time.time()

    # build model
    model = AblationModel(
        use_audio=exp['use_audio'],
        use_speaker=exp['use_speaker'],
        freeze_text=True,
        dropout=0.3,
    ).to(DEVICE)
    model = model.float()

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters()
                    if p.requires_grad)
    print(f"  Parameters: {total:,} total | {trainable:,} trainable")

    # loss & optimizer
    emo_crit  = nn.CrossEntropyLoss(
        weight=EMOTION_WEIGHTS.to(DEVICE)
    )
    sent_crit = nn.CrossEntropyLoss()
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=0.01
    )
    total_steps = len(train_loader) * EPOCHS
    scheduler   = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps
    )

    best_val_f1 = 0
    best_path   = os.path.join(SAVE_DIR, f'{exp["name"]}_best.pt')

    for epoch in range(1, EPOCHS + 1):
        print(f"\n  Epoch {epoch}/{EPOCHS}")

        # unfreeze BERT at epoch 4
        if epoch == UNFREEZE_EPOCH:
            print("  🔓 Unfreezing BERT...")
            for p in model.text_encoder.parameters():
                p.requires_grad = True
            optimizer = AdamW([
                {'params': model.text_encoder.parameters(), 'lr': 5e-6},
                {'params': [p for n, p in model.named_parameters()
                            if 'text_encoder' not in n], 'lr': 2e-5},
            ], weight_decay=0.01)
            remaining = len(train_loader) * (EPOCHS - epoch + 1)
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=remaining // 10,
                num_training_steps=remaining
            )

        train_loss, train_f1 = train_epoch(
            model, train_loader, optimizer, scheduler,
            emo_crit, sent_crit
        )
        val_loss, val_f1 = evaluate(
            model, dev_loader, emo_crit, sent_crit, 'Val'
        )

        print(f"  Train Loss: {train_loss:.4f} | Train F1: {train_f1:.4f}")
        print(f"  Val   Loss: {val_loss:.4f}  | Val   F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), best_path)
            print(f"  ✅ Best saved! Val F1: {val_f1:.4f}")

    # final test
    print(f"\n  {'='*40}")
    print(f"  Final Test — {exp['name']}")
    print(f"  {'='*40}")
    model.load_state_dict(torch.load(best_path, weights_only=True))
    test_loss, test_f1 = evaluate(
        model, test_loader, emo_crit, sent_crit, 'Test'
    )
    print(f"  Test Loss: {test_loss:.4f} | Test F1: {test_f1:.4f}")

    elapsed = (time.time() - start_time) / 60
    print(f"  ⏱️  Time: {elapsed:.1f} minutes")

    return {
        'experiment':  exp['name'],
        'description': exp['description'],
        'use_audio':   exp['use_audio'],
        'use_speaker': exp['use_speaker'],
        'context':     exp['context'],
        'best_val_f1': round(best_val_f1, 4),
        'test_f1':     round(test_f1, 4),
        'test_loss':   round(test_loss, 4),
        'time_min':    round(elapsed, 1),
    }


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

def main():
    print(f"{'='*60}")
    print(f"  ABLATION STUDY — Multimodal Emotion Recognition (MELD)")
    print(f"  Device : {DEVICE}")
    print(f"  Epochs : {EPOCHS} per experiment")
    print(f"  Experiments: {len(EXPERIMENTS)}")
    print(f"{'='*60}")

    all_results = []

    for exp in EXPERIMENTS:
        # load data with correct context window
        print(f"\n📂 Loading data (context={exp['context']})...")
        train_loader, dev_loader, test_loader = get_dataloaders(
            batch_size=BATCH_SIZE,
            max_text_len=MAX_LEN,
            context_window=exp['context'],
        )

        result = run_experiment(exp, train_loader, dev_loader, test_loader)
        all_results.append(result)

        # save intermediate results after each experiment
        _save_results(all_results)
        print(f"\n💾 Results saved to {SAVE_DIR}/ablation_results.csv")

    # ── Final Summary ───────────────────────────────────────────
    print(f"\n\n{'='*60}")
    print(f"  ABLATION STUDY — FINAL RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Experiment':<35} {'Val F1':>8} {'Test F1':>8} {'Time':>8}")
    print(f"  {'-'*60}")

    baseline = all_results[0]['test_f1']
    for r in all_results:
        gain = r['test_f1'] - baseline
        gain_str = f"(+{gain:.4f})" if gain > 0 else f"({gain:.4f})"
        print(f"  {r['description']:<35} "
              f"{r['best_val_f1']:>8.4f} "
              f"{r['test_f1']:>8.4f} "
              f"{gain_str:>10} "
              f"{r['time_min']:>6.1f}m")

    print(f"\n✅ All experiments complete!")
    print(f"📊 Full results: {SAVE_DIR}/ablation_results.csv")


def _save_results(results: list):
    """Save results to CSV after each experiment."""
    csv_path = os.path.join(SAVE_DIR, 'ablation_results.csv')
    keys = ['experiment', 'description', 'use_audio', 'use_speaker',
            'context', 'best_val_f1', 'test_f1', 'test_loss', 'time_min']
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)


if __name__ == '__main__':
    main()