import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
from transformers import AutoModel

TEXT_MODEL_PATH = '/data8/zhangxin/comp5423/gte-modernbert-base'


# ── Audio Encoder ──────────────────────────────────────────────
class AudioEncoder(nn.Module):
    """
    Extracts audio features using Mel Spectrogram + CNN.
    Input : raw waveform [B, 1, T]
    Output: audio features [B, hidden_size]
    """
    def __init__(self, hidden_size=768, dropout=0.3):
        super().__init__()

        self.mel_transform = T.MelSpectrogram(
            sample_rate=16000,
            n_fft=512,
            hop_length=160,
            n_mels=80
        )
        self.amplitude_to_db = T.AmplitudeToDB()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 16))
        )

        cnn_out_size = 128 * 4 * 16
        self.projection = nn.Sequential(
            nn.Linear(cnn_out_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, waveform):
        mel      = self.mel_transform(waveform)
        mel      = self.amplitude_to_db(mel)
        features = self.cnn(mel)
        features = features.view(features.size(0), -1)
        return self.projection(features)   # [B, hidden_size]


# ── Speaker Embedding ──────────────────────────────────────────
class SpeakerEmbedding(nn.Module):
   
    def __init__(self, num_speakers=7, embedding_dim=64, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(num_speakers, embedding_dim)
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, speaker_ids):
        # speaker_ids: [B]
        emb = self.embedding(speaker_ids)      # [B, embedding_dim]
        return self.projection(emb)            # [B, embedding_dim]


# ── Cross Modal Attention ──────────────────────────────────────
class CrossModalAttention(nn.Module):
    
    def __init__(self, hidden_size=768, num_heads=8, dropout=0.1):
        super().__init__()

        self.text_to_audio = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.audio_to_text = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm1   = nn.LayerNorm(hidden_size)
        self.norm2   = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text_feat, audio_feat):
        text_seq  = text_feat.unsqueeze(1)   # [B, 1, H]
        audio_seq = audio_feat.unsqueeze(1)  # [B, 1, H]

        text_attended, _ = self.text_to_audio(
            query=text_seq, key=audio_seq, value=audio_seq
        )
        text_out = self.norm1(text_seq + self.dropout(text_attended))

        audio_attended, _ = self.audio_to_text(
            query=audio_seq, key=text_seq, value=text_seq
        )
        audio_out = self.norm2(audio_seq + self.dropout(audio_attended))

        return text_out.squeeze(1), audio_out.squeeze(1)  # [B, H], [B, H]


# ── Gated Fusion ───────────────────────────────────────────────
class GatedFusion(nn.Module):
    """
    Dynamically weights text vs audio contribution.
    Gate ≈ 1 → trust text more.
    Gate ≈ 0 → trust audio more.
    """
    def __init__(self, hidden_size=768, dropout=0.3):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid()
        )
        self.fusion  = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text_feat, audio_feat):
        combined = torch.cat([text_feat, audio_feat], dim=-1)
        gate     = self.gate(combined)
        fused    = self.fusion(combined)
        output   = gate * text_feat + (1 - gate) * audio_feat + fused
        return self.dropout(output)   # [B, hidden_size]


# ── Full Multimodal Model ──────────────────────────────────────
class MultimodalEmotionModel(nn.Module):
    """
    Full multimodal model for emotion recognition on MELD.

    Modalities:
      - Text    : GTE-ModernBERT (CLS token) + conversation context
      - Audio   : CNN + MelSpectrogram
      - Speaker : learned speaker embeddings (Chandler, Monica, etc.)

    Fusion:
      - Cross-Modal Attention (bidirectional text ↔ audio)
      - Gated Fusion
      - Speaker embedding concatenated before classifier

    Outputs:
      - emotion logits   [B, 7]
      - sentiment logits [B, 3]
    """

    def __init__(self, num_emotions=7, num_sentiments=3,
                 hidden_size=768, speaker_emb_dim=64,
                 num_speakers=7, dropout=0.3, freeze_text=True):
        super().__init__()

        # ── text encoder ───────────────────────────────────────
        self.text_encoder = AutoModel.from_pretrained(
            TEXT_MODEL_PATH,
            local_files_only=True,
            low_cpu_mem_usage=True
        )
        if freeze_text:
            for param in self.text_encoder.parameters():
                param.requires_grad = False
        print(f"[Model] Text encoder hidden size : "
              f"{self.text_encoder.config.hidden_size}")
        print(f"[Model] freeze_text              : {freeze_text}")

        # ── audio encoder ──────────────────────────────────────
        self.audio_encoder = AudioEncoder(
            hidden_size=hidden_size, dropout=dropout
        )
        print(f"[Model] Audio encoder            : CNN + MelSpec")

        # ── speaker embedding ──────────────────────────────────
        self.speaker_embedding = SpeakerEmbedding(
            num_speakers=num_speakers,
            embedding_dim=speaker_emb_dim,
            dropout=dropout
        )
        print(f"[Model] Speaker embedding dim    : {speaker_emb_dim}")

        # ── cross-modal attention ──────────────────────────────
        self.cross_attention = CrossModalAttention(
            hidden_size=hidden_size, num_heads=8, dropout=dropout
        )

        # ── gated fusion ───────────────────────────────────────
        self.gated_fusion = GatedFusion(
            hidden_size=hidden_size, dropout=dropout
        )

        # ── dropout ────────────────────────────────────────────
        self.dropout = nn.Dropout(dropout)

        # ── classifier input size ──────────────────────────────
        # fused (768) + speaker_emb (64) = 832
        classifier_input = hidden_size + speaker_emb_dim

        # ── emotion head ───────────────────────────────────────
        self.emotion_head = nn.Sequential(
            nn.Linear(classifier_input, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_emotions)
        )

        # ── sentiment head ─────────────────────────────────────
        self.sentiment_head = nn.Sequential(
            nn.Linear(classifier_input, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_sentiments)
        )

    def forward(self, input_ids, attention_mask, waveform, speaker_ids):
        # ── text encoding ──────────────────────────────────────
        text_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        text_feat = text_outputs.last_hidden_state[:, 0, :]  # CLS
        text_feat = self.dropout(text_feat)                   # [B, 768]

        # ── audio encoding ─────────────────────────────────────
        audio_feat = self.audio_encoder(waveform)             # [B, 768]

        # ── cross-modal attention ──────────────────────────────
        text_attended, audio_attended = self.cross_attention(
            text_feat, audio_feat
        )                                                      # [B, 768]

        # ── gated fusion ───────────────────────────────────────
        fused = self.gated_fusion(text_attended, audio_attended)
        fused = self.dropout(fused)                           # [B, 768]

        # ── speaker embedding ──────────────────────────────────
        speaker_emb = self.speaker_embedding(speaker_ids)     # [B, 64]

        # ── combine fused + speaker ────────────────────────────
        combined = torch.cat([fused, speaker_emb], dim=-1)    # [B, 832]

        # ── predictions ────────────────────────────────────────
        emotion_logits   = self.emotion_head(combined)        # [B, 7]
        sentiment_logits = self.sentiment_head(combined)      # [B, 3]

        return emotion_logits, sentiment_logits


# ── Quick Test ─────────────────────────────────────────────────
if __name__ == '__main__':
    print("Testing MultimodalEmotionModel with Speaker Embedding...\n")
    device = torch.device('cpu')

    model = MultimodalEmotionModel(
        freeze_text=True,
        dropout=0.3
    ).to(device)
    model = model.float()

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters()
                    if p.requires_grad)
    print(f"\nTotal parameters    : {total:,}")
    print(f"Trainable parameters: {trainable:,}")

    # dummy inputs
    B = 4
    input_ids      = torch.randint(0, 1000, (B, 128))
    attention_mask = torch.ones(B, 128, dtype=torch.long)
    waveform       = torch.randn(B, 1, 16000 * 5)
    speaker_ids    = torch.tensor([0, 1, 2, 6])  # chandler, monica, rachel, other

    emo_logits, sent_logits = model(
        input_ids, attention_mask, waveform, speaker_ids
    )

    print(f"\n✅ Forward pass successful!")
    print(f"  emotion_logits shape   : {emo_logits.shape}")
    print(f"  sentiment_logits shape : {sent_logits.shape}")
    print(f"\n✅ Model with Speaker Embedding ready!")