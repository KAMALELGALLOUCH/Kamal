import torch
import torch.nn as nn
from transformers import AutoModel

MODEL_PATH = '/data8/zhangxin/comp5423/gte-modernbert-base'


class TextBaselineModel(nn.Module):
    """
    Baseline: text-only model using GTE-ModernBERT
    Predicts both emotion (7 classes) and sentiment (3 classes)
    """
    def __init__(self, num_emotions=7, num_sentiments=3, dropout=0.3):
        super().__init__()

        # text encoder
        
        self.encoder = AutoModel.from_pretrained(
            MODEL_PATH, 
            local_files_only=True,
            low_cpu_mem_usage=True
        )
        for param in self.encoder.parameters():
            param.requires_grad = False

        hidden_size = self.encoder.config.hidden_size
        print(f"[TextBaseline] Encoder hidden size: {hidden_size}")

        # shared dropout
        self.dropout = nn.Dropout(dropout)

        # emotion head
        self.emotion_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_emotions)
        )

        # sentiment head
        self.sentiment_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_sentiments)
        )

    def forward(self, input_ids, attention_mask):
        # encode text
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # use [CLS] token representation
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)

        # predictions
        emotion_logits   = self.emotion_head(cls_output)
        sentiment_logits = self.sentiment_head(cls_output)

        return emotion_logits, sentiment_logits


if __name__ == '__main__':
    print("Testing TextBaselineModel...")
    device = torch.device('cpu')
    print(f"Using device: {device}")

    model = TextBaselineModel().to(device)
    model = model.float()  # ensure float32 throughout

    total_params = sum(p.numel() for p in model.parameters())
    trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters    : {total_params:,}")
    print(f"Trainable parameters: {trainable:,}")

    input_ids      = torch.randint(0, 1000, (4, 128))
    attention_mask = torch.ones(4, 128).long()
   

    emotion_logits, sentiment_logits = model(input_ids, attention_mask)

    print(f"\n✅ Forward pass successful!")
    print(f"  emotion_logits shape   : {emotion_logits.shape}")
    print(f"  sentiment_logits shape : {sentiment_logits.shape}")
    print(f"\n✅ Baseline model ready!")