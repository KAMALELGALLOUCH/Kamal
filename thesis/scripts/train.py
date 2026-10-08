import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6'
import sys
import torch
import torch.nn as nn
import numpy as np
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, classification_report
from tqdm import tqdm

sys.path.append('/data8/luoyan/Kamal/thesis')
from data.meld_dataset import get_dataloaders, EMOTION2ID, SENTIMENT2ID
from models.text_baselines import TextBaselineModel

# config
DEVICE = torch.device('cpu')
EPOCHS     = 2
BATCH_SIZE = 4
LR         = 2e-5
MAX_LEN    = 64
SAVE_DIR   = '/data8/luoyan/Kamal/thesis/outputs'
os.makedirs(SAVE_DIR, exist_ok=True)

# emotion class weights (to handle imbalance)
EMOTION_COUNTS = [4710, 1743, 1205, 1109, 683, 271, 268]
EMOTION_WEIGHTS = torch.tensor(
    [1.0 / c for c in EMOTION_COUNTS], dtype=torch.float32
)
EMOTION_WEIGHTS = EMOTION_WEIGHTS / EMOTION_WEIGHTS.sum()
EMOTION_WEIGHTS = EMOTION_WEIGHTS.to(DEVICE)

ID2EMOTION   = {v: k for k, v in EMOTION2ID.items()}
ID2SENTIMENT = {v: k for k, v in SENTIMENT2ID.items()}


def train_epoch(model, loader, optimizer, scheduler,
                emo_criterion, sent_criterion):
    model.train()
    total_loss = 0
    all_emo_preds, all_emo_labels = [], []

    for batch in tqdm(loader, desc='Training'):
        input_ids      = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        emo_labels     = batch['emotion_label'].to(DEVICE)
        sent_labels    = batch['sentiment_label'].to(DEVICE)

        optimizer.zero_grad()
        emo_logits, sent_logits = model(input_ids, attention_mask)

        emo_loss  = emo_criterion(emo_logits, emo_labels)
        sent_loss = sent_criterion(sent_logits, sent_labels)
        loss = emo_loss + 0.5 * sent_loss  # emotion is primary task

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        preds = torch.argmax(emo_logits, dim=1).cpu().numpy()
        all_emo_preds.extend(preds)
        all_emo_labels.extend(emo_labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    f1 = f1_score(all_emo_labels, all_emo_preds, average='weighted')
    return avg_loss, f1


def evaluate(model, loader, emo_criterion, sent_criterion, split='Val'):
    model.eval()
    total_loss = 0
    all_emo_preds, all_emo_labels = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f'Evaluating {split}'):
            input_ids      = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            emo_labels     = batch['emotion_label'].to(DEVICE)
            sent_labels    = batch['sentiment_label'].to(DEVICE)

            emo_logits, sent_logits = model(input_ids, attention_mask)
            emo_loss  = emo_criterion(emo_logits, emo_labels)
            sent_loss = sent_criterion(sent_logits, sent_labels)
            loss = emo_loss + 0.5 * sent_loss

            total_loss += loss.item()
            preds = torch.argmax(emo_logits, dim=1).cpu().numpy()
            all_emo_preds.extend(preds)
            all_emo_labels.extend(emo_labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    f1 = f1_score(all_emo_labels, all_emo_preds, average='weighted')

    # full report on test
    if split == 'Test':
        labels     = list(ID2EMOTION.keys())
        label_names = [ID2EMOTION[i] for i in labels]
        print("\n" + classification_report(
            all_emo_labels, all_emo_preds,
            labels=labels, target_names=label_names
        ))

    return avg_loss, f1


def main():
    print(f"Device: {DEVICE}")
    print("Loading data...")
    train_loader, dev_loader, test_loader = get_dataloaders(
        batch_size=BATCH_SIZE, max_text_len=MAX_LEN
    )

    print("Building model...")
    torch.cuda.empty_cache()
    model = TextBaselineModel()
    model = model.float()
    model = model.to(DEVICE)

    # loss functions
    emo_criterion  = nn.CrossEntropyLoss(weight=EMOTION_WEIGHTS)
    sent_criterion = nn.CrossEntropyLoss()

    # optimizer & scheduler
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps
    )

    print(f"\nStarting training for {EPOCHS} epochs...")
    best_val_f1   = 0
    best_model_path = os.path.join(SAVE_DIR, 'best_model.pt')

    for epoch in range(1, EPOCHS + 1):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{EPOCHS}")
        print(f"{'='*50}")

        train_loss, train_f1 = train_epoch(
            model, train_loader, optimizer, scheduler,
            emo_criterion, sent_criterion
        )
        val_loss, val_f1 = evaluate(
            model, dev_loader, emo_criterion, sent_criterion, 'Val'
        )

        print(f"\nTrain Loss: {train_loss:.4f} | Train F1: {train_f1:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Val   F1: {val_f1:.4f}")

        # save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), best_model_path)
            print(f"✅ New best model saved! Val F1: {val_f1:.4f}")

    # final test evaluation
    print(f"\n{'='*50}")
    print("Final Test Evaluation")
    print(f"{'='*50}")
    model.load_state_dict(torch.load(best_model_path))
    test_loss, test_f1 = evaluate(
        model, test_loader, emo_criterion, sent_criterion, 'Test'
    )
    print(f"Test Loss: {test_loss:.4f} | Test F1: {test_f1:.4f}")
    print(f"\n🎉 Training complete! Best Val F1: {best_val_f1:.4f}")


if __name__ == '__main__':
    main()