"""
LLM authorship classification demo: CNN vs BiLSTM in PyTorch (Automated RQ2 Multi-Input).

Install:
    pip install torch pandas scikit-learn matplotlib

Run:
    python train.py --data dataset_demo.csv --epochs 12
"""
import argparse
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix

PAD, UNK = "<pad>", "<unk>"

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def tokenize(text):
    return re.findall(r"[a-z]+(?:'[a-z]+)?|[0-9]+|[^\\w\\s]", str(text).lower())

def build_vocab(texts, min_freq=1, max_vocab=20000):
    counts = Counter(tok for text in texts for tok in tokenize(text))
    vocab = {PAD: 0, UNK: 1}
    for tok, freq in counts.most_common(max_vocab - len(vocab)):
        if freq >= min_freq and tok not in vocab:
            vocab[tok] = len(vocab)
    return vocab

def encode(text, vocab, max_len=220):
    ids = [vocab.get(tok, vocab[UNK]) for tok in tokenize(text)[:max_len]]
    if not ids:
        ids = [vocab[UNK]]
    return ids + [vocab[PAD]] * (max_len - len(ids))

class TextDataset(Dataset):
    def __init__(self, frame, vocab, label_to_id, text_col, max_len):
        self.texts = frame[text_col].astype(str).tolist()
        self.labels = [label_to_id[x] for x in frame["LLM_name"].tolist()]
        self.vocab, self.max_len = vocab, max_len
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return (torch.tensor(encode(self.texts[idx], self.vocab, self.max_len), dtype=torch.long),
                torch.tensor(self.labels[idx], dtype=torch.long))

class TextCNN(nn.Module):
    def __init__(self, vocab_size, n_classes, emb_dim=96, channels=96, kernels=(3,4,5), dropout=0.35):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(emb_dim, channels, k, padding=k//2) for k in kernels])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(channels * len(kernels), n_classes)
    def forward(self, x):
        emb = self.embedding(x).transpose(1, 2)
        pooled = [torch.relu(conv(emb)).amax(dim=2) for conv in self.convs]
        return self.fc(self.dropout(torch.cat(pooled, dim=1)))

class TextBiLSTM(nn.Module):
    def __init__(self, vocab_size, n_classes, emb_dim=96, hidden=96, dropout=0.35):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden * 2, n_classes)
    def forward(self, x):
        emb = self.embedding(x)
        outputs, _ = self.lstm(emb)
        mask = (x != 0).unsqueeze(-1).to(outputs.dtype)
        pooled = (outputs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return self.fc(self.dropout(pooled))

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, total, correct = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)

@torch.no_grad()
def evaluate(model, loader, device, labels):
    model.eval()
    ys, preds = [], []
    for x, y in loader:
        pred = model(x.to(device)).argmax(1).cpu().tolist()
        preds.extend(pred)
        ys.extend(y.tolist())
    acc = accuracy_score(ys, preds)
    p, r, f1, _ = precision_recall_fscore_support(
        ys, preds, labels=list(range(len(labels))), average="macro", zero_division=0
    )
    return {"accuracy": acc, "macro_precision": p, "macro_recall": r, "macro_f1": f1}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset_demo.csv")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=220)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    seed_everything(args.seed)

    df = pd.read_csv(args.data).dropna(subset=["LLM_name", "LLM_Input", "LLM_output", "prompt_id"]).copy()
    
    # Pre-construct the combined text representation column
    df["input_plus_output"] = df["LLM_Input"].astype(str) + " [SEP] " + df["LLM_output"].astype(str)

    labels = sorted(df["LLM_name"].unique().tolist())
    if len(labels) < 3:
        raise ValueError("Dataset must contain at least 3 distinct LLM_name labels.")
    label_to_id = {name: i for i, name in enumerate(labels)}

    # Split by prompt_id so the same prompt does not appear in both train and test.
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=args.seed)
    train_idx, test_idx = next(splitter.split(df, groups=df["prompt_id"]))
    train_df, test_df = df.iloc[train_idx].copy(), df.iloc[test_idx].copy()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Labels: {label_to_id}")

    # Automate RQ2 Multi-Input Comparison: Input Only, Output Only, Combined
    input_modes = {
        "Input_Only": "LLM_Input",
        "Output_Only": "LLM_output",
        "Combined": "input_plus_output"
    }

    all_results = []

    for mode_name, text_col in input_modes.items():
        print(f"\n==============================================")
        print(f"Running RQ2 Evaluation Scenario: {mode_name} (Using column: {text_col})")
        print(f"==============================================")

        # Build vocabulary specific to this input configuration using training text only
        vocab = build_vocab(train_df[text_col].astype(str).tolist())
        train_ds = TextDataset(train_df, vocab, label_to_id, text_col, args.max_len)
        test_ds = TextDataset(test_df, vocab, label_to_id, text_col, args.max_len)
        
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size)

        for model_name, model_class in [
            ("CNN", TextCNN),
            ("BiLSTM", TextBiLSTM),
        ]:
            model = model_class(len(vocab), len(labels)).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            criterion = nn.CrossEntropyLoss()

            print(f"\n--- Training {model_name} on {mode_name} ---")
            for epoch in range(1, args.epochs + 1):
                loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
                if epoch % 4 == 0 or epoch == args.epochs:
                    print(f"Epoch {epoch:02d}/{args.epochs} | loss={loss:.4f} | train_acc={train_acc:.3f}")

            metrics = evaluate(model, test_loader, device, labels)
            print(f"Test Result [{model_name} | {mode_name}] -> Accuracy: {metrics['accuracy']:.3f}")

            # Record results with identifiers
            record = {"Model": model_name, "Input_Mode": mode_name}
            record.update(metrics)
            all_results.append(record)

            # Save checkpoint
            ckpt_filename = f"{model_name.lower()}_{mode_name.lower()}_checkpoint.pt"
            torch.save({
                "model_state_dict": model.state_dict(), 
                "vocab": vocab,
                "labels": labels, 
                "model_name": model_name, 
                "input_mode": mode_name,
                "max_len": args.max_len
            }, ckpt_filename)

    # Compile and save final summary table for RQ2
    results_df = pd.DataFrame(all_results)
    results_df.to_csv("rq2_mult_input_metrics_summary.csv", index=False)
    
    print("\n==============================================")
    print("RQ2 Multi-Input Comparison Summary:")
    print("==============================================")
    print(results_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nSaved all checkpoints and 'rq2_mult_input_metrics_summary.csv'")

if __name__ == "__main__":
    main()
