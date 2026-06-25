"""
SentimentScope — Aspect-Based Sentiment Analysis Engine
Author  : Tejas D S  |  github.com/tejas-ds6/sentimentscope
"""

import torch
import torch.nn as nn
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict
import os

LABELS    = ["negative", "neutral", "positive"]
LABEL2ID  = {l: i for i, l in enumerate(LABELS)}
ID2LABEL  = {i: l for i, l in enumerate(LABELS)}
MODEL_NAME = "distilbert-base-uncased"


# ──────────────────────────────────────────────────────────────────────────────
class AspectSentimentDataset(Dataset):
    """
    Each sample: {"text": "...", "aspect": "battery life", "label": "positive"}
    The model receives concatenated "[text] [SEP] [aspect]" so it can attend
    to the specific aspect while reading the review.
    """
    def __init__(self, samples: List[Dict], tokenizer, max_len: int = 128):
        self.samples   = samples
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s   = self.samples[idx]
        enc = self.tokenizer(
            s["text"], s["aspect"],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "label":          torch.tensor(LABEL2ID[s["label"]], dtype=torch.long),
        }


# ──────────────────────────────────────────────────────────────────────────────
class SentimentScopeModel:
    """Fine-tuned DistilBERT for aspect-level sentiment classification."""

    def __init__(self, model_path: str = None, device: str = None):
        self.device    = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
        if model_path and os.path.isdir(model_path):
            self.model = DistilBertForSequenceClassification.from_pretrained(
                model_path, num_labels=3, id2label=ID2LABEL, label2id=LABEL2ID
            )
        else:
            self.model = DistilBertForSequenceClassification.from_pretrained(
                MODEL_NAME, num_labels=3, id2label=ID2LABEL, label2id=LABEL2ID
            )
        self.model.to(self.device)

    # ── Training ──────────────────────────────────────────────────────────────
    def train(self, train_data: List[Dict], val_data: List[Dict],
              epochs: int = 3, batch_size: int = 16, lr: float = 2e-5,
              save_path: str = "model_output") -> List[Dict]:
        train_ds = AspectSentimentDataset(train_data, self.tokenizer)
        val_ds   = AspectSentimentDataset(val_data,   self.tokenizer)
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_dl   = DataLoader(val_ds,   batch_size=batch_size)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        total_steps = epochs * len(train_dl)
        scheduler   = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=0.1, total_iters=total_steps
        )
        history = []
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for batch in train_dl:
                optimizer.zero_grad()
                out = self.model(
                    input_ids=batch["input_ids"].to(self.device),
                    attention_mask=batch["attention_mask"].to(self.device),
                    labels=batch["label"].to(self.device),
                )
                out.loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                total_loss += out.loss.item()

            val_acc  = self._evaluate(val_dl)
            avg_loss = total_loss / len(train_dl)
            history.append({"epoch": epoch + 1, "loss": round(avg_loss, 4),
                             "val_acc": round(val_acc, 4)})
            print(f"Epoch {epoch+1}/{epochs} | loss={avg_loss:.4f} | val_acc={val_acc:.4f}")

        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"Model saved to {save_path}")
        return history

    # ── Evaluation ────────────────────────────────────────────────────────────
    def _evaluate(self, dl: DataLoader) -> float:
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in dl:
                out    = self.model(
                    input_ids=batch["input_ids"].to(self.device),
                    attention_mask=batch["attention_mask"].to(self.device),
                )
                preds   = out.logits.argmax(dim=-1).cpu()
                correct += (preds == batch["label"]).sum().item()
                total   += len(batch["label"])
        return correct / total if total else 0.0

    # ── Inference ─────────────────────────────────────────────────────────────
    def predict(self, text: str, aspects: List[str]) -> List[Dict]:
        """
        Returns aspect-level sentiment for each aspect term.

        Example:
            predict("Battery is great but the screen is terrible",
                    ["battery", "screen"])
            → [{"aspect": "battery", "sentiment": "positive", "confidence": 0.97, ...},
               {"aspect": "screen",  "sentiment": "negative", "confidence": 0.93, ...}]
        """
        self.model.eval()
        results = []
        for aspect in aspects:
            enc = self.tokenizer(
                text, aspect,
                max_length=128, padding="max_length",
                truncation=True, return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                logits = self.model(**enc).logits[0]
            probs     = torch.softmax(logits, dim=-1).cpu().tolist()
            label_idx = int(torch.argmax(logits).item())
            results.append({
                "aspect":     aspect,
                "sentiment":  ID2LABEL[label_idx],
                "confidence": round(probs[label_idx], 4),
                "scores":     {ID2LABEL[i]: round(p, 4) for i, p in enumerate(probs)},
            })
        return results
