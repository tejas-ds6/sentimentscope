"""
Data loading utilities for SentimentScope.
Supports SemEval-2014 format and a simple CSV format.
"""

import json, csv, random
from typing import List, Dict, Tuple


def load_semeval_json(path: str) -> List[Dict]:
    """Load pre-processed SemEval-2014 data from JSON."""
    with open(path) as f:
        return json.load(f)


def load_csv(path: str) -> List[Dict]:
    """
    Load from CSV with columns: text, aspect, label
    label must be one of: positive / neutral / negative
    """
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row["label"].strip().lower()
            if label in ("positive", "neutral", "negative"):
                rows.append({
                    "text":   row["text"].strip(),
                    "aspect": row["aspect"].strip().lower(),
                    "label":  label,
                })
    return rows


def train_val_split(data: List[Dict], val_ratio: float = 0.15,
                    seed: int = 42) -> Tuple[List[Dict], List[Dict]]:
    random.seed(seed)
    shuffled = data[:]
    random.shuffle(shuffled)
    cut = int(len(shuffled) * (1 - val_ratio))
    return shuffled[:cut], shuffled[cut:]


def make_demo_dataset(n: int = 200) -> List[Dict]:
    """
    Generate a small synthetic dataset for quick local testing.
    Do NOT use for real evaluation.
    """
    templates = [
        ("The {aspect} is absolutely amazing and works perfectly.",  "positive"),
        ("I love the {aspect}, it exceeded all my expectations.",    "positive"),
        ("The {aspect} is terrible and broke after a week.",         "negative"),
        ("Worst {aspect} I have ever seen, complete waste of money.","negative"),
        ("The {aspect} is okay, nothing special about it.",          "neutral"),
        ("The {aspect} works as expected, average quality.",         "neutral"),
    ]
    aspects = ["battery", "screen", "camera", "performance", "design",
               "speaker", "price", "keyboard", "touchpad", "charging"]
    random.seed(0)
    dataset = []
    for _ in range(n):
        tmpl, label = random.choice(templates)
        aspect = random.choice(aspects)
        dataset.append({"text": tmpl.format(aspect=aspect), "aspect": aspect, "label": label})
    return dataset
