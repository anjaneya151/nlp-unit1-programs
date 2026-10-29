"""
Program 24: Fine-Tuning BERT for text classification using Hugging Face
Fine-tunes bert-base-uncased for sentiment classification.
Uses a small slice of the IMDB dataset if downloadable, else a built-in toy dataset.
Run: python bert_finetune.py
"""
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          Trainer, TrainingArguments)

MODEL = "bert-base-uncased"


def load_data():
    try:
        from datasets import load_dataset
        ds = load_dataset("imdb")
        tr = ds["train"].shuffle(seed=42).select(range(1000))
        te = ds["test"].shuffle(seed=42).select(range(300))
        print("Using IMDB subset (1000 train / 300 test)")
        return tr["text"], tr["label"], te["text"], te["label"]
    except Exception as e:
        print("IMDB not available (%s). Using built-in toy dataset." % type(e).__name__)
        pos = ["I loved this movie", "Great acting and a wonderful story", "Absolutely brilliant film",
               "Superb direction and music", "An amazing experience", "Really enjoyable and moving",
               "Outstanding performance", "Fantastic characters and plot"]
        neg = ["Terrible movie, waste of time", "Boring plot and bad acting", "I hated this film",
               "Dull and disappointing", "Worst movie ever", "Poor script and direction",
               "Horrible experience", "Weak characters and boring story"]
        X = (pos + neg) * 4
        y = ([1] * len(pos) + [0] * len(neg)) * 4
        return X, y, ["What a great film", "Awful and boring", "I really loved it", "Terrible acting"], [1, 0, 1, 0]


class TextDataset(Dataset):
    def __init__(self, texts, labels, tok):
        self.enc = tok(list(texts), truncation=True, padding=True, max_length=128)
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: torch.tensor(v[i]) for k, v in self.enc.items()}
        item["labels"] = torch.tensor(self.labels[i])
        return item


def metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return {"accuracy": accuracy_score(p.label_ids, preds), "f1": f1_score(p.label_ids, preds)}


if __name__ == "__main__":
    Xtr, ytr, Xte, yte = load_data()
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2)
    train_ds, test_ds = TextDataset(Xtr, ytr, tok), TextDataset(Xte, yte, tok)

    args = TrainingArguments(output_dir="./bert_out", num_train_epochs=2,
                             per_device_train_batch_size=16, per_device_eval_batch_size=32,
                             learning_rate=2e-5, weight_decay=0.01, logging_steps=20,
                             save_strategy="no", report_to="none")
    trainer = Trainer(model=model, args=args, train_dataset=train_ds,
                      eval_dataset=test_ds, compute_metrics=metrics)
    trainer.train()
    print("Evaluation:", trainer.evaluate())

    model.eval()
    for s in ["This movie was absolutely wonderful!", "It was a boring, terrible waste of time."]:
        enc = tok(s, return_tensors="pt", truncation=True).to(model.device)
        with torch.no_grad():
            pred = model(**enc).logits.argmax(-1).item()
        print(f"{'POSITIVE' if pred else 'NEGATIVE'} | {s}")
    model.save_pretrained("./bert_sentiment_model")
    tok.save_pretrained("./bert_sentiment_model")
