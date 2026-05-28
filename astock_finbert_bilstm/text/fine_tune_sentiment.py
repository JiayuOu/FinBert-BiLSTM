from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("USE_TF", "0")

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from ..config import DEFAULT_FINBERT_MODEL


LABEL_MAP = {
    "negative": 0,
    "neg": 0,
    "bearish": 0,
    "0": 0,
    "neutral": 1,
    "neu": 1,
    "1": 1,
    "positive": 2,
    "pos": 2,
    "bullish": 2,
    "2": 2,
}


def normalize_labels(series: pd.Series) -> list[int]:
    labels = []
    for value in series.astype(str).str.strip().str.lower():
        if value not in LABEL_MAP:
            raise ValueError(f"Unsupported label `{value}`. Use negative/neutral/positive or 0/1/2.")
        labels.append(LABEL_MAP[value])
    return labels


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    pred = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, pred, average="macro", zero_division=0)
    return {
        "accuracy": accuracy_score(labels, pred),
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a Chinese BERT/FinBERT sentiment classifier.")
    parser.add_argument("--train-csv", required=True, help="CSV with columns: text,label")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-model", default=DEFAULT_FINBERT_MODEL)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--max-samples", type=int, default=None, help="Optional quick-run sample size before splitting.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from datasets import Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

    df = pd.read_csv(args.train_csv)
    if not {"text", "label"}.issubset(df.columns):
        raise ValueError("sentiment training CSV must contain columns: text,label")
    df = df.dropna(subset=["text", "label"]).copy()
    if args.max_samples and len(df) > args.max_samples:
        df = df.sample(n=args.max_samples, random_state=args.seed).reset_index(drop=True)
    df["label"] = normalize_labels(df["label"])

    train_df, val_df = train_test_split(
        df[["text", "label"]],
        test_size=args.val_ratio,
        random_state=args.seed,
        stratify=df["label"] if df["label"].nunique() > 1 else None,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    def tokenize(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=args.max_length)

    train_ds = Dataset.from_pandas(train_df.reset_index(drop=True)).map(tokenize, batched=True)
    val_ds = Dataset.from_pandas(val_df.reset_index(drop=True)).map(tokenize, batched=True)
    train_ds = train_ds.remove_columns(["text"])
    val_ds = val_ds.remove_columns(["text"])

    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=3,
        id2label={0: "negative", 1: "neutral", 2: "positive"},
        label2id={"negative": 0, "neutral": 1, "positive": 2},
    )

    output_dir = Path(args.output_dir)
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        seed=args.seed,
        report_to=[],
    )

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_ds,
        "eval_dataset": val_ds,
        "compute_metrics": compute_metrics,
    }
    try:
        trainer = Trainer(**trainer_kwargs, processing_class=tokenizer)
    except TypeError as exc:
        if "processing_class" not in str(exc):
            raise
        trainer = Trainer(**trainer_kwargs, tokenizer=tokenizer)
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"saved fine-tuned sentiment model to {output_dir}")


if __name__ == "__main__":
    main()
