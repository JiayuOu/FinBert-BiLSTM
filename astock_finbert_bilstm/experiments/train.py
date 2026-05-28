from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from ..config import ExperimentConfig
from ..data.dataset import (
    StockSequenceDataset,
    chronological_split,
    create_sequences,
    fit_transform_market_scaler,
    fit_transform_text_scaler,
)
from ..modeling.metrics import classification_metrics
from ..modeling.models import build_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _predict(model: nn.Module, loader: DataLoader, device: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_weights: list[np.ndarray] = []
    with torch.no_grad():
        for market, text, labels in loader:
            market = market.to(device)
            text = text.to(device)
            logits, weights = model(market, text)
            all_probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
            all_labels.append(labels.numpy())
            all_weights.append(weights.cpu().numpy())
    return np.concatenate(all_labels), np.concatenate(all_probs), np.concatenate(all_weights)


def train_experiment(dataset_csv: str | Path, output_dir: str | Path, config: ExperimentConfig) -> dict[str, object]:
    set_seed(config.seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(dataset_csv)
    arrays = create_sequences(df, config.market_features, config.text_features, config.window_size)
    train_idx, val_idx, test_idx = chronological_split(arrays.meta, config.test_ratio, config.val_ratio)
    market_scaled, market_scaler = fit_transform_market_scaler(arrays.market, train_idx)
    text_values = arrays.text
    text_scaler = None
    if config.scale_text_features:
        text_values, text_scaler = fit_transform_text_scaler(arrays.text, train_idx)

    loaders = {
        "train": DataLoader(
            StockSequenceDataset(market_scaled[train_idx], text_values[train_idx], arrays.labels[train_idx]),
            batch_size=config.batch_size,
            shuffle=True,
        ),
        "val": DataLoader(
            StockSequenceDataset(market_scaled[val_idx], text_values[val_idx], arrays.labels[val_idx]),
            batch_size=config.batch_size,
            shuffle=False,
        ),
        "test": DataLoader(
            StockSequenceDataset(market_scaled[test_idx], text_values[test_idx], arrays.labels[test_idx]),
            batch_size=config.batch_size,
            shuffle=False,
        ),
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(
        config.model,
        market_dim=len(config.market_features),
        text_dim=len(config.text_features),
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        dropout=config.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()
    history: list[dict[str, float]] = []
    best_val_f1 = -1.0
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(1, config.epochs + 1):
        model.train()
        total_loss = 0.0
        total_seen = 0
        for market, text, labels in loaders["train"]:
            market = market.to(device)
            text = text.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits, _ = model(market, text)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * labels.shape[0]
            total_seen += int(labels.shape[0])

        val_y, val_probs, _ = _predict(model, loaders["val"], device)
        val_metrics = classification_metrics(val_y, val_probs)
        row = {"epoch": epoch, "train_loss": total_loss / max(total_seen, 1), "val_f1": float(val_metrics["f1"])}
        history.append(row)

        if float(val_metrics["f1"]) > best_val_f1:
            best_val_f1 = float(val_metrics["f1"])
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    test_y, test_probs, test_weights = _predict(model, loaders["test"], device)
    test_metrics = classification_metrics(test_y, test_probs)
    attention_summary = {
        "market_weight_mean": float(test_weights[:, 0].mean()),
        "text_weight_mean": float(test_weights[:, 1].mean()),
    }
    results = {"config": config.to_dict(), "history": history, "test": test_metrics, "attention": attention_summary}

    torch.save(
        {
            "model_state": model.state_dict(),
            "config": config.to_dict(),
            "market_scaler_path": "market_scaler.joblib",
            "text_scaler_path": "text_scaler.joblib" if text_scaler is not None else None,
        },
        output_dir / "model.pt",
    )
    config.save_json(output_dir / "config.json")
    joblib.dump(market_scaler, output_dir / "market_scaler.joblib")
    if text_scaler is not None:
        joblib.dump(text_scaler, output_dir / "text_scaler.joblib")
    arrays.meta.iloc[test_idx].assign(
        y_true=test_y,
        prob_down=test_probs[:, 0],
        prob_up=test_probs[:, 1],
        market_weight=test_weights[:, 0],
        text_weight=test_weights[:, 1],
    ).to_csv(output_dir / "test_predictions.csv", index=False)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train A-share FinBERT + BiLSTM fusion models.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="attention_fusion", choices=["market_lstm", "market_bilstm", "concat_fusion", "attention_fusion"])
    parser.add_argument("--window-size", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--scale-text-features", action="store_true", help="Standardize text features using the train split only.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.dataset, nrows=1)
    text_features = [c for c in df.columns if c.startswith("text_") and c != "text_only"]
    config = ExperimentConfig(
        model=args.model,
        window_size=args.window_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_size=args.hidden_size,
        learning_rate=args.learning_rate,
        text_features=text_features or ExperimentConfig().text_features,
        scale_text_features=args.scale_text_features,
    )
    results = train_experiment(args.dataset, args.output_dir, config)
    print(json.dumps(results["test"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
