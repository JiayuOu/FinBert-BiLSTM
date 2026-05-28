from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from ..config import ExperimentConfig
from ..data.dataset import create_sequences
from ..modeling.models import build_model


def predict_one(checkpoint_path: str | Path, dataset_csv: str | Path, stock_code: str, trade_date: str) -> dict[str, object]:
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.is_dir():
        checkpoint_path = checkpoint_path / "model.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = ExperimentConfig(**checkpoint["config"])
    df = pd.read_csv(dataset_csv)
    arrays = create_sequences(df, config.market_features, config.text_features, config.window_size)
    mask = (arrays.meta["stock_code"].astype(str) == str(stock_code)) & (arrays.meta["trade_date"].astype(str) == trade_date)
    if not mask.any():
        raise ValueError(f"No sequence found for {stock_code} at {trade_date}.")
    idx = int(np.flatnonzero(mask.to_numpy())[0])

    scaler = joblib.load(checkpoint_path.parent / checkpoint.get("market_scaler_path", "market_scaler.joblib"))
    market = scaler.transform(arrays.market[idx].reshape(-1, arrays.market.shape[-1])).reshape(1, config.window_size, -1)
    text = arrays.text[idx].reshape(1, -1)

    model = build_model(
        config.model,
        market_dim=len(config.market_features),
        text_dim=len(config.text_features),
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        dropout=config.dropout,
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    with torch.no_grad():
        logits, weights = model(torch.tensor(market, dtype=torch.float32), torch.tensor(text, dtype=torch.float32))
        probs = torch.softmax(logits, dim=-1).numpy()[0]
        branch_weights = weights.numpy()[0]
    return {
        "stock_code": stock_code,
        "trade_date": trade_date,
        "prob_down": float(probs[0]),
        "prob_up": float(probs[1]),
        "prediction": "up" if probs[1] >= probs[0] else "down",
        "market_weight": float(branch_weights[0]),
        "text_weight": float(branch_weights[1]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict one A-share movement sample.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--stock-code", required=True)
    parser.add_argument("--trade-date", required=True)
    args = parser.parse_args()
    print(json.dumps(predict_one(args.checkpoint, args.dataset, args.stock_code, args.trade_date), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
