from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ..config import ExperimentConfig
from .train import train_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run thesis ablation baselines.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--window-size", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--scale-text-features", action="store_true", help="Standardize text features using the train split only.")
    args = parser.parse_args()

    df = pd.read_csv(args.dataset, nrows=1)
    text_features = [c for c in df.columns if c.startswith("text_") and c != "text_only"]
    models = ["market_lstm", "market_bilstm", "concat_fusion", "attention_fusion"]
    summary = {}
    for model_name in models:
        config = ExperimentConfig(
            model=model_name,
            epochs=args.epochs,
            window_size=args.window_size,
            batch_size=args.batch_size,
            hidden_size=args.hidden_size,
            learning_rate=args.learning_rate,
            text_features=text_features or ExperimentConfig().text_features,
            scale_text_features=args.scale_text_features,
        )
        run_dir = Path(args.output_dir) / model_name
        result = train_experiment(args.dataset, run_dir, config)
        summary[model_name] = result["test"]

    out = Path(args.output_dir) / "ablation_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
