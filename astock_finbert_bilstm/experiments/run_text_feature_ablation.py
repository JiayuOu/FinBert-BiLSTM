from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from ..config import ExperimentConfig
from ..data.dataset import create_sequences
from .text_feature_sets import TEXT_FEATURE_SET_CHOICES, select_text_features
from .train import train_experiment

MODELS = ["market_lstm", "market_bilstm", "concat_fusion", "attention_fusion"]
DEFAULT_FEATURE_SETS = ["all", "sentiment_event", "embedding_only", "event_strength_only"]


def _run_name(feature_set: str, scale_text_features: bool) -> str:
    suffix = "scaled" if scale_text_features else "unscaled"
    return f"{feature_set}_{suffix}"


def _sequence_summary(df: pd.DataFrame, text_features: list[str], window_size: int) -> dict[str, Any]:
    arrays = create_sequences(
        df,
        market_features=ExperimentConfig().market_features,
        text_features=text_features,
        window_size=window_size,
    )
    labels = pd.Series(arrays.labels).value_counts().sort_index()
    return {
        "sequences": int(len(arrays.labels)),
        "label_0": int(labels.get(0, 0)),
        "label_1": int(labels.get(1, 0)),
        "start": str(arrays.meta["trade_date"].min()),
        "end": str(arrays.meta["trade_date"].max()),
    }


def _metrics_row(
    run_name: str,
    feature_set: str,
    scale_text_features: bool,
    n_text_features: int,
    sequence_summary: dict[str, Any],
    model_name: str,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    cm = metrics.get("confusion_matrix", [[None, None], [None, None]])
    row = {
        "run_name": run_name,
        "feature_set": feature_set,
        "scale_text_features": scale_text_features,
        "model": model_name,
        "n_text_features": n_text_features,
        **sequence_summary,
    }
    for key in ["accuracy", "precision", "recall", "f1", "auc"]:
        row[key] = metrics.get(key)
    row.update({"tn": cm[0][0], "fp": cm[0][1], "fn": cm[1][0], "tp": cm[1][1]})
    return row


def run_text_feature_ablation(
    dataset: str | Path,
    output_dir: str | Path,
    feature_sets: list[str],
    scale_text_features: bool,
    epochs: int,
    window_size: int,
    batch_size: int,
    hidden_size: int,
    learning_rate: float,
) -> pd.DataFrame:
    df = pd.read_csv(dataset)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for feature_set in feature_sets:
        text_features = select_text_features(list(df.columns), feature_set)
        sequence_summary = _sequence_summary(df, text_features, window_size)
        run_name = _run_name(feature_set, scale_text_features)
        run_dir = output_dir / run_name
        run_summary: dict[str, Any] = {}

        for model_name in MODELS:
            config = ExperimentConfig(
                model=model_name,
                epochs=epochs,
                window_size=window_size,
                batch_size=batch_size,
                hidden_size=hidden_size,
                learning_rate=learning_rate,
                text_features=text_features,
                scale_text_features=scale_text_features,
            )
            result = train_experiment(dataset, run_dir / model_name, config)
            metrics = result["test"]
            run_summary[model_name] = metrics
            rows.append(
                _metrics_row(
                    run_name,
                    feature_set,
                    scale_text_features,
                    len(text_features),
                    sequence_summary,
                    model_name,
                    metrics,
                )
            )

        with (run_dir / "ablation_summary.json").open("w", encoding="utf-8") as f:
            json.dump(run_summary, f, ensure_ascii=False, indent=2)

    comparison = pd.DataFrame(rows)
    comparison.to_csv(output_dir / "text_feature_comparison.csv", index=False)
    pivot = comparison.pivot(index="run_name", columns="model", values="auc")
    pivot.to_csv(output_dir / "text_feature_auc_pivot.csv")
    return comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run text feature-set ablations for A-share fusion models.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--feature-set", action="append", choices=TEXT_FEATURE_SET_CHOICES, default=None)
    parser.add_argument("--scale-text-features", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--window-size", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    feature_sets = args.feature_set or DEFAULT_FEATURE_SETS
    comparison = run_text_feature_ablation(
        dataset=args.dataset,
        output_dir=args.output_dir,
        feature_sets=feature_sets,
        scale_text_features=args.scale_text_features,
        epochs=args.epochs,
        window_size=args.window_size,
        batch_size=args.batch_size,
        hidden_size=args.hidden_size,
        learning_rate=args.learning_rate,
    )
    print(comparison.to_json(orient="records", force_ascii=False, indent=2))


if __name__ == "__main__":
    main()
