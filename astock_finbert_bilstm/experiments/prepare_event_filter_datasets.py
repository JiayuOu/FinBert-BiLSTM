from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from ..config import ExperimentConfig
from ..data.dataset import build_aligned_dataset, create_sequences


EVENT_FILTERS: list[dict[str, Any]] = [
    {"name": "baseline", "filters": {}},
    {"name": "news_count_ge2", "filters": {"min_text_news_count": 2}},
    {"name": "news_count_ge3", "filters": {"min_text_news_count": 3}},
    {"name": "source_count_ge2", "filters": {"min_text_source_count": 2}},
    {"name": "news_ge2_source_ge2", "filters": {"min_text_news_count": 2, "min_text_source_count": 2}},
    {"name": "sentiment_abs_top50", "filters": {"text_sentiment_abs_quantile": 0.50}},
    {"name": "sentiment_abs_top30", "filters": {"text_sentiment_abs_quantile": 0.70}},
]


def _dataset_summary(df: pd.DataFrame, window_size: int) -> dict[str, Any]:
    text_features = [c for c in df.columns if c.startswith("text_") and c != "text_only"]
    summary: dict[str, Any] = {
        "rows": int(len(df)),
        "raw_text_rows": int(df.get("has_text_raw", pd.Series(False, index=df.index)).sum()),
        "event_text_rows": int(df.get("has_text", pd.Series(False, index=df.index)).sum()),
        "row_labels": {str(k): int(v) for k, v in df["label"].value_counts().sort_index().items()},
    }
    try:
        arrays = create_sequences(
            df,
            market_features=ExperimentConfig().market_features,
            text_features=text_features or ExperimentConfig().text_features,
            window_size=window_size,
        )
        summary["sequences"] = int(len(arrays.labels))
        summary["sequence_labels"] = {str(k): int(v) for k, v in pd.Series(arrays.labels).value_counts().sort_index().items()}
        summary["sequence_start_date"] = str(arrays.meta["trade_date"].min())
        summary["sequence_end_date"] = str(arrays.meta["trade_date"].max())
    except ValueError as exc:
        summary["sequences"] = 0
        summary["sequence_error"] = str(exc)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare strong-news event-filter datasets and sample summaries.")
    parser.add_argument("--market-csv", required=True)
    parser.add_argument("--text-features-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prefix", default="aligned_dataset_event")
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--label-mode", default="future_mean", choices=["future_close", "future_mean", "future_excess_mean"])
    parser.add_argument("--return-threshold", type=float, default=0.01)
    parser.add_argument("--drop-neutral", action="store_true")
    parser.add_argument("--text-only", action="store_true")
    parser.add_argument("--window-size", type=int, default=20)
    args = parser.parse_args()

    market = pd.read_csv(args.market_csv)
    text = pd.read_csv(args.text_features_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for spec in EVENT_FILTERS:
        name = spec["name"]
        filters = spec["filters"]
        df = build_aligned_dataset(
            market,
            text,
            horizon=args.horizon,
            label_mode=args.label_mode,
            return_threshold=args.return_threshold,
            drop_neutral=args.drop_neutral,
            text_only=args.text_only,
            **filters,
        )
        output_csv = output_dir / f"{args.prefix}_{name}.csv"
        df.to_csv(output_csv, index=False)
        summary = {"name": name, "output_csv": str(output_csv), "filters": filters}
        summary.update(_dataset_summary(df, args.window_size))
        summaries.append(summary)

    summary_path = output_dir / f"{args.prefix}_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
