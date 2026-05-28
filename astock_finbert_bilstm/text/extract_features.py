from __future__ import annotations

import argparse
import torch

from ..config import DEFAULT_FINBERT_MODEL
from .features import TextFeatureConfig, extract_text_feature_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract FinBERT-style text features.")
    parser.add_argument("--text-csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default=DEFAULT_FINBERT_MODEL)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--aggregation-mode",
        default="concat",
        choices=["concat", "per_text_mean"],
        help="Text aggregation strategy: concat groups same-day text before encoding; per_text_mean encodes each row and averages by stock-date.",
    )
    parser.add_argument(
        "--include-source",
        action="append",
        default=None,
        help="Keep only this text source. Can be passed multiple times.",
    )
    parser.add_argument(
        "--exclude-source",
        action="append",
        default=None,
        help="Drop this text source before feature extraction. Can be passed multiple times.",
    )
    parser.add_argument(
        "--include-sentiment",
        action="store_true",
        help="Also export negative/neutral/positive probabilities. Use only with a sentiment-fine-tuned classifier.",
    )
    args = parser.parse_args()

    config = TextFeatureConfig(
        model_name=args.model,
        max_length=args.max_length,
        batch_size=args.batch_size,
        device=args.device or ("cuda" if torch.cuda.is_available() else "cpu"),
        include_sentiment=args.include_sentiment,
        aggregation_mode=args.aggregation_mode,
    )
    extract_text_feature_csv(
        args.text_csv,
        args.output,
        config,
        include_sources=args.include_source,
        exclude_sources=args.exclude_source,
    )
    print(f"saved text features to {args.output}")


if __name__ == "__main__":
    main()
