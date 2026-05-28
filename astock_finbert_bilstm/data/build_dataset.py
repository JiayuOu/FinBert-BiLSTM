from __future__ import annotations

import argparse

from .dataset import save_aligned_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Build aligned A-share market/text dataset.")
    parser.add_argument("--market-csv", required=True)
    parser.add_argument("--text-features-csv", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--horizon", type=int, default=1, help="Prediction horizon in trading days.")
    parser.add_argument("--label-mode", default="future_close", choices=["future_close", "future_mean", "future_excess_mean"], help="Label target: close(t+horizon), mean(close(t+1..t+horizon)), or excess mean return versus the equal-weight stock-pool benchmark.")
    parser.add_argument("--return-threshold", type=float, default=0.0, help="Minimum future return for positive labels. Use with --drop-neutral to remove weak moves.")
    parser.add_argument("--drop-neutral", action="store_true", help="Drop rows with abs(label return) below --return-threshold.")
    parser.add_argument("--text-only", action="store_true", help="Keep only stock-date rows that have matched text features.")
    parser.add_argument("--min-text-news-count", type=int, default=None, help="Require at least this many news items on the target stock-date.")
    parser.add_argument("--min-text-source-count", type=int, default=None, help="Require at least this many distinct text sources on the target stock-date.")
    parser.add_argument("--min-abs-text-sentiment-score", type=float, default=None, help="Require abs(text_sentiment_score) to be at least this value.")
    parser.add_argument("--text-sentiment-abs-quantile", type=float, default=None, help="Require abs(text_sentiment_score) to be at or above this quantile among raw text days.")
    parser.add_argument("--min-text-sentiment-score-std", type=float, default=None, help="Require text_sentiment_score_std to be at least this value.")
    args = parser.parse_args()
    save_aligned_dataset(
        args.market_csv,
        args.text_features_csv,
        args.output,
        horizon=args.horizon,
        label_mode=args.label_mode,
        return_threshold=args.return_threshold,
        drop_neutral=args.drop_neutral,
        text_only=args.text_only,
        min_text_news_count=args.min_text_news_count,
        min_text_source_count=args.min_text_source_count,
        min_abs_text_sentiment_score=args.min_abs_text_sentiment_score,
        text_sentiment_abs_quantile=args.text_sentiment_abs_quantile,
        min_text_sentiment_score_std=args.min_text_sentiment_score_std,
    )
    print(f"saved aligned dataset to {args.output}")


if __name__ == "__main__":
    main()
