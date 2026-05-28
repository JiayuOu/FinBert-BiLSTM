from __future__ import annotations

import argparse
import json
from pathlib import Path

from .sentiment_data import DEFAULT_SOURCE_NAMES, HF_SOURCES, label_stats, list_sources, merge_sources


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and normalize three-class financial sentiment datasets.")
    parser.add_argument("--list-sources", action="store_true", help="Show built-in public dataset sources and exit.")
    parser.add_argument(
        "--source",
        action="append",
        choices=sorted(HF_SOURCES),
        help="Dataset source to download. Can be passed multiple times.",
    )
    parser.add_argument("--output", default="data/raw/sentiment_train.csv")
    parser.add_argument("--min-text-length", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.list_sources:
        print(list_sources().to_string(index=False))
        return

    sources = args.source or DEFAULT_SOURCE_NAMES
    frame = merge_sources(
        sources,
        output=args.output,
        min_text_length=args.min_text_length,
        max_samples=args.max_samples,
        seed=args.seed,
    )
    summary = {
        "output": str(Path(args.output)),
        "sources": sources,
        "rows": int(len(frame)),
        "label_stats": label_stats(frame).to_dict(orient="records"),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
