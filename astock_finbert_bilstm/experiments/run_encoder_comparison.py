from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..config import ExperimentConfig
from .train import train_experiment


def parse_dataset_arg(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Use name=path, for example finbert=data/processed/finbert_dataset.csv")
    name, path = value.split("=", 1)
    name = name.strip()
    path = path.strip()
    if not name or not path:
        raise argparse.ArgumentTypeError("Both name and path are required in name=path.")
    return name, path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare text encoders such as ordinary Chinese BERT and FinBERT under the same downstream model."
    )
    parser.add_argument(
        "--dataset",
        action="append",
        required=True,
        type=parse_dataset_arg,
        help="Aligned dataset in name=path format. Pass this option multiple times.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="attention_fusion", choices=["concat_fusion", "attention_fusion"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--window-size", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=64)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    summary = {}
    for name, dataset_path in args.dataset:
        config = ExperimentConfig(
            model=args.model,
            epochs=args.epochs,
            window_size=args.window_size,
            batch_size=args.batch_size,
            hidden_size=args.hidden_size,
        )
        result = train_experiment(dataset_path, output_dir / name, config)
        summary[name] = result["test"]

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "encoder_comparison_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
