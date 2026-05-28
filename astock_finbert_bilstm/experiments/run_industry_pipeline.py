from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def run_step(args: list[str]) -> None:
    print(f"\n[RUN] {' '.join(args)}", flush=True)
    subprocess.run(args, check=True)


def remove_path(path: str | Path) -> None:
    target = Path(path)
    if not target.exists():
        return
    if target.is_dir():
        shutil.rmtree(target)
    else:
        target.unlink()
    print(f"[CLEAN] removed {target}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a full industry-stock FinBERT + BiLSTM experiment pipeline.")
    parser.add_argument("--stock-list", default="data/stock_pools/new_energy.csv")
    parser.add_argument("--start-date", default="2019-01-01")
    parser.add_argument("--end-date", default="2024-12-31")
    parser.add_argument("--sentiment-train-csv", default="data/raw/sentiment_train.csv")
    parser.add_argument("--base-model", default="models/FinBERT2-base")
    parser.add_argument("--sentiment-model-dir", default="runs/finbert_cn")
    parser.add_argument("--market-csv", default="data/raw/market.csv")
    parser.add_argument("--news-csv", default="data/raw/news.csv")
    parser.add_argument("--text-features-csv", default="data/processed/text_features_finbert.csv")
    parser.add_argument("--dataset-csv", default="data/processed/aligned_dataset_finbert.csv")
    parser.add_argument("--fusion-output-dir", default="runs/attention_fusion")
    parser.add_argument("--ablation-output-dir", default="runs/ablation")
    parser.add_argument("--finetune-epochs", type=float, default=3)
    parser.add_argument("--train-epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--max-text-rows-per-stock", type=int, default=None)
    parser.add_argument("--text-sleep", type=float, default=0.5)
    parser.add_argument("--window-size", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--skip-finetune", action="store_true")
    parser.add_argument("--skip-download-news", action="store_true")
    parser.add_argument("--no-proxy", action="store_true")
    parser.add_argument("--clean", action="store_true", help="Remove generated raw/processed/run outputs before running.")
    args = parser.parse_args()

    if args.clean:
        for path in [
            args.market_csv,
            args.news_csv,
            args.text_features_csv,
            args.dataset_csv,
            args.fusion_output_dir,
            args.ablation_output_dir,
        ]:
            remove_path(path)

    py = sys.executable

    if not args.skip_finetune:
        run_step(
            [
                py,
                "-m",
                "astock_finbert_bilstm.text.fine_tune_sentiment",
                "--train-csv",
                args.sentiment_train_csv,
                "--output-dir",
                args.sentiment_model_dir,
                "--base-model",
                args.base_model,
                "--epochs",
                str(args.finetune_epochs),
                "--batch-size",
                str(args.batch_size),
                "--max-length",
                str(args.max_length),
            ]
        )

    market_cmd = [
        py,
        "-m",
        "astock_finbert_bilstm.data.download_market",
        "--stock-list",
        args.stock_list,
        "--start-date",
        args.start_date,
        "--end-date",
        args.end_date,
        "--output",
        args.market_csv,
    ]
    if args.no_proxy:
        market_cmd.append("--no-proxy")
    run_step(market_cmd)

    if not args.skip_download_news:
        news_cmd = [
            py,
            "-m",
            "astock_finbert_bilstm.text.download_stock_texts",
            "--stock-list",
            args.stock_list,
            "--start-date",
            args.start_date,
            "--end-date",
            args.end_date,
            "--output",
            args.news_csv,
            "--sleep",
            str(args.text_sleep),
        ]
        if args.max_text_rows_per_stock:
            news_cmd.extend(["--max-rows-per-stock", str(args.max_text_rows_per_stock)])
        if args.no_proxy:
            news_cmd.append("--no-proxy")
        run_step(news_cmd)

    run_step(
        [
            py,
            "-m",
            "astock_finbert_bilstm.text.extract_features",
            "--text-csv",
            args.news_csv,
            "--output",
            args.text_features_csv,
            "--model",
            args.sentiment_model_dir,
            "--include-sentiment",
            "--batch-size",
            str(args.batch_size),
            "--max-length",
            str(args.max_length),
        ]
    )

    run_step(
        [
            py,
            "-m",
            "astock_finbert_bilstm.data.build_dataset",
            "--market-csv",
            args.market_csv,
            "--text-features-csv",
            args.text_features_csv,
            "--output",
            args.dataset_csv,
            "--horizon",
            str(args.horizon),
        ]
    )

    run_step(
        [
            py,
            "-m",
            "astock_finbert_bilstm.experiments.train",
            "--dataset",
            args.dataset_csv,
            "--output-dir",
            args.fusion_output_dir,
            "--model",
            "attention_fusion",
            "--epochs",
            str(args.train_epochs),
            "--batch-size",
            str(args.batch_size),
            "--window-size",
            str(args.window_size),
            "--learning-rate",
            str(args.learning_rate),
        ]
    )

    run_step(
        [
            py,
            "-m",
            "astock_finbert_bilstm.experiments.run_ablation",
            "--dataset",
            args.dataset_csv,
            "--output-dir",
            args.ablation_output_dir,
            "--epochs",
            str(args.train_epochs),
            "--batch-size",
            str(args.batch_size),
            "--window-size",
            str(args.window_size),
            "--learning-rate",
            str(args.learning_rate),
        ]
    )


if __name__ == "__main__":
    main()
