from __future__ import annotations

import argparse
import json
import os
import signal
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from ..config import ExperimentConfig
from ..data.dataset import build_aligned_dataset, create_sequences
from ..data.download_market import disable_proxy_env, read_stock_codes
from ..data.features import normalize_market_columns
from ..data.sources import fetch_akshare_daily
from ..text.align_text_dates import align_text_dates_to_market
from ..text.download_tushare_news import download_tushare_news
from ..text.extract_features import TextFeatureConfig, extract_text_feature_csv
from ..text.import_stock_texts import import_astock_csv, merge_text_csvs
from .train import train_experiment

MODELS = ["market_lstm", "market_bilstm", "concat_fusion", "attention_fusion"]
TUSHARE_SOURCES = ["sina", "eastmoney", "10jqka"]


class _MarketDownloadTimeout(RuntimeError):
    pass


def _raise_market_timeout(signum: int, frame: object) -> None:
    raise _MarketDownloadTimeout("single-stock market download timed out")


def _write_market_outputs(frames: list[pd.DataFrame], failures: list[tuple[str, str]], output: Path) -> None:
    if frames:
        result = normalize_market_columns(pd.concat(frames, ignore_index=True))
        keep_first = [
            "stock_code",
            "trade_date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "amount",
            "turnover_rate",
            "pct_chg",
        ]
        ordered = [col for col in keep_first if col in result.columns]
        ordered += [col for col in result.columns if col not in ordered]
        result = result[ordered].drop_duplicates(subset=["stock_code", "trade_date"])
        result = result.sort_values(["stock_code", "trade_date"]).reset_index(drop=True)
        output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output, index=False, encoding="utf-8-sig")
    if failures:
        pd.DataFrame(failures, columns=["stock_code", "error"]).to_csv(
            output.with_suffix(".failures.csv"),
            index=False,
            encoding="utf-8-sig",
        )


def _download_market_data_incremental(
    stock_codes: list[str],
    start_date: str,
    end_date: str,
    output: str | Path,
    adjust: str,
    sleep_seconds: float,
    timeout_seconds: int,
) -> pd.DataFrame:
    output = Path(output)
    frames: list[pd.DataFrame] = []
    failures: list[tuple[str, str]] = []
    done_codes: set[str] = set()

    if output.exists():
        existing = normalize_market_columns(pd.read_csv(output))
        if not existing.empty and "stock_code" in existing.columns:
            frames.append(existing)
            done_codes = set(existing["stock_code"].astype(str).unique())
            print(f"[INFO] resuming market download from {output}, completed={len(done_codes)}")

    old_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _raise_market_timeout)
    try:
        for stock_code in stock_codes:
            if stock_code in done_codes:
                print(f"[SKIP] {stock_code}: already in {output}")
                continue
            try:
                signal.alarm(timeout_seconds)
                one = fetch_akshare_daily(stock_code, start_date, end_date, adjust=adjust)
                signal.alarm(0)
                source_code_columns = [col for col in ["股票代码", "代码", "ts_code"] if col in one.columns]
                if source_code_columns:
                    one = one.drop(columns=source_code_columns)
                one["stock_code"] = stock_code
                frames.append(one)
                done_codes.add(stock_code)
                _write_market_outputs(frames, failures, output)
                print(f"[OK] {stock_code}: {len(one)} rows")
            except Exception as exc:
                signal.alarm(0)
                failures.append((stock_code, str(exc)))
                _write_market_outputs(frames, failures, output)
                print(f"[FAIL] {stock_code}: {exc}")
            time.sleep(sleep_seconds)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

    if not frames:
        detail = "; ".join(f"{code}: {message}" for code, message in failures)
        raise RuntimeError(f"No market data downloaded. {detail}")
    result = normalize_market_columns(pd.concat(frames, ignore_index=True))
    print(f"[DONE] market data saved to {output}, rows={len(result)}, stocks={result['stock_code'].nunique()}")
    return result


def _limited_stock_codes(stock_list: str | Path, max_stocks: int | None) -> list[str]:
    codes = read_stock_codes(stock_list)
    if max_stocks is not None:
        codes = codes[:max_stocks]
    if not codes:
        raise ValueError(f"No stock codes found in {stock_list}")
    return codes


def _make_stock_pool_subset(stock_list: str | Path, stock_codes: list[str], output: str | Path) -> Path:
    source = pd.read_csv(stock_list)
    if "stock_code" not in source.columns:
        source = source.rename(columns={source.columns[0]: "stock_code"})
    source["stock_code"] = source["stock_code"].astype(str)
    subset = source[source["stock_code"].isin(stock_codes)].copy()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    subset.to_csv(output, index=False, encoding="utf-8-sig")
    return output


def _text_features_from_dataset(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("text_") and c != "text_only"]


def _dataset_summary(df: pd.DataFrame, window_size: int) -> dict[str, Any]:
    text_features = _text_features_from_dataset(df)
    summary: dict[str, Any] = {
        "rows": int(len(df)),
        "stocks": int(df["stock_code"].nunique()),
        "raw_text_rows": int(df.get("has_text_raw", pd.Series(False, index=df.index)).sum()),
        "event_text_rows": int(df.get("has_text", pd.Series(False, index=df.index)).sum()),
        "row_labels": {str(k): int(v) for k, v in df["label"].value_counts().sort_index().items()},
        "text_features": int(len(text_features)),
    }
    arrays = create_sequences(df, text_features=text_features, window_size=window_size)
    summary.update(
        {
            "sequences": int(len(arrays.labels)),
            "sequence_labels": {str(k): int(v) for k, v in pd.Series(arrays.labels).value_counts().sort_index().items()},
            "sequence_start_date": str(arrays.meta["trade_date"].min()),
            "sequence_end_date": str(arrays.meta["trade_date"].max()),
        }
    )
    return summary


def _run_ablation(dataset_csv: str | Path, output_dir: str | Path, args: argparse.Namespace) -> dict[str, Any]:
    df = pd.read_csv(dataset_csv, nrows=1)
    text_features = _text_features_from_dataset(df)
    summary: dict[str, Any] = {}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for model_name in MODELS:
        config = ExperimentConfig(
            model=model_name,
            epochs=args.train_epochs,
            window_size=args.window_size,
            batch_size=args.batch_size,
            hidden_size=args.hidden_size,
            learning_rate=args.learning_rate,
            text_features=text_features or ExperimentConfig().text_features,
            scale_text_features=False,
        )
        result = train_experiment(dataset_csv, output_dir / model_name, config)
        summary[model_name] = result["test"]

    with (output_dir / "ablation_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    rows = []
    for model_name, metrics in summary.items():
        row = {"model": model_name}
        for key in ["accuracy", "precision", "recall", "f1", "auc"]:
            row[key] = metrics[key]
        cm = metrics["confusion_matrix"]
        row.update({"tn": cm[0][0], "fp": cm[0][1], "fn": cm[1][0], "tp": cm[1][1]})
        rows.append(row)
    pd.DataFrame(rows).to_csv(output_dir / "ablation_comparison.csv", index=False)
    return summary


def _paths(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "market": args.raw_dir / "market.csv",
        "tushare": args.raw_dir / "news_tushare.csv",
        "astock": args.raw_dir / "news_astock.csv",
        "merged_news": args.raw_dir / "news_merged.csv",
        "aligned_news": args.raw_dir / "news_aligned.csv",
        "text_features": args.processed_dir / "text_features_finbert_news_per_text_mean.csv",
        "dataset": args.processed_dir / "aligned_dataset_h10_mean_thr1_source_count_ge2_textonly.csv",
        "dataset_summary": args.processed_dir / "aligned_dataset_h10_mean_thr1_source_count_ge2_textonly_summary.json",
    }


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    if args.no_proxy:
        disable_proxy_env()
    stock_codes = _limited_stock_codes(args.stock_list, args.max_stocks)
    stock_list = Path(args.stock_list)
    if args.max_stocks is not None:
        stock_list = _make_stock_pool_subset(args.stock_list, stock_codes, args.work_dir / f"stock_pool_first_{args.max_stocks}.csv")

    args.raw_dir.mkdir(parents=True, exist_ok=True)
    args.processed_dir.mkdir(parents=True, exist_ok=True)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    paths = _paths(args)

    if not args.skip_market:
        _download_market_data_incremental(
            stock_codes=stock_codes,
            start_date=args.start_date,
            end_date=args.end_date,
            output=paths["market"],
            adjust=args.adjust,
            sleep_seconds=args.market_sleep,
            timeout_seconds=args.market_timeout,
        )

    if not args.skip_tushare:
        token = args.tushare_token or os.environ.get("TUSHARE_TOKEN")
        if not token:
            raise SystemExit("Missing Tushare token. Pass --tushare-token or set TUSHARE_TOKEN.")
        download_tushare_news(
            stock_list=stock_list,
            start_date=args.start_date,
            end_date=args.end_date,
            output=paths["tushare"],
            token=token,
            sources=TUSHARE_SOURCES,
            window_days=args.tushare_window_days,
            sleep_seconds=args.tushare_sleep,
        )


    if not args.skip_astock:
        import_astock_csv(
            input_path=args.astock_input,
            stock_list=stock_list,
            output=paths["astock"],
            start_date=args.start_date,
            end_date=args.end_date,
            source_name="astock_stock_news",
        )

    if not args.skip_merge:
        inputs = []
        if paths["tushare"].exists():
            inputs.append(paths["tushare"])
        if paths["astock"].exists():
            inputs.append(paths["astock"])
        if not inputs:
            raise SystemExit("No news inputs found for merge.")
        merge_text_csvs(inputs, paths["merged_news"])
        align_text_dates_to_market(paths["merged_news"], paths["market"], paths["aligned_news"])

    if not args.skip_features:
        config = TextFeatureConfig(
            model_name=args.finbert_model,
            max_length=args.max_length,
            batch_size=args.batch_size,
            device=args.device or ("cuda" if torch.cuda.is_available() else "cpu"),
            include_sentiment=True,
            aggregation_mode="per_text_mean",
        )
        extract_text_feature_csv(paths["aligned_news"], paths["text_features"], config)

    if not args.skip_dataset:
        market = pd.read_csv(paths["market"])
        text = pd.read_csv(paths["text_features"])
        dataset = build_aligned_dataset(
            market,
            text,
            horizon=10,
            label_mode="future_mean",
            return_threshold=0.01,
            drop_neutral=True,
            text_only=True,
            min_text_source_count=2,
        )
        paths["dataset"].parent.mkdir(parents=True, exist_ok=True)
        dataset.to_csv(paths["dataset"], index=False)
        summary = _dataset_summary(dataset, args.window_size)
        with paths["dataset_summary"].open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)
        print(json.dumps(summary, ensure_ascii=False, indent=2))

    result: dict[str, Any] = {
        "stock_count": len(stock_codes),
        "market_csv": str(paths["market"]),
        "tushare_csv": str(paths["tushare"]),
        "astock_csv": str(paths["astock"]),
        "aligned_news_csv": str(paths["aligned_news"]),
        "text_features_csv": str(paths["text_features"]),
        "dataset_csv": str(paths["dataset"]),
        "dataset_summary_json": str(paths["dataset_summary"]),
        "run_dir": str(args.run_dir),
    }
    if not args.skip_train:
        result["ablation"] = _run_ablation(paths["dataset"], args.run_dir, args)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run expanded new-energy strong-news A-share experiment pipeline.")
    parser.add_argument("--stock-list", default="data/stock_pools/new_energy_50.csv")
    parser.add_argument("--max-stocks", type=int, default=None, help="Use only the first N stocks for smoke tests.")
    parser.add_argument("--start-date", default="2019-01-01")
    parser.add_argument("--end-date", default="2024-12-31")
    parser.add_argument("--work-dir", type=Path, default=Path("data/raw/expanded_new_energy_50"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/expanded_new_energy_50"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed/expanded_new_energy_50"))
    parser.add_argument("--run-dir", type=Path, default=Path("runs/expanded_new_energy_50_source_count_ge2"))
    parser.add_argument("--astock-input", default="/tmp/Astock/data/df_all_year_srl.csv")
    parser.add_argument("--tushare-token", default=None)
    parser.add_argument("--finbert-model", default="runs/finbert_cn")
    parser.add_argument("--adjust", default="qfq", choices=["", "qfq", "hfq"])
    parser.add_argument("--market-sleep", type=float, default=0.2)
    parser.add_argument("--market-timeout", type=int, default=180, help="Per-stock market download timeout in seconds.")
    parser.add_argument("--tushare-window-days", type=int, default=7)
    parser.add_argument("--tushare-sleep", type=float, default=0.5)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default=None)
    parser.add_argument("--train-epochs", type=int, default=20)
    parser.add_argument("--window-size", type=int, default=20)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--no-proxy", action="store_true", help="Clear proxy environment variables before network requests.")
    parser.add_argument("--skip-market", action="store_true")
    parser.add_argument("--skip-tushare", action="store_true")
    parser.add_argument("--skip-astock", action="store_true")
    parser.add_argument("--skip-merge", action="store_true")
    parser.add_argument("--skip-features", action="store_true")
    parser.add_argument("--skip-dataset", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_pipeline(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
