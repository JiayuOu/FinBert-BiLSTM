from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


ARTIFACT_PATHS = [
    "data/raw/expanded_new_energy_50/market.csv",
    "data/raw/expanded_new_energy_50/market.failures.csv",
    "data/raw/expanded_new_energy_50/news_aligned.csv",
    "data/raw/expanded_new_energy_50/news_astock.csv",
    "data/raw/expanded_new_energy_50/news_merged.csv",
    "data/raw/expanded_new_energy_50/news_tushare.csv",
    "data/raw/expanded_new_energy_50/news_tushare.failures.csv",
    "data/raw/market.csv",
    "data/raw/news.csv",
    "data/raw/news_astock.csv",
    "data/raw/news_super_enriched.csv",
    "data/raw/news_super_enriched_aligned.csv",
    "data/raw/news_tushare.csv",
    "data/raw/sentiment_train.csv",
    "data/raw/sentiment_train.stats.csv",
    "data/processed/expanded_new_energy_50/text_features_finbert_news_per_text_mean.csv",
    "data/processed/expanded_new_energy_50/aligned_dataset_h10_mean_thr1_source_count_ge2_textonly.csv",
    "data/processed/expanded_new_energy_50/aligned_dataset_h10_mean_thr1_source_count_ge2_textonly_summary.json",
    "data/processed/event_filters/aligned_dataset_finbert_news_event_h10_mean_thr1_textonly_source_count_ge2.csv",
    "data/processed/event_filters/aligned_dataset_finbert_news_event_h10_mean_thr1_textonly_summary.json",
    "data/processed/text_features_finbert_super_enriched_news_only_per_text_mean.csv",
    "models/FinBERT2-base/.gitattributes",
    "models/FinBERT2-base/README.md",
    "models/FinBERT2-base/added_tokens.json",
    "models/FinBERT2-base/config.json",
    "models/FinBERT2-base/pytorch_model.bin",
    "models/FinBERT2-base/special_tokens_map.json",
    "models/FinBERT2-base/tokenizer.json",
    "models/FinBERT2-base/tokenizer_config.json",
    "models/FinBERT2-base/vocab.txt",
    "runs/finbert_cn/config.json",
    "runs/finbert_cn/model.safetensors",
    "runs/finbert_cn/tokenizer.json",
    "runs/finbert_cn/tokenizer_config.json",
    "runs/finbert_cn/training_args.bin",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the external artifact zip with repo-relative paths.")
    parser.add_argument("--output", default="project_artifacts.zip", help="Output zip path.")
    args = parser.parse_args()

    missing = [path for path in ARTIFACT_PATHS if not Path(path).exists()]
    if missing:
        print("Cannot build artifact zip; missing files:")
        for path in missing:
            print(f"  - {path}")
        return 1

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in ARTIFACT_PATHS:
            zf.write(path, arcname=path)
            print(f"added {path}")

    print(f"\nWrote {output}")
    print("Users should unzip this file at the repository root.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
