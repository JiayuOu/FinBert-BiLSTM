from __future__ import annotations

from pathlib import Path


REQUIRED_PATHS = [
    "data/stock_pools/new_energy_50.csv",
    "data/raw/expanded_new_energy_50/market.csv",
    "data/raw/expanded_new_energy_50/news_aligned.csv",
    "data/raw/expanded_new_energy_50/news_tushare.csv",
    "data/raw/expanded_new_energy_50/news_astock.csv",
    "data/processed/expanded_new_energy_50/text_features_finbert_news_per_text_mean.csv",
    "data/processed/expanded_new_energy_50/aligned_dataset_h10_mean_thr1_source_count_ge2_textonly.csv",
    "data/processed/event_filters/aligned_dataset_finbert_news_event_h10_mean_thr1_textonly_source_count_ge2.csv",
    "data/processed/text_features_finbert_super_enriched_news_only_per_text_mean.csv",
    "models/FinBERT2-base/pytorch_model.bin",
    "models/FinBERT2-base/config.json",
    "models/FinBERT2-base/tokenizer.json",
    "runs/finbert_cn/model.safetensors",
    "runs/finbert_cn/config.json",
    "runs/finbert_cn/tokenizer.json",
]


def main() -> int:
    missing = [path for path in REQUIRED_PATHS if not Path(path).exists()]
    if missing:
        print("Missing artifact files:")
        for path in missing:
            print(f"  - {path}")
        print("\nDownload project_artifacts.zip and unzip it at the repository root.")
        return 1

    print("All required artifact files are present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
