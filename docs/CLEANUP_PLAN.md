# 项目清理建议清单

这个文件记录建议保留和建议删除的文件。由于删除会影响实验复现，实际删除前建议先确认本清单。

## 保留核心代码

```text
astock_finbert_bilstm/config.py
astock_finbert_bilstm/data/
astock_finbert_bilstm/text/download_tushare_news.py
astock_finbert_bilstm/text/import_stock_texts.py
astock_finbert_bilstm/text/align_text_dates.py
astock_finbert_bilstm/text/extract_features.py
astock_finbert_bilstm/text/features.py
astock_finbert_bilstm/text/fine_tune_sentiment.py
astock_finbert_bilstm/experiments/train.py
astock_finbert_bilstm/experiments/run_ablation.py
astock_finbert_bilstm/experiments/prepare_event_filter_datasets.py
astock_finbert_bilstm/experiments/run_text_feature_ablation.py
astock_finbert_bilstm/experiments/text_feature_sets.py
astock_finbert_bilstm/experiments/run_expanded_strong_news_pipeline.py
astock_finbert_bilstm/modeling/
tests/
```

## 可考虑删除的旧代码

```text
astock_finbert_bilstm/experiments/explain.py
astock_finbert_bilstm/experiments/infer.py
astock_finbert_bilstm/experiments/run_encoder_comparison.py
astock_finbert_bilstm/experiments/run_industry_pipeline.py
astock_finbert_bilstm/text/download_announcements.py
astock_finbert_bilstm/text/download_stock_texts.py
```

## 保留最终数据

```text
data/stock_pools/new_energy.csv
data/stock_pools/new_energy_50.csv

data/raw/market.csv
data/raw/news_tushare.csv
data/raw/news_astock.csv
data/raw/news_super_enriched.csv
data/raw/news_super_enriched_aligned.csv

data/raw/expanded_new_energy_50/market.csv
data/raw/expanded_new_energy_50/market.failures.csv
data/raw/expanded_new_energy_50/news_tushare.csv
data/raw/expanded_new_energy_50/news_tushare.failures.csv
data/raw/expanded_new_energy_50/news_astock.csv
data/raw/expanded_new_energy_50/news_merged.csv
data/raw/expanded_new_energy_50/news_aligned.csv

data/processed/text_features_finbert_super_enriched_news_only_per_text_mean.csv
data/processed/event_filters/aligned_dataset_finbert_news_event_h10_mean_thr1_textonly_source_count_ge2.csv
data/processed/event_filters/aligned_dataset_finbert_news_event_h10_mean_thr1_textonly_summary.json

data/processed/expanded_new_energy_50/text_features_finbert_news_per_text_mean.csv
data/processed/expanded_new_energy_50/aligned_dataset_h10_mean_thr1_source_count_ge2_textonly.csv
data/processed/expanded_new_energy_50/aligned_dataset_h10_mean_thr1_source_count_ge2_textonly_summary.json
```

## 可考虑删除的旧数据与大型中间产物

```text
.venv/
.vscode/
**/__pycache__/

data/raw/market_test.csv
data/raw/news.failures.csv
data/raw/news_enriched.csv
data/raw/news_enriched_aligned.csv
data/raw/news_test.csv
data/raw/news_test.failures.csv
data/raw/news_tushare_smoke.csv
data/raw/sentiment_train_small.csv
data/raw/sentiment_train_small.stats.csv
data/raw/expanded_new_energy_50/stock_pool_first_2.csv
data/raw/expanded_new_energy_50/stock_pool_first_5.csv

data/processed/source_ablation/
data/processed/text_features_finbert.csv
data/processed/text_features_finbert_enriched.csv
data/processed/text_features_finbert_super_enriched.csv
data/processed/text_features_finbert_super_enriched_news_only_per_text_mean_len256.csv
data/processed/text_features_finbert_test.csv
data/processed/aligned_dataset_finbert_super_enriched_news_only_per_text_mean_h10_excess_mean_thr1_source_count_ge2_textonly.csv
data/processed/aligned_dataset_finbert_super_enriched_news_only_per_text_mean_h10_mean_thr1_textonly.csv
data/processed/aligned_dataset_finbert_super_enriched_news_only_per_text_mean_len256_h10_mean_thr1_textonly.csv
data/processed/event_filters/aligned_dataset_finbert_news_event_h10_mean_thr1_textonly_baseline.csv
data/processed/event_filters/aligned_dataset_finbert_news_event_h10_mean_thr1_textonly_news_count_ge2.csv
data/processed/event_filters/aligned_dataset_finbert_news_event_h10_mean_thr1_textonly_news_count_ge3.csv
data/processed/event_filters/aligned_dataset_finbert_news_event_h10_mean_thr1_textonly_news_ge2_source_ge2.csv
data/processed/event_filters/aligned_dataset_finbert_news_event_h10_mean_thr1_textonly_sentiment_abs_top30.csv
data/processed/event_filters/aligned_dataset_finbert_news_event_h10_mean_thr1_textonly_sentiment_abs_top50.csv
```

## 保留最终结果

```text
runs/event_filter_ablation/
runs/source_ablation/
runs/text_feature_ablation/
runs/excess_return_source_count_ge2/
runs/expanded_new_energy_50_source_count_ge2/
runs/finbert_cn/
models/FinBERT2-base/
```

其中 `runs/finbert_cn/` 建议只保留最终模型文件，删除中间 checkpoint：

```text
runs/finbert_cn/checkpoint-223/
runs/finbert_cn/checkpoint-446/
runs/finbert_cn/checkpoint-669/
```

## 可考虑删除的旧结果目录

```text
runs/ablation/
runs/ablation_news_only_per_text_mean_h10_mean_thr1_textonly/
runs/ablation_news_only_per_text_mean_h10_mean_thr1_textonly_threshold_tuned/
runs/ablation_news_only_per_text_mean_len256_h10_mean_thr1_textonly/
runs/ablation_super_enriched_h10_mean/
runs/ablation_super_enriched_h10_mean_thr1_textonly/
runs/attention_fusion*/
runs/text_feature_ablation_unscaled/
```

## 当前清理状态

已执行清理：

```text
- 删除 .venv、.vscode、__pycache__ 等本地环境/缓存
- 删除旧 FinBERT checkpoint，只保留 runs/finbert_cn 最终模型
- 删除旧窗口、旧阈值、烟测、非主线 run 目录
- 删除 source_ablation 等大型 processed 中间数据
- 删除旧 event_filter 中除 source_count_ge2 外的大型对齐数据集
- 将 runs/event_filter_ablation、runs/source_ablation、runs/text_feature_ablation、runs/excess_return_source_count_ge2、runs/expanded_new_energy_50_source_count_ge2 压缩为 README 中列出的汇总文件
```

当前仍保留的 CSV：

```text
data/processed/event_filters/aligned_dataset_finbert_news_event_h10_mean_thr1_textonly_source_count_ge2.csv
data/processed/expanded_new_energy_50/aligned_dataset_h10_mean_thr1_source_count_ge2_textonly.csv
data/processed/expanded_new_energy_50/text_features_finbert_news_per_text_mean.csv
data/processed/text_features_finbert_super_enriched_news_only_per_text_mean.csv
data/raw/expanded_new_energy_50/market.csv
data/raw/expanded_new_energy_50/market.failures.csv
data/raw/expanded_new_energy_50/news_aligned.csv
data/raw/expanded_new_energy_50/news_astock.csv
data/raw/expanded_new_energy_50/news_merged.csv
data/raw/expanded_new_energy_50/news_tushare.csv
data/raw/expanded_new_energy_50/news_tushare.failures.csv
data/raw/market.csv
data/raw/news.csv
data/raw/news_astock.csv
data/raw/news_super_enriched.csv
data/raw/news_super_enriched_aligned.csv
data/raw/news_tushare.csv
data/raw/sentiment_train.csv
data/raw/sentiment_train.stats.csv
data/stock_pools/new_energy.csv
data/stock_pools/new_energy_50.csv
```

其中 `data/raw/news*.csv`、`data/raw/sentiment_train*.csv`、`data/stock_pools/new_energy.csv` 属于早期 20 只实验和 FinBERT 情感训练输入。它们体积不大，但若只保留 49 只扩展主实验，可以进一步删除。
