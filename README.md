# A-Share FinBERT + BiLSTM Thesis Experiment

This repository contains the thesis experiment for:

```text
基于 FinBERT 与 BiLSTM 融合模型的 A 股股票短期涨跌预测研究
```

The maintained project guide is in Chinese: [README.zh-CN.md](README.zh-CN.md).

Main experiment:

```text
future_mean label, horizon = 10, threshold = 1%
text-only samples, source_count >= 2 strong-news filter
FinBERT text features + BiLSTM market sequence
```

Large artifacts are hosted on Hugging Face:

```text
https://huggingface.co/datasets/KallonOu/FinBert-BiLSTM-artifacts
```

After cloning the repo, install dependencies and restore artifacts with:

```bash
pip install -r requirements.txt
python scripts/download_artifacts.py
```

Key result: on the expanded 49-stock pool, `attention_fusion` reaches AUC 0.5633 versus `market_bilstm` AUC 0.5498, giving an incremental AUC of about +0.0135 under the strong-news event setting.
