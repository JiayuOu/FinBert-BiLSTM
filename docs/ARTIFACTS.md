# 大文件下载与放置说明

本仓库建议上传到 GitHub 的部分只包含代码、README、股票池和小型结果汇总。大模型和大型 CSV 统一放在 Hugging Face Dataset：

```text
https://huggingface.co/datasets/KallonOu/FinBert-BiLSTM-artifacts
```

请下载其中的：

```text
project_artifacts.zip
```

以下大文件不要直接提交到普通 GitHub 仓库：

```text
models/FinBERT2-base/pytorch_model.bin
runs/finbert_cn/model.safetensors
data/processed/expanded_new_energy_50/aligned_dataset_h10_mean_thr1_source_count_ge2_textonly.csv
data/processed/event_filters/aligned_dataset_finbert_news_event_h10_mean_thr1_textonly_source_count_ge2.csv
data/processed/expanded_new_energy_50/text_features_finbert_news_per_text_mean.csv
```

## 使用者如何放置数据

推荐在仓库根目录运行一条命令自动下载、解压并检查：

```bash
python scripts/download_artifacts.py
```

也可以手动下载 `project_artifacts.zip`，然后在仓库根目录解压：

```bash
unzip project_artifacts.zip -d .
python scripts/check_artifacts.py
```

Windows 如果没有 `unzip`，可以右键解压到项目根目录，或使用 PowerShell：

```powershell
Expand-Archive .\project_artifacts.zip -DestinationPath . -Force
python scripts\check_artifacts.py
```

解压后路径必须保持为：

```text
data/raw/expanded_new_energy_50/market.csv
data/raw/expanded_new_energy_50/news_aligned.csv
data/processed/expanded_new_energy_50/text_features_finbert_news_per_text_mean.csv
data/processed/expanded_new_energy_50/aligned_dataset_h10_mean_thr1_source_count_ge2_textonly.csv
models/FinBERT2-base/pytorch_model.bin
runs/finbert_cn/model.safetensors
```

这样项目里的相对路径不需要修改。

## 检查文件是否放对

```bash
python scripts/check_artifacts.py
```

看到下面输出即表示大文件已放置正确：

```text
All required artifact files are present.
```

## 维护者如何生成压缩包

在本地完整项目根目录执行：

```bash
python scripts/make_artifacts.py --output project_artifacts.zip
```

生成的 zip 内部会保留项目相对路径。使用者只需要在仓库根目录解压。
