# 基于 FinBERT 与 BiLSTM 融合模型的 A 股股票短期涨跌预测研究

本项目是论文实验的代码、数据与结果整理版。研究主线是：

```text
高质量股票级新闻事件 + FinBERT 文本特征
能否为 BiLSTM 行情模型提供增量预测信息
```

最终主实验采用：

```text
label_mode = future_mean
horizon = 10
return_threshold = 0.01
drop_neutral = True
text_only = True
min_text_source_count = 2
```

含义：用目标日之后 10 个交易日的平均收益构造标签；未来 10 日平均收益大于等于 1% 记为上涨，低于等于 -1% 记为下跌，中间弱波动样本丢弃。实验只保留目标日有新闻事件的样本，并要求同一股票-日期至少被两个新闻来源共同报道。

## 1. 实验流程总览

完整流程如下：

```text
股票池
  -> 行情数据
  -> 股票级新闻数据
  -> 新闻日期对齐到交易日
  -> FinBERT 文本特征提取
  -> 行情-文本对齐与标签构造
  -> LSTM/BiLSTM/融合模型训练
  -> 主实验与消融结果汇总
```

对应到本项目中的主要输入、代码和输出：

| 步骤 | 作用 | 主要代码 | 输入文件 | 输出文件 |
|---|---|---|---|---|
| 1. 股票池 | 确定实验股票范围 | `astock_finbert_bilstm/data/download_market.py` 读取股票池 | `data/stock_pools/new_energy_50.csv` | 后续行情和新闻下载使用的股票列表 |
| 2. 行情数据 | 获取 A 股日线 OHLCV 数据 | `astock_finbert_bilstm/data/download_market.py`、`astock_finbert_bilstm/data/sources.py` | 股票池、起止日期 | `data/raw/expanded_new_energy_50/market.csv` |
| 3. 新闻数据 | 获取多来源股票级新闻 | `astock_finbert_bilstm/text/download_tushare_news.py`、`astock_finbert_bilstm/text/import_stock_texts.py` | 股票池、Tushare/Astock 新闻源 | `news_tushare.csv`、`news_astock.csv`、`news_merged.csv` |
| 4. 交易日对齐 | 将非交易日新闻对齐到有效交易日 | `astock_finbert_bilstm/text/align_text_dates.py` | `news_merged.csv`、`market.csv` | `news_aligned.csv` |
| 5. 文本特征 | 用 FinBERT 提取情绪概率和语义向量 | `astock_finbert_bilstm/text/extract_features.py`、`astock_finbert_bilstm/text/features.py` | `news_aligned.csv`、`runs/finbert_cn` | `text_features_finbert_news_per_text_mean.csv` |
| 6. 标签与数据集 | 构造未来收益标签、筛选强新闻事件 | `astock_finbert_bilstm/data/build_dataset.py`、`astock_finbert_bilstm/data/dataset.py` | `market.csv`、文本特征 CSV | `aligned_dataset_h10_mean_thr1_source_count_ge2_textonly.csv` |
| 7. 模型训练 | 训练行情基准和文本融合模型 | `astock_finbert_bilstm/experiments/train.py`、`astock_finbert_bilstm/experiments/run_ablation.py` | 最终对齐数据集 | `runs/expanded_new_energy_50_source_count_ge2/` |
| 8. 消融分析 | 比较事件筛选、来源、文本特征设置 | `astock_finbert_bilstm/experiments/prepare_event_filter_datasets.py`、`astock_finbert_bilstm/experiments/run_text_feature_ablation.py` | 不同对齐数据集/特征组 | `runs/event_filter_ablation/`、`runs/source_ablation/`、`runs/text_feature_ablation/` |


## 2. GitHub 上传与大文件放置

普通 GitHub 仓库不适合直接提交大模型和大型 CSV。建议仓库只提交代码、README、股票池、小型结果汇总和辅助脚本；大文件放到 Hugging Face Dataset：

```text
https://huggingface.co/datasets/KallonOu/FinBert-BiLSTM-artifacts
```

使用者复现时不需要改代码路径。安装依赖后，在项目根目录运行一条命令即可下载、解压并检查：

```bash
python scripts/download_artifacts.py
```

也可以手动下载 `project_artifacts.zip` 后解压：

```bash
unzip project_artifacts.zip -d .
python scripts/check_artifacts.py
```

Windows 可以右键解压，或使用 PowerShell：

```powershell
Expand-Archive .\project_artifacts.zip -DestinationPath . -Force
python scripts\check_artifacts.py
```

压缩包内部保持项目相对路径，例如：

```text
data/processed/expanded_new_energy_50/aligned_dataset_h10_mean_thr1_source_count_ge2_textonly.csv
runs/finbert_cn/model.safetensors
models/FinBERT2-base/pytorch_model.bin
```

维护者可用下面命令重新生成压缩包：

```bash
python scripts/make_artifacts.py --output project_artifacts.zip
```

详细说明见 `docs/ARTIFACTS.md`。

## 3. 当前保留的数据文件

### 3.1 股票池

| 文件 | 中文解释 | 用途 |
|---|---|---|
| `data/stock_pools/new_energy_50.csv` | 扩展新能源股票池，原计划 50 只 | 49 只扩展股票池主实验入口；`002594.SZ` 比亚迪行情下载失败 |
| `data/stock_pools/new_energy.csv` | 早期 20 只新能源股票池 | 20 只旧主实验和事件筛选消融的股票范围 |

`new_energy_50.csv` 的典型字段：

```text
stock_code,stock_name
300750.SZ,宁德时代
```

### 3.2 原始行情与新闻数据

| 文件 | 中文解释 | 数据角色 |
|---|---|---|
| `data/raw/expanded_new_energy_50/market.csv` | 49 只股票的日线行情数据 | 49 只主实验的行情输入 |
| `data/raw/expanded_new_energy_50/market.failures.csv` | 行情下载失败记录 | 说明比亚迪等失败原因 |
| `data/raw/expanded_new_energy_50/news_tushare.csv` | Tushare 多来源新闻 | 49 只主实验的主要新闻来源，包含新浪、东方财富、同花顺等 |
| `data/raw/expanded_new_energy_50/news_astock.csv` | Astock 新闻导入结果 | 辅助新闻来源 |
| `data/raw/expanded_new_energy_50/news_merged.csv` | 合并后的新闻数据 | Tushare 与 Astock 合并后的中间文件 |
| `data/raw/expanded_new_energy_50/news_aligned.csv` | 对齐交易日后的新闻数据 | FinBERT 特征提取的直接输入 |
| `data/raw/expanded_new_energy_50/news_tushare.failures.csv` | 新闻下载失败记录 | 检查新闻源覆盖情况 |
| `data/raw/market.csv` | 20 只股票旧实验行情数据 | 20 只实验和事件筛选消融输入 |
| `data/raw/news_tushare.csv`、`news_astock.csv`、`news_super_enriched*.csv` | 20 只实验新闻数据 | 20 只实验的文本输入和增强新闻版本 |
| `data/raw/sentiment_train.csv`、`sentiment_train.stats.csv` | 金融情感训练数据及统计 | 可用于重新微调 `runs/finbert_cn` 情感分类头 |

行情 CSV 的关键字段：

```text
stock_code,trade_date,open,high,low,close,volume,amount,turnover_rate,pct_chg,振幅,涨跌额
```

字段含义：

| 字段 | 中文含义 |
|---|---|
| `stock_code` | 股票代码，如 `300750.SZ` |
| `trade_date` | 交易日期 |
| `open/high/low/close` | 开盘价、最高价、最低价、收盘价 |
| `volume` | 成交量 |
| `amount` | 成交额 |
| `turnover_rate` | 换手率 |
| `pct_chg` | 当日涨跌幅 |
| `振幅`、`涨跌额` | 数据源保留的补充行情字段 |

新闻 CSV 的关键字段：

```text
stock_code,trade_date,text,source,url
```

字段含义：

| 字段 | 中文含义 |
|---|---|
| `stock_code` | 新闻关联股票 |
| `trade_date` | 新闻对应交易日；非交易日新闻已经对齐到相邻有效交易日 |
| `text` | 新闻标题、摘要或正文拼接文本 |
| `source` | 新闻来源，如 `sina`、`eastmoney`、`10jqka`、`astock_stock_news` |
| `url` | 新闻链接，部分来源可能为空 |

### 3.3 文本特征与最终训练数据

| 文件 | 中文解释 | 数据角色 |
|---|---|---|
| `data/processed/expanded_new_energy_50/text_features_finbert_news_per_text_mean.csv` | 49 只实验 FinBERT 文本特征 | 最终训练数据集的文本输入 |
| `data/processed/expanded_new_energy_50/aligned_dataset_h10_mean_thr1_source_count_ge2_textonly.csv` | 49 只实验最终对齐训练数据 | 主实验四模型训练输入 |
| `data/processed/expanded_new_energy_50/aligned_dataset_h10_mean_thr1_source_count_ge2_textonly_summary.json` | 49 只最终数据统计 | 记录样本数、标签分布、日期范围 |
| `data/processed/text_features_finbert_super_enriched_news_only_per_text_mean.csv` | 20 只实验 FinBERT 文本特征 | 20 只事件筛选消融输入 |
| `data/processed/event_filters/aligned_dataset_finbert_news_event_h10_mean_thr1_textonly_source_count_ge2.csv` | 20 只 source_count>=2 最终数据集 | 20 只旧主实验训练输入 |
| `data/processed/event_filters/aligned_dataset_finbert_news_event_h10_mean_thr1_textonly_summary.json` | 20 只事件筛选数据统计 | 记录不同事件筛选策略的数据规模 |

文本特征 CSV 的关键字段：

```text
stock_code,trade_date,
text_prob_negative,text_prob_neutral,text_prob_positive,text_sentiment_score,
text_emb_0,...,text_emb_767,
text_news_count,text_source_count,text_length_mean,text_length_sum,
text_sentiment_score_max,text_sentiment_score_min,text_sentiment_score_std
```

字段含义：

| 字段 | 中文含义 |
|---|---|
| `text_prob_negative/text_prob_neutral/text_prob_positive` | FinBERT 输出的负面/中性/正面概率 |
| `text_sentiment_score` | 情绪分数，通常为正面概率减负面概率 |
| `text_emb_0 ... text_emb_767` | FinBERT `[CLS]` 语义向量，共 768 维 |
| `text_news_count` | 当日该股票新闻数量 |
| `text_source_count` | 当日该股票被多少个不同来源报道 |
| `text_length_mean/text_length_sum` | 文本长度统计 |
| `text_sentiment_score_max/min/std` | 当日多条新闻情绪分数的统计量 |

最终训练数据集的关键字段：

```text
stock_code,trade_date,
open,high,low,close,volume,amount,turnover_rate,pct_chg,
ma5,ma10,ma20,rsi14,macd,macd_signal,macd_hist,
future_mean_close,future_return,label,
label_mode,label_horizon,return_threshold,drop_neutral,text_only,
text_prob_negative,...,text_emb_767,text_news_count,text_source_count,...,
has_text_raw,has_text,event_min_text_source_count
```

字段含义：

| 字段 | 中文含义 |
|---|---|
| `ma5/ma10/ma20` | 5/10/20 日移动平均线 |
| `rsi14` | 14 日 RSI 技术指标 |
| `macd/macd_signal/macd_hist` | MACD 相关指标 |
| `future_mean_close` | 未来 10 个交易日平均收盘价 |
| `future_return` | 相对目标日收盘价的未来平均收益 |
| `label` | 二分类标签，1 为上涨，0 为下跌 |
| `label_mode` | 标签模式，主实验为 `future_mean` |
| `label_horizon` | 标签预测窗口，主实验为 10 |
| `return_threshold` | 涨跌阈值，主实验为 0.01 |
| `drop_neutral` | 是否丢弃中性样本，主实验为 True |
| `text_only` | 是否只保留有新闻样本，主实验为 True |
| `has_text_raw/has_text` | 原始是否有文本、事件筛选后是否有文本 |
| `event_min_text_source_count` | 事件筛选的最小来源数，主实验为 2 |

## 4. 当前保留的代码文件

### 4.1 数据与标签构造

| 文件 | 中文解释 |
|---|---|
| `astock_finbert_bilstm/data/download_market.py` | 行情下载入口；读取股票池，下载日线行情 |
| `astock_finbert_bilstm/data/sources.py` | 具体行情源封装，供下载脚本调用 |
| `astock_finbert_bilstm/data/features.py` | 清洗行情字段，计算 MA、RSI、MACD 等技术指标 |
| `astock_finbert_bilstm/data/dataset.py` | 核心数据集逻辑：行情与文本按 `stock_code + trade_date` 对齐，构造未来收益标签，筛选事件样本，生成模型序列 |
| `astock_finbert_bilstm/data/build_dataset.py` | 构建对齐数据集的命令行入口 |

### 4.2 文本数据与 FinBERT 特征

| 文件 | 中文解释 |
|---|---|
| `astock_finbert_bilstm/text/download_tushare_news.py` | 下载 Tushare 多来源新闻 |
| `astock_finbert_bilstm/text/import_stock_texts.py` | 导入 Astock 新闻，并与其他文本 CSV 合并 |
| `astock_finbert_bilstm/text/align_text_dates.py` | 将新闻日期对齐到有效交易日 |
| `astock_finbert_bilstm/text/extract_features.py` | 使用 FinBERT 输出情绪概率、情绪分数和 768 维语义向量 |
| `astock_finbert_bilstm/text/features.py` | 聚合多条新闻文本特征，生成新闻数量、来源数量、长度、情绪统计等事件强度特征 |
| `astock_finbert_bilstm/text/fine_tune_sentiment.py` | 可选：使用 `sentiment_train.csv` 微调 FinBERT 情感分类头 |
| `astock_finbert_bilstm/text/prepare_sentiment_dataset.py`、`sentiment_data.py` | 可选：准备金融情感训练数据 |

### 4.3 模型与实验

| 文件 | 中文解释 |
|---|---|
| `astock_finbert_bilstm/modeling/models.py` | 定义 `market_lstm`、`market_bilstm`、`concat_fusion`、`attention_fusion` |
| `astock_finbert_bilstm/modeling/metrics.py` | 计算 Accuracy、Precision、Recall、F1、AUC 和混淆矩阵 |
| `astock_finbert_bilstm/experiments/train.py` | 单个模型训练入口，负责数据划分、标准化、训练、评估和结果保存 |
| `astock_finbert_bilstm/experiments/run_ablation.py` | 四模型对比入口，同时训练行情基准和融合模型 |
| `astock_finbert_bilstm/experiments/run_expanded_strong_news_pipeline.py` | 49 只扩展股票池完整流程：行情、新闻、合并、对齐、特征、数据集、训练 |
| `astock_finbert_bilstm/experiments/prepare_event_filter_datasets.py` | 生成事件筛选消融数据集 |
| `astock_finbert_bilstm/experiments/run_text_feature_ablation.py`、`text_feature_sets.py` | 文本特征组和标准化消融 |

仍保留但非最终主线的脚本：`infer.py`、`explain.py`、`run_encoder_comparison.py`、`run_industry_pipeline.py`、`download_announcements.py`、`download_stock_texts.py`。它们属于早期推理、解释、编码器比较或旧数据源流程，阅读主实验时可以先跳过。

## 5. 复现路径

### 5.1 只复现最终四模型结果

这是最直接的复现方式。输入已经是最终对齐数据集，不需要重新下载新闻或重新提取 FinBERT 特征。

```bash
python -m astock_finbert_bilstm.experiments.run_ablation \
  --dataset data/processed/expanded_new_energy_50/aligned_dataset_h10_mean_thr1_source_count_ge2_textonly.csv \
  --output-dir runs/expanded_new_energy_50_source_count_ge2 \
  --epochs 20 \
  --window-size 20 \
  --batch-size 64
```

输入：

```text
data/processed/expanded_new_energy_50/aligned_dataset_h10_mean_thr1_source_count_ge2_textonly.csv
```

输出：

```text
runs/expanded_new_energy_50_source_count_ge2/ablation_summary.json
runs/expanded_new_energy_50_source_count_ge2/ablation_comparison.csv
```

### 5.2 从已保留的中间数据重建最终数据集并训练

如果想重新执行“文本特征 + 行情 -> 最终对齐数据集 -> 训练”这一步，可以使用扩展股票池 pipeline，并跳过已经完成的下载、合并和特征提取。

```bash
python -m astock_finbert_bilstm.experiments.run_expanded_strong_news_pipeline \
  --stock-list data/stock_pools/new_energy_50.csv \
  --raw-dir data/raw/expanded_new_energy_50 \
  --processed-dir data/processed/expanded_new_energy_50 \
  --run-dir runs/expanded_new_energy_50_source_count_ge2 \
  --finbert-model runs/finbert_cn \
  --skip-market \
  --skip-tushare \
  --skip-astock \
  --skip-merge \
  --skip-features
```

主要输入：

```text
data/raw/expanded_new_energy_50/market.csv
data/processed/expanded_new_energy_50/text_features_finbert_news_per_text_mean.csv
```

主要输出：

```text
data/processed/expanded_new_energy_50/aligned_dataset_h10_mean_thr1_source_count_ge2_textonly.csv
runs/expanded_new_energy_50_source_count_ge2/ablation_summary.json
runs/expanded_new_energy_50_source_count_ge2/ablation_comparison.csv
```

### 5.3 从新闻文本重新提取 FinBERT 特征

如果要从 `news_aligned.csv` 重新生成文本特征，需要保留的 FinBERT 模型目录是：

```text
runs/finbert_cn
```

命令：

```bash
python -m astock_finbert_bilstm.text.extract_features \
  --text-csv data/raw/expanded_new_energy_50/news_aligned.csv \
  --output data/processed/expanded_new_energy_50/text_features_finbert_news_per_text_mean.csv \
  --model runs/finbert_cn \
  --include-sentiment \
  --aggregation-mode per_text_mean \
  --batch-size 64 \
  --max-length 128
```

输入：

```text
data/raw/expanded_new_energy_50/news_aligned.csv
runs/finbert_cn/
```

输出：

```text
data/processed/expanded_new_energy_50/text_features_finbert_news_per_text_mean.csv
```

### 5.4 重新构建 20 只股票 source_count>=2 数据集

20 只旧主实验和事件筛选消融使用以下命令构建不同筛选策略的数据集：

```bash
python -m astock_finbert_bilstm.experiments.prepare_event_filter_datasets \
  --market-csv data/raw/market.csv \
  --text-features-csv data/processed/text_features_finbert_super_enriched_news_only_per_text_mean.csv \
  --output-dir data/processed/event_filters \
  --prefix aligned_dataset_finbert_news_event_h10_mean_thr1_textonly \
  --horizon 10 \
  --label-mode future_mean \
  --return-threshold 0.01 \
  --drop-neutral \
  --text-only
```

当前清理后只保留了主线使用的 `source_count>=2` 大型数据集，以及消融汇总结果。

## 6. 最终主实验结果

### 6.1 20 只股票旧主实验

主实验数据集：

```text
data/processed/event_filters/aligned_dataset_finbert_news_event_h10_mean_thr1_textonly_source_count_ge2.csv
```

对应结果：

```text
runs/event_filter_ablation/source_count_ge2/ablation_summary.json
```

| Model | Accuracy | F1 | AUC |
|---|---:|---:|---:|
| market_lstm | 0.4983 | 0.5773 | 0.5024 |
| market_bilstm | 0.5121 | 0.5635 | 0.5615 |
| concat_fusion | 0.4394 | 0.5781 | 0.4710 |
| attention_fusion | 0.5087 | 0.5590 | 0.5655 |

结论：`attention_fusion` 的 AUC 为 0.5655，略高于 `market_bilstm` 的 0.5615，文本增益约 +0.0040。

### 6.2 49 只扩展股票池主实验

股票池：

```text
data/stock_pools/new_energy_50.csv
```

实际成功下载行情 49 只股票，`002594.SZ` 比亚迪行情下载失败。

数据统计：

```text
rows = 56186
stocks = 49
raw_text_rows = 5662
event_text_rows = 2061
sequences = 2030
sequence_labels: 0 = 1045, 1 = 985
date range: 2019-02-25 到 2024-12-16
text_features = 779
```

结果文件：

```text
runs/expanded_new_energy_50_source_count_ge2/ablation_comparison.csv
runs/expanded_new_energy_50_source_count_ge2/ablation_summary.json
```

| Model | Accuracy | F1 | AUC |
|---|---:|---:|---:|
| market_lstm | 0.5320 | 0.6058 | 0.5537 |
| market_bilstm | 0.5345 | 0.5356 | 0.5498 |
| concat_fusion | 0.4828 | 0.6154 | 0.4702 |
| attention_fusion | 0.5394 | 0.5641 | 0.5633 |

结论：扩展股票池后，`attention_fusion` 的 AUC 为 0.5633，仍高于 `market_bilstm` 的 0.5498，相对 BiLSTM 的 AUC 增益约 +0.0135。

需要注意：49 只实验中 `attention_fusion` 的绝对 AUC 并没有高于 20 只实验，而是从 0.5655 略降到 0.5633；但纯行情 `market_bilstm` 从 0.5615 降至 0.5498，因此文本融合模型相对纯行情模型的优势更明显。

## 7. 消融实验结果

### 7.1 事件筛选消融

结果文件：

```text
runs/event_filter_ablation/event_filter_comparison.csv
runs/event_filter_ablation/event_filter_auc_pivot.csv
```

`attention_fusion` 结果：

| 筛选策略 | 样本数 | Accuracy | F1 | AUC |
|---|---:|---:|---:|---:|
| baseline | 3866 | 0.5401 | 0.4855 | 0.5447 |
| news_count >= 2 | 2238 | 0.5446 | 0.5507 | 0.5586 |
| news_count >= 3 | 1490 | 0.5000 | 0.5329 | 0.5307 |
| source_count >= 2 | 1442 | 0.5087 | 0.5590 | 0.5655 |
| sentiment_abs top50 | 1927 | 0.4585 | 0.5778 | 0.5208 |
| sentiment_abs top30 | 1157 | 0.4914 | 0.5462 | 0.4932 |

结论：`source_count >= 2` 是当前最有解释力的强新闻事件筛选策略。多来源共同报道比单纯新闻数量或情绪强度筛选更可靠。

### 7.2 新闻来源消融

结果文件：

```text
runs/source_ablation/source_ablation_comparison.csv
runs/source_ablation/source_ablation_auc_pivot.csv
```

结论：`drop_astock` 和 `10jqka+sina` 的 `attention_fusion` AUC 表现较好，说明 Tushare 多来源新闻对主结果贡献较大。Astock 在扩展实验中匹配较少，贡献相对较弱。

### 7.3 文本特征组与标准化消融

结果文件：

```text
runs/text_feature_ablation/text_feature_comparison.csv
runs/text_feature_ablation/text_feature_auc_pivot.csv
```

关键 AUC：

| Run | market_bilstm | concat_fusion | attention_fusion |
|---|---:|---:|---:|
| all_unscaled | 0.5615 | 0.4710 | 0.5655 |
| all_scaled | 0.5615 | 0.4694 | 0.4806 |
| sentiment_event_scaled | 0.5615 | 0.5530 | 0.5059 |
| embedding_only_scaled | 0.5615 | 0.4722 | 0.4759 |
| event_strength_only_scaled | 0.5615 | 0.5557 | 0.5361 |

结论：文本标准化没有提升主模型，`all_unscaled + attention_fusion` 仍是最佳主线。embedding-only 表现较弱，事件强度特征有一定信息但不足以超过完整文本特征组合。

### 7.4 超额收益标签补充实验

结果文件：

```text
runs/excess_return_source_count_ge2/ablation_comparison.csv
runs/excess_return_source_count_ge2/ablation_summary.json
```

| Model | Accuracy | F1 | AUC |
|---|---:|---:|---:|
| market_lstm | 0.5187 | 0.5442 | 0.5207 |
| market_bilstm | 0.5560 | 0.5103 | 0.5497 |
| concat_fusion | 0.5336 | 0.6944 | 0.5660 |
| attention_fusion | 0.5410 | 0.5176 | 0.5343 |

该实验作为补充实验，不作为主线。结果说明当前文本特征对绝对方向和事件关注度的解释更强，对个股相对板块表现的解释能力有限。

## 8. 论文表述口径

可以写：

```text
本文通过多来源共同报道定义强新闻事件，并将 FinBERT 文本特征与 BiLSTM 行情序列进行融合。
实验表明，在 source_count >= 2 的强新闻事件日中，attention_fusion 模型在 20 只股票与扩展 49 只股票实验中均取得最高 AUC。
尤其在扩展股票池后，attention_fusion AUC 达到 0.5633，高于 market_bilstm 的 0.5498，说明文本特征在高质量新闻事件样本中提供了行情序列之外的增量排序信息。
```

也需要诚实说明：

```text
整体 AUC 绝对值不高，说明短期涨跌预测仍然困难。
Accuracy 提升有限。
有效强新闻序列为 2030，样本规模仍不算大。
文本增益主要体现在 AUC 排序能力，而不是分类准确率的大幅提升。
```
