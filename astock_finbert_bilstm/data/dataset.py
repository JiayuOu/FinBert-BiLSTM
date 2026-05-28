from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from ..config import DEFAULT_MARKET_FEATURES, DEFAULT_TEXT_FEATURES
from .features import add_technical_indicators


def build_aligned_dataset(
    market_df: pd.DataFrame,
    text_feature_df: pd.DataFrame | None = None,
    horizon: int = 1,
    label_mode: str = "future_close",
    return_threshold: float = 0.0,
    drop_neutral: bool = False,
    text_only: bool = False,
    min_text_news_count: int | None = None,
    min_text_source_count: int | None = None,
    min_abs_text_sentiment_score: float | None = None,
    text_sentiment_abs_quantile: float | None = None,
    min_text_sentiment_score_std: float | None = None,
) -> pd.DataFrame:
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if label_mode not in {"future_close", "future_mean", "future_excess_mean"}:
        raise ValueError("label_mode must be one of: future_close, future_mean, future_excess_mean")
    if return_threshold < 0:
        raise ValueError("return_threshold must be >= 0")
    if min_text_news_count is not None and min_text_news_count < 1:
        raise ValueError("min_text_news_count must be >= 1")
    if min_text_source_count is not None and min_text_source_count < 1:
        raise ValueError("min_text_source_count must be >= 1")
    if min_abs_text_sentiment_score is not None and min_abs_text_sentiment_score < 0:
        raise ValueError("min_abs_text_sentiment_score must be >= 0")
    if text_sentiment_abs_quantile is not None and not 0 <= text_sentiment_abs_quantile <= 1:
        raise ValueError("text_sentiment_abs_quantile must be between 0 and 1")
    if min_text_sentiment_score_std is not None and min_text_sentiment_score_std < 0:
        raise ValueError("min_text_sentiment_score_std must be >= 0")

    market = add_technical_indicators(market_df)
    market = market.sort_values(["stock_code", "trade_date"]).copy()

    grouped_close = market.groupby("stock_code")["close"]
    if label_mode == "future_close":
        market["future_close"] = grouped_close.shift(-horizon)
        label_target_col = "future_close"
    else:
        future_mean = sum(grouped_close.shift(-step) for step in range(1, horizon + 1)) / horizon
        market["future_mean_close"] = future_mean
        label_target_col = "future_mean_close"
    market["future_return"] = market[label_target_col] / market["close"] - 1.0
    market = market.dropna(subset=[label_target_col, "future_return"]).copy()
    label_return_col = "future_return"
    if label_mode == "future_excess_mean":
        market["benchmark_future_return"] = market.groupby("trade_date")["future_return"].transform("mean")
        market["future_excess_return"] = market["future_return"] - market["benchmark_future_return"]
        market = market.dropna(subset=["benchmark_future_return", "future_excess_return"]).copy()
        label_return_col = "future_excess_return"
    if drop_neutral and return_threshold > 0:
        market = market[market[label_return_col].abs() >= return_threshold].copy()
    market["label"] = (market[label_return_col] >= return_threshold).astype(int)
    market["label_mode"] = label_mode
    market["label_horizon"] = horizon
    market["return_threshold"] = return_threshold
    market["drop_neutral"] = drop_neutral
    market["text_only"] = text_only

    if text_feature_df is not None:
        text = text_feature_df.copy()
        text["trade_date"] = pd.to_datetime(text["trade_date"]).dt.strftime("%Y-%m-%d")
        text["stock_code"] = text["stock_code"].astype(str)
        merged = market.merge(text, on=["stock_code", "trade_date"], how="left")
    else:
        merged = market

    text_cols = [c for c in merged.columns if c.startswith("text_") and c != "text_only"]
    if "text_news_count" in merged.columns:
        has_text_raw = merged["text_news_count"].notna() & (merged["text_news_count"] > 0)
    elif text_cols:
        has_text_raw = merged[text_cols].notna().any(axis=1)
    else:
        has_text_raw = pd.Series(False, index=merged.index)

    event_mask = has_text_raw.copy()
    if min_text_news_count is not None:
        if "text_news_count" not in merged.columns:
            raise ValueError("min_text_news_count requires text_news_count in text features")
        event_mask &= merged["text_news_count"].fillna(0) >= min_text_news_count
    if min_text_source_count is not None:
        if "text_source_count" not in merged.columns:
            raise ValueError("min_text_source_count requires text_source_count in text features")
        event_mask &= merged["text_source_count"].fillna(0) >= min_text_source_count
    if min_abs_text_sentiment_score is not None or text_sentiment_abs_quantile is not None:
        if "text_sentiment_score" not in merged.columns:
            raise ValueError("sentiment strength filters require text_sentiment_score in text features")
        abs_score = merged["text_sentiment_score"].abs()
        if text_sentiment_abs_quantile is not None:
            quantile_base = abs_score[has_text_raw & abs_score.notna()]
            quantile_threshold = float(quantile_base.quantile(text_sentiment_abs_quantile)) if not quantile_base.empty else float("inf")
            event_mask &= abs_score.fillna(-1) >= quantile_threshold
            merged["event_abs_sentiment_quantile_threshold"] = quantile_threshold
        if min_abs_text_sentiment_score is not None:
            event_mask &= abs_score.fillna(-1) >= min_abs_text_sentiment_score
    if min_text_sentiment_score_std is not None:
        if "text_sentiment_score_std" not in merged.columns:
            raise ValueError("min_text_sentiment_score_std requires text_sentiment_score_std in text features")
        event_mask &= merged["text_sentiment_score_std"].fillna(0) >= min_text_sentiment_score_std

    merged["has_text_raw"] = has_text_raw
    merged["has_text"] = event_mask
    merged["event_min_text_news_count"] = min_text_news_count if min_text_news_count is not None else 0
    merged["event_min_text_source_count"] = min_text_source_count if min_text_source_count is not None else 0
    merged["event_min_abs_text_sentiment_score"] = min_abs_text_sentiment_score if min_abs_text_sentiment_score is not None else 0.0
    merged["event_text_sentiment_abs_quantile"] = text_sentiment_abs_quantile if text_sentiment_abs_quantile is not None else 0.0
    merged["event_min_text_sentiment_score_std"] = min_text_sentiment_score_std if min_text_sentiment_score_std is not None else 0.0
    # Keep full market history so sequence windows remain consecutive trading days.
    # create_sequences filters target rows when text_only is enabled.
    for col in text_cols:
        merged[col] = merged[col].fillna(0.0)

    for col in DEFAULT_MARKET_FEATURES:
        if col not in merged.columns:
            merged[col] = 0.0
    for col in DEFAULT_TEXT_FEATURES:
        if col not in merged.columns:
            merged[col] = 0.0

    numeric_cols = DEFAULT_MARKET_FEATURES + [c for c in merged.columns if c.startswith("text_") and c != "text_only"]
    merged[numeric_cols] = merged[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return merged.sort_values(["stock_code", "trade_date"]).reset_index(drop=True)


def save_aligned_dataset(
    market_csv: str | Path,
    text_features_csv: str | Path | None,
    output: str | Path,
    horizon: int = 1,
    label_mode: str = "future_close",
    return_threshold: float = 0.0,
    drop_neutral: bool = False,
    text_only: bool = False,
    min_text_news_count: int | None = None,
    min_text_source_count: int | None = None,
    min_abs_text_sentiment_score: float | None = None,
    text_sentiment_abs_quantile: float | None = None,
    min_text_sentiment_score_std: float | None = None,
) -> None:
    market = pd.read_csv(market_csv)
    text = pd.read_csv(text_features_csv) if text_features_csv else None
    result = build_aligned_dataset(
        market,
        text,
        horizon=horizon,
        label_mode=label_mode,
        return_threshold=return_threshold,
        drop_neutral=drop_neutral,
        text_only=text_only,
        min_text_news_count=min_text_news_count,
        min_text_source_count=min_text_source_count,
        min_abs_text_sentiment_score=min_abs_text_sentiment_score,
        text_sentiment_abs_quantile=text_sentiment_abs_quantile,
        min_text_sentiment_score_std=min_text_sentiment_score_std,
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)


@dataclass
class SequenceArrays:
    market: np.ndarray
    text: np.ndarray
    labels: np.ndarray
    meta: pd.DataFrame


def create_sequences(
    df: pd.DataFrame,
    market_features: list[str] | None = None,
    text_features: list[str] | None = None,
    window_size: int = 20,
) -> SequenceArrays:
    market_features = market_features or DEFAULT_MARKET_FEATURES
    text_features = text_features or DEFAULT_TEXT_FEATURES

    markets: list[np.ndarray] = []
    texts: list[np.ndarray] = []
    labels: list[int] = []
    meta_rows: list[dict[str, object]] = []

    for stock_code, stock in df.sort_values(["stock_code", "trade_date"]).groupby("stock_code"):
        stock = stock.reset_index(drop=True)
        if len(stock) < window_size:
            continue
        for end in range(window_size - 1, len(stock)):
            window = stock.iloc[end - window_size + 1 : end + 1]
            target = stock.iloc[end]
            target_text_only = str(target.get("text_only", False)).lower() == "true"
            target_has_text = str(target.get("has_text", False)).lower() == "true"
            if target_text_only and not target_has_text:
                continue
            markets.append(window[market_features].to_numpy(dtype=np.float32))
            texts.append(target[text_features].to_numpy(dtype=np.float32))
            labels.append(int(target["label"]))
            meta_rows.append({"stock_code": stock_code, "trade_date": target["trade_date"]})

    if not markets:
        raise ValueError("No sequences created. Check stock history length and window_size.")

    return SequenceArrays(
        market=np.stack(markets),
        text=np.stack(texts),
        labels=np.asarray(labels, dtype=np.int64),
        meta=pd.DataFrame(meta_rows),
    )


def chronological_split(meta: pd.DataFrame, test_ratio: float, val_ratio: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dates = pd.to_datetime(meta["trade_date"])
    order = np.argsort(dates.to_numpy())
    n = len(meta)
    test_start = int(n * (1 - test_ratio))
    val_start = int(test_start * (1 - val_ratio))
    train_idx = order[:val_start]
    val_idx = order[val_start:test_start]
    test_idx = order[test_start:]
    return train_idx, val_idx, test_idx


def fit_transform_market_scaler(market: np.ndarray, train_idx: np.ndarray) -> tuple[np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    n_samples, window, n_features = market.shape
    train_flat = market[train_idx].reshape(-1, n_features)
    scaler.fit(train_flat)
    scaled = scaler.transform(market.reshape(-1, n_features)).reshape(n_samples, window, n_features)
    return scaled.astype(np.float32), scaler


def fit_transform_text_scaler(text: np.ndarray, train_idx: np.ndarray) -> tuple[np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    scaler.fit(text[train_idx])
    scaled = scaler.transform(text)
    return scaled.astype(np.float32), scaler


class StockSequenceDataset(Dataset):
    def __init__(self, market: np.ndarray, text: np.ndarray, labels: np.ndarray):
        self.market = torch.tensor(market, dtype=torch.float32)
        self.text = torch.tensor(text, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.market[index], self.text[index], self.labels[index]
