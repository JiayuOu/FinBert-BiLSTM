from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

os.environ.setdefault("USE_TF", "0")

import numpy as np
import pandas as pd
import torch

from ..config import DEFAULT_FINBERT_MODEL


@dataclass
class TextFeatureConfig:
    model_name: str = DEFAULT_FINBERT_MODEL
    max_length: int = 128
    batch_size: int = 16
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    include_sentiment: bool = False
    aggregation_mode: Literal["concat", "per_text_mean"] = "concat"


class FinBertFeatureExtractor:
    """Extract [CLS] embeddings and sentiment probabilities from a HF model.

    For thesis experiments, replace model_name with your fine-tuned Chinese
    financial sentiment checkpoint. The model should be a sequence classifier.
    """

    def __init__(self, config: TextFeatureConfig):
        from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        model_cls = AutoModelForSequenceClassification if config.include_sentiment else AutoModel
        self.model = model_cls.from_pretrained(config.model_name, output_hidden_states=True).to(config.device)
        self.model.eval()

    @torch.no_grad()
    def transform(self, texts: list[str]) -> pd.DataFrame:
        rows: list[dict[str, float]] = []
        for start in range(0, len(texts), self.config.batch_size):
            batch = texts[start : start + self.config.batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
                return_tensors="pt",
            ).to(self.config.device)
            output = self.model(**encoded)
            cls = output.hidden_states[-1][:, 0, :].detach().cpu().numpy()
            probs = None
            if self.config.include_sentiment:
                probs = torch.softmax(output.logits, dim=-1).detach().cpu().numpy()

            for idx, emb in enumerate(cls):
                row: dict[str, float] = {}
                if probs is not None:
                    neg, neu, pos = _normalize_three_class_probs(probs[idx])
                    row.update(
                        {
                            "text_prob_negative": float(neg),
                            "text_prob_neutral": float(neu),
                            "text_prob_positive": float(pos),
                            "text_sentiment_score": float(pos - neg),
                        }
                    )
                row.update({f"text_emb_{i}": float(v) for i, v in enumerate(emb)})
                rows.append(row)
        return pd.DataFrame(rows)


def _normalize_three_class_probs(probs: np.ndarray) -> tuple[float, float, float]:
    if probs.shape[0] >= 3:
        first_three = probs[:3]
        total = first_three.sum()
        if total <= 0:
            return 1 / 3, 1 / 3, 1 / 3
        return tuple((first_three / total).tolist())  # type: ignore[return-value]
    if probs.shape[0] == 2:
        neg = float(probs[0])
        pos = float(probs[1])
        neu = max(0.0, 1.0 - neg - pos)
        return neg, neu, pos
    return 1 / 3, 1 / 3, 1 / 3


def aggregate_texts(text_df: pd.DataFrame) -> pd.DataFrame:
    required = {"stock_code", "trade_date", "text"}
    missing = required - set(text_df.columns)
    if missing:
        raise ValueError(f"text data missing required columns: {sorted(missing)}")
    out = text_df.copy()
    out["trade_date"] = pd.to_datetime(out["trade_date"]).dt.strftime("%Y-%m-%d")
    out["stock_code"] = out["stock_code"].astype(str)
    return (
        out.groupby(["stock_code", "trade_date"], as_index=False)["text"]
        .apply(lambda values: " ".join(str(v) for v in values if pd.notna(v)))
        .reset_index(drop=True)
    )


def _prepare_text_rows(
    text_df: pd.DataFrame,
    include_sources: list[str] | None = None,
    exclude_sources: list[str] | None = None,
) -> pd.DataFrame:
    required = {"stock_code", "trade_date", "text"}
    missing = required - set(text_df.columns)
    if missing:
        raise ValueError(f"text data missing required columns: {sorted(missing)}")

    out = text_df.copy()
    out["trade_date"] = pd.to_datetime(out["trade_date"]).dt.strftime("%Y-%m-%d")
    out["stock_code"] = out["stock_code"].astype(str)
    out["text"] = out["text"].fillna("").astype(str).str.strip()
    out = out[out["text"] != ""].copy()

    if "source" not in out.columns:
        out["source"] = "unknown"
    out["source"] = out["source"].fillna("unknown").astype(str)
    if include_sources:
        out = out[out["source"].isin(include_sources)].copy()
    if exclude_sources:
        out = out[~out["source"].isin(exclude_sources)].copy()
    if out.empty:
        raise ValueError("No text rows remain after source filtering.")
    return out.reset_index(drop=True)


def _text_strength_features(text_df: pd.DataFrame) -> pd.DataFrame:
    out = text_df.copy()
    out["text_length"] = out["text"].astype(str).str.len()
    grouped = out.groupby(["stock_code", "trade_date"], as_index=False)
    return grouped.agg(
        text_news_count=("text", "size"),
        text_source_count=("source", "nunique"),
        text_length_mean=("text_length", "mean"),
        text_length_sum=("text_length", "sum"),
    )


def _aggregate_feature_rows(feature_df: pd.DataFrame, raw_text_df: pd.DataFrame) -> pd.DataFrame:
    id_cols = ["stock_code", "trade_date"]
    feature_cols = [c for c in feature_df.columns if c not in id_cols]
    mean_features = feature_df.groupby(id_cols, as_index=False)[feature_cols].mean()
    strength = _text_strength_features(raw_text_df)
    result = mean_features.merge(strength, on=id_cols, how="left")
    if "text_sentiment_score" in feature_df.columns:
        sentiment_stats = feature_df.groupby(id_cols)["text_sentiment_score"].agg(["max", "min", "std"]).reset_index()
        sentiment_stats = sentiment_stats.rename(
            columns={
                "max": "text_sentiment_score_max",
                "min": "text_sentiment_score_min",
                "std": "text_sentiment_score_std",
            }
        )
        result = result.merge(sentiment_stats, on=id_cols, how="left")
        result["text_sentiment_score_std"] = result["text_sentiment_score_std"].fillna(0.0)
    return result


def extract_text_feature_csv(
    text_csv: str | Path,
    output: str | Path,
    config: TextFeatureConfig,
    include_sources: list[str] | None = None,
    exclude_sources: list[str] | None = None,
) -> None:
    raw_text_df = _prepare_text_rows(pd.read_csv(text_csv), include_sources=include_sources, exclude_sources=exclude_sources)
    extractor = FinBertFeatureExtractor(config)
    if config.aggregation_mode == "concat":
        text_df = aggregate_texts(raw_text_df)
        features = extractor.transform(text_df["text"].astype(str).tolist())
        result = pd.concat([text_df[["stock_code", "trade_date"]].reset_index(drop=True), features], axis=1)
    elif config.aggregation_mode == "per_text_mean":
        features = extractor.transform(raw_text_df["text"].astype(str).tolist())
        feature_rows = pd.concat([raw_text_df[["stock_code", "trade_date"]].reset_index(drop=True), features], axis=1)
        result = _aggregate_feature_rows(feature_rows, raw_text_df)
    else:
        raise ValueError(f"Unsupported aggregation_mode: {config.aggregation_mode}")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
