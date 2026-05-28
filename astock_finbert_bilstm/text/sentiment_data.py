from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


LABEL_ALIASES = {
    "negative": "negative",
    "neg": "negative",
    "bearish": "negative",
    "0": "negative",
    "负面": "negative",
    "負面": "negative",
    "消极": "negative",
    "利空": "negative",
    "neutral": "neutral",
    "neu": "neutral",
    "1": "neutral",
    "中性": "neutral",
    "中立": "neutral",
    "普通": "neutral",
    "positive": "positive",
    "pos": "positive",
    "bullish": "positive",
    "2": "positive",
    "正面": "positive",
    "积极": "positive",
    "利好": "positive",
}

TEXT_COLUMNS = ["text", "sentence", "content", "headline", "title", "news", "summary"]
LABEL_COLUMNS = ["label", "sentiment", "label_name", "label_cn", "polarity"]
LANGUAGE_COLUMNS = ["language", "lang", "locale"]


@dataclass(frozen=True)
class HuggingFaceSentimentSource:
    name: str
    dataset_id: str
    subset: str | None = None
    language: str | None = None
    description: str = ""


HF_SOURCES = {
    "kenpache_zh": HuggingFaceSentimentSource(
        name="kenpache_zh",
        dataset_id="Kenpache/multilingual-financial-sentiment",
        language="zh",
        description="Multilingual financial news sentiment dataset; keep Chinese rows only.",
    ),
    "eland_zh_tw": HuggingFaceSentimentSource(
        name="eland_zh_tw",
        dataset_id="p988744/eland-sentiment-zh-data",
        description="Chinese financial sentiment dataset for Taiwan stock market text.",
    ),
}

DEFAULT_SOURCE_NAMES = ["kenpache_zh"]


def normalize_label(value: Any) -> str | None:
    key = str(value).strip().lower()
    return LABEL_ALIASES.get(key)


def list_sources() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "name": source.name,
                "dataset_id": source.dataset_id,
                "language": source.language or "",
                "description": source.description,
            }
            for source in HF_SOURCES.values()
        ]
    )


def prepare_huggingface_source(source_name: str) -> pd.DataFrame:
    if source_name not in HF_SOURCES:
        raise ValueError(f"Unknown source `{source_name}`. Available: {sorted(HF_SOURCES)}")
    source = HF_SOURCES[source_name]

    try:
        from datasets import Dataset, DatasetDict, load_dataset
    except ImportError as exc:
        raise ImportError("Install `datasets` first: pip install datasets") from exc

    loaded = load_dataset(source.dataset_id, source.subset) if source.subset else load_dataset(source.dataset_id)
    frames = []
    if isinstance(loaded, DatasetDict):
        for split_name, split in loaded.items():
            frame = split.to_pandas()
            frame["source_split"] = split_name
            frames.append(frame)
    elif isinstance(loaded, Dataset):
        frame = loaded.to_pandas()
        frame["source_split"] = "train"
        frames.append(frame)
    else:
        raise TypeError(f"Unsupported Hugging Face dataset object: {type(loaded)!r}")

    raw = pd.concat(frames, ignore_index=True)
    return normalize_sentiment_frame(raw, source_name=source.name, language=source.language)


def normalize_sentiment_frame(
    raw: pd.DataFrame,
    source_name: str,
    language: str | None = None,
) -> pd.DataFrame:
    frame = raw.copy()
    if language:
        language_col = _first_existing(frame, LANGUAGE_COLUMNS)
        if language_col:
            frame = frame[frame[language_col].astype(str).str.lower().str.startswith(language.lower())].copy()

    text_col = _first_existing(frame, TEXT_COLUMNS)
    label_col = _first_existing(frame, LABEL_COLUMNS)
    if not text_col or not label_col:
        raise ValueError(
            f"Cannot find text/label columns for {source_name}. "
            f"Columns found: {list(frame.columns)}. "
            f"Text candidates: {TEXT_COLUMNS}; label candidates: {LABEL_COLUMNS}."
        )

    out = pd.DataFrame(
        {
            "text": frame[text_col].astype(str).str.strip(),
            "label": frame[label_col].map(normalize_label),
            "source": source_name,
        }
    )
    out = out.dropna(subset=["text", "label"])
    out = out[out["text"].str.len() > 0]
    return out.reset_index(drop=True)


def merge_sources(
    source_names: list[str],
    output: str | Path,
    min_text_length: int = 4,
    max_samples: int | None = None,
    seed: int = 42,
    skip_failed: bool = True,
) -> pd.DataFrame:
    frames = []
    failed = []
    for name in source_names:
        try:
            frames.append(prepare_huggingface_source(name))
        except Exception as exc:
            if not skip_failed:
                raise
            failed.append((name, exc))

    if not frames:
        details = "; ".join(f"{name}: {exc}" for name, exc in failed)
        raise RuntimeError(f"No sentiment dataset source could be loaded. Failures: {details}")

    for name, exc in failed:
        print(f"[WARN] Skip failed sentiment source `{name}`: {exc}")

    merged = pd.concat(frames, ignore_index=True)
    merged = merged[merged["text"].str.len() >= min_text_length]
    merged = merged.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)
    if max_samples and len(merged) > max_samples:
        merged = merged.sample(n=max_samples, random_state=seed).reset_index(drop=True)

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    merged[["text", "label"]].to_csv(output, index=False)
    stats = label_stats(merged)
    stats.to_csv(output.with_suffix(".stats.csv"), index=False)
    return merged


def label_stats(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total = len(frame)
    for label, count in frame["label"].value_counts().sort_index().items():
        rows.append({"label": label, "count": int(count), "ratio": float(count / max(total, 1))})
    return pd.DataFrame(rows)


def _first_existing(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    lowered = {col.lower(): col for col in frame.columns}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None
