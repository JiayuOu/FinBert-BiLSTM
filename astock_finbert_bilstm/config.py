from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import json


DEFAULT_FINBERT_MODEL = "valuesimplex-ai-lab/FinBERT2-base"

DEFAULT_MARKET_FEATURES = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "turnover_rate",
    "pct_chg",
    "ma5",
    "ma10",
    "ma20",
    "rsi14",
    "macd",
    "macd_signal",
    "macd_hist",
]

DEFAULT_TEXT_FEATURES = [
    "text_prob_negative",
    "text_prob_neutral",
    "text_prob_positive",
    "text_sentiment_score",
]


@dataclass
class ExperimentConfig:
    model: str = "attention_fusion"
    window_size: int = 20
    horizon: int = 1
    market_features: list[str] = field(default_factory=lambda: DEFAULT_MARKET_FEATURES.copy())
    text_features: list[str] = field(default_factory=lambda: DEFAULT_TEXT_FEATURES.copy())
    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.2
    learning_rate: float = 1e-3
    batch_size: int = 64
    epochs: int = 20
    seed: int = 42
    test_ratio: float = 0.2
    val_ratio: float = 0.1
    scale_text_features: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, path: str | Path) -> "ExperimentConfig":
        with Path(path).open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
