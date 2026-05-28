from __future__ import annotations

TEXT_FEATURE_SET_CHOICES = [
    "all",
    "sentiment_only",
    "embedding_only",
    "event_strength_only",
    "sentiment_event",
]

_SENTIMENT_FEATURES = {
    "text_sentiment_score",
    "text_sentiment_score_max",
    "text_sentiment_score_min",
    "text_sentiment_score_std",
}
_EVENT_STRENGTH_FEATURES = {
    "text_news_count",
    "text_source_count",
    "text_length_mean",
    "text_length_sum",
}


def all_text_features(columns: list[str]) -> list[str]:
    return [c for c in columns if c.startswith("text_") and c != "text_only"]


def select_text_features(columns: list[str], feature_set: str) -> list[str]:
    text_features = all_text_features(columns)
    if feature_set == "all":
        selected = text_features
    elif feature_set == "sentiment_only":
        selected = [c for c in text_features if c.startswith("text_prob_") or c in _SENTIMENT_FEATURES]
    elif feature_set == "embedding_only":
        selected = [c for c in text_features if c.startswith("text_emb_")]
    elif feature_set == "event_strength_only":
        selected = [c for c in text_features if c in _EVENT_STRENGTH_FEATURES]
    elif feature_set == "sentiment_event":
        selected = [
            c
            for c in text_features
            if c.startswith("text_prob_") or c in _SENTIMENT_FEATURES or c in _EVENT_STRENGTH_FEATURES
        ]
    else:
        raise ValueError(f"unknown text feature set: {feature_set}")

    if not selected:
        raise ValueError(f"text feature set {feature_set!r} selected no columns")
    return selected
