from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def summarize_attention(prediction_csv: str | Path) -> dict[str, float]:
    df = pd.read_csv(prediction_csv)
    return {
        "market_weight_mean": float(df["market_weight"].mean()),
        "text_weight_mean": float(df["text_weight"].mean()),
        "market_weight_std": float(df["market_weight"].std()),
        "text_weight_std": float(df["text_weight"].std()),
    }


def permutation_importance_frame(
    baseline_probs: np.ndarray,
    permuted_probs_by_feature: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Model-agnostic feature importance if SHAP is unavailable.

    Importance is the mean absolute change of the upward probability after a
    feature is permuted.
    """
    rows = []
    base_up = baseline_probs[:, 1]
    for feature, probs in permuted_probs_by_feature.items():
        rows.append({"feature": feature, "importance": float(np.mean(np.abs(base_up - probs[:, 1])))})
    return pd.DataFrame(rows).sort_values("importance", ascending=False).reset_index(drop=True)
