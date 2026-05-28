from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
import torch

from astock_finbert_bilstm.data.dataset import build_aligned_dataset, create_sequences, fit_transform_text_scaler
from astock_finbert_bilstm.experiments.text_feature_sets import select_text_features
from astock_finbert_bilstm.modeling.models import AttentionFusionClassifier, build_model


class DatasetAndModelTests(unittest.TestCase):
    def test_alignment_label_uses_next_day_without_future_features(self) -> None:
        market = pd.DataFrame(
            {
                "stock_code": ["000001.SZ"] * 5,
                "trade_date": pd.date_range("2024-01-01", periods=5).strftime("%Y-%m-%d"),
                "open": [10, 11, 12, 11, 13],
                "high": [11, 12, 13, 12, 14],
                "low": [9, 10, 11, 10, 12],
                "close": [10, 11, 10, 12, 11],
                "volume": [100, 110, 90, 120, 100],
                "turnover_rate": [1, 1, 1, 1, 1],
            }
        )
        text = pd.DataFrame(
            {
                "stock_code": ["000001.SZ"] * 5,
                "trade_date": pd.date_range("2024-01-01", periods=5).strftime("%Y-%m-%d"),
                "text_prob_negative": [0.2] * 5,
                "text_prob_neutral": [0.5] * 5,
                "text_prob_positive": [0.3] * 5,
                "text_sentiment_score": [0.1] * 5,
            }
        )
        aligned = build_aligned_dataset(market, text, horizon=1)
        self.assertEqual(aligned.loc[0, "label"], 1)
        self.assertEqual(aligned.loc[1, "label"], 0)
        self.assertEqual(aligned.loc[2, "label"], 1)
        self.assertEqual(len(aligned), 4)

        arrays = create_sequences(aligned, window_size=3)
        self.assertEqual(arrays.market.shape[1], 3)
        self.assertEqual(arrays.labels.tolist(), [1, 0])


    def test_text_only_event_filter_keeps_continuous_market_window(self) -> None:
        market = pd.DataFrame(
            {
                "stock_code": ["000001.SZ"] * 6,
                "trade_date": pd.date_range("2024-01-01", periods=6).strftime("%Y-%m-%d"),
                "open": [10, 11, 12, 13, 14, 15],
                "high": [11, 12, 13, 14, 15, 16],
                "low": [9, 10, 11, 12, 13, 14],
                "close": [10, 11, 12, 13, 14, 15],
                "volume": [100, 110, 120, 130, 140, 150],
                "turnover_rate": [1, 1, 1, 1, 1, 1],
            }
        )
        text = pd.DataFrame(
            {
                "stock_code": ["000001.SZ"] * 5,
                "trade_date": pd.date_range("2024-01-01", periods=5).strftime("%Y-%m-%d"),
                "text_prob_negative": [0.2] * 5,
                "text_prob_neutral": [0.5] * 5,
                "text_prob_positive": [0.3] * 5,
                "text_sentiment_score": [0.1] * 5,
                "text_news_count": [1, 1, 1, 2, 3],
                "text_source_count": [1, 1, 1, 1, 2],
            }
        )
        aligned = build_aligned_dataset(market, text, horizon=1, text_only=True, min_text_news_count=2)
        self.assertTrue(aligned.loc[aligned["trade_date"] == "2024-01-03", "has_text_raw"].item())
        self.assertFalse(aligned.loc[aligned["trade_date"] == "2024-01-03", "has_text"].item())
        self.assertTrue(aligned.loc[aligned["trade_date"] == "2024-01-04", "has_text"].item())

        arrays = create_sequences(aligned, window_size=3)
        self.assertEqual(arrays.meta["trade_date"].tolist(), ["2024-01-04", "2024-01-05"])
        self.assertEqual(arrays.market.shape[1], 3)

    def test_future_excess_mean_label_uses_equal_weight_benchmark(self) -> None:
        market = pd.DataFrame(
            {
                "stock_code": ["000001.SZ"] * 4 + ["000002.SZ"] * 4,
                "trade_date": list(pd.date_range("2024-01-01", periods=4).strftime("%Y-%m-%d")) * 2,
                "open": [100, 110, 110, 110, 100, 100, 90, 90],
                "high": [101, 111, 111, 111, 101, 101, 91, 91],
                "low": [99, 109, 109, 109, 99, 99, 89, 89],
                "close": [100, 110, 110, 110, 100, 100, 90, 90],
                "volume": [100] * 8,
                "turnover_rate": [1] * 8,
            }
        )

        aligned = build_aligned_dataset(
            market,
            horizon=1,
            label_mode="future_excess_mean",
            return_threshold=0.03,
            drop_neutral=True,
        )

        labels = {
            (row.stock_code, row.trade_date): int(row.label)
            for row in aligned[["stock_code", "trade_date", "label"]].itertuples(index=False)
        }
        self.assertEqual(labels[("000001.SZ", "2024-01-01")], 1)
        self.assertEqual(labels[("000002.SZ", "2024-01-01")], 0)
        self.assertEqual(labels[("000001.SZ", "2024-01-02")], 1)
        self.assertEqual(labels[("000002.SZ", "2024-01-02")], 0)
        self.assertEqual(len(aligned), 4)
        self.assertIn("benchmark_future_return", aligned.columns)
        self.assertIn("future_excess_return", aligned.columns)


    def test_text_scaler_uses_train_split_and_preserves_shape(self) -> None:
        text = np.asarray(
            [
                [1.0, 10.0],
                [3.0, 14.0],
                [100.0, 200.0],
            ],
            dtype=np.float32,
        )
        scaled, scaler = fit_transform_text_scaler(text, np.asarray([0, 1]))

        self.assertEqual(scaled.shape, text.shape)
        self.assertTrue(np.allclose(scaler.mean_, [2.0, 12.0]))
        self.assertTrue(np.allclose(scaled[[0, 1]].mean(axis=0), [0.0, 0.0], atol=1e-6))

    def test_text_feature_set_selection(self) -> None:
        columns = [
            "stock_code",
            "text_prob_negative",
            "text_prob_neutral",
            "text_prob_positive",
            "text_sentiment_score",
            "text_sentiment_score_std",
            "text_emb_0",
            "text_emb_1",
            "text_news_count",
            "text_source_count",
            "text_length_mean",
            "text_length_sum",
            "text_only",
        ]

        self.assertEqual(select_text_features(columns, "embedding_only"), ["text_emb_0", "text_emb_1"])
        self.assertEqual(
            select_text_features(columns, "event_strength_only"),
            ["text_news_count", "text_source_count", "text_length_mean", "text_length_sum"],
        )
        self.assertNotIn("text_emb_0", select_text_features(columns, "sentiment_event"))
        self.assertIn("text_news_count", select_text_features(columns, "sentiment_event"))


    def test_attention_fusion_shapes(self) -> None:
        model = AttentionFusionClassifier(market_dim=14, text_dim=4, hidden_size=8)
        logits, weights = model(torch.randn(2, 5, 14), torch.randn(2, 4))
        self.assertEqual(tuple(logits.shape), (2, 2))
        self.assertEqual(tuple(weights.shape), (2, 2))
        self.assertTrue(torch.allclose(weights.sum(dim=1), torch.ones(2), atol=1e-6))

    def test_ablation_model_factory(self) -> None:
        for name in ["market_lstm", "market_bilstm", "concat_fusion", "attention_fusion"]:
            model = build_model(name, market_dim=14, text_dim=4, hidden_size=8, num_layers=1, dropout=0.1)
            logits, weights = model(torch.randn(3, 6, 14), torch.randn(3, 4))
            self.assertEqual(tuple(logits.shape), (3, 2))
            self.assertEqual(tuple(weights.shape), (3, 2))


if __name__ == "__main__":
    unittest.main()
