from __future__ import annotations

import torch
from torch import nn


class MarketEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.2,
        bidirectional: bool = True,
    ):
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
            bidirectional=bidirectional,
        )
        self.output_dim = hidden_size * (2 if bidirectional else 1)

    def forward(self, market_window: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(market_window)
        if self.lstm.bidirectional:
            return torch.cat([hidden[-2], hidden[-1]], dim=-1)
        return hidden[-1]


class AttentionFusionClassifier(nn.Module):
    def __init__(
        self,
        market_dim: int,
        text_dim: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.2,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.market_encoder = MarketEncoder(market_dim, hidden_size, num_layers, dropout, bidirectional)
        shared_dim = self.market_encoder.output_dim
        self.text_projection = nn.Sequential(
            nn.Linear(text_dim, shared_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.branch_score = nn.Linear(shared_dim, 1)
        self.classifier = nn.Sequential(
            nn.Linear(shared_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 2),
        )

    def forward(self, market_window: torch.Tensor, text_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        market_vec = self.market_encoder(market_window)
        text_vec = self.text_projection(text_features)
        branches = torch.stack([market_vec, text_vec], dim=1)
        weights = torch.softmax(self.branch_score(branches).squeeze(-1), dim=-1)
        fused = (branches * weights.unsqueeze(-1)).sum(dim=1)
        return self.classifier(fused), weights


class MarketOnlyClassifier(nn.Module):
    def __init__(
        self,
        market_dim: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.2,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.market_encoder = MarketEncoder(market_dim, hidden_size, num_layers, dropout, bidirectional)
        self.classifier = nn.Sequential(
            nn.Linear(self.market_encoder.output_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 2),
        )

    def forward(self, market_window: torch.Tensor, text_features: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.classifier(self.market_encoder(market_window))
        weights = torch.ones((market_window.shape[0], 2), device=market_window.device)
        weights[:, 1] = 0.0
        return logits, weights


class ConcatFusionClassifier(nn.Module):
    def __init__(
        self,
        market_dim: int,
        text_dim: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.2,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.market_encoder = MarketEncoder(market_dim, hidden_size, num_layers, dropout, bidirectional)
        self.classifier = nn.Sequential(
            nn.Linear(self.market_encoder.output_dim + text_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 2),
        )

    def forward(self, market_window: torch.Tensor, text_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        market_vec = self.market_encoder(market_window)
        logits = self.classifier(torch.cat([market_vec, text_features], dim=-1))
        weights = torch.full((market_window.shape[0], 2), 0.5, device=market_window.device)
        return logits, weights


def build_model(
    model_name: str,
    market_dim: int,
    text_dim: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
) -> nn.Module:
    if model_name == "market_bilstm":
        return MarketOnlyClassifier(market_dim, hidden_size, num_layers, dropout, bidirectional=True)
    if model_name == "market_lstm":
        return MarketOnlyClassifier(market_dim, hidden_size, num_layers, dropout, bidirectional=False)
    if model_name == "concat_fusion":
        return ConcatFusionClassifier(market_dim, text_dim, hidden_size, num_layers, dropout, bidirectional=True)
    if model_name == "attention_fusion":
        return AttentionFusionClassifier(market_dim, text_dim, hidden_size, num_layers, dropout, bidirectional=True)
    raise ValueError(f"unknown model: {model_name}")
