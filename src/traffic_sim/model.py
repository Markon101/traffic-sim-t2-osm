from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn


@dataclass
class ModelConfig:
    input_dim: int
    max_signals: int
    num_actions: int
    d_model: int = 128
    num_layers: int = 4
    num_heads: int = 8
    dropout: float = 0.1


class TransformerDQN(nn.Module):
    """Transformer encoder with a CLS token for Q-value projection."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Linear(cfg.input_dim, cfg.d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        self.positional = nn.Parameter(torch.randn(1, cfg.max_signals + 1, cfg.d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.d_model * 4,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=cfg.num_layers,
            enable_nested_tensor=False,
        )
        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.num_actions)
        self._initialize()

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # obs: (batch, max_signals, input_dim)
        batch, seq_len, _ = obs.shape
        x = self.embedding(obs)

        cls_tokens = self.cls_token.expand(batch, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        pos = self.positional[:, : seq_len + 1, :]
        x = x + pos

        padding_mask = self._padding_mask(obs)
        cls_padding = torch.zeros((batch, 1), dtype=torch.bool, device=obs.device)
        mask = torch.cat([cls_padding, padding_mask], dim=1)

        encoded = self.transformer(x, src_key_padding_mask=mask)
        cls_state = encoded[:, 0, :]
        cls_state = self.norm(cls_state)
        q_values = self.head(cls_state)
        return q_values

    def _padding_mask(self, obs: torch.Tensor) -> torch.Tensor:
        # treat rows with all zeros as padding
        mask = obs.abs().sum(dim=-1) < 1e-5
        return mask

    def _initialize(self) -> None:
        nn.init.trunc_normal_(self.embedding.weight, std=0.02)
        nn.init.zeros_(self.embedding.bias)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)


class ResidualMLPDQN(nn.Module):
    """Simpler residual MLP baseline for comparison."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        hidden = 1024
        input_size = cfg.input_dim * cfg.max_signals

        self.trunk = nn.Sequential(
            nn.LayerNorm(input_size),
            nn.Linear(input_size, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.proj = nn.Linear(hidden, cfg.num_actions)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        batch = obs.shape[0]
        x = obs.view(batch, -1)
        trunk_output = self.trunk(x)
        q_values = self.proj(trunk_output)
        return q_values


def build_model(
    model_type: Literal["transformer", "residual"],
    cfg: ModelConfig,
) -> nn.Module:
    if model_type == "transformer":
        return TransformerDQN(cfg)
    if model_type == "residual":
        return ResidualMLPDQN(cfg)
    raise ValueError(f"Unknown model_type: {model_type}")
