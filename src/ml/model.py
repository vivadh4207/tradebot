"""LSTM price-direction model. Torch import is kept at module top here —
this file is only imported by the trainer and the LSTMSignal AT CONSTRUCTION
time, both of which already require torch.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .checkpoint import CheckpointMeta


def _torch():
    import torch
    import torch.nn as nn
    return torch, nn


class LSTMPriceModel:
    """Wrapper namespace — we use a factory so the actual nn.Module class
    is only materialized after torch is imported. See `from_meta`.
    """

    @staticmethod
    def from_meta(meta: "CheckpointMeta"):
        torch, nn = _torch()

        class _Net(nn.Module):
            def __init__(self, input_size: int, hidden_size: int,
                          num_layers: int, dropout: float, num_classes: int):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout if num_layers > 1 else 0.0,
                    batch_first=True,
                )
                self.norm = nn.LayerNorm(hidden_size)
                self.head = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size // 2, num_classes),
                )

            def forward(self, x):
                # x: (batch, seq_len, features)
                out, _ = self.lstm(x)
                out = out[:, -1, :]          # take the final timestep
                out = self.norm(out)
                return self.head(out)         # logits, shape (batch, num_classes)

        return _Net(
            input_size=meta.input_size,
            hidden_size=meta.hidden_size,
            num_layers=meta.num_layers,
            dropout=meta.dropout,
            num_classes=meta.num_classes,
        )
