"""Checkpoint save/load that stays torch-free on systems without PyTorch.

State-dict itself is always torch — we just avoid importing torch at
module load time so non-Jetson hosts can still run the bot and backtest
without installing PyTorch.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .features import FeatureStats, FEATURE_COLS


@dataclass
class CheckpointMeta:
    """Everything we need to run inference identical to training."""
    version: str = "lstm-v1"
    seq_len: int = 30
    horizon: int = 5
    input_size: int = 7
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    num_classes: int = 3
    up_thr: float = 0.001
    down_thr: float = -0.001
    features: List[str] = field(default_factory=lambda: list(FEATURE_COLS))
    stats: Dict[str, Any] = field(default_factory=lambda: {"means": [], "stds": []})
    trained_symbols: List[str] = field(default_factory=list)
    train_bar_count: int = 0
    val_accuracy: float = 0.0
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CheckpointMeta":
        # fill in missing fields with defaults (forward-compat for older ckpts)
        defaults = cls()
        base = defaults.to_dict()
        base.update(d)
        return cls(**base)


def save_checkpoint(path: str | Path, model, meta: CheckpointMeta) -> None:
    """Write model state-dict + metadata to disk.

    The meta is stored both inside the torch file AND as a side-car .json
    so you can inspect what's in a checkpoint without loading torch.
    """
    import torch   # lazy
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "meta": meta.to_dict()},
        path,
    )
    (path.with_suffix(".json")).write_text(json.dumps(meta.to_dict(), indent=2))


def load_checkpoint(path: str | Path, model_factory, map_location: str = "cpu"):
    """Load a checkpoint. Returns (model, CheckpointMeta).

    `model_factory` is a callable that takes a CheckpointMeta and returns
    a new (untrained) nn.Module — normally LSTMPriceModel.from_meta.
    """
    import torch   # lazy
    blob = torch.load(str(path), map_location=map_location, weights_only=False)
    meta = CheckpointMeta.from_dict(blob["meta"])
    model = model_factory(meta)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model, meta
