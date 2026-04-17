"""Training loop. Uses GPU if available (auto-detect), falls back to CPU.

This file imports torch eagerly. Call it only when you're training.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .checkpoint import CheckpointMeta, save_checkpoint
from .features import FeatureStats
from .model import LSTMPriceModel


@dataclass
class TrainResult:
    best_val_loss: float
    best_val_accuracy: float
    best_epoch: int
    final_train_loss: float
    n_train: int
    n_val: int


def _pick_device():
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _class_weights(y: np.ndarray, num_classes: int = 3):
    import torch
    counts = np.bincount(y, minlength=num_classes).astype(np.float32)
    # inverse frequency, capped
    inv = (counts.sum() / (counts + 1e-6))
    inv = np.clip(inv, 0.5, 5.0)
    return torch.tensor(inv, dtype=torch.float32)


def train(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    meta: CheckpointMeta,
    *,
    epochs: int = 40, batch_size: int = 256, lr: float = 1e-3,
    weight_decay: float = 1e-4, early_stop_patience: int = 5,
    checkpoint_path: str = "checkpoints/lstm_best.pt",
    verbose: bool = True,
) -> TrainResult:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    device = _pick_device()
    model = LSTMPriceModel.from_meta(meta).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    cls_w = _class_weights(y_train, meta.num_classes).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=cls_w)

    def _loader(X, y, shuffle):
        ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=0, pin_memory=(device.type == "cuda"))

    train_loader = _loader(X_train, y_train, shuffle=True)
    val_loader = _loader(X_val, y_val, shuffle=False)

    best_val = float("inf")
    best_acc = 0.0
    best_epoch = 0
    patience = 0
    last_train_loss = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        total, total_loss = 0, 0.0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += yb.size(0)
            total_loss += float(loss.item()) * yb.size(0)
        last_train_loss = total_loss / max(1, total)

        model.eval()
        vl_total, vl_loss, vl_correct = 0, 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                logits = model(xb)
                loss = loss_fn(logits, yb)
                vl_total += yb.size(0)
                vl_loss += float(loss.item()) * yb.size(0)
                vl_correct += int((logits.argmax(dim=-1) == yb).sum().item())
        vl_loss /= max(1, vl_total)
        vl_acc = vl_correct / max(1, vl_total)

        if verbose:
            print(f"  epoch {epoch:3d}  train_loss={last_train_loss:.4f}  "
                  f"val_loss={vl_loss:.4f}  val_acc={vl_acc:.4f}")

        if vl_loss < best_val - 1e-4:
            best_val = vl_loss
            best_acc = vl_acc
            best_epoch = epoch
            patience = 0
            meta.val_accuracy = vl_acc
            save_checkpoint(checkpoint_path, model, meta)
        else:
            patience += 1
            if patience >= early_stop_patience:
                if verbose:
                    print(f"  early stopping at epoch {epoch} (no improvement for "
                          f"{early_stop_patience} epochs)")
                break

    return TrainResult(
        best_val_loss=float(best_val),
        best_val_accuracy=float(best_acc),
        best_epoch=best_epoch,
        final_train_loss=float(last_train_loss),
        n_train=int(X_train.shape[0]),
        n_val=int(X_val.shape[0]),
    )
