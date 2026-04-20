"""LSTMSignal — wraps a trained checkpoint as a SignalSource.

Lazy torch import. If torch isn't installed OR the checkpoint doesn't exist,
`emit` silently returns None and logs once at construction. The rest of
the bot continues normally — the existing momentum/ORB/VWAP signals still
fire.

This signal is deliberately a WEAK predictor on its own. The value is
in the 14-filter chain voting against or alongside it — never as a
standalone entry trigger.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Optional

import numpy as np

from ..core.types import Signal, Side, OptionRight
from ..core.logger import get_logger
from ..ml.features import build_feature_matrix, FeatureStats
from ..ml.checkpoint import CheckpointMeta, load_checkpoint
from .base import SignalSource, SignalContext


log = get_logger(__name__)


class LSTMSignal(SignalSource):
    name = "lstm"

    def __init__(self,
                 checkpoint_path: Optional[str] = None,
                 min_confidence: float = 0.55,
                 device: Optional[str] = None,
                 journal=None,
                 timeframe_minutes: int = 5,
                 log_all_predictions: bool = True):
        self.checkpoint_path = checkpoint_path or os.getenv(
            "LSTM_MODEL_PATH", "checkpoints/lstm_best.pt",
        )
        self.min_confidence = min_confidence
        self._device = device or os.getenv("LSTM_DEVICE", "").strip() or None
        self._model = None
        self._meta: Optional[CheckpointMeta] = None
        self._stats: Optional[FeatureStats] = None
        self._lock = threading.Lock()
        self._journal = journal
        self._timeframe_min = int(timeframe_minutes)
        self._log_all_predictions = bool(log_all_predictions)
        self._load()

    def _load(self) -> None:
        p = Path(self.checkpoint_path)
        if not p.exists():
            log.info("lstm_signal_disabled", reason="no_checkpoint",
                     path=str(p))
            return
        try:
            import torch  # lazy
            from ..ml.model import LSTMPriceModel

            dev = self._pick_device(torch)
            self._model, self._meta = load_checkpoint(
                str(p),
                model_factory=LSTMPriceModel.from_meta,
                map_location=str(dev),
            )
            self._model.to(dev)
            self._model.eval()
            self._stats = FeatureStats.from_dict(self._meta.stats)
            self._device = dev
            log.info("lstm_signal_ready", path=str(p),
                     device=str(dev),
                     val_accuracy=round(self._meta.val_accuracy, 4))
        except Exception as e:                         # noqa: BLE001
            log.warning("lstm_signal_init_failed", err=str(e))
            self._model = None

    @staticmethod
    def _pick_device(torch):
        """Device preference: CUDA > MPS (Apple Silicon) > CPU.

        CUDA handles Jetson (aarch64 + NVIDIA). MPS accelerates Apple
        Silicon Macs that don't expose CUDA. CPU is the universal
        fallback — older Macs, Linux x86_64 without a GPU, CI runners.

        Env overrides:
          TRADEBOT_TORCH_DEVICE=cpu   — force CPU (useful for debugging)
          TRADEBOT_TORCH_DEVICE=cuda  — fail loudly if CUDA missing
          TRADEBOT_TORCH_DEVICE=mps   — likewise for MPS
        """
        import os as _os
        forced = _os.getenv("TRADEBOT_TORCH_DEVICE", "").strip().lower()
        if forced in ("cpu", "cuda", "mps"):
            return torch.device(forced)
        if torch.cuda.is_available():
            return torch.device("cuda")
        # MPS = Metal Performance Shaders backend on Apple Silicon. Guarded
        # with getattr because older torch builds don't have the attribute.
        mps_mod = getattr(torch.backends, "mps", None)
        if mps_mod is not None and getattr(mps_mod, "is_available", lambda: False)():
            return torch.device("mps")
        return torch.device("cpu")

    def emit(self, ctx: SignalContext) -> Optional[Signal]:
        if self._model is None or self._meta is None or self._stats is None:
            return None
        need = self._meta.seq_len + 25  # +warm-up for RSI / rolling stats
        if len(ctx.bars) < need:
            return None
        try:
            feats = build_feature_matrix(ctx.bars[-need:])
            # Drop the first 25 warm-up rows → last seq_len rows are the window
            window = feats[-self._meta.seq_len:]
            x = self._stats.transform(window).astype(np.float32)
            import torch   # lazy
            with self._lock:
                with torch.no_grad():
                    xb = torch.from_numpy(x).unsqueeze(0).to(self._device)
                    logits = self._model(xb)
                    probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
            cls = int(np.argmax(probs))
            conf = float(probs[cls])

            # Log every prediction (including neutral and low-conf) to the
            # journal for calibration analysis. Silent on errors.
            if self._journal is not None and self._log_all_predictions:
                try:
                    from ..storage.journal import MLPrediction
                    self._journal.record_ml_prediction(MLPrediction(
                        id=None, ts=ctx.now, symbol=ctx.symbol,
                        model=self._meta.version,
                        pred_class=cls, confidence=conf,
                        p_bearish=float(probs[0]),
                        p_neutral=float(probs[1]),
                        p_bullish=float(probs[2]),
                        horizon_minutes=self._meta.horizon * self._timeframe_min,
                        up_thr=self._meta.up_thr,
                        down_thr=self._meta.down_thr,
                        entry_price=float(ctx.spot) if ctx.spot else None,
                    ))
                except Exception:
                    pass

            if conf < self.min_confidence:
                return None
            if cls == 2:                                 # bullish
                return Signal(source=self.name, symbol=ctx.symbol,
                              side=Side.BUY, option_right=OptionRight.CALL,
                              confidence=conf,
                              rationale=f"lstm bullish p={conf:.2f}",
                              meta={"direction": "bullish",
                                    "entry_tag": "directional_momentum"})
            if cls == 0:                                 # bearish
                return Signal(source=self.name, symbol=ctx.symbol,
                              side=Side.BUY, option_right=OptionRight.PUT,
                              confidence=conf,
                              rationale=f"lstm bearish p={conf:.2f}",
                              meta={"direction": "bearish",
                                    "entry_tag": "directional_momentum"})
            return None   # neutral
        except Exception as e:                           # noqa: BLE001
            log.warning("lstm_emit_error", symbol=ctx.symbol, err=str(e))
            return None
