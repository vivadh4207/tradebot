"""Claude-AI powered signal: analyzes candles + options flow + intel, returns
a structured bullish/bearish/flat call with confidence.

Calls the Anthropic API via the python `anthropic` package IF installed and
`ANTHROPIC_API_KEY` is set. Otherwise returns None (silent no-op).

Minimum confidence to emit: 0.65 (configurable).
"""
from __future__ import annotations

import json
import os
from typing import Optional, List

from ..core.types import Signal, Side, OptionRight, Bar
from .base import SignalSource, SignalContext


class ClaudeAISignal(SignalSource):
    name = "claude_ai"

    def __init__(self, min_confidence: float = 0.65,
                 model: str = "claude-sonnet-4-6"):
        self.min_conf = min_confidence
        self.model = model
        self._client = None
        try:
            import anthropic
            key = os.getenv("ANTHROPIC_API_KEY")
            if key:
                self._client = anthropic.Anthropic(api_key=key)
        except Exception:
            self._client = None

    def emit(self, ctx: SignalContext) -> Optional[Signal]:
        if self._client is None or len(ctx.bars) < 20:
            return None
        prompt = self._build_prompt(ctx)
        try:
            msg = self._client.messages.create(
                model=self.model,
                max_tokens=400,
                system=(
                    "You are an options trading analyst. Respond ONLY as JSON with keys: "
                    "decision ('bullish'|'bearish'|'flat'), confidence (0..1), rationale. "
                    "Never promise guaranteed profit. Be concise."
                ),
                messages=[{"role": "user", "content": prompt}],
            )
            text = msg.content[0].text if msg.content else "{}"
            data = json.loads(text.strip().split("```")[0].strip().strip("```json").strip())
        except Exception:
            return None
        decision = data.get("decision", "flat")
        conf = float(data.get("confidence", 0.0))
        if conf < self.min_conf or decision == "flat":
            return None
        right = OptionRight.CALL if decision == "bullish" else OptionRight.PUT
        return Signal(source=self.name, symbol=ctx.symbol,
                      side=Side.BUY, option_right=right,
                      confidence=conf,
                      rationale=str(data.get("rationale", ""))[:200],
                      meta={"direction": decision})

    def _build_prompt(self, ctx: SignalContext) -> str:
        tail: List[Bar] = ctx.bars[-20:]
        closes = [round(b.close, 2) for b in tail]
        vols = [int(b.volume) for b in tail]
        return (
            f"Symbol: {ctx.symbol}\n"
            f"Spot: {ctx.spot}\nVWAP: {ctx.vwap}\n"
            f"Last 20 closes: {closes}\n"
            f"Last 20 volumes: {vols}\n"
            f"ATM IV 30d: {ctx.atm_iv_30d}, RV 20d: {ctx.rv_20d}\n"
            f"IV 30d: {ctx.iv_30d}, IV 90d: {ctx.iv_90d}\n"
            "Call it."
        )
