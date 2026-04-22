"""LLMOriginationSignal — reads the LLM autotrade queue and emits
Signals so the ensemble + filter chain + order path treat LLM ideas
the same as rules-engine signals.

Async by design: the research agent runs in a separate process and
writes to a file. This signal's emit() does a cheap mtime-check file
read (~1-2 ms). No LLM calls on the hot path."""
from __future__ import annotations

import logging
import os
from typing import Optional

from ..core.types import Signal, Side, OptionRight
from .base import SignalSource, SignalContext
from ..intelligence.llm_autotrade_queue import LLMAutotradeQueue, QueuedIdea


_log = logging.getLogger(__name__)


# Map confidence string → numeric confidence. LLM ideas with "low"
# confidence are already filtered out at the queue layer; this is
# just the score the ensemble sees.
_CONF_MAP = {"high": 0.90, "medium": 0.72}


class LLMOriginationSignal(SignalSource):
    """Emits a Signal when the research agent has dropped an idea for
    this symbol in the queue. Never emits when:
      - LLM_AUTOTRADE env not set to 1
      - Kill switch file exists
      - Daily cap reached
      - No fresh idea for this symbol in the queue
    """
    name = "llm_origination"

    def __init__(self,
                 *, max_age_min: int = 30,
                 max_trades_per_day: int = 3,
                 min_confidence: str = "medium"):
        allowed = {"high"}
        if min_confidence == "medium":
            allowed.add("medium")
        elif min_confidence == "low":
            allowed.update(["medium", "low"])
        self._queue = LLMAutotradeQueue(
            max_age_min=max_age_min,
            allowed_confidences=allowed,
            max_trades_per_day=max_trades_per_day,
        )

    def emit(self, ctx: SignalContext) -> Optional[Signal]:
        # Runtime env gate. Checked every tick so the operator can set
        # LLM_AUTOTRADE=1 + restart and this signal lights up without
        # a code change.
        if os.getenv("LLM_AUTOTRADE", "").strip() not in ("1", "true", "yes"):
            return None

        idea = self._queue.next_idea_for(ctx.symbol)
        if idea is None:
            return None

        right = OptionRight.CALL if idea.direction == "call" else OptionRight.PUT
        confidence = _CONF_MAP.get(idea.confidence, 0.60)

        _log.info(
            "llm_origination_emit symbol=%s dir=%s strike=%s expiry=%s "
            "conf=%s id=%s",
            idea.symbol, idea.direction, idea.strike, idea.expiry,
            idea.confidence, idea.id,
        )

        # Pass strike + expiry + risk management hints to the contract
        # picker via meta so it uses the LLM's specific strike instead
        # of defaulting to ATM.
        meta = {
            "direction": "bullish" if right == OptionRight.CALL else "bearish",
            "source": "llm_origination",
            "idea_id": idea.id,
            "proposed_strike": idea.strike,
            "proposed_expiry": idea.expiry,
            "proposed_entry": idea.entry,
            "proposed_pt": idea.profit_target,
            "proposed_sl": idea.stop_loss,
            "llm_confidence": idea.confidence,
            "time_horizon": idea.time_horizon,
            "llm_rationale": (idea.rationale or "")[:200],
        }
        return Signal(
            source=self.name,
            symbol=ctx.symbol,
            side=Side.BUY,
            option_right=right,
            confidence=confidence,
            rationale=f"llm_origination[{idea.confidence}]: {idea.rationale[:120]}",
            meta=meta,
        )

    def peek_status(self) -> dict:
        """For the !llm-autotrade Discord status command."""
        return self._queue.peek_state()

    def set_killed(self, killed: bool) -> None:
        """For !llm-autotrade on/off."""
        self._queue.set_killed(killed)
