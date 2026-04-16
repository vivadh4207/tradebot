"""MI + Edge Score composite.

Combines Market Intelligence flags (HIGH_VIX_CAUTION, BREADTH_DIVERGENCE,
AGAINST_GAMMA_REGIME, RSI_OVERBOUGHT, BELOW_VWAP_FOR_LONG,
STRUCTURE_MISALIGN, etc.) into a single integer score. Matches your spec:
blocks trade only if combined score < -5.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class MIFlag:
    name: str
    weight: int                 # negative = adverse
    rationale: str = ""


@dataclass
class MIEdgeResult:
    score: int
    flags: List[MIFlag] = field(default_factory=list)


class MIEdgeScorer:
    BLOCK_THRESHOLD = -5

    def score(self,
              vix: float,
              breadth_divergence: float,         # -1..+1
              against_gamma_regime: bool,
              rsi: float,
              spot_vs_vwap: float,               # spot/vwap - 1
              structure_align: bool,
              direction: str,                    # 'bullish' | 'bearish' | 'premium_harvest'
              ) -> MIEdgeResult:
        flags: List[MIFlag] = []
        score = 0

        if vix > 30:
            flags.append(MIFlag("HIGH_VIX_CAUTION", -2, f"vix={vix:.1f}"))
            score -= 2
        if breadth_divergence < -0.3:
            flags.append(MIFlag("BREADTH_DIVERGENCE", -2, f"ad={breadth_divergence:.2f}"))
            score -= 2
        if against_gamma_regime:
            flags.append(MIFlag("AGAINST_GAMMA_REGIME", -2, "dealer flow compressive"))
            score -= 2
        if rsi > 75:
            flags.append(MIFlag("RSI_OVERBOUGHT", -1, f"rsi={rsi:.1f}"))
            if direction == "bullish":
                score -= 1
        if rsi < 25:
            flags.append(MIFlag("RSI_OVERSOLD", -1, f"rsi={rsi:.1f}"))
            if direction == "bearish":
                score -= 1
        if spot_vs_vwap < -0.002 and direction == "bullish":
            flags.append(MIFlag("BELOW_VWAP_FOR_LONG", -1, f"diff={spot_vs_vwap:.4f}"))
            score -= 1
        if spot_vs_vwap > 0.002 and direction == "bearish":
            flags.append(MIFlag("ABOVE_VWAP_FOR_SHORT", -1, f"diff={spot_vs_vwap:.4f}"))
            score -= 1
        if not structure_align:
            flags.append(MIFlag("STRUCTURE_MISALIGN", -1, "price vs HTF alignment failed"))
            score -= 1

        # positive conditions add edge
        if vix < 20 and direction in {"bullish", "premium_harvest"}:
            flags.append(MIFlag("LOW_VIX_OK", +1))
            score += 1
        if breadth_divergence > 0.3:
            flags.append(MIFlag("BROAD_BREADTH_CONFIRM", +1))
            score += 1

        return MIEdgeResult(score=score, flags=flags)

    @classmethod
    def is_blocked(cls, result: MIEdgeResult) -> bool:
        return result.score < cls.BLOCK_THRESHOLD
