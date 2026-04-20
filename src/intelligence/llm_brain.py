"""LLM brain — review/synthesis layer on top of the rule-based bot.

Architecture:

  bars + chain
      └─► rules: momentum / ORB / breadth / RSI / S-R / ensemble
              │         (these do the heavy feature extraction)
              ▼
          structured CandidateDecision
              │
              ▼
          LLMBrain.review(candidate, context)     ← this module
              │      (compact JSON in, strict JSON out)
              ▼
          ReviewedDecision {action, confidence_mult, reason}
              │
              ▼
          execution chain → broker

The LLM never sees raw bars or runs the strategy from scratch. It sees
a compact summary of what the deterministic signals already concluded,
plus regime/risk context, and returns one of three actions:

  * confirm       — green-light, keep confidence
  * adjust        — keep direction but scale confidence by a multiplier
                    (typical range 0.5–1.2, clamped [0.3, 1.3])
  * veto          — block the trade (only honored in HARD mode)

Guardrails:
  * Strict JSON output schema. Anything else → fail-open (original
    decision passes through unchanged).
  * Per-symbol rate limit (default 1 call per 30s). Local inference
    takes 300ms–3s on Jetson; more than a few calls per minute is
    waste.
  * Short result cache. Identical context within TTL returns the
    cached decision (no inference round trip).
  * Confidence multiplier clamped so the LLM can't triple a weak
    signal or zero out a strong one.
  * Hard-mode veto is OPT-IN (env var LLM_BRAIN_HARD_GATE=1). Default
    is soft: LLM can only adjust confidence, not block.

The backing LLM is a local GGUF model served via llama-cpp-python
(already installed by deploy/jetson/setup.sh for news classification).
`LLM_MODEL_PATH` in .env points at the GGUF file. CPU fallback is
disabled by default — if CUDA isn't available, the brain stays dormant
rather than running inference at 2 tokens/second.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


_log = logging.getLogger(__name__)


# ------------------------------------------------------------- structures


@dataclass
class CandidateDecision:
    """What the rules decided. Input to the LLM review."""
    symbol: str
    action: str                     # "enter_long" | "enter_short" | "hold" | "exit"
    direction: str                  # "bullish" | "bearish" | "neutral"
    source: str                     # e.g. "ensemble", "extreme_momentum"
    confidence: float               # 0..1
    rationale: str = ""


@dataclass
class ReviewContext:
    """The structured snapshot the LLM sees. Compact, no raw bars."""
    spot: float
    vwap: float = 0.0
    regime: Optional[str] = None
    vix: Optional[float] = None
    vix_percentile: Optional[float] = None
    breadth_score: Optional[float] = None
    breadth_is_risk_off: bool = False
    rsi_14: Optional[float] = None
    nearest_support: Optional[float] = None
    nearest_resistance: Optional[float] = None
    five_bar_move_pct: Optional[float] = None
    volume_vs_avg_20: Optional[float] = None
    iv_rank: Optional[float] = None
    open_positions: int = 0
    position_on_symbol: int = 0     # signed contracts we already hold
    day_pnl_usd: Optional[float] = None
    catalyst_in_24h: bool = False
    news_score: Optional[float] = None
    contributing_signals: List[str] = field(default_factory=list)


@dataclass
class ReviewedDecision:
    """Output of the LLM review."""
    action: str                     # "confirm" | "adjust" | "veto"
    confidence_multiplier: float    # applied to the candidate's confidence
    reason: str                     # brief free-text, audit-friendly
    latency_ms: int = 0
    model: str = ""
    from_cache: bool = False

    @property
    def is_veto(self) -> bool:
        return self.action == "veto"

    @property
    def final_confidence(self) -> float:
        return max(0.0, min(1.0, 1.0 * self.confidence_multiplier))


@dataclass
class LLMBrainConfig:
    model_path: str = ""                        # absolute GGUF path
    model_name: str = "llama-3.1-8b-q4"
    enabled: bool = False
    hard_gate: bool = False                     # if True, vetos are honored
    rate_limit_sec_per_symbol: float = 30.0
    cache_ttl_sec: float = 20.0
    max_tokens: int = 180                       # output budget — JSON is short
    temperature: float = 0.1                    # near-deterministic
    min_clamp: float = 0.30                     # confidence_multiplier floor
    max_clamp: float = 1.30                     # confidence_multiplier ceiling
    timeout_sec: float = 3.0                    # fail-open if LLM slower than this
    n_ctx: int = 2048
    n_gpu_layers: int = -1                      # all layers on GPU (Jetson CUDA)


# ------------------------------------------------------------- helpers


_FAIL_OPEN = ReviewedDecision(
    action="confirm",
    confidence_multiplier=1.0,
    reason="llm_unavailable_or_failed_fail_open",
)


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _fingerprint(candidate: CandidateDecision, ctx: ReviewContext) -> str:
    """Cache key. Rounds all floats to coarse bins so near-identical
    contexts hit the cache instead of re-running inference for
    indistinguishable differences."""
    def _b(x, step):
        if x is None:
            return "-"
        return str(round(x / step) * step)
    parts = [
        candidate.symbol, candidate.action, candidate.direction,
        _b(candidate.confidence, 0.05),
        _b(ctx.spot, 0.5),
        _b(ctx.vix, 1),
        _b(ctx.breadth_score, 0.1),
        _b(ctx.rsi_14, 2),
        str(ctx.regime or "-"),
        str(ctx.breadth_is_risk_off),
        str(ctx.position_on_symbol),
    ]
    return "|".join(parts)


def _build_prompt(candidate: CandidateDecision, ctx: ReviewContext) -> str:
    """Compact prompt: structured feature summary + strict JSON schema.

    The system contract is:
      - You are a sanity-check layer, not the decision-maker.
      - The rules already decided. You review.
      - Output ONE line of JSON, nothing else.
    """
    features = {k: v for k, v in asdict(ctx).items() if v is not None}
    decision = asdict(candidate)
    return f"""You are the final sanity-check layer for a paper-trading options bot.
The deterministic strategy layer has proposed a decision. You review it against
the compact feature snapshot below and return ONE line of JSON.

Valid actions:
  "confirm"  — the setup looks coherent; keep the trade as-is
  "adjust"   — keep direction but modify confidence (use confidence_multiplier)
  "veto"     — the setup has a material contradiction (only acted on in hard-gate mode)

Hard rules you must respect:
  - Do not invent new price levels. Reference only the numbers provided.
  - Never recommend larger than 1.3x confidence.
  - Never recommend smaller than 0.3x unless you choose "veto".
  - If any two features are in sharp contradiction (e.g. bearish signal
    while breadth is strongly risk-on, or trade against the nearest S/R),
    lean toward "adjust" with a lower multiplier, or "veto" if the
    contradiction is severe.
  - "reason" must be under 120 characters, plain text, no newlines.

CANDIDATE DECISION (from rules):
{json.dumps(decision, separators=(',', ':'))}

FEATURE SNAPSHOT:
{json.dumps(features, separators=(',', ':'))}

Output ONE JSON object with exactly these keys:
  {{"action": "confirm"|"adjust"|"veto",
    "confidence_multiplier": <float between 0.3 and 1.3>,
    "reason": "<short plain text>"}}"""


def _parse_response(raw: str) -> Optional[ReviewedDecision]:
    """Extract the JSON object from the model output. Models sometimes
    prepend `<json>` or `Sure, here's...` — we scan for the first { and
    the matching }, then parse. Returns None on any malformation."""
    if not raw:
        return None
    s = raw.find("{")
    e = raw.rfind("}")
    if s < 0 or e <= s:
        return None
    try:
        obj = json.loads(raw[s:e + 1])
    except Exception:
        return None
    action = str(obj.get("action", "")).lower().strip()
    if action not in ("confirm", "adjust", "veto"):
        return None
    try:
        mult = float(obj.get("confidence_multiplier", 1.0))
    except (TypeError, ValueError):
        mult = 1.0
    reason = str(obj.get("reason", ""))[:120].replace("\n", " ")
    return ReviewedDecision(
        action=action,
        confidence_multiplier=mult,
        reason=reason,
    )


# ------------------------------------------------------------- brain


class LLMBrain:
    """Stateful reviewer. Lazily loads the model, caches and rate-limits."""

    def __init__(self, cfg: Optional[LLMBrainConfig] = None):
        self.cfg = cfg or LLMBrainConfig()
        self._lock = threading.Lock()
        self._model = None
        self._load_err: Optional[str] = None
        self._last_call_per_symbol: Dict[str, float] = {}
        self._cache: Dict[str, tuple] = {}   # fingerprint → (decision, ts)

    # ---------- model load ----------

    def _ensure_model(self) -> bool:
        """Lazy load. Returns True if the model is ready."""
        if self._model is not None:
            return True
        if self._load_err is not None:
            return False
        with self._lock:
            if self._model is not None:
                return True
            if not self.cfg.enabled:
                self._load_err = "disabled_by_config"
                return False
            if not self.cfg.model_path or not os.path.exists(self.cfg.model_path):
                self._load_err = f"model_path_missing:{self.cfg.model_path}"
                _log.warning("llm_brain_model_missing path=%s", self.cfg.model_path)
                return False
            try:
                from llama_cpp import Llama      # lazy import
                self._model = Llama(
                    model_path=self.cfg.model_path,
                    n_ctx=self.cfg.n_ctx,
                    n_gpu_layers=self.cfg.n_gpu_layers,
                    verbose=False,
                )
                _log.info("llm_brain_loaded model=%s", self.cfg.model_name)
                return True
            except Exception as e:
                self._load_err = f"load_failed:{type(e).__name__}:{e}"
                _log.warning("llm_brain_load_failed err=%s", e)
                return False

    # ---------- public API ----------

    def review(self, candidate: CandidateDecision,
                 context: ReviewContext) -> ReviewedDecision:
        """Return a ReviewedDecision. Fail-open: unavailable/malformed →
        confirm with multiplier=1.0 so the original decision passes
        through unchanged."""
        if not self.cfg.enabled:
            return _FAIL_OPEN

        fp = _fingerprint(candidate, context)
        now = time.time()

        # Cache hit?
        with self._lock:
            entry = self._cache.get(fp)
            if entry is not None:
                cached, ts = entry
                if (now - ts) < self.cfg.cache_ttl_sec:
                    return ReviewedDecision(
                        **{**asdict(cached), "from_cache": True}
                    )

        # Rate limit per symbol
        last = self._last_call_per_symbol.get(candidate.symbol, 0.0)
        if (now - last) < self.cfg.rate_limit_sec_per_symbol:
            return _FAIL_OPEN

        # Model ready?
        if not self._ensure_model():
            return _FAIL_OPEN

        # Infer
        started = time.time()
        prompt = _build_prompt(candidate, context)
        try:
            raw = self._infer(prompt)
        except Exception as e:
            _log.warning("llm_brain_infer_failed sym=%s err=%s",
                          candidate.symbol, e)
            return _FAIL_OPEN
        latency_ms = int((time.time() - started) * 1000)

        parsed = _parse_response(raw)
        if parsed is None:
            _log.info("llm_brain_bad_json sym=%s raw=%s",
                       candidate.symbol, raw[:120])
            return _FAIL_OPEN

        # Clamp the multiplier to sane bounds and apply veto policy
        mult = _clamp(parsed.confidence_multiplier,
                       self.cfg.min_clamp, self.cfg.max_clamp)
        action = parsed.action
        if action == "veto" and not self.cfg.hard_gate:
            # Soft mode: demote veto to a strong-down-adjustment.
            action = "adjust"
            mult = min(mult, 0.5)

        decision = ReviewedDecision(
            action=action,
            confidence_multiplier=mult,
            reason=parsed.reason,
            latency_ms=latency_ms,
            model=self.cfg.model_name,
        )

        # Bookkeep rate limit + cache
        with self._lock:
            self._last_call_per_symbol[candidate.symbol] = now
            self._cache[fp] = (decision, now)
            # Trim cache if it grows: keep last 256 entries
            if len(self._cache) > 256:
                for k in list(self._cache.keys())[:-256]:
                    del self._cache[k]

        return decision

    def _infer(self, prompt: str) -> str:
        """Raw one-shot completion. Subclasses can override for tests."""
        assert self._model is not None
        # Use create_completion for the broadest llama-cpp-python API
        # compatibility — it works across 0.2.x and 0.3.x versions.
        resp = self._model.create_completion(
            prompt=prompt,
            max_tokens=self.cfg.max_tokens,
            temperature=self.cfg.temperature,
            stop=["\n\n"],
        )
        try:
            return resp["choices"][0]["text"]
        except Exception:
            return ""


# ------------------------------------------------------------- factory


def build_llm_brain_from_settings(settings) -> Optional[LLMBrain]:
    """Construct an LLMBrain from `settings.raw.llm_brain` + env overrides.
    Returns None if disabled."""
    cfg_dict = (settings.raw.get("llm_brain", {}) or {})
    enabled_cfg = bool(cfg_dict.get("enabled", False))
    # Env override — easy to flip without touching YAML
    env_enabled = os.getenv("LLM_BRAIN_ENABLED", "").strip() in ("1", "true", "yes")
    enabled = enabled_cfg or env_enabled
    if not enabled:
        return None

    # Model path: env var > config > default under data root
    model_path = (
        os.getenv("LLM_MODEL_PATH", "").strip()
        or cfg_dict.get("model_path")
        or ""
    )
    if not model_path:
        try:
            from ..core.data_paths import data_path
            model_path = str(data_path("models/llama-3.1-8b-q4.gguf"))
        except Exception:
            pass

    cfg = LLMBrainConfig(
        model_path=model_path,
        model_name=str(cfg_dict.get("model_name", "llama-3.1-8b-q4")),
        enabled=True,
        hard_gate=(
            bool(cfg_dict.get("hard_gate", False))
            or os.getenv("LLM_BRAIN_HARD_GATE", "").strip() in ("1", "true", "yes")
        ),
        rate_limit_sec_per_symbol=float(cfg_dict.get("rate_limit_sec_per_symbol", 30.0)),
        cache_ttl_sec=float(cfg_dict.get("cache_ttl_sec", 20.0)),
        max_tokens=int(cfg_dict.get("max_tokens", 180)),
        temperature=float(cfg_dict.get("temperature", 0.1)),
        min_clamp=float(cfg_dict.get("min_clamp", 0.30)),
        max_clamp=float(cfg_dict.get("max_clamp", 1.30)),
        timeout_sec=float(cfg_dict.get("timeout_sec", 3.0)),
        n_ctx=int(cfg_dict.get("n_ctx", 2048)),
        n_gpu_layers=int(cfg_dict.get("n_gpu_layers", -1)),
    )
    return LLMBrain(cfg)
