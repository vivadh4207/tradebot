"""Offline strategy auditor — uses a larger (70B) local LLM to review
the WHOLE bot setup for flaws, misconfigurations, and regime
mismatches. Not in the hot path.

Contrast with `llm_brain.py` (8B, reviews each entry in real time).
The auditor is slow — 70B on a Jetson runs at ~3-6 tok/sec. Designed
to run once nightly (cron) and on demand (dashboard button), never
per-tick.

What it sees:

  - Current settings.yaml (parameters, enabled strategies, filters)
  - Recent trade journal summary (last N days: N trades, win rate, EV)
  - Recent walk-forward verdict
  - Current market snapshot (VIX, breadth, recent regime distribution)
  - The list of configured strategies

What it returns (strict JSON):

  {"overall_health": 0-100,
   "summary": "one-line high-level read",
   "issues": [{"severity": "low|medium|high",
               "area": "sizing|strategy|regime|filter|...",
               "detail": "...",
               "fix": "..."}],
   "strengths": ["..."]}

Results are appended to logs/strategy_audit.jsonl. The dashboard
surfaces the latest entry plus a history panel.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


_log = logging.getLogger(__name__)


# --------------------------------------------------------------- dataclasses


@dataclass
class AuditIssue:
    severity: str                    # "low" | "medium" | "high"
    area: str                         # "sizing" | "strategy" | "regime" | ...
    detail: str
    fix: str = ""


@dataclass
class AuditReport:
    ts: str
    overall_health: int               # 0-100
    summary: str
    issues: List[AuditIssue] = field(default_factory=list)
    strengths: List[str] = field(default_factory=list)
    model: str = ""
    latency_sec: float = 0.0
    input_snapshot_digest: str = ""   # short hash so repeat-audits are visible

    def to_jsonl(self) -> str:
        """Serialize one-line. JSONL means append-friendly + tailable."""
        return json.dumps({
            "ts": self.ts,
            "overall_health": self.overall_health,
            "summary": self.summary,
            "issues": [asdict(i) for i in self.issues],
            "strengths": self.strengths,
            "model": self.model,
            "latency_sec": round(self.latency_sec, 2),
            "snapshot": self.input_snapshot_digest,
        }, separators=(",", ":"))


@dataclass
class StrategyAuditorConfig:
    # backend: "llama_cpp" (GGUF load) or "ollama" (HTTP to localhost:11434)
    backend: str = "llama_cpp"
    model_path: str = ""                           # llama_cpp: GGUF path
    model_name: str = "llama-3.1-70b-q4"            # ollama: tag like "llama3.1:70b"
    n_ctx: int = 4096                              # 70B needs more room
    n_gpu_layers: int = -1
    max_tokens: int = 900                          # long structured output
    temperature: float = 0.15
    log_path: str = "logs/strategy_audit.jsonl"
    trade_lookback_days: int = 14


# --------------------------------------------------------------- helpers


def _build_snapshot(settings, journal=None, extra_context: Optional[Dict] = None,
                     cfg: Optional[StrategyAuditorConfig] = None,
                     political_news=None) -> Dict[str, Any]:
    """Compact structured snapshot for the auditor. NO raw bars, NO
    per-trade rows — just aggregates. Keeps the LLM's context usage low
    and the signal-to-noise high.

    Args:
      political_news: optional PoliticalNewsProvider whose
        snapshot_for_auditor() output is included under "political_news".
    """
    cfg = cfg or StrategyAuditorConfig()
    snap: Dict[str, Any] = {}

    # Config subset — just the knobs that materially shape strategy
    raw = settings.raw if hasattr(settings, "raw") else {}
    for key in ("strategy_mode", "universe", "vix", "iv_rank",
                 "signal", "execution", "sizing", "exits",
                 "ensemble", "credit_spreads", "putcall_oi",
                 "breadth", "signals", "wheel"):
        if key in raw:
            snap[f"cfg.{key}"] = raw[key]

    # Trade history aggregate
    if journal is not None:
        try:
            since = datetime.now(tz=timezone.utc) - timedelta(
                days=cfg.trade_lookback_days,
            )
            trades = journal.closed_trades(since=since)
            wins = [t for t in trades if (t.pnl or 0) > 0]
            losses = [t for t in trades if (t.pnl or 0) < 0]
            n = len(trades)
            snap["trade_history"] = {
                "lookback_days": cfg.trade_lookback_days,
                "n_trades": n,
                "n_wins": len(wins),
                "n_losses": len(losses),
                "win_rate": round(len(wins) / max(1, n), 3),
                "total_pnl": round(sum(t.pnl or 0 for t in trades), 2),
                "avg_win_pct":  round(sum(t.pnl_pct or 0 for t in wins) / max(1, len(wins)), 4),
                "avg_loss_pct": round(sum(abs(t.pnl_pct or 0) for t in losses) / max(1, len(losses)), 4),
                "unique_symbols": sorted({t.symbol for t in trades}),
            }
        except Exception as e:
            snap["trade_history_error"] = str(e)

    if extra_context:
        snap["current_market"] = extra_context

    # Political news — structured headlines the 70B sees alongside the
    # config/journal. Lets the auditor flag issues like "FOMC in 2 days
    # but credit_spreads.enabled=true" or "tariff announcement —
    # universe concentrated in SPY/QQQ is extra vulnerable".
    if political_news is not None:
        try:
            snap["political_news"] = political_news.snapshot_for_auditor()
        except Exception as e:
            snap["political_news_error"] = str(e)

    return snap


def _build_prompt(snapshot: Dict[str, Any]) -> str:
    """System + user prompt. Strict JSON output."""
    return f"""You are an experienced options-trading quant reviewing the
configuration of a paper-trading bot. Identify flaws, misconfigurations,
regime mismatches, and opportunities. Return STRICT JSON — no preamble,
no prose outside the JSON object.

SNAPSHOT:
{json.dumps(snapshot, indent=2, default=str)}

Output ONE JSON object with exactly these keys:
  {{"overall_health": <int 0-100>,
    "summary": "<one sentence high-level read>",
    "issues": [
      {{"severity": "low|medium|high",
        "area": "sizing|strategy|regime|filter|universe|exit|signal|data",
        "detail": "<1-2 sentences, plain text>",
        "fix": "<concrete actionable change>"}}
    ],
    "strengths": ["<short bullets, each ≤ 80 chars>"]}}

Scoring guidance:
  - 90-100: cohesive setup, no material issues
  - 70-89:  healthy, minor polish items
  - 50-69:  working but has one or more medium issues
  - 30-49:  serious structural problem (strategy / regime / sizing)
  - 0-29:   will likely lose money as configured

Be specific. Refuse vague feedback. Every "issue.detail" must reference
specific numbers from the snapshot."""


def _parse(raw: str) -> Optional[Dict[str, Any]]:
    """Extract the first balanced JSON object from the model output."""
    if not raw:
        return None
    s = raw.find("{")
    e = raw.rfind("}")
    if s < 0 or e <= s:
        return None
    try:
        return json.loads(raw[s:e + 1])
    except Exception:
        return None


def _digest(obj: Dict[str, Any]) -> str:
    """Short digest of the audited snapshot — so the operator can see
    at a glance whether a new audit was of the same setup or not."""
    import hashlib
    s = json.dumps(obj, sort_keys=True, default=str)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]


# --------------------------------------------------------------- auditor


class StrategyAuditor:
    """Offline auditor. Lazy-loads the 70B model; does NOT keep it in
    memory between audits (frees the GPU for the LIVE review brain)."""

    def __init__(self, cfg: Optional[StrategyAuditorConfig] = None):
        self.cfg = cfg or StrategyAuditorConfig()

    def audit(self, settings, journal=None,
               extra_context: Optional[Dict] = None,
               political_news=None) -> Optional[AuditReport]:
        """Run one audit. Returns None if the model can't be loaded
        (e.g. GGUF file not on disk) or if output is unparseable.

        Args:
          political_news: optional PoliticalNewsProvider whose recent
            headlines are embedded in the prompt.
        """
        snapshot = _build_snapshot(settings, journal=journal,
                                     extra_context=extra_context,
                                     cfg=self.cfg,
                                     political_news=political_news)

        import time as _time
        started = _time.time()
        prompt = _build_prompt(snapshot)

        # Dispatch on backend: ollama (HTTP, preferred on Jetson) or
        # llama_cpp (direct GGUF load).
        if self.cfg.backend == "ollama":
            from .ollama_client import build_ollama_client
            client = build_ollama_client()
            if not client.ping():
                _log.warning("strategy_auditor_ollama_unreachable base_url=%s",
                              client.cfg.base_url)
                return None
            try:
                raw = client.generate(
                    model=self.cfg.model_name,
                    prompt=prompt,
                    temperature=self.cfg.temperature,
                    max_tokens=self.cfg.max_tokens,
                    num_ctx=self.cfg.n_ctx,
                    stop=["\n\n\n"],
                )
            except Exception as e:
                _log.warning("strategy_auditor_ollama_infer_failed err=%s", e)
                return None
        else:
            # llama_cpp path — loads GGUF from model_path
            if not self.cfg.model_path or not os.path.exists(self.cfg.model_path):
                _log.warning("strategy_auditor_model_missing path=%s",
                              self.cfg.model_path)
                return None
            try:
                from llama_cpp import Llama
                llm = Llama(
                    model_path=self.cfg.model_path,
                    n_ctx=self.cfg.n_ctx,
                    n_gpu_layers=self.cfg.n_gpu_layers,
                    verbose=False,
                )
            except Exception as e:
                _log.warning("strategy_auditor_load_failed err=%s", e)
                return None
            try:
                resp = llm.create_completion(
                    prompt=prompt,
                    max_tokens=self.cfg.max_tokens,
                    temperature=self.cfg.temperature,
                    stop=["\n\n\n"],
                )
                raw = resp["choices"][0]["text"]
            except Exception as e:
                _log.warning("strategy_auditor_infer_failed err=%s", e)
                return None
            finally:
                # Free the GPU for the live review brain.
                try:
                    del llm
                except Exception:
                    pass

        parsed = _parse(raw)
        if parsed is None:
            _log.info("strategy_auditor_bad_json raw=%s", raw[:200])
            return None

        issues = []
        for i in parsed.get("issues", []):
            try:
                issues.append(AuditIssue(
                    severity=str(i.get("severity", "medium")).lower(),
                    area=str(i.get("area", "unknown")),
                    detail=str(i.get("detail", ""))[:500],
                    fix=str(i.get("fix", ""))[:500],
                ))
            except Exception:
                continue

        try:
            health = int(parsed.get("overall_health", 50))
        except (TypeError, ValueError):
            health = 50
        health = max(0, min(100, health))

        return AuditReport(
            ts=datetime.now(tz=timezone.utc).isoformat(),
            overall_health=health,
            summary=str(parsed.get("summary", ""))[:300],
            issues=issues,
            strengths=[str(s)[:120] for s in parsed.get("strengths", [])][:8],
            model=self.cfg.model_name,
            latency_sec=_time.time() - started,
            input_snapshot_digest=_digest(snapshot),
        )

    def append(self, report: AuditReport, root: Optional[Path] = None) -> Path:
        """Append a report to the audit log. Creates parent dir if missing."""
        from ..core.data_paths import data_path
        log_path = data_path(self.cfg.log_path) if root is None else (root / self.cfg.log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(report.to_jsonl() + "\n")
        return log_path


def read_recent_audits(limit: int = 20) -> List[Dict[str, Any]]:
    """Dashboard helper — return the latest N audits, newest first.
    Returns [] if the log file is missing."""
    from ..core.data_paths import data_path
    cfg = StrategyAuditorConfig()
    path = data_path(cfg.log_path)
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return []
    out = []
    for ln in reversed(lines):
        if not ln.strip():
            continue
        try:
            out.append(json.loads(ln))
        except Exception:
            continue
        if len(out) >= limit:
            break
    return out


def build_auditor_from_settings(settings) -> Optional[StrategyAuditor]:
    """Factory. Returns None when no 70B model path is configured."""
    cfg_d = (settings.raw.get("strategy_auditor", {}) or {})
    model_path = (os.getenv("LLM_AUDITOR_MODEL_PATH", "").strip()
                   or cfg_d.get("model_path")
                   or "")
    if not model_path:
        try:
            from ..core.data_paths import data_path
            model_path = str(data_path("models/llama-3.1-70b-q4.gguf"))
        except Exception:
            pass
    # Shared backend flag — same LLM_BACKEND env the brain reads.
    backend = (os.getenv("LLM_BACKEND", "").strip().lower()
                or str(cfg_d.get("backend", "llama_cpp")).lower())
    if backend not in ("ollama", "llama_cpp"):
        backend = "llama_cpp"

    # Model name — ollama tag (e.g. "llama3.1:70b") or informational name.
    model_name = (os.getenv("LLM_AUDITOR_MODEL", "").strip()
                  or str(cfg_d.get("model_name", "llama-3.1-70b-q4")))

    cfg = StrategyAuditorConfig(
        backend=backend,
        model_path=model_path,
        model_name=model_name,
        n_ctx=int(cfg_d.get("n_ctx", 4096)),
        n_gpu_layers=int(cfg_d.get("n_gpu_layers", -1)),
        max_tokens=int(cfg_d.get("max_tokens", 900)),
        temperature=float(cfg_d.get("temperature", 0.15)),
        log_path=str(cfg_d.get("log_path", "logs/strategy_audit.jsonl")),
        trade_lookback_days=int(cfg_d.get("trade_lookback_days", 14)),
    )
    return StrategyAuditor(cfg)
