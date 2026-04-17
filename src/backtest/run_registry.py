"""Backtest run registry — reproducibility by git SHA + config hash.

Every backtest writes a one-line record capturing:
  - git commit SHA (what code was run)
  - SHA256 of the settings.yaml contents (what config was used)
  - seed
  - data source + window
  - final metrics (Sharpe, max DD, total return)
  - timestamp

Use to reproduce any historical result exactly: identify the commit, check
out, reconstruct the config with the same hash, re-run.

Storage: JSONL at `logs/backtest_runs.jsonl`. Append-only.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class RunRecord:
    ts: str
    git_sha: str
    config_sha256: str
    seed: Optional[int]
    data_source: str
    window_days: int
    total_bars: int
    final_equity: float
    total_return_pct: float
    sharpe: float
    max_drawdown_pct: float
    n_trades: int
    notes: str = ""


def git_sha(repo_dir: Optional[str | Path] = None) -> str:
    """Current commit SHA or 'unknown' if not in a repo / git missing."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_dir) if repo_dir else None,
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()[:12]
    except Exception:
        return "unknown"


def config_hash(settings_path: str | Path) -> str:
    try:
        data = Path(settings_path).read_bytes()
        return hashlib.sha256(data).hexdigest()[:16]
    except Exception:
        return "unknown"


def register_run(
    registry_path: str | Path,
    *,
    settings_path: str | Path,
    seed: Optional[int],
    data_source: str,
    window_days: int,
    total_bars: int,
    final_equity: float,
    metrics: Dict[str, Any],
    notes: str = "",
) -> RunRecord:
    """Append a new RunRecord. Returns the record."""
    rec = RunRecord(
        ts=datetime.now(tz=timezone.utc).isoformat(),
        git_sha=git_sha(),
        config_sha256=config_hash(settings_path),
        seed=seed, data_source=data_source,
        window_days=int(window_days), total_bars=int(total_bars),
        final_equity=float(final_equity),
        total_return_pct=float(metrics.get("total_return_pct", 0.0)),
        sharpe=float(metrics.get("sharpe", 0.0)),
        max_drawdown_pct=float(metrics.get("max_drawdown_pct", 0.0)),
        n_trades=int(metrics.get("n_trades", 0)),
        notes=notes,
    )
    p = Path(registry_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as f:
        f.write(json.dumps(asdict(rec)) + "\n")
    return rec


def read_registry(path: str | Path):
    p = Path(path)
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        try:
            d = json.loads(line)
            out.append(RunRecord(**d))
        except Exception:
            continue
    return out
