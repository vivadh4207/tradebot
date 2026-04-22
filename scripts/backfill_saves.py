"""Backfill the saves tracker from log tail + SQLite journal.

Run ONCE after deploying saves_tracker.py to populate exit_saves.jsonl
with historical defensive exits that fired before record_exit() was
wired in. After this runs, !saves will show real data.

Parses logs/tradebot.out for `fast_exit` events with defensive reasons,
cross-references the fills in the journal, writes matching save records.

Usage:
  python scripts/backfill_saves.py
  python scripts/backfill_saves.py --since-hours 72
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_TS_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?)"
)
# Parse key=value pairs independent of order. structlog emits them
# alphabetically (pnl_pct first then price then reason...) so a single
# positional regex brittle. Use per-field pattern instead.
def _parse_kv(line: str) -> dict:
    out = {}
    # symbol=WORD  price=NUM  pnl_pct=NUM  realized_usd=NUM
    for m in re.finditer(r"(\w+)=(?:'([^']*)'|\"([^\"]*)\"|(\S+))", line):
        key = m.group(1)
        val = m.group(2) if m.group(2) is not None else (
            m.group(3) if m.group(3) is not None else m.group(4)
        )
        out[key] = val
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since-hours", type=int, default=48)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    from src.intelligence.saves_tracker import (
        _is_defensive, _store_path,
    )
    import json

    log_path = ROOT / "logs" / "tradebot.out"
    if not log_path.exists():
        print(f"[!] log not found: {log_path}")
        return 2

    # Read last ~10MB to cover ~48h of session logs
    with log_path.open("rb") as f:
        import os as _os
        sz = _os.fstat(f.fileno()).st_size
        f.seek(max(0, sz - 10_000_000))
        if sz > 10_000_000:
            f.readline()
        text = f.read().decode("utf-8", errors="replace")
    text = _ANSI_RE.sub("", text)

    cutoff = datetime.now(tz=timezone.utc) - timedelta(hours=args.since_hours)
    found = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if "fast_exit" not in line:
            continue
        ts_m = _TS_RE.match(line)
        if not ts_m:
            continue
        try:
            ts_str = ts_m.group("ts").rstrip("Z")
            ts = datetime.fromisoformat(ts_str).replace(tzinfo=timezone.utc)
        except Exception:
            continue
        if ts < cutoff:
            continue
        kv = _parse_kv(line)
        reason = (kv.get("reason") or "").strip("'\"")
        symbol = kv.get("symbol")
        if not reason or not symbol:
            continue
        if not _is_defensive(reason):
            continue
        try:
            price = float(kv.get("price", 0))
            pnl_pct = float(kv.get("pnl_pct", 0)) / 100.0     # pct -> frac
        except Exception:
            continue

        # Infer qty + dte where possible — use journal if available
        # else fall back to conservative defaults.
        qty = 1
        dte = 7
        underlying = symbol
        try:
            from src.storage.journal import SqliteJournal
            from src.core.data_paths import data_path
            jpath = data_path("logs/tradebot.sqlite")
            j = SqliteJournal(str(jpath))
            matches = [
                t for t in j.closed_trades(since=cutoff)
                if t.symbol == symbol and t.closed_at
                   and abs((t.closed_at - ts).total_seconds()) < 60
            ]
            if matches:
                t = matches[0]
                qty = abs(int(t.qty))
                if t.entry_tag:
                    dm = re.search(r"\|dte=(\d+)", t.entry_tag)
                    if dm:
                        dte = int(dm.group(1))
                    # Just the ticker — the tag has sym=QQQ|right=...
                    um = re.search(r"\|sym=([A-Z]{1,6})\b", t.entry_tag)
                    if um:
                        underlying = um.group(1)
        except Exception:
            pass
        # Extract ticker from OCC symbol (e.g. QQQ260506C00655000 -> QQQ)
        if underlying == symbol:
            occ_m = re.match(r"^([A-Z]{1,6})\d{6}[CP]\d{8}$", symbol)
            if occ_m:
                underlying = occ_m.group(1)

        found.append({
            "ts": ts.timestamp(),
            "symbol": symbol,
            "underlying": underlying,
            "exit_reason": reason,
            "exit_price": price,
            "qty": qty,
            "peak_pnl_pct": max(pnl_pct, 0.01),   # unknown, conservative
            "exit_pnl_pct": pnl_pct,
            "dte": dte,
            "recheck_30m_price": None,
            "recheck_2h_price": None,
            "saved_usd_30m": None,
            "saved_usd_2h": None,
        })

    if not found:
        print("[backfill] no defensive exits in window")
        return 0

    path = _store_path()
    if args.dry_run:
        print(f"[backfill] would write {len(found)} records to {path}:")
        for r in found:
            dt = datetime.fromtimestamp(r["ts"]).strftime("%Y-%m-%d %H:%M")
            print(f"  {dt} · {r['symbol']} · {r['exit_reason'][:60]} · "
                   f"${r['exit_price']:.2f}")
        return 0

    path.parent.mkdir(parents=True, exist_ok=True)
    # Append (don't overwrite) so future record_exit calls keep stacking
    existing_keys = set()
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                r = json.loads(line)
                existing_keys.add(f"{r['symbol']}:{int(r['ts'])}")
            except Exception:
                continue
    written = 0
    with path.open("a") as f:
        for r in found:
            key = f"{r['symbol']}:{int(r['ts'])}"
            if key in existing_keys:
                continue
            f.write(json.dumps(r) + "\n")
            written += 1
    print(f"[backfill] wrote {written} new records to {path}")
    print(f"[backfill] ({len(found) - written} duplicates skipped)")
    print()
    print("Run `!saves` in Discord (or "
           "`python scripts/run_saves_report.py`) to see the report.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
