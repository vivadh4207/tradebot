"""End-of-day comprehensive summary generator.

Produces a structured Markdown + JSON audit of the day's session that:
  - Reads tradebot.out log + Tradier orders/balances
  - Tabulates every fill + matched round-trip P&L
  - Surfaces all exec_chain blocks by filter/reason
  - Flags potential bugs (rejected order patterns, phantom positions,
    reconcile drift events, circuit breaker trips)
  - Computes performance metrics (win rate, avg win/loss, drawdown)
  - Writes:
      logs/daily_summary_YYYY-MM-DD.md   — for human + LLM reading
      logs/daily_summary_YYYY-MM-DD.json  — structured for code
  - Posts the markdown to Discord via DISCORD_WEBHOOK_URL (or _AUDIT)

Run manually:
    python scripts/generate_daily_summary.py
Or schedule via launchd at 16:30 ET to auto-fire after EOD flatten.

Auditor LLM (Groq llama-3.3-70b-versatile) reads the markdown for
flagging "did we see bugs today like the phantom-position thrash."
"""
from __future__ import annotations

import json
import os
import re
import sys
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from datetime import date as _d, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except Exception:
    pass

LOG_FILE = ROOT / "logs" / "tradebot.out"
OUTPUT_DIR = ROOT / "logs"


# Bug-pattern signatures the LLM auditor should flag.
BUG_PATTERNS = {
    "phantom_position_pruned": "Local PaperBroker had a position Tradier didn't — pruned. Indicates mirror dropped a buy without local knowing.",
    "tradier_circuit_breaker_tripped": "3+ rejects in 60s on a symbol — operational issue (sandbox throttle, broker issue, or position desync).",
    "reconcile_drift_alarm": "Local broker cost-basis differs from Tradier by > $200 — likely phantom or stale position.",
    "tradier_terminal_reject": "Order accepted by Tradier syntactically but rejected at routing. Common: closing a position Tradier doesn't have.",
    "tradier_pending_at_timeout": "Order didn't reach terminal status within poll window — sandbox slow OR network issue.",
    "fast_loop_error": "Fast-exit thread errored — investigate stack trace.",
    "alpaca_quote_error_falling_back": "Primary data feed couldn't quote a symbol. If it's an option (esp 0DTE), Tradier-quote fallback should fire.",
    "groq_http_err": "Groq API rejected a request — usually rate-limit on free tier.",
    "tradier_auth_err": "Tradier token invalid — bot can't read positions or trade.",
    "watchdog heartbeat_stale": "Bot was killed by watchdog for stale heartbeat (loop wedged).",
}


def _read_log_lines() -> List[str]:
    if not LOG_FILE.exists():
        return []
    try:
        return LOG_FILE.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return []


def _today_iso() -> str:
    return _d.today().isoformat()


def _filter_today(lines: List[str], ymd: str) -> List[str]:
    return [l for l in lines if ymd in l[:30]]


def _http_get_json(url: str, headers: dict, timeout: float = 10.0) -> Optional[dict]:
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except Exception:
        return None


def _tradier_data(ymd: str) -> Dict[str, Any]:
    """Pull today's Tradier positions, balance, orders."""
    tok = (os.getenv("TRADIER_TOKEN") or "").strip()
    acct = (os.getenv("TRADIER_ACCOUNT") or "").strip()
    base = "https://sandbox.tradier.com/v1/accounts/" + acct
    h = {"Authorization": f"Bearer {tok}", "Accept": "application/json"}
    if not tok or not acct:
        return {}
    bal = (_http_get_json(f"{base}/balances", h) or {}).get("balances", {})
    pos_obj = (_http_get_json(f"{base}/positions", h) or {}).get("positions", {})
    pos = pos_obj.get("position", []) if isinstance(pos_obj, dict) else []
    if isinstance(pos, dict):
        pos = [pos]
    ord_obj = (_http_get_json(f"{base}/orders", h) or {}).get("orders", {})
    orders = ord_obj.get("order", []) if isinstance(ord_obj, dict) else []
    if isinstance(orders, dict):
        orders = [orders]
    today_orders = [
        o for o in orders
        if o.get("transaction_date", "")[:10] == ymd
    ]
    return {"balances": bal, "positions": pos, "orders_today": today_orders}


def _round_trip_pairs(orders: List[dict]) -> Tuple[List[dict], float, float]:
    """FIFO pair buys with sells per option_symbol. Returns
    (closed_trips, gross_win_usd, gross_loss_usd)."""
    by_sym: Dict[str, List[dict]] = defaultdict(list)
    for o in orders:
        if o.get("status") != "filled":
            continue
        sym = o.get("option_symbol") or o.get("symbol", "")
        by_sym[sym].append(o)
    pairs: List[dict] = []
    win, loss = 0.0, 0.0
    for sym, ords in by_sym.items():
        ords.sort(key=lambda x: x.get("transaction_date", ""))
        opens = []  # list of (price, qty) FIFO
        for o in ords:
            side = o.get("side", "")
            qty = int(float(o.get("exec_quantity") or 0))
            px = float(o.get("avg_fill_price") or 0)
            if "buy_to_open" in side or "sell_to_open" in side:
                opens.append({"price": px, "qty": qty,
                                "ts": o.get("transaction_date", ""),
                                "side": side})
            elif "to_close" in side:
                remaining = qty
                while remaining > 0 and opens:
                    op = opens[0]
                    matched = min(remaining, op["qty"])
                    if "buy_to_open" in op["side"]:
                        pnl = (px - op["price"]) * matched * 100
                    else:
                        pnl = (op["price"] - px) * matched * 100
                    pairs.append({
                        "symbol": sym,
                        "open_ts": op["ts"], "close_ts": o.get("transaction_date", ""),
                        "open_px": op["price"], "close_px": px,
                        "qty": matched, "pnl": pnl,
                    })
                    if pnl > 0: win += pnl
                    else:       loss += pnl
                    op["qty"] -= matched
                    if op["qty"] <= 0:
                        opens.pop(0)
                    remaining -= matched
    return pairs, win, loss


def build_summary(ymd: str) -> Dict[str, Any]:
    """Build the structured summary for date `ymd` (YYYY-MM-DD)."""
    lines = _read_log_lines()
    today_log = _filter_today(lines, ymd)
    tradier = _tradier_data(ymd)

    # Fills
    pairs, gross_win, gross_loss = _round_trip_pairs(
        tradier.get("orders_today", [])
    )
    n_round_trips = len(pairs)
    n_winners = sum(1 for p in pairs if p["pnl"] > 0)
    n_losers = sum(1 for p in pairs if p["pnl"] < 0)
    win_rate = n_winners / max(1, n_round_trips)
    avg_win = (gross_win / max(1, n_winners)) if n_winners else 0.0
    avg_loss = (gross_loss / max(1, n_losers)) if n_losers else 0.0
    biggest_winner = max(pairs, key=lambda p: p["pnl"]) if pairs else None
    biggest_loser  = min(pairs, key=lambda p: p["pnl"]) if pairs else None

    # Filter blocks histogram
    block_filters = Counter()
    for ln in today_log:
        if "exec_chain_block" in ln:
            m = re.search(r"filter=([A-Za-z0-9_]+)", ln)
            if m:
                block_filters[m.group(1)] += 1

    # Skip reasons
    skip_reasons = Counter()
    for ln in today_log:
        for tag in ("entry_skip_already_holding",
                     "entry_skip_per_underlying_cap",
                     "entry_skip_daily_budget",
                     "entry_skip_daily_pnl_lock",
                     "entry_skip_chop",
                     "entry_skip_crash_bias",
                     "entry_skip_rush_bias",
                     "entry_skip_qty_zero",
                     "entry_skip_no_liquid_strike"):
            if tag in ln:
                skip_reasons[tag] += 1

    # Bug-pattern hits
    bugs_seen = []
    for tag, desc in BUG_PATTERNS.items():
        n = sum(1 for ln in today_log if tag in ln)
        if n > 0:
            bugs_seen.append({"pattern": tag, "count": n, "desc": desc})

    # Tradier balances
    bal = tradier.get("balances", {})
    equity = float(bal.get("total_equity") or 0)
    cash = float(bal.get("total_cash") or 0)
    open_pl = float(bal.get("open_pl") or 0)
    close_pl = float(bal.get("close_pl") or 0)

    # Filled order counts
    filled = sum(1 for o in tradier.get("orders_today", []) if o.get("status") == "filled")
    rejected = sum(1 for o in tradier.get("orders_today", []) if o.get("status") == "rejected")

    return {
        "date": ymd,
        "tradier_equity": equity,
        "tradier_cash": cash,
        "tradier_open_pl": open_pl,
        "tradier_close_pl": close_pl,
        "tradier_net_day": open_pl + close_pl,
        "open_positions": tradier.get("positions", []),
        "filled_orders": filled,
        "rejected_orders": rejected,
        "round_trips": n_round_trips,
        "winners": n_winners,
        "losers": n_losers,
        "win_rate": win_rate,
        "gross_winners_usd": round(gross_win, 2),
        "gross_losers_usd": round(gross_loss, 2),
        "avg_win_usd": round(avg_win, 2),
        "avg_loss_usd": round(avg_loss, 2),
        "biggest_winner": biggest_winner,
        "biggest_loser": biggest_loser,
        "filter_blocks": dict(block_filters.most_common()),
        "entry_skips": dict(skip_reasons.most_common()),
        "bugs_seen": bugs_seen,
        "trade_pairs": pairs,
    }


def render_markdown(s: Dict[str, Any]) -> str:
    """Render the summary as Markdown for human + LLM reading."""
    out = []
    out.append(f"# 📊 Daily Trading Summary — {s['date']}")
    out.append("")
    out.append("## Account State (Tradier truth)")
    out.append(f"- **Equity**: ${s['tradier_equity']:,.2f}")
    out.append(f"- **Cash**: ${s['tradier_cash']:,.2f}")
    out.append(f"- **Closed P&L**: ${s['tradier_close_pl']:+,.2f}")
    out.append(f"- **Open P&L**: ${s['tradier_open_pl']:+,.2f}")
    out.append(f"- **Net day**: **${s['tradier_net_day']:+,.2f}**")
    out.append("")
    out.append("## Trading Activity")
    out.append(f"- Filled orders: {s['filled_orders']}")
    out.append(f"- Rejected orders: {s['rejected_orders']}")
    out.append(f"- Round trips: {s['round_trips']}")
    out.append(f"- Win rate: {s['win_rate']:.1%}  ({s['winners']} W / {s['losers']} L)")
    out.append(f"- Avg win: ${s['avg_win_usd']:+,.2f}")
    out.append(f"- Avg loss: ${s['avg_loss_usd']:+,.2f}")
    if s["biggest_winner"]:
        bw = s["biggest_winner"]
        out.append(f"- Biggest winner: **{bw['symbol']}** ${bw['pnl']:+,.2f} ({bw['open_px']} → {bw['close_px']})")
    if s["biggest_loser"]:
        bl = s["biggest_loser"]
        out.append(f"- Biggest loser: **{bl['symbol']}** ${bl['pnl']:+,.2f} ({bl['open_px']} → {bl['close_px']})")
    out.append("")
    out.append("## Open Positions at EOD")
    if not s["open_positions"]:
        out.append("- (flat)")
    else:
        for p in s["open_positions"]:
            out.append(f"- {p.get('symbol')} qty={p.get('quantity')} cost=${p.get('cost_basis')}")
    out.append("")
    out.append("## Entries Skipped (discipline working)")
    if not s["entry_skips"]:
        out.append("- (none)")
    else:
        for tag, n in s["entry_skips"].items():
            out.append(f"- `{tag}`: {n}")
    out.append("")
    out.append("## Filter Chain Blocks")
    if not s["filter_blocks"]:
        out.append("- (none)")
    else:
        for f, n in s["filter_blocks"].items():
            out.append(f"- `{f}`: {n}")
    out.append("")
    out.append("## ⚠️ Bug-Pattern Detection")
    if not s["bugs_seen"]:
        out.append("- ✅ No known bug patterns detected today.")
    else:
        for b in s["bugs_seen"]:
            out.append(f"- ⚠️ **`{b['pattern']}`** seen {b['count']}x — {b['desc']}")
    out.append("")
    out.append("## All Round-Trip Trades")
    if not s["trade_pairs"]:
        out.append("- (none)")
    else:
        out.append("| Symbol | Open | Close | Qty | P&L |")
        out.append("|---|---|---|---|---|")
        for p in s["trade_pairs"]:
            out.append(f"| {p['symbol']} | ${p['open_px']:.2f} | ${p['close_px']:.2f} | {p['qty']} | ${p['pnl']:+,.2f} |")
    out.append("")
    out.append("---")
    out.append("*Generated by `scripts/generate_daily_summary.py`. "
                 "Read this with a fresh LLM session for end-of-day audit.*")
    return "\n".join(out)


def post_to_discord(markdown: str) -> bool:
    url = (os.getenv("DISCORD_WEBHOOK_URL_AUDIT")
            or os.getenv("DISCORD_WEBHOOK_URL")
            or "").strip()
    if not url:
        print("[summary] no DISCORD_WEBHOOK_URL_AUDIT — skipping post")
        return False
    # Discord 2000 char limit per message — chunk if needed
    body = markdown
    while body:
        chunk = body[:1900]
        if len(body) > 1900:
            cut = chunk.rfind("\n")
            if cut > 0:
                chunk = body[:cut]
        body = body[len(chunk):].lstrip("\n")
        payload = json.dumps({"content": chunk}).encode("utf-8")
        req = urllib.request.Request(
            url, data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            urllib.request.urlopen(req, timeout=10)
        except Exception as e:
            print(f"[summary] discord post err: {e}")
            return False
    return True


def main() -> int:
    ymd = (sys.argv[1] if len(sys.argv) > 1 else _today_iso())
    print(f"[summary] generating for {ymd}")
    s = build_summary(ymd)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_DIR / f"daily_summary_{ymd}.json"
    md_path = OUTPUT_DIR / f"daily_summary_{ymd}.md"
    json_path.write_text(json.dumps(s, indent=2, default=str))
    md = render_markdown(s)
    md_path.write_text(md, encoding="utf-8")
    print(f"[summary] wrote {json_path.name} + {md_path.name}")
    if post_to_discord(md):
        print(f"[summary] posted to Discord")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
