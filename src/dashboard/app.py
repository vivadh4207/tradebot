"""FastAPI read-only dashboard: equity curve, trades, open positions.

Reads the configured journal (SQLite or CockroachDB). No authentication —
bind to localhost and reach it through an SSH tunnel or a reverse-proxy
with auth. Do NOT expose to the public internet.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

import json
from collections import Counter, defaultdict

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse

from ..core.config import load_settings
from ..storage.journal import build_journal


def _load_journal():
    root = Path(__file__).resolve().parents[2]
    s = load_settings(root / "config" / "settings.yaml")
    return build_journal(
        backend=s.get("storage.backend", "sqlite"),
        sqlite_path=s.get("storage.sqlite_path", str(root / "logs" / "tradebot.sqlite")),
        dsn_env_var=s.get("storage.cockroach_dsn_env", "COCKROACH_DSN"),
    )


app = FastAPI(title="tradebot dashboard", docs_url=None, redoc_url=None)


@app.get("/api/equity", response_class=JSONResponse)
def equity(days: int = Query(30, ge=1, le=365)):
    j = _load_journal()
    try:
        since = datetime.now(tz=timezone.utc) - timedelta(days=days)
        rows = j.equity_series(since=since, limit=20000)
        return {"points": [{"ts": r[0], "equity": r[1], "cash": r[2], "day_pnl": r[3]}
                            for r in rows]}
    finally:
        j.close()


@app.get("/api/trades", response_class=JSONResponse)
def trades(days: int = Query(30, ge=1, le=365), limit: int = Query(500, le=5000)):
    j = _load_journal()
    try:
        since = datetime.now(tz=timezone.utc) - timedelta(days=days)
        ts = j.closed_trades(since=since)[-limit:]
        return {"trades": [
            {"symbol": t.symbol,
             "opened_at": t.opened_at.isoformat() if t.opened_at else None,
             "closed_at": t.closed_at.isoformat() if t.closed_at else None,
             "side": t.side, "qty": t.qty,
             "entry_price": t.entry_price, "exit_price": t.exit_price,
             "pnl": t.pnl, "pnl_pct": t.pnl_pct,
             "entry_tag": t.entry_tag, "exit_reason": t.exit_reason,
             "is_option": t.is_option}
            for t in ts
        ]}
    finally:
        j.close()


@app.get("/api/metrics", response_class=JSONResponse)
def metrics(days: int = Query(30, ge=1, le=365)):
    j = _load_journal()
    try:
        since = datetime.now(tz=timezone.utc) - timedelta(days=days)
        ts = j.closed_trades(since=since)
        wins = [t for t in ts if (t.pnl or 0) > 0]
        losses = [t for t in ts if (t.pnl or 0) < 0]
        n_decided = len(wins) + len(losses)
        win_rate = (len(wins) / n_decided) if n_decided else 0.0
        avg_win = (sum(t.pnl_pct or 0 for t in wins) / len(wins)) if wins else 0.0
        avg_loss = (sum(abs(t.pnl_pct or 0) for t in losses) / len(losses)) if losses else 0.0
        total_pnl = sum((t.pnl or 0) for t in ts)
        return {"n_trades": len(ts), "n_wins": len(wins), "n_losses": len(losses),
                "win_rate": round(win_rate, 4),
                "avg_win_pct": round(avg_win, 4),
                "avg_loss_pct": round(avg_loss, 4),
                "total_pnl": round(total_pnl, 2),
                "expected_value_pct":
                    round(win_rate * avg_win - (1 - win_rate) * avg_loss, 4)}
    finally:
        j.close()


@app.get("/api/ensemble", response_class=JSONResponse)
def ensemble(days: int = Query(14, ge=1, le=365),
             recent_limit: int = Query(50, ge=1, le=500)):
    """Per-regime stats + recent decisions.

    Joins ensemble_decisions with closed_trades on approximate time
    (decision_ts ≤ opened_at ≤ decision_ts + 60min) for a per-regime
    win-rate readout.
    """
    j = _load_journal()
    try:
        since = datetime.now(tz=timezone.utc) - timedelta(days=days)
        decisions = j.ensemble_decisions(since=since)
        trades = j.closed_trades(since=since)
    finally:
        j.close()

    if not decisions:
        return {
            "n_decisions": 0,
            "n_emitted": 0,
            "global_emit_rate": 0.0,
            "by_regime": {},
            "recent": [],
            "contributors_overall": [],
        }

    # Index trades by symbol → sorted by opened_at for fast matching.
    tr_by_symbol: Dict[str, List] = defaultdict(list)
    for t in trades:
        if t.opened_at is None:
            continue
        tr_by_symbol[t.symbol].append(t)
    for k in tr_by_symbol:
        tr_by_symbol[k].sort(key=lambda t: t.opened_at)

    def _match(decision):
        """Nearest closed trade opened within 60 min of the decision."""
        ts = decision.ts if decision.ts.tzinfo else decision.ts.replace(tzinfo=timezone.utc)
        window = ts + timedelta(minutes=60)
        for t in tr_by_symbol.get(decision.symbol, []):
            ot = t.opened_at
            if ot.tzinfo is None:
                ot = ot.replace(tzinfo=timezone.utc)
            if ts <= ot <= window:
                return t
            if ot > window:
                break
        return None

    by_regime: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
        "n": 0, "emits": 0, "matched": 0, "wins": 0,
        "pnl_pcts": [], "contributors": Counter(),
    })
    emitted_count = 0
    for d in decisions:
        b = by_regime[d.regime]
        b["n"] += 1
        if d.emitted:
            b["emits"] += 1
            emitted_count += 1
            t = _match(d)
            if t is not None:
                b["matched"] += 1
                if (t.pnl or 0) > 0:
                    b["wins"] += 1
                if t.pnl_pct is not None:
                    b["pnl_pcts"].append(float(t.pnl_pct))
        if d.contributors:
            try:
                for c in json.loads(d.contributors):
                    b["contributors"][c.get("source", "?")] += 1
            except Exception:
                pass

    by_regime_out: Dict[str, Any] = {}
    overall_contrib: Counter = Counter()
    for regime, b in by_regime.items():
        n = b["n"]
        matched = b["matched"]
        pnl_list = b["pnl_pcts"]
        by_regime_out[regime] = {
            "n": n,
            "emits": b["emits"],
            "emit_rate": round(b["emits"] / n, 4) if n else 0.0,
            "matched_trades": matched,
            "win_rate": round(b["wins"] / matched, 4) if matched else None,
            "mean_pnl_pct": round(sum(pnl_list) / len(pnl_list), 6) if pnl_list else None,
            "contributors": [{"source": s, "count": c}
                              for s, c in b["contributors"].most_common(8)],
        }
        overall_contrib.update(b["contributors"])

    recent = []
    for d in decisions[-recent_limit:][::-1]:
        contributors = []
        if d.contributors:
            try:
                contributors = json.loads(d.contributors)
            except Exception:
                contributors = []
        recent.append({
            "ts": d.ts.isoformat() if hasattr(d.ts, "isoformat") else str(d.ts),
            "symbol": d.symbol, "regime": d.regime,
            "emitted": bool(d.emitted),
            "direction": d.dominant_direction,
            "score": round(d.dominant_score, 4) if d.dominant_score is not None else None,
            "opposing": round(d.opposing_score, 4) if d.opposing_score is not None else None,
            "n_inputs": d.n_inputs,
            "reason": d.reason,
            "contributors": contributors,
        })

    return {
        "n_decisions": len(decisions),
        "n_emitted": emitted_count,
        "global_emit_rate": round(emitted_count / len(decisions), 4) if decisions else 0.0,
        "by_regime": by_regime_out,
        "recent": recent,
        "contributors_overall": [{"source": s, "count": c}
                                  for s, c in overall_contrib.most_common(12)],
    }


_INDEX_HTML = """<!DOCTYPE html>
<html><head>
<meta charset="utf-8"/>
<title>tradebot</title>
<style>
  body { font-family: -apple-system, system-ui, sans-serif; background: #0b1020;
         color: #e6e9f5; margin: 0; padding: 24px; }
  h1 { font-size: 20px; margin: 0 0 16px; }
  h2 { font-size: 14px; text-transform: uppercase; letter-spacing: .08em;
       color: #9aa3c7; margin: 24px 0 10px; }
  .grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }
  .card { background: #141a33; border: 1px solid #223; border-radius: 10px;
          padding: 14px; }
  .card .k { color: #9aa3c7; font-size: 11px; text-transform: uppercase; }
  .card .v { font-size: 22px; font-weight: 600; margin-top: 4px; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th, td { padding: 8px 10px; text-align: left; border-bottom: 1px solid #1e2540; }
  th { color: #9aa3c7; font-weight: 500; font-size: 11px; text-transform: uppercase; }
  tr:hover { background: #131a35; }
  .pos { color: #6ee7b7; } .neg { color: #fca5a5; } .mut { color: #9aa3c7; }
  canvas { max-height: 280px; }
  .row { display: flex; gap: 12px; align-items: center; margin-bottom: 6px; }
  select { background: #141a33; color: #e6e9f5; border: 1px solid #2b3360;
           border-radius: 6px; padding: 4px 8px; }
  .pill { display: inline-block; padding: 1px 7px; border-radius: 10px;
          font-size: 11px; font-weight: 500; }
  .pill.emit  { background: #14291e; color: #6ee7b7; }
  .pill.block { background: #2a1a1f; color: #fca5a5; }
  .pill.regime { background: #182045; color: #a2b7ff; font-family: ui-monospace, monospace; }
  .chip { display: inline-block; background: #1a2142; color: #cbd5ff;
          border-radius: 4px; padding: 1px 6px; font-size: 11px; margin-right: 4px; }
</style>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head><body>
<h1>tradebot &nbsp;·&nbsp; <span class="mut">read-only journal view</span></h1>
<div class="row">
  <label>lookback</label>
  <select id="days">
    <option value="1">1d</option><option value="7">7d</option>
    <option value="30" selected>30d</option>
    <option value="90">90d</option><option value="365">365d</option>
  </select>
</div>
<div class="grid" id="metrics"></div>
<h2>equity curve</h2>
<div class="card"><canvas id="chart"></canvas></div>
<h2>closed trades</h2>
<div class="card"><table id="trades"><thead><tr>
  <th>closed</th><th>symbol</th><th>side</th><th>qty</th>
  <th>entry</th><th>exit</th><th>pnl</th><th>pnl %</th>
  <th>entry tag</th><th>exit reason</th>
</tr></thead><tbody></tbody></table></div>

<h2>ensemble &nbsp;·&nbsp; <span class="mut">regime-weighted signal coordinator</span></h2>
<div class="grid" id="ens-summary"></div>
<h2>regime breakdown</h2>
<div class="card"><table id="ens-regimes"><thead><tr>
  <th>regime</th><th>decisions</th><th>emits</th><th>emit rate</th>
  <th>matched trades</th><th>win rate</th><th>mean pnl %</th><th>top contributors</th>
</tr></thead><tbody></tbody></table></div>
<h2>recent ensemble decisions</h2>
<div class="card"><table id="ens-recent"><thead><tr>
  <th>time</th><th>symbol</th><th>regime</th><th>result</th><th>direction</th>
  <th>score</th><th>opposing</th><th>n inputs</th><th>reason</th>
</tr></thead><tbody></tbody></table></div>
<script>
let chart;
function fmt(n, d=2) { return (n===null||n===undefined) ? '' : Number(n).toFixed(d); }
function cls(n) { return n>0 ? 'pos' : (n<0 ? 'neg' : 'mut'); }
async function refresh() {
  const d = document.getElementById('days').value;
  const [m, eq, tr, en] = await Promise.all([
    fetch('/api/metrics?days='+d).then(r=>r.json()),
    fetch('/api/equity?days='+d).then(r=>r.json()),
    fetch('/api/trades?days='+d+'&limit=200').then(r=>r.json()),
    fetch('/api/ensemble?days='+d+'&recent_limit=50').then(r=>r.json()).catch(()=>null),
  ]);
  const mel = document.getElementById('metrics');
  mel.innerHTML = '';
  const cards = [
    ['trades', m.n_trades],
    ['win rate', (m.win_rate*100).toFixed(1)+'%'],
    ['avg win / avg loss', fmt(m.avg_win_pct*100,2)+'% / '+fmt(m.avg_loss_pct*100,2)+'%'],
    ['total pnl', '$'+fmt(m.total_pnl,2)],
  ];
  cards.forEach(([k,v]) => {
    const el = document.createElement('div'); el.className='card';
    el.innerHTML = `<div class="k">${k}</div><div class="v">${v}</div>`;
    mel.appendChild(el);
  });
  const labels = eq.points.map(p => p.ts);
  const data = eq.points.map(p => p.equity);
  if (chart) chart.destroy();
  chart = new Chart(document.getElementById('chart'), {
    type: 'line',
    data: { labels, datasets: [{ label: 'equity', data, borderColor: '#8ab4ff',
                                  backgroundColor: 'rgba(138,180,255,.08)',
                                  pointRadius: 0, tension: 0.2, fill: true }] },
    options: { responsive: true, scales: {
      x: { ticks: { color: '#9aa3c7', maxTicksLimit: 8 } },
      y: { ticks: { color: '#9aa3c7' } } },
      plugins: { legend: { display: false } } },
  });
  const tb = document.querySelector('#trades tbody');
  tb.innerHTML = '';
  tr.trades.slice().reverse().forEach(t => {
    const row = document.createElement('tr');
    row.innerHTML =
      `<td class="mut">${t.closed_at ? t.closed_at.replace('T',' ').slice(0,19) : ''}</td>`+
      `<td>${t.symbol}</td><td>${t.side}</td><td>${t.qty}</td>`+
      `<td>${fmt(t.entry_price)}</td><td>${fmt(t.exit_price)}</td>`+
      `<td class="${cls(t.pnl)}">${fmt(t.pnl)}</td>`+
      `<td class="${cls(t.pnl_pct)}">${t.pnl_pct!==null ? (t.pnl_pct*100).toFixed(2)+'%' : ''}</td>`+
      `<td class="mut">${t.entry_tag ?? ''}</td>`+
      `<td class="mut">${t.exit_reason ?? ''}</td>`;
    tb.appendChild(row);
  });
  renderEnsemble(en);
}

function renderEnsemble(en) {
  const sum = document.getElementById('ens-summary');
  const regTbody = document.querySelector('#ens-regimes tbody');
  const recTbody = document.querySelector('#ens-recent tbody');
  sum.innerHTML = '';
  regTbody.innerHTML = '';
  recTbody.innerHTML = '';
  if (!en) {
    sum.innerHTML = '<div class="card"><div class="k">ensemble</div>'+
      '<div class="v mut">n/a</div></div>';
    return;
  }
  const cards = [
    ['decisions', en.n_decisions],
    ['emitted', en.n_emitted],
    ['emit rate', (en.global_emit_rate*100).toFixed(1)+'%'],
    ['top contributors',
     (en.contributors_overall||[]).slice(0,4)
        .map(c=>c.source+':'+c.count).join(' · ') || '—'],
  ];
  cards.forEach(([k,v]) => {
    const el = document.createElement('div'); el.className='card';
    el.innerHTML = `<div class="k">${k}</div><div class="v">${v}</div>`;
    sum.appendChild(el);
  });
  // per-regime table
  const order = ['trend_lowvol','trend_highvol','range_lowvol','range_highvol','opening','closing'];
  const seen = new Set();
  const orderedRegimes = [];
  order.forEach(r => { if (en.by_regime[r]) { orderedRegimes.push(r); seen.add(r); } });
  Object.keys(en.by_regime).forEach(r => { if (!seen.has(r)) orderedRegimes.push(r); });
  orderedRegimes.forEach(r => {
    const d = en.by_regime[r];
    const wr = d.win_rate===null || d.win_rate===undefined ? '—' : (d.win_rate*100).toFixed(1)+'%';
    const mp = d.mean_pnl_pct===null || d.mean_pnl_pct===undefined ? '—' : (d.mean_pnl_pct*100).toFixed(2)+'%';
    const topC = (d.contributors||[]).slice(0,4)
      .map(c=>`<span class="chip">${c.source}:${c.count}</span>`).join('');
    const row = document.createElement('tr');
    row.innerHTML =
      `<td><span class="pill regime">${r}</span></td>`+
      `<td>${d.n}</td>`+
      `<td>${d.emits}</td>`+
      `<td>${(d.emit_rate*100).toFixed(1)}%</td>`+
      `<td class="mut">${d.matched_trades||0}</td>`+
      `<td class="${d.win_rate>0.5?'pos':(d.win_rate<0.45 && d.win_rate!==null?'neg':'mut')}">${wr}</td>`+
      `<td class="${d.mean_pnl_pct>0?'pos':(d.mean_pnl_pct<0?'neg':'mut')}">${mp}</td>`+
      `<td>${topC}</td>`;
    regTbody.appendChild(row);
  });
  // recent decisions
  (en.recent||[]).forEach(d => {
    const row = document.createElement('tr');
    const pill = d.emitted ?
      '<span class="pill emit">emit</span>' :
      '<span class="pill block">block</span>';
    row.innerHTML =
      `<td class="mut">${(d.ts||'').replace('T',' ').slice(0,19)}</td>`+
      `<td>${d.symbol}</td>`+
      `<td><span class="pill regime">${d.regime}</span></td>`+
      `<td>${pill}</td>`+
      `<td>${d.direction||'—'}</td>`+
      `<td>${d.score===null||d.score===undefined?'—':Number(d.score).toFixed(3)}</td>`+
      `<td class="mut">${d.opposing===null||d.opposing===undefined?'—':Number(d.opposing).toFixed(3)}</td>`+
      `<td class="mut">${d.n_inputs}</td>`+
      `<td class="mut">${d.reason}</td>`;
    recTbody.appendChild(row);
  });
}
document.getElementById('days').addEventListener('change', refresh);
refresh(); setInterval(refresh, 60_000);
</script>
</body></html>
"""


@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(_INDEX_HTML)
