# News filter — defensive gating on fresh, direction-contradicting news

## Honest framing

News is **not a source of alpha** for a bot like this. By the time a
headline reaches your feed, liquid US equity prices have already moved.
Chasing news sentiment is how retail loses money.

What news is good for: **avoiding disasters**.

- Don't buy calls into a downgrade that hit 3 minutes ago.
- Don't sell puts on a name that just announced a lawsuit.
- Don't be long-vol through an unscheduled M&A that kills implied vol.

The 15th filter (`f15_news_filter`) enforces exactly that, and nothing
more. It's a **block**, not a signal.

## How it works

```
TradeBot
   │
   ├─ AlpacaNewsProvider ──► last 2h of headlines for a symbol
   │     (uses your existing Alpaca API keys — no new signup)
   │
   ├─ ClaudeNewsClassifier ──► score ∈ [-1, +1] + short rationale
   │     (uses ANTHROPIC_API_KEY; falls back to keyword classifier)
   │
   ├─ CachedNewsSentiment ──► 5-minute TTL per symbol (saves API calls)
   │
   └─ f15_news_filter ──► BLOCK if direction contradicts a strong score
```

## Thresholds (tunable in `config/settings.yaml`)

```yaml
news:
  enabled: true
  lookback_hours: 2
  cache_ttl_seconds: 300
  block_score: 0.50                  # directional trades
  premium_harvest_block_score: 0.75  # VRP / Wheel / premium selling
```

| Signal direction | Blocked when |
|---|---|
| `bullish` (call) | `news_score <= -0.50` |
| `bearish` (put) | `news_score >= +0.50` |
| `premium_harvest` | `|news_score| >= 0.75` |

Anything weaker is **advisory only** — logged so you can audit, not blocking.

## What goes into the journal + dashboard

Every blocked trade is logged at INFO with the filter name and the
rationale (`news_negative_for_long: -0.85 downgrade, guidance cut`).
Blocked-by-news events are also pushed to Discord/Slack as `[warn] news
block: SPY momentum blocked by news: news_negative_for_long: ...`.

## Costs

- **Alpaca news API:** free with any Alpaca account. No extra signup.
- **Claude classifier:** ~1 API call per symbol per 5 min = ~120 calls per
  symbol per trading day. With Sonnet on short prompts, well under $1/day
  for a 10-symbol universe. Still, it's optional — set `ANTHROPIC_API_KEY`
  to enable, otherwise the keyword classifier (free, weak) is used.

## Disabling

If you want to paper-test without the news filter (for clean A/B stats):

```yaml
news:
  enabled: false
```

Or at runtime:

```bash
unset ANTHROPIC_API_KEY        # drops back to keyword classifier
```

## What's NOT included (and why)

- Twitter/Reddit/Stocktwits parsing. Noisy, expensive, adversarial.
- Earnings calendar integration. That's what `econ_calendar.py` is for —
  known catalysts get a pre-event blackout window, which is cleaner than
  reading headlines.
- Sentiment-based entries. Intentional: news is a gate, not a signal.
