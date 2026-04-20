# Security Checklist (OWASP-aligned for a Trading Bot)

Adapted from agamm/claude-code-owasp. These apply BOTH to the code and to
the operational environment running the bot.

## Credentials & secrets

- Store API keys only in `.env` (gitignored). NEVER in `.env.example`
  (that file is a committed template — anything in it will end up public
  the moment you push). If a real key ever lands in `.env.example`, rotate
  it immediately in the provider's dashboard.
- Rotate credentials after any local disk handoff or repo clone.
- Never log the DSN, API key, or account number. Log symbols, sides,
  quantities, fills — nothing that identifies the account.
- For live trading: use a dedicated brokerage account with no margin you
  aren't willing to lose, separate from your primary account.
- Keys must have **read+trade** scope only, never withdrawal.
- The CockroachDB DSN is a secret too — treat it like a broker API key.

## Input validation

- Every price, quantity, strike, and expiry passes `OrderValidator`.
- Reject NaN, negative, crossed, or stale quotes via `QuoteValidator`.
- Enforce TTL on `Signal` and `Order` (5s default). Drop stale commands.

## Injection / deserialization

- `anthropic` responses are parsed as JSON inside a strict try/except, and
  the parsed object's keys are type-coerced before use.
- YAML is loaded with `yaml.safe_load`, never `yaml.load`.

## Authentication & session

- Broker adapter imports SDK lazily; missing credentials raise a clear error
  and do NOT fall through to fake fills in live mode.
- `LIVE_TRADING=true` is required AND `broker.name != paper`. Either alone
  must not route real orders.

## Logging

- Never log API keys, account numbers, or fill prices tied to a personal ID.
- Do log: symbol, qty, side, reason, filter result, PnL%.

## Dependency hygiene

- Pin minimum versions in `requirements.txt`.
- Run `pip-audit` on every dependency change.

## Operational

- Daily equity guardrail: -2% → halt entries.
- Kill switch: `touch KILL` at the repo root → main loop exits cleanly after
  closing open positions.
- Automatic EOD sweep at 15:45 ET.
- No new entries after 15:30 ET.
- Never run the bot on the same machine you use for personal banking.

## Incident response

1. `touch KILL` or Ctrl-C the process; if runaway, revoke the API key in
   the broker dashboard.
2. Reconcile positions via broker UI before restarting.
3. Submit a post-mortem in `docs/postmortems/` describing root cause +
   prevention.
