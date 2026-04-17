# CLAUDE.md — Project Conventions

This file tells future Claude-Code sessions how to work in this repo without
breaking the risk model. Style rules lean on `hardikpandya/stop-slop`.

## What this repo is

A paper-trading-first options/equities framework. Every architectural change
must preserve three invariants:

1. **LIVE_TRADING default is `false`.** No PR may flip this.
2. **Every order passes the 14-filter chain + order validator + portfolio
   risk check.** No bypass paths.
3. **Every entry sets `auto_profit_target` and `auto_stop_loss`.** Exits are
   event-driven; hope is not a strategy.

## Style rules (stop-slop)

- No hype adjectives ("trillion-dollar", "guaranteed", "world-class") in
  code, comments, docstrings, or logs.
- No more than 3 sentences in any docstring unless it documents math.
- Don't add features Claude wasn't asked for. Don't pad imports, don't
  emit unused helpers.
- No emojis in code or commit messages.
- Keep pure functions pure. Side effects only in `brokers/*` and `main.py`.
- Default to `from __future__ import annotations`.

## Code review checklist

Before merging:

1. `pytest -q` passes.
2. `python scripts/run_backtest.py` runs to completion, no crashes.
3. New module has at least one unit test.
4. If it touches order flow, it has a test for both accept AND reject paths.
5. If it adds a runtime network call, it imports the SDK lazily and has a
   synthetic fallback.

## Don't do

- Don't vendor the uploaded playbook text into code files. Reference by
  `docs/playbook_mapping.md`.
- Don't introduce threads other than the fast-exit thread without a design doc.
- Don't add Robinhood adapters — it has no official API and violates ToS.
- Don't promise returns in any surface (logs, README, docstrings).

## Do

- Add new strategies as `SignalSource` subclasses in `src/signals/`.
- Add new brokers as `BrokerAdapter` subclasses with lazy SDK imports.
- Log filter decisions at `INFO` with the filter name and reason.

## Research vs production

- `research/` — notebooks, scratch, candidate signals not yet reviewed.
- `src/` — production. Typed, tested, shippable.
- Production code must NEVER import from `research/`. Graduation
  checklist: `research/README.md`.
