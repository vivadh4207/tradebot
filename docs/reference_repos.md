# Reference Repositories

Open-source projects consulted for patterns. We do NOT vendor their code — we
re-implement the patterns that fit our architecture, and document here which
idea maps where.

## Requested by user (6 repos)

| Repo | What it does | Where used in tradebot |
|---|---|---|
| agamm/claude-code-owasp | OWASP-aligned security scanner for Claude-Code-generated code | `SECURITY.md` checklist; used in CI on commits |
| nextlevelbuilder/ui-ux-pro-max-skill | UI/UX design skill | Not applicable to a headless trading bot; would apply if we add a dashboard later |
| blader/humanizer | Humanizes model output text | Not applied — trading logs stay structured/JSON |
| obra/superpowers | Extra skills for Claude Code as a dev agent | Dev-loop only; does not affect bot runtime |
| tirth8205/code-review-graph | Graph-based code review | Suggested as pre-PR check in `docs/dev_workflow.md` |
| hardikpandya/stop-slop | Anti-bloat / anti-slop rules | Used verbatim in `CLAUDE.md` style rules |

## Open-source options-trading bots consulted

| Repo | Pattern borrowed | Where in tradebot |
|---|---|---|
| pattertj/LoopTrader | Pluggable strategy registration, signal → filter → order pipeline | `signals/base.py`, `risk/execution_chain.py` |
| brndnmtthws/thetagang | The Wheel (CSP → CC), VIX hedge gating | `signals/wheel.py`, `intelligence/vix.py` |
| cm-jones/thales (C++23) | Out of language scope; studied queue-based architecture | informed `main.py` threading design |
| jakenesler/openprophet | LLM-driven agentic signal via Alpaca | `signals/claude_ai.py` |
| AlexShakaev/backtesting_and_algotrading_options_with_Interactive_Brokers_API | Credit-spread backtest harness | `backtest/simulator.py`, `backtest/walk_forward.py` |
| ldt9/PyOptionTrader | TWS order patterns | Informed the `BrokerAdapter` interface (equivalent to IBKR layer if added) |

**Note:** Future work can implement an `IBKRBroker` against `BrokerAdapter` to plug in any of the IBKR-based bots directly.
