# Playbook → Code Mapping

Traceability between the uploaded options-trading playbook and the source files.

| Playbook Section | Concept | File(s) |
|---|---|---|
| 0. Architecture | Async + queue-based components, command TTL | `core/types.py` (Order/Signal TTL), `main.py` (fast loop) |
| 1.1 BS + Greeks | `bs_price`, `bs_greeks` (Δ, Γ, Vega, Θ, ρ, Vanna, Charm) | `math_tools/pricer.py` |
| 1.1 IV solver | Brent's method IV back-solve from mid | `math_tools/pricer.py::implied_vol` |
| 2.3 / 8 | HAR-RV realized vol forecast | `math_tools/har_rv.py` |
| 3.1 | VRP + z-score signal | `signals/vrp.py`, `signals/master_stack.py` |
| 3.2 | Daily put-write (short-dated OTM) | `signals/vrp.py` (chain-aware), `signals/wheel.py` |
| 4.1 | SVI surface fit | `math_tools/svi.py` |
| 5.1 | Kelly fraction (quarter-Kelly, 5% cap) | `math_tools/sizing.py::kelly_fraction` |
| 5.2 | VIX-regime sizing | `math_tools/sizing.py::vix_regime_multiplier` |
| 5.3 | Hybrid Kelly + VIX + VRP sizing | `math_tools/sizing.py::hybrid_sizing`, `risk/position_sizer.py` |
| 6 | Portfolio Greek limits + stress test | `risk/portfolio_risk.py` |
| 7.1 | Smart limit pricing (aggression = 0.3) | `risk/order_validator.py` (rounding) + call sites |
| 7.2 | QuoteValidator | `brokers/quote_validator.py` |
| 8 | Master composite signal stack | `signals/master_stack.py` |
| 9 | Walk-forward backtest | `backtest/walk_forward.py` |

## Priority-order recommendations (from playbook §10)

1. QuoteValidator + TTL on commands ✅ (`quote_validator.py`, Signal/Order TTL)
2. Accurate Greeks + IV from mid ✅ (`pricer.py`)
3. PortfolioRiskManager + stress ✅ (`portfolio_risk.py`)
4. VRP signal + IV-rank gate ✅ (`vrp.py`, 14-filter chain #7)
5. Hybrid Kelly+VIX sizing ✅ (`sizing.py`)
6. Adaptive delta threshold — pending (only needed once we add gamma-scalping)
7. SVI surface fit ✅ (`svi.py`)
8. HAR-RV forecast ✅ (`har_rv.py`)
