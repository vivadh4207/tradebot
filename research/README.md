# research/

Notebooks, scratch code, and experimental signals. **Production code in
`src/` must NOT import from this directory.** If a hypothesis proves out,
graduate it into `src/` with proper tests and types first.

## Rule

- `src/*.py` → production code. Typed, tested, reviewed.
- `research/*.py` + `research/*.ipynb` → research. Anything goes.

## Suggested layout

```
research/
├── README.md
├── notebooks/          # Jupyter
├── prototype_signals/  # candidate SignalSource implementations before graduation
└── data_exploration/   # one-off analyses
```

## Graduation checklist — from `research/` to `src/`

1. Code moves to `src/<module>/<feature>.py`, typed with `from __future__ import annotations`.
2. Unit tests in `tests/test_<feature>.py`, covering happy path AND at least one failure/edge case.
3. If it's a `SignalSource`, fits the ensemble weights schema in `config/settings.yaml`.
4. If it touches order flow, has tests for BOTH accept AND reject paths.
5. Lazy-import any runtime dependencies that aren't in `requirements.txt`.

A signal that can't clear the graduation checklist isn't ready for the
execution chain — keep iterating in `research/` until it does.
