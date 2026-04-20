"""Bar replay — iterate historical bars in chronological order for a universe."""
from __future__ import annotations

from typing import Dict, Iterator, List

from ..core.types import Bar


class BarReplayer:
    def __init__(self, bars_by_symbol: Dict[str, List[Bar]]):
        # Ensure sorted ascending
        self._data = {s: sorted(b, key=lambda x: x.ts) for s, b in bars_by_symbol.items()}

    def timeline(self) -> Iterator[Bar]:
        # merge-sort across all symbols
        pointers = {s: 0 for s in self._data}
        while True:
            next_symbol = None
            next_ts = None
            for s, arr in self._data.items():
                i = pointers[s]
                if i < len(arr):
                    if next_ts is None or arr[i].ts < next_ts:
                        next_ts = arr[i].ts
                        next_symbol = s
            if next_symbol is None:
                return
            yield self._data[next_symbol][pointers[next_symbol]]
            pointers[next_symbol] += 1
