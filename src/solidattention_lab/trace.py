"""Minimal Chrome Trace writer; no opaque profiling dependency required."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class Event:
    name: str
    cat: str
    ts: float
    dur: float
    pid: int
    tid: str
    args: dict

    def chrome(self) -> dict:
        item = asdict(self)
        item.update({"ph": "X"})
        return item


def write_trace(path: str | Path, events: list[Event], metadata: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {"traceEvents": [event.chrome() for event in events], "metadata": metadata}
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
