#!/usr/bin/env python3
"""Summarize repeated native C2 selection/packing metrics."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


FIELDS = (
    "representative_build_ms",
    "selection_ms",
    "batch_read_ms",
    "h2d_ms",
    "attention_kernel_ms",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics", nargs="+", type=Path)
    args = parser.parse_args()
    rows = [json.loads(path.read_text()) for path in args.metrics]
    identities = {(row["version"], row["backend"]) for row in rows}
    if len(identities) != 1:
        raise SystemExit(f"mixed inputs: {sorted(identities)}")
    print(f"{next(iter(identities))}, runs={len(rows)}")
    print("selected ids:", sorted({tuple(row["selected_block_ids"]) for row in rows}))
    for field in FIELDS:
        values = [float(row[field]) for row in rows]
        deviation = statistics.stdev(values) if len(values) > 1 else 0.0
        print(f"{field}: {statistics.mean(values):.9f} ± {deviation:.9f}")
    print(
        "quality:",
        max(float(row["max_error_vs_sparse_cpu"]) for row in rows),
        min(float(row["cosine_vs_sparse_cpu"]) for row in rows),
        min(float(row["selected_vs_dense_cosine"]) for row in rows),
    )


if __name__ == "__main__":
    main()
