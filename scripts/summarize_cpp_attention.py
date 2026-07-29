#!/usr/bin/env python3
"""Summarize repeated native C1/C1.1 attention metrics."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


FIELDS = ("mean_read_ms", "mean_h2d_ms", "mean_kernel_ms", "mean_d2h_ms")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics", nargs="+", type=Path)
    args = parser.parse_args()
    reports = [json.loads(path.read_text()) for path in args.metrics]
    versions = {(row["version"], row["backend"]) for row in reports}
    if len(versions) != 1:
        raise SystemExit(f"mixed version/backend inputs: {sorted(versions)}")
    print(f"{next(iter(versions))}, runs={len(reports)}")
    for field in FIELDS:
        values = [float(row[field]) for row in reports]
        deviation = statistics.stdev(values) if len(values) > 1 else 0.0
        print(f"{field}: {statistics.mean(values):.9f} ± {deviation:.9f}")
    stage_fields = list(FIELDS)
    if all("mean_consumer_ms" in row for row in reports):
        consumer = [float(row["mean_consumer_ms"]) for row in reports]
        deviation = statistics.stdev(consumer) if len(consumer) > 1 else 0.0
        print(
            f"mean_consumer_ms: {statistics.mean(consumer):.9f} "
            f"± {deviation:.9f}"
        )
        stage_fields.append("mean_consumer_ms")
    stage_sums = [
        sum(float(row[field]) for field in stage_fields) for row in reports
    ]
    deviation = statistics.stdev(stage_sums) if len(stage_sums) > 1 else 0.0
    print(f"instrumented_stage_sum_ms: {statistics.mean(stage_sums):.9f} ± {deviation:.9f}")
    if all("mean_device_call_wall_ms" in row for row in reports):
        wall = [float(row["mean_device_call_wall_ms"]) for row in reports]
        deviation = statistics.stdev(wall) if len(wall) > 1 else 0.0
        print(
            f"mean_device_call_wall_ms: {statistics.mean(wall):.9f} "
            f"± {deviation:.9f}"
        )
    print(
        "quality:",
        max(float(row["max_absolute_error"]) for row in reports),
        min(float(row["cosine_vs_cpu"]) for row in reports),
    )


if __name__ == "__main__":
    main()
