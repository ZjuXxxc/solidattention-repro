#!/usr/bin/env python3
"""Repeat P0 without rebuilding and publish distributional scheduler metrics."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--ffn-iterations", type=int, default=512)
    parser.add_argument("--tag", default="P0-cuda-dual-pipeline")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    binary = root / "build/cpp/solidattention-p0"
    scratch = root / "artifacts/cpp-p0-benchmark"
    scratch.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    compatibility = root / "vendor/nvidia-535.288/usr/lib/x86_64-linux-gnu"
    environment["LD_LIBRARY_PATH"] = (
        str(compatibility)
        + (":" + environment["LD_LIBRARY_PATH"] if environment.get("LD_LIBRARY_PATH") else "")
    )
    runs: list[dict[str, object]] = []
    for repeat in range(args.repeats):
        run_dir = scratch / f"repeat-{repeat:02d}"
        subprocess.run(
            [
                str(binary),
                "--output",
                str(run_dir),
                "--steps",
                str(args.steps),
                "--ffn-iterations",
                str(args.ffn_iterations),
            ],
            check=True,
            env=environment,
        )
        runs.append(json.loads((run_dir / "pipeline-metrics.json").read_text()))

    speedups = [float(run["speedup"]) for run in runs]
    serial = [float(run["serial_layer_ms"]) for run in runs]
    pipeline = [float(run["pipeline_layer_ms"]) for run in runs]
    waits = [float(run["pipeline_exposed_read_wait_ms"]) for run in runs]
    summary = {
        "version": args.tag,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "repeats": args.repeats,
        "steps_per_repeat": args.steps,
        "layers": runs[0]["layers"],
        "ffn_iterations": args.ffn_iterations,
        "scope": runs[0]["scope"],
        "serial_layer_ms_median": statistics.median(serial),
        "serial_layer_ms_p10": percentile(serial, 0.10),
        "serial_layer_ms_p90": percentile(serial, 0.90),
        "pipeline_layer_ms_median": statistics.median(pipeline),
        "pipeline_layer_ms_p10": percentile(pipeline, 0.10),
        "pipeline_layer_ms_p90": percentile(pipeline, 0.90),
        "speedup_median": statistics.median(speedups),
        "speedup_p10": percentile(speedups, 0.10),
        "speedup_p90": percentile(speedups, 0.90),
        "exposed_ssd_wait_ms_per_repeat_median": statistics.median(waits),
        "max_abs_error_max": max(
            float(run["serial_pipeline_max_abs_error"]) for run in runs
        ),
        "raw_runs": runs,
    }
    published = root / "artifacts/runs"
    published.mkdir(parents=True, exist_ok=True)
    metrics = published / f"{args.tag}-metrics.json"
    trace = published / f"{args.tag}-trace.json"
    metrics.write_text(json.dumps(summary, indent=2) + "\n")
    shutil.copyfile(
        scratch / f"repeat-{args.repeats - 1:02d}" / "pipeline-trace.json", trace
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "raw_runs"}, indent=2))
    print(f"metrics={metrics}")
    print(f"trace={trace}")


if __name__ == "__main__":
    main()
