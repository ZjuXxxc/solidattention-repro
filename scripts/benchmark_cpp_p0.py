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
    parser.add_argument("--tag")
    parser.add_argument("--history-correction", action="store_true")
    parser.add_argument("--infllm-selection", action="store_true")
    parser.add_argument("--generation-tickets", action="store_true")
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
    tag = args.tag or (
        "P1.3a-generation-safe-history-correction"
        if args.generation_tickets
        else "P1.1-infllm-history-correction"
        if args.infllm_selection
        else "P1.0-history-correction"
        if args.history_correction
        else "P0-cuda-dual-pipeline"
    )
    runs: list[dict[str, object]] = []
    for repeat in range(args.repeats):
        run_dir = scratch / f"repeat-{repeat:02d}"
        command = [
                str(binary),
                "--output",
                str(run_dir),
                "--steps",
                str(args.steps),
                "--ffn-iterations",
                str(args.ffn_iterations),
            ]
        if args.generation_tickets:
            command.append("--generation-tickets")
        elif args.infllm_selection:
            command.append("--infllm-selection")
        elif args.history_correction:
            command.append("--history-correction")
        subprocess.run(
            command,
            check=True,
            env=environment,
        )
        runs.append(json.loads((run_dir / "pipeline-metrics.json").read_text()))

    speedups = [float(run["speedup"]) for run in runs]
    serial = [float(run["serial_layer_ms"]) for run in runs]
    pipeline = [float(run["pipeline_layer_ms"]) for run in runs]
    waits = [float(run["pipeline_exposed_read_wait_ms"]) for run in runs]
    correction_reads = [float(run["correction_read_ms"]) for run in runs]
    correction_h2d = [float(run["correction_h2d_ms"]) for run in runs]
    summary = {
        "version": tag,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "repeats": args.repeats,
        "steps_per_repeat": args.steps,
        "layers": runs[0]["layers"],
        "ffn_iterations": args.ffn_iterations,
        "representative_build_ms_median": statistics.median(
            float(run["representative_build_ms"]) for run in runs
        ),
        "selection_ms_per_repeat_median": statistics.median(
            float(run["pipeline_selection_ms"]) for run in runs
        ),
        "dense_selected_attention_mass": float(
            runs[0]["dense_selected_attention_mass"]
        ),
        "dense_oracle_block_recall": float(
            runs[0]["dense_oracle_block_recall"]
        ),
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
        "history_hit_rate": float(runs[0]["history_hit_rate"]),
        "miss_blocks_per_repeat": int(runs[0]["miss_blocks"]),
        "verified_prefetch_tickets_per_repeat": int(
            runs[0].get("verified_prefetch_tickets", 0)
        ),
        "rejected_stale_tickets": int(runs[0].get("rejected_stale_tickets", 0)),
        "stale_ticket_self_test_passed": bool(
            runs[0].get("stale_ticket_self_test_passed", False)
        ),
        "correction_read_ms_per_repeat_median": statistics.median(
            correction_reads
        ),
        "correction_h2d_ms_per_repeat_median": statistics.median(
            correction_h2d
        ),
        "max_abs_error_max": max(
            float(run["serial_pipeline_max_abs_error"]) for run in runs
        ),
        "raw_runs": runs,
    }
    published = root / "artifacts/runs"
    published.mkdir(parents=True, exist_ok=True)
    metrics = published / f"{tag}-metrics.json"
    trace = published / f"{tag}-trace.json"
    metrics.write_text(json.dumps(summary, indent=2) + "\n")
    shutil.copyfile(
        scratch / f"repeat-{args.repeats - 1:02d}" / "pipeline-trace.json", trace
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "raw_runs"}, indent=2))
    print(f"metrics={metrics}")
    print(f"trace={trace}")


if __name__ == "__main__":
    main()
