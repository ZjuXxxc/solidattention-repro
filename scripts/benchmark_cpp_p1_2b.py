#!/usr/bin/env python3
"""Repeat the exported real-Qwen SSD sparse layer without rebuilding/exporting."""

from __future__ import annotations

import argparse
import json
import os
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
    parser.add_argument("--pipeline-next", action="store_true")
    parser.add_argument("--io-repeats", type=int, default=1)
    parser.add_argument(
        "--fixture", type=Path, default=Path("artifacts/qwen-layer0-sparse")
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fixture = (root / args.fixture).resolve()
    manifest = json.loads((fixture / "manifest.json").read_text())
    selected = ",".join(str(value) for value in manifest["selected_blocks"])
    environment = os.environ.copy()
    compatibility = root / "vendor/nvidia-535.288/usr/lib/x86_64-linux-gnu"
    environment["LD_LIBRARY_PATH"] = (
        str(compatibility)
        + (":" + environment["LD_LIBRARY_PATH"] if environment.get("LD_LIBRARY_PATH") else "")
    )
    scratch = root / "artifacts/cpp-p1-2b-benchmark"
    scratch.mkdir(parents=True, exist_ok=True)
    runs = []
    for repeat in range(args.repeats):
        metrics = scratch / f"repeat-{repeat:02d}.json"
        command = [
                str(root / "build/cpp/solidattention-p1-2"),
                "--sparse",
                "--selected",
                selected,
                "--input",
                str(fixture),
                "--metrics",
                str(metrics),
                "--io-repeats",
                str(args.io_repeats),
            ]
        if args.pipeline_next:
            command.append("--pipeline-next")
        subprocess.run(
            command,
            check=True,
            env=environment,
            stdout=subprocess.DEVNULL,
        )
        runs.append(json.loads(metrics.read_text()))
    native = [float(run["native_wall_ms"]) for run in runs]
    reads = [float(run["ssd_read_ms"]) for run in runs]
    h2d = [float(run["sparse_h2d_ms"]) for run in runs]
    version = (
        "P1.2c-persistent-real-qwen-pipeline"
        if args.pipeline_next
        else "P1.2b-real-qwen-ssd-sparse"
    )
    summary = {
        "version": version,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "repeats": args.repeats,
        "model": runs[0]["model"],
        "layer": runs[0]["layer"],
        "prompt_tokens": runs[0]["prompt_tokens"],
        "selected_tokens": runs[0]["attention_tokens"],
        "selected_blocks": manifest["selected_blocks"],
        "native_compute_wall_ms_median": statistics.median(native),
        "native_compute_wall_ms_p10": percentile(native, 0.1),
        "native_compute_wall_ms_p90": percentile(native, 0.9),
        "ssd_read_ms_median": statistics.median(reads),
        "ssd_read_ms_p10": percentile(reads, 0.1),
        "ssd_read_ms_p90": percentile(reads, 0.9),
        "h2d_ms_median": statistics.median(h2d),
        "io_repeats_per_process": args.io_repeats,
        "persistent_read_median_ms": statistics.median(
            float(run["persistent_read_median_ms"]) for run in runs
        ),
        "next_read_submit_to_cqe_ms_median": statistics.median(
            float(run["next_read_ms"]) for run in runs
        ),
        "exposed_next_read_wait_ms_median": statistics.median(
            float(run["exposed_next_read_wait_ms"]) for run in runs
        ),
        "next_h2d_ms_median": statistics.median(
            float(run["next_h2d_ms"]) for run in runs
        ),
        "sparse_teacher_layer_max_error_max": max(
            float(run["audits"]["sparse_layer_output"]["max_abs_error"])
            for run in runs
        ),
        "sparse_teacher_layer_cosine_min": min(
            float(run["audits"]["sparse_layer_output"]["cosine"]) for run in runs
        ),
        "sparse_vs_dense_layer_max_abs_error": runs[0][
            "sparse_vs_dense_layer_max_abs_error"
        ],
        "sparse_vs_dense_layer_cosine": runs[0]["sparse_vs_dense_layer_cosine"],
        "raw_runs": runs,
    }
    output = root / f"artifacts/runs/{version}-metrics.json"
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "raw_runs"}, indent=2))
    print(f"metrics={output}")


if __name__ == "__main__":
    main()
