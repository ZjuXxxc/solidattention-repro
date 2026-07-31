#!/usr/bin/env python3
"""Create a compact chain plan and run all 28 layers in one native process."""

from __future__ import annotations

import json
import os
import argparse
import statistics
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fixture = root / "artifacts/qwen-28-layers"
    manifest = json.loads((fixture / "manifest.json").read_text())
    plan = root / "artifacts/qwen-28-layers/chain-plan.tsv"
    plan.write_text("".join(
        f"{entry['layer']}\t{entry['directory']}\t{entry['kv_offset']}\t"
        f"{','.join(map(str, entry['chain_selected_blocks']))}\n"
        for entry in manifest["layers"]
    ))
    environment = os.environ.copy()
    compatibility = root / "vendor/nvidia-535.288/usr/lib/x86_64-linux-gnu"
    environment["LD_LIBRARY_PATH"] = (
        str(compatibility)
        + (":" + environment["LD_LIBRARY_PATH"] if environment.get("LD_LIBRARY_PATH") else "")
    )
    scratch = root / "artifacts/cpp-p1-2f"
    scratch.mkdir(parents=True, exist_ok=True)
    runs = []
    for repeat in range(args.repeats):
        raw_metrics = scratch / f"repeat-{repeat:02d}.json"
        subprocess.run(
            [
            str(root / "build/cpp/solidattention-p1-2f"),
            "--fixture", str(fixture), "--plan", str(plan),
            "--metrics", str(raw_metrics),
            ],
            check=True, env=environment, stdout=subprocess.DEVNULL,
        )
        runs.append(json.loads(raw_metrics.read_text()))
    result = runs[-1]
    details = result["layers_detail"]
    def totals(name: str) -> list[float]:
        return [sum(row[name] for row in run["layers_detail"]) for run in runs]
    summary = {
        "version": result["version"],
        "layers": result["layers"],
        "repeats": args.repeats,
        "total_wall_ms_median": statistics.median(run["total_wall_ms"] for run in runs),
        "weight_read_ms_sum_median": statistics.median(totals("weight_read_ms")),
        "weight_h2d_ms_sum_median": statistics.median(totals("weight_h2d_ms")),
        "kv_read_ms_sum_median": statistics.median(totals("kv_read_ms")),
        "kv_h2d_ms_sum_median": statistics.median(totals("kv_h2d_ms")),
        "compute_ms_sum_median": statistics.median(totals("compute_ms")),
        "maximum_input_error": max(row["input_max_error"] for row in details),
        "minimum_input_cosine": min(row["input_cosine"] for row in details),
        "maximum_output_error": max(row["output_max_error"] for row in details),
        "minimum_output_cosine": min(row["output_cosine"] for row in details),
        "layers_detail": details,
        "raw_runs": runs,
    }
    metrics = root / "artifacts/runs/P1.2f-single-process-qwen-chain-metrics.json"
    metrics.write_text(json.dumps(summary, indent=2) + "\n")
    cursor_us = 0.0
    events = []
    stages = [
        ("weight_read_ms", "layer weights SSD→DRAM", "Weight I/O"),
        ("weight_h2d_ms", "layer weights DRAM→VRAM", "Weight H2D"),
        ("kv_read_ms", "selected KV SSD→pinned DRAM", "KV I/O"),
        ("kv_h2d_ms", "selected KV pinned DRAM→VRAM", "KV H2D"),
        ("compute_ms", "real Qwen sparse layer", "GPU compute"),
    ]
    for row in details:
        for field, name, lane in stages:
            duration_us = row[field] * 1000
            events.append({
                "name": name, "cat": "P1.2f", "ph": "X",
                "ts": cursor_us, "dur": duration_us, "pid": 1, "tid": lane,
                "args": {"layer": row["layer"]},
            })
            cursor_us += duration_us
    trace = root / "artifacts/runs/P1.2f-single-process-qwen-chain-trace.json"
    trace.write_text(json.dumps({"traceEvents": events, "displayTimeUnit": "ms"}) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "layers_detail"}, indent=2))


if __name__ == "__main__":
    main()
