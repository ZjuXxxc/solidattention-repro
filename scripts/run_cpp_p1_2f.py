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
    parser.add_argument("--resident-weights", action="store_true")
    parser.add_argument("--pipeline-kv", action="store_true")
    parser.add_argument("--final-audit-only", action="store_true")
    parser.add_argument("--dram-prefetch-all", action="store_true")
    parser.add_argument("--read-ahead", type=int, default=1)
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
        command = [
            str(root / "build/cpp/solidattention-p1-2f"),
            "--fixture", str(fixture), "--plan", str(plan),
            "--metrics", str(raw_metrics),
            ]
        if args.resident_weights:
            command.append("--resident-weights")
        if args.pipeline_kv:
            command.append("--pipeline-kv")
        if args.final_audit_only:
            command.append("--final-audit-only")
        if args.dram_prefetch_all:
            command.append("--dram-prefetch-all")
        if args.read_ahead != 1:
            command.extend(["--read-ahead", str(args.read_ahead)])
        subprocess.run(
            command,
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
        "resident_weights": result["resident_weights"],
        "resident_preload_ms_median": statistics.median(
            run["resident_preload_ms"] for run in runs
        ),
        "resident_weight_bytes": result["resident_weight_bytes"],
        "pipeline_kv": result["pipeline_kv"],
        "final_audit_only": result.get("final_audit_only", False),
        "dram_prefetch_all": result.get("dram_prefetch_all", False),
        "read_ahead": result.get("read_ahead", 1),
        "dram_prefetch_ms_median": statistics.median(
            run.get("dram_prefetch_ms", 0.0) for run in runs
        ),
        "pinned_dram_bytes": result.get("pinned_dram_bytes", 0),
        "total_wall_ms_median": statistics.median(run["total_wall_ms"] for run in runs),
        "weight_read_ms_sum_median": statistics.median(totals("weight_read_ms")),
        "weight_h2d_ms_sum_median": statistics.median(totals("weight_h2d_ms")),
        "kv_read_ms_sum_median": statistics.median(totals("kv_read_ms")),
        "kv_h2d_ms_sum_median": statistics.median(totals("kv_h2d_ms")),
        "compute_ms_sum_median": statistics.median(totals("compute_ms")),
        "exposed_prefetch_wait_ms_sum_median": statistics.median(
            totals("exposed_prefetch_wait_ms")
        ),
        "maximum_input_error": (
            None if args.final_audit_only else max(row["input_max_error"] for row in details)
        ),
        "minimum_input_cosine": (
            None if args.final_audit_only else min(row["input_cosine"] for row in details)
        ),
        "maximum_output_error": result.get(
            "final_output_max_error", max(row["output_max_error"] for row in details)
        ),
        "minimum_output_cosine": result.get(
            "final_output_cosine", min(row["output_cosine"] for row in details)
        ),
        "layers_detail": details,
        "raw_runs": runs,
    }
    metrics = root / f"artifacts/runs/{summary['version']}-metrics.json"
    metrics.write_text(json.dumps(summary, indent=2) + "\n")
    cursor_us = 0.0
    events = []
    if args.pipeline_kv:
        if args.dram_prefetch_all or args.read_ahead > 1:
            preload_us = summary["dram_prefetch_ms_median"] * 1000
            events.append({
                "name": (
                    "all selected KV SSD→pinned DRAM (outside decode timer)"
                    if args.dram_prefetch_all
                    else f"initial depth-{args.read_ahead} SSD read-ahead"
                ),
                "cat": "scheduler-setup", "ph": "X", "ts": 0.0,
                "dur": preload_us, "pid": 0, "tid": "Setup",
                "args": {"bytes": summary["pinned_dram_bytes"]},
            })
            cursor_us = preload_us
        # This is a scheduler reconstruction from measured durations, not a
        # CUPTI trace.  Read i+1 is submitted at the start of layer i; H2D i+1
        # is launched after attention/post-norm and overlaps the tail MLP.
        for index, row in enumerate(details):
            critical_us = row["compute_ms"] * 1000
            events.append({
                "name": "Qwen layer critical section", "cat": "P1.2g.1", "ph": "X",
                "ts": cursor_us, "dur": critical_us, "pid": 1, "tid": "Compute stream",
                "args": {"layer": row["layer"], "includes_exposed_io_wait": True},
            })
            if index + 1 < len(details):
                next_row = details[index + 1]
                h2d_us = next_row["kv_h2d_ms"] * 1000
                target_index = index + args.read_ahead
                if not args.dram_prefetch_all and target_index < len(details):
                    target_row = details[target_index]
                    read_us = target_row["kv_read_ms"] * 1000
                    events.append({
                        "name": "bounded KV read-ahead", "cat": "P1.2h", "ph": "X",
                        "ts": cursor_us, "dur": read_us, "pid": 1, "tid": "liburing SSD→DRAM",
                        "args": {"producer_layer": row["layer"], "target_layer": target_row["layer"]},
                    })
                events.append({
                    "name": "copy next-layer KV", "cat": "P1.2g.1", "ph": "X",
                    "ts": cursor_us + max(0.0, critical_us - h2d_us), "dur": h2d_us,
                    "pid": 1, "tid": "CUDA copy stream",
                    "args": {"producer_layer": row["layer"], "target_layer": next_row["layer"]},
                })
                wait_us = row["exposed_prefetch_wait_ms"] * 1000
                if wait_us:
                    events.append({
                        "name": "exposed prefetch wait", "cat": "P1.2g.1", "ph": "X",
                        "ts": cursor_us + max(0.0, critical_us - wait_us), "dur": wait_us,
                        "pid": 1, "tid": "Host scheduler",
                        "args": {"layer": row["layer"], "target_layer": next_row["layer"]},
                    })
            cursor_us += critical_us
    else:
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
    trace = root / f"artifacts/runs/{summary['version']}-trace.json"
    trace.write_text(json.dumps({"traceEvents": events, "displayTimeUnit": "ms"}) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "layers_detail"}, indent=2))


if __name__ == "__main__":
    main()
