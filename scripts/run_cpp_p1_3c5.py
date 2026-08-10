#!/usr/bin/env python3
import json
import os
import statistics
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[1]
env = os.environ.copy()
compat = root / "vendor/nvidia-535.288/usr/lib/x86_64-linux-gnu"
env["LD_LIBRARY_PATH"] = str(compat) + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
env["PYTHONPATH"] = str(root / "src")
subprocess.run([str(root / ".venv/bin/python"), str(root / "scripts/export_qwen_prompt_representatives.py")], check=True, env=env)
for script in ("build_cpp_p1_3c4.sh", "build_cpp_p1_3c5.sh"):
    subprocess.run([str(root / "scripts" / script)], check=True)
subprocess.run([
    str(root / "build/cpp/solidattention-p1-3c4"), "--tail", str(root / "artifacts/qwen-lm-head/c4-tail.bin"),
    "--queries", str(root / "artifacts/qwen-lm-head/c4-queries-layer-head-token.f32"),
    "--prompt-store", str(root / "artifacts/qwen-28-layers/all-layers-kv-fp16.bin"),
    "--output", str(root / "artifacts/cpp-p1-3c4"),
], check=True)
runs = []
for repeat in range(5):
    path = root / f"artifacts/cpp-p1-3c5-repeat-{repeat}.json"
    subprocess.run([
        str(root / "build/cpp/solidattention-p1-3c5"),
        "--prompt-representatives", str(root / "artifacts/qwen-28-layers/prompt-representatives.f32"),
        "--decode-representatives", str(root / "artifacts/cpp-p1-3c4/representatives.f32"),
        "--queries", str(root / "artifacts/qwen-lm-head/c4-queries-layer-head-token.f32"),
        "--store", str(root / "artifacts/cpp-p1-3c4/main-kv-store.bin"),
        "--plan", str(root / "artifacts/qwen-28-layers/chain-plan.tsv"), "--output", str(path),
    ], check=True)
    runs.append(json.loads(path.read_text()))
summary = dict(runs[0])
for key in ("selection_ms_total", "selected_read_ms_total", "correction_read_ms_total"):
    summary[key + "_median_5"] = statistics.median(run[key] for run in runs)
summary["runs"] = runs
output = root / "artifacts/runs/P1.3c.5-merged-selection-correction-metrics.json"
output.write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps({key: value for key, value in summary.items() if key != "runs"}, indent=2))
