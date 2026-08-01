#!/usr/bin/env python3
import argparse, json, statistics, subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--real-qwen", action="store_true")
args = parser.parse_args()
subprocess.run([str(root / "scripts/build_cpp_p1_3b0.sh")], check=True)
runs = []
for repeat in range(5):
    output = root / "artifacts/cpp-p1-3b0-benchmark" / f"repeat-{repeat:02d}"
    command = [
        str(root / "build/cpp/solidattention-p1-3b0"), "--output", str(output),
        "--tokens", "512",
    ]
    if args.real_qwen:
        command += ["--input-kv", str(root / "artifacts/qwen-lifecycle-kv/layer-major-token-interleaved-kv-fp16.bin")]
    subprocess.run(command, check=True)
    runs.append(json.loads((output / "metrics.json").read_text()))
summary = dict(runs[-1])
summary.update({
    "repeats": 5,
    "wall_ms_median": statistics.median(run["wall_ms"] for run in runs),
    "wall_ms_min": min(run["wall_ms"] for run in runs),
    "wall_ms_max": max(run["wall_ms"] for run in runs),
    "raw_runs": runs,
})
if args.real_qwen:
    summary["fixture"] = json.loads(
        (root / "artifacts/qwen-lifecycle-kv/manifest.json").read_text()
    )
name = ("P1.3b.1-real-qwen-kv-lifecycle" if args.real_qwen
        else "P1.3b.0-native-main-store-lifecycle")
path = root / f"artifacts/runs/{name}-metrics.json"
path.write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps({k: v for k, v in summary.items() if k != "raw_runs"}, indent=2))
