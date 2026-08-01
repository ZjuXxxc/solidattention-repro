#!/usr/bin/env python3
import json, statistics, subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[1]
subprocess.run([str(root / "scripts/build_cpp_p1_3b0.sh")], check=True)
runs = []
for repeat in range(5):
    output = root / "artifacts/cpp-p1-3b0-benchmark" / f"repeat-{repeat:02d}"
    subprocess.run([
        str(root / "build/cpp/solidattention-p1-3b0"), "--output", str(output),
        "--tokens", "512",
    ], check=True)
    runs.append(json.loads((output / "metrics.json").read_text()))
summary = dict(runs[-1])
summary.update({
    "repeats": 5,
    "wall_ms_median": statistics.median(run["wall_ms"] for run in runs),
    "wall_ms_min": min(run["wall_ms"] for run in runs),
    "wall_ms_max": max(run["wall_ms"] for run in runs),
    "raw_runs": runs,
})
path = root / "artifacts/runs/P1.3b.0-native-main-store-lifecycle-metrics.json"
path.write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps({k: v for k, v in summary.items() if k != "raw_runs"}, indent=2))
