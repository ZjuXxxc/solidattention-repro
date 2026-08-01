#!/usr/bin/env python3
import json, os, statistics, subprocess
from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parents[1]
subprocess.run([str(root / "scripts/build_cpp_p1_2.sh")], check=True)
subprocess.run([str(root / "scripts/build_cpp_p1_3b0.sh")], check=True)
env = os.environ.copy()
compat = root / "vendor/nvidia-535.288/usr/lib/x86_64-linux-gnu"
env["LD_LIBRARY_PATH"] = str(compat) + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
scratch = root / "artifacts/cpp-p1-3b2"
scratch.mkdir(parents=True, exist_ok=True)
native_store = scratch / "native-projected-kv-fp16.bin"
with native_store.open("wb") as combined:
    for layer in range(28):
        layer_file = scratch / f"layer-{layer:02d}.bin"
        subprocess.run([
            str(root / "build/cpp/solidattention-p1-2"), "--input",
            str(root / f"artifacts/qwen-28-layers/layer-{layer:02d}"),
            "--kv-projection-only", "--kv-output", str(layer_file),
        ], check=True, env=env, stdout=subprocess.DEVNULL)
        combined.write(layer_file.read_bytes())
native = np.fromfile(native_store, dtype=np.float16).astype(np.float32)
teacher = np.fromfile(root / "artifacts/qwen-28-layers/all-layers-kv-fp16.bin",
                      dtype=np.float16).astype(np.float32)
difference = np.abs(native - teacher)
dot = float(np.dot(native.astype(np.float64), teacher.astype(np.float64)))
cosine = dot / float(np.linalg.norm(native.astype(np.float64)) * np.linalg.norm(teacher.astype(np.float64)))
layer_elements = native.size // 28
layer_audits = []
for layer in range(28):
    begin, end = layer * layer_elements, (layer + 1) * layer_elements
    delta = difference[begin:end]
    layer_audits.append({
        "layer": layer, "mismatch_rate": float(np.mean(delta != 0)),
        "maximum_absolute_error": float(delta.max()),
        "mean_absolute_error": float(delta.mean()),
    })
runs = []
for repeat in range(5):
    out = scratch / f"lifecycle-{repeat:02d}"
    subprocess.run([
        str(root / "build/cpp/solidattention-p1-3b0"), "--output", str(out),
        "--tokens", "480", "--input-kv", str(native_store),
    ], check=True, stdout=subprocess.DEVNULL)
    runs.append(json.loads((out / "metrics.json").read_text()))
summary = {
    "version": "P1.3b.2-native-qwen-kv-projection-lifecycle",
    "scope": "native CUDA/cuBLAS projection and native physical lifecycle",
    "layers": 28, "projected_tokens": 512, "decode_tokens": 480,
    "elements": int(native.size),
    "fp16_mismatch_elements": int(np.count_nonzero(difference)),
    "fp16_mismatch_rate": float(np.mean(difference != 0)),
    "maximum_absolute_error": float(difference.max()),
    "mean_absolute_error": float(difference.mean()),
    "cosine": cosine,
    "worst_layer_by_max_error": max(layer_audits, key=lambda row: row["maximum_absolute_error"])["layer"],
    "layer_audits": layer_audits,
    "sealed_blocks": runs[0]["sealed_blocks"],
    "verified_main_store_reads": runs[0]["verified_main_store_reads"],
    "resident_tail_tokens_per_layer": runs[0]["resident_tail_tokens_per_layer"],
    "selection_generation_per_layer": runs[0]["selection_generation_per_layer"],
    "lifecycle_wall_ms_median": statistics.median(run["wall_ms"] for run in runs),
    "raw_lifecycle_runs": runs,
}
path = root / "artifacts/runs/P1.3b.2-native-qwen-kv-projection-lifecycle-metrics.json"
path.write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps({k:v for k,v in summary.items() if k != "raw_lifecycle_runs"}, indent=2))
