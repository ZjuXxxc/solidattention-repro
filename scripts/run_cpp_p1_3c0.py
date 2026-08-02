#!/usr/bin/env python3
import json, os, statistics, subprocess
from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parents[1]
subprocess.run([str(root / "scripts/build_cpp_p1_2f.sh")], check=True)
subprocess.run([str(root / "scripts/build_cpp_p1_3c0.sh")], check=True)
subprocess.run([str(root / ".venv/bin/python"), str(root / "scripts/export_qwen_lm_head.py")], check=True)
env = os.environ.copy(); compat = root / "vendor/nvidia-535.288/usr/lib/x86_64-linux-gnu"
env["LD_LIBRARY_PATH"] = str(compat) + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
fixture = root / "artifacts/qwen-lm-head"; runs = []
for repeat in range(5):
    chain_metrics = fixture / f"chain-{repeat}.json"
    hidden = fixture / f"hidden-{repeat}.f32"
    subprocess.run([str(root / "build/cpp/solidattention-p1-2f"),
        "--fixture", str(root / "artifacts/qwen-28-layers"), "--plan",
        str(root / "artifacts/qwen-28-layers/chain-plan.tsv"), "--metrics",
        str(chain_metrics), "--hidden-output", str(hidden), "--resident-weights",
        "--pipeline-kv", "--final-audit-only", "--read-ahead", "16"],
        check=True, env=env, stdout=subprocess.DEVNULL)
    lm_metrics = fixture / f"lm-{repeat}.json"
    subprocess.run([str(root / "build/cpp/solidattention-p1-3c0"), "--fixture",
        str(fixture), "--hidden", str(hidden), "--metrics", str(lm_metrics)],
        check=True, env=env)
    runs.append({"chain": json.loads(chain_metrics.read_text()),
                 "lm_head": json.loads(lm_metrics.read_text())})
h = np.fromfile(fixture / "hidden-4.f32", np.float32)
n = np.fromfile(fixture / "final_norm.f32", np.float32)
w = np.memmap(fixture / "lm_head.f32", np.float32, "r", shape=(151936,1024))
x = h / np.sqrt(np.mean(h*h) + 1e-6) * n
teacher = np.asarray(w @ x); native = np.fromfile(fixture / "native_logits.f32", np.float32)
delta = np.abs(native-teacher)
summary = {"version":"P1.3c.0-native-hidden-to-token", "repeats":5,
    "chain_wall_ms_median":statistics.median(r["chain"]["total_wall_ms"] for r in runs),
    "lm_head_wall_ms_median":statistics.median(r["lm_head"]["native_wall_ms"] for r in runs),
    "native_argmax_token":int(native.argmax()), "teacher_argmax_token":int(teacher.argmax()),
    "native_top5":native.argsort()[-5:][::-1].tolist(), "teacher_top5":teacher.argsort()[-5:][::-1].tolist(),
    "logit_max_abs_error":float(delta.max()), "logit_mean_abs_error":float(delta.mean()),
    "logit_cosine":float(np.dot(native.astype(np.float64),teacher.astype(np.float64))/(np.linalg.norm(native.astype(np.float64))*np.linalg.norm(teacher.astype(np.float64)))),
    "raw_runs":runs}
path=root/"artifacts/runs/P1.3c.0-native-hidden-to-token-metrics.json"
path.write_text(json.dumps(summary,indent=2)+"\n")
print(json.dumps({k:v for k,v in summary.items() if k!="raw_runs"},indent=2))
