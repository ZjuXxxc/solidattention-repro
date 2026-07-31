#!/usr/bin/env python3
"""Run the native sparse executor over 28 distinct streamed Qwen layer bundles."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, default=Path("artifacts/qwen-28-layers"))
    parser.add_argument("--max-layers", type=int, default=28)
    parser.add_argument("--chain", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fixture = (root / args.fixture).resolve()
    manifest = json.loads((fixture / "manifest.json").read_text())
    environment = os.environ.copy()
    compatibility = root / "vendor/nvidia-535.288/usr/lib/x86_64-linux-gnu"
    environment["LD_LIBRARY_PATH"] = (
        str(compatibility)
        + (":" + environment["LD_LIBRARY_PATH"] if environment.get("LD_LIBRARY_PATH") else "")
    )
    scratch = root / "artifacts/cpp-p1-2d"
    scratch.mkdir(parents=True, exist_ok=True)
    results = []
    wall_begin = time.perf_counter()
    for entry in manifest["layers"][: args.max_layers]:
        layer = int(entry["layer"])
        selected_key = "chain_selected_blocks" if args.chain else "selected_blocks"
        selected = ",".join(str(value) for value in entry[selected_key])
        suffix = "chain" if args.chain else "independent"
        metrics = scratch / f"layer-{layer:02d}-{suffix}-metrics.json"
        hidden_output = scratch / f"layer-{layer:02d}-chain-output.f32"
        command = [
                str(root / "build/cpp/solidattention-p1-2"),
                "--sparse", "--layer", str(layer), "--selected", selected,
                "--kv-store", str(fixture / manifest["kv_store"]),
                "--kv-offset", str(entry["kv_offset"]),
                "--input", str(fixture / entry["directory"]),
                "--metrics", str(metrics),
            ]
        if args.chain:
            command.extend(["--chain", "--hidden-output", str(hidden_output)])
            if layer > 0:
                command.extend([
                    "--hidden-input",
                    str(scratch / f"layer-{layer - 1:02d}-chain-output.f32"),
                ])
        subprocess.run(
            command,
            check=True, env=environment, stdout=subprocess.DEVNULL,
        )
        result = json.loads(metrics.read_text())
        results.append(result)
        audit_name = "chain_sparse_layer_output" if args.chain else "sparse_layer_output"
        audit = result["audits"][audit_name]
        print(
            f"layer={layer:02d} pass={result['pass']} "
            f"max_error={audit['max_abs_error']:.9g} cosine={audit['cosine']:.9f}"
        )
    bundle_sizes = [
        sum(path.stat().st_size for path in (fixture / entry["directory"]).glob("*.f32"))
        for entry in manifest["layers"][: args.max_layers]
    ]
    audit_name = "chain_sparse_layer_output" if args.chain else "sparse_layer_output"
    summary = {
        "version": (
            "P1.2e-28-layer-native-hidden-chain"
            if args.chain else "P1.2d-28-distinct-qwen-layer-audit"
        ),
        "model": manifest["model"],
        "revision": manifest["revision"],
        "layers": len(results),
        "prompt_tokens": manifest["tokens"],
        "selected_blocks_by_layer": [
            entry[selected_key] for entry in manifest["layers"][: args.max_layers]
        ],
        "unique_selected_block_sets": len({
            tuple(entry[selected_key])
            for entry in manifest["layers"][: args.max_layers]
        }),
        "all_layers_pass": all(bool(result["pass"]) for result in results),
        "maximum_sparse_teacher_layer_error": max(
            float(result["audits"][audit_name]["max_abs_error"])
            for result in results
        ),
        "minimum_sparse_teacher_layer_cosine": min(
            float(result["audits"][audit_name]["cosine"])
            for result in results
        ),
        "minimum_sparse_vs_dense_layer_cosine": min(
            float(result["sparse_vs_dense_layer_cosine"]) for result in results
        ),
        "maximum_sparse_vs_dense_layer_error": max(
            float(result["sparse_vs_dense_layer_max_abs_error"]) for result in results
        ),
        "maximum_layer_bundle_bytes": max(bundle_sizes),
        "total_layer_bundle_bytes": sum(bundle_sizes),
        "shared_kv_store_bytes": (fixture / manifest["kv_store"]).stat().st_size,
        "orchestration_wall_seconds": time.perf_counter() - wall_begin,
        "maximum_chain_input_error": (
            max(float(result["audits"]["chain_input"]["max_abs_error"])
                for result in results) if args.chain else 0.0
        ),
        "minimum_chain_input_cosine": (
            min(float(result["audits"]["chain_input"]["cosine"])
                for result in results) if args.chain else 1.0
        ),
        "layers_detail": results,
    }
    output = root / f"artifacts/runs/{summary['version']}-metrics.json"
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "layers_detail"}, indent=2))
    print(f"metrics={output}")
    if not summary["all_layers_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
