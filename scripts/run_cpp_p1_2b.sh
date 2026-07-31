#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fixture="$project_dir/artifacts/qwen-layer0-sparse"
"$project_dir/.venv/bin/python" "$project_dir/scripts/export_qwen_layer.py" \
  --tokens 512 --position-start 0 --sparse-export --output "$fixture"
"$project_dir/scripts/build_cpp_p1_2.sh"
selected="$("$project_dir/.venv/bin/python" -c \
  'import json,sys; print(",".join(map(str,json.load(open(sys.argv[1]))["selected_blocks"])))' \
  "$fixture/manifest.json")"
compat_lib="$project_dir/vendor/nvidia-535.288/usr/lib/x86_64-linux-gnu"
export LD_LIBRARY_PATH="$compat_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$project_dir/build/cpp/solidattention-p1-2" \
  --sparse --selected "$selected" --input "$fixture" \
  --metrics "$project_dir/artifacts/qwen-layer0-sparse/native-sparse-metrics.json" \
  --trace "$project_dir/artifacts/qwen-layer0-sparse/native-sparse-trace.json" "$@"
