#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
version="${1:?usage: run_versioned_decode.sh VERSION [decode arguments...]}"
shift
run_id="$(date -u +%Y%m%dT%H%M%SZ)"
safe_version="${version//[^A-Za-z0-9_.-]/_}"
prefix="artifacts/runs/${run_id}-${safe_version}"
mkdir -p artifacts/runs
PYTHONPATH=src .venv/bin/python -m solidattention_lab.qwen_decode \
  --version "$version" --trace "${prefix}-trace.json" "$@" | tee "${prefix}-metrics.json"
PYTHONPATH=src .venv/bin/python -m solidattention_lab.dashboard \
  "${prefix}-trace.json" "${prefix}-dashboard.html"
echo "immutable run prefix: $project_dir/$prefix"
