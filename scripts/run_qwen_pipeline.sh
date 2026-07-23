#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
mkdir -p artifacts
PYTHONPATH=src .venv/bin/python -m solidattention_lab.qwen_pipeline "$@"
for mode in serial overlap; do
  PYTHONPATH=src .venv/bin/python -m solidattention_lab.dashboard \
    "artifacts/qwen-pipeline-${mode}.json" "artifacts/qwen-pipeline-${mode}.html"
done
