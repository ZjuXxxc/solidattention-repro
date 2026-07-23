#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
mkdir -p artifacts
PYTHONPATH=src .venv/bin/python -m solidattention_lab.qwen_decode "$@"
PYTHONPATH=src .venv/bin/python -m solidattention_lab.dashboard \
  artifacts/qwen-decode-trace.json artifacts/qwen-decode-dashboard.html
