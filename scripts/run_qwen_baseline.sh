#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
PYTHONPATH=src .venv/bin/python -m solidattention_lab.qwen_baseline "$@"
