#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
mkdir -p artifacts
PYTHONPATH=src python3 -m solidattention_lab.simulator --trace artifacts/trace.json "$@"
PYTHONPATH=src python3 -m solidattention_lab.dashboard artifacts/trace.json artifacts/dashboard.html
