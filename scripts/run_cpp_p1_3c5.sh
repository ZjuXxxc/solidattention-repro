#!/usr/bin/env bash
set -euo pipefail
p="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$p/.venv/bin/python" "$p/scripts/run_cpp_p1_3c5.py"
