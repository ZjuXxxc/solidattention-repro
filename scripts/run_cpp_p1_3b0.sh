#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"$project_dir/scripts/build_cpp_p1_3b0.sh"
exec "$project_dir/build/cpp/solidattention-p1-3b0" \
  --output "$project_dir/artifacts/cpp-p1-3b0" "$@"
