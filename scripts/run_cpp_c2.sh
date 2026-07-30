#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"$project_dir/scripts/build_cpp_c2.sh"
compat_lib="$project_dir/vendor/nvidia-535.288/usr/lib/x86_64-linux-gnu"
export LD_LIBRARY_PATH="$compat_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$project_dir/build/cpp/solidattention-c2" \
  --backend cuda --output "$project_dir/artifacts/cpp-c2" "$@"
