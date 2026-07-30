#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"$project_dir/scripts/build_cpp_p0.sh"
compat_lib="$project_dir/vendor/nvidia-535.288/usr/lib/x86_64-linux-gnu"
export LD_LIBRARY_PATH="$compat_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$project_dir/build/cpp/solidattention-p0" \
  --output "$project_dir/artifacts/cpp-p0" "$@"
