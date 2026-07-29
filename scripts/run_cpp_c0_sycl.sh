#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"$project_dir/scripts/build_cpp_sycl.sh"
oneapi_prefix="$project_dir/vendor/oneapi-conda"
export LD_LIBRARY_PATH="$oneapi_prefix/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$project_dir/build/cpp/solidattention-c0-sycl" \
  --backend sycl --output "$project_dir/artifacts/cpp-c0-sycl" "$@"
