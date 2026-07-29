#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="$project_dir/build/cpp"
liburing="$project_dir/vendor/liburing/usr"
sycl_compiler="${CXX_SYCL:-icpx}"
if ! command -v "$sycl_compiler" >/dev/null 2>&1; then
  echo "missing oneAPI compiler: $sycl_compiler" >&2
  echo "set CXX_SYCL to an icpx/clang++ binary with -fsycl support" >&2
  exit 2
fi
mkdir -p "$build_dir"
"$sycl_compiler" -std=c++20 -O2 -pthread -fsycl \
  -DSOLIDATTENTION_ENABLE_SYCL=1 \
  -I"$project_dir/cpp/include" -I"$liburing/include" \
  "$project_dir/cpp/src/main.cpp" \
  "$project_dir/cpp/src/trace.cpp" \
  "$project_dir/cpp/src/uring_reader.cpp" \
  "$project_dir/cpp/src/sycl_backend.cpp" \
  "$liburing/lib/x86_64-linux-gnu/liburing.a" \
  -o "$build_dir/solidattention-c0-sycl"
echo "built $build_dir/solidattention-c0-sycl"
