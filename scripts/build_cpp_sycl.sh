#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="$project_dir/build/cpp"
liburing="$project_dir/vendor/liburing/usr"
local_compiler="$project_dir/vendor/oneapi-conda/bin/icpx"
if [[ -x "$local_compiler" ]]; then
  default_compiler="$local_compiler"
else
  default_compiler="icpx"
fi
sycl_compiler="${CXX_SYCL:-$default_compiler}"
host_gcc="$project_dir/vendor/oneapi-conda/lib/gcc/x86_64-conda-linux-gnu/15.2.0"
if ! command -v "$sycl_compiler" >/dev/null 2>&1; then
  echo "missing oneAPI compiler: $sycl_compiler" >&2
  echo "set CXX_SYCL to an icpx/clang++ binary with -fsycl support" >&2
  exit 2
fi
mkdir -p "$build_dir"
"$sycl_compiler" -std=c++20 -O2 -pthread -fsycl \
  --gcc-toolchain="$project_dir/vendor/oneapi-conda" \
  -B"$host_gcc" -L"$host_gcc" \
  -DSOLIDATTENTION_ENABLE_SYCL=1 \
  -isystem "$host_gcc/include/c++" \
  -isystem "$host_gcc/include/c++/x86_64-conda-linux-gnu" \
  -isystem "$host_gcc/include/c++/backward" \
  -I"$project_dir/cpp/include" -I"$liburing/include" \
  "$project_dir/cpp/src/main.cpp" \
  "$project_dir/cpp/src/trace.cpp" \
  "$project_dir/cpp/src/uring_reader.cpp" \
  "$project_dir/cpp/src/sycl_backend.cpp" \
  "$liburing/lib/x86_64-linux-gnu/liburing.a" \
  -o "$build_dir/solidattention-c0-sycl"
echo "built $build_dir/solidattention-c0-sycl"
