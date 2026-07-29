#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="$project_dir/build/cpp"
liburing="$project_dir/vendor/liburing/usr"
oneapi="$project_dir/vendor/oneapi-conda"
compiler="${CXX_SYCL:-$oneapi/bin/icpx}"
host_gcc="$oneapi/lib/gcc/x86_64-conda-linux-gnu/15.2.0"
if [[ ! -x "$compiler" ]]; then
  echo "missing oneAPI compiler: $compiler" >&2
  exit 2
fi
mkdir -p "$build_dir"
"$compiler" -std=c++20 -O1 -pthread -fsycl \
  --gcc-toolchain="$oneapi" -B"$host_gcc" -L"$host_gcc" \
  -DSOLIDATTENTION_ENABLE_SYCL=1 \
  -isystem "$host_gcc/include/c++" \
  -isystem "$host_gcc/include/c++/x86_64-conda-linux-gnu" \
  -isystem "$host_gcc/include/c++/backward" \
  -I"$project_dir/cpp/include" -I"$liburing/include" \
  "$project_dir/cpp/src/attention_main.cpp" \
  "$project_dir/cpp/src/attention_reference.cpp" \
  "$project_dir/cpp/src/trace.cpp" \
  "$project_dir/cpp/src/uring_reader.cpp" \
  "$project_dir/cpp/src/sycl_backend.cpp" \
  "$liburing/lib/x86_64-linux-gnu/liburing.a" \
  -o "$build_dir/solidattention-c1-sycl"
echo "built $build_dir/solidattention-c1-sycl"
