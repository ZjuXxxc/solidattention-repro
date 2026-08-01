#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$project_dir/build/cpp"
compiler=("$project_dir/.venv/bin/python" -m ziglang c++)
"${compiler[@]}" -std=c++20 -O2 -I"$project_dir/cpp/include" \
  -I"$project_dir/vendor/liburing/usr/include" \
  "$project_dir/cpp/src/kv_lifecycle_main.cpp" \
  "$project_dir/cpp/src/uring_reader.cpp" \
  "$project_dir/vendor/liburing/usr/lib/x86_64-linux-gnu/liburing.a" \
  -o "$project_dir/build/cpp/solidattention-p1-3b0"
echo "built $project_dir/build/cpp/solidattention-p1-3b0"
