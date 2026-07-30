#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="$project_dir/build/cpp"
mkdir -p "$build_dir"
cuda_runtime="$project_dir/.venv/lib/python3.12/site-packages/nvidia/cuda_runtime"
cuda_nvrtc="$project_dir/.venv/lib/python3.12/site-packages/nvidia/cuda_nvrtc"
cuda_headers="$project_dir/.venv/lib/python3.12/site-packages/triton/backends/nvidia/include"
liburing="$project_dir/vendor/liburing/usr"
compiler=("$project_dir/.venv/bin/python" -m ziglang c++)
"${compiler[@]}" -std=c++20 -O2 -pthread -Wno-nullability-completeness \
  -I"$project_dir/cpp/include" -I"$liburing/include" \
  -I"$cuda_headers" -I"$cuda_runtime/include" -I"$cuda_nvrtc/include" \
  "$project_dir/cpp/src/pipeline_main.cpp" \
  "$project_dir/cpp/src/selection.cpp" \
  "$project_dir/cpp/src/attention_reference.cpp" \
  "$project_dir/cpp/src/trace.cpp" \
  "$project_dir/cpp/src/uring_reader.cpp" \
  "$liburing/lib/x86_64-linux-gnu/liburing.a" \
  "$cuda_runtime/lib/libcudart.so.12" \
  "$cuda_nvrtc/lib/libnvrtc.so.12" \
  /usr/lib/x86_64-linux-gnu/libcuda.so.535.309.01 \
  -Wl,-rpath,"$cuda_runtime/lib" -Wl,-rpath,"$cuda_nvrtc/lib" \
  -ldl -o "$build_dir/solidattention-p0"
echo "built $build_dir/solidattention-p0"
