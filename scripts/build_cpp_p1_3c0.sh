#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cuda_runtime="$project_dir/.venv/lib/python3.12/site-packages/nvidia/cuda_runtime"
cuda_nvrtc="$project_dir/.venv/lib/python3.12/site-packages/nvidia/cuda_nvrtc"
cublas="$project_dir/.venv/lib/python3.12/site-packages/nvidia/cublas"
liburing="$project_dir/vendor/liburing/usr"
headers="$project_dir/.venv/lib/python3.12/site-packages/triton/backends/nvidia/include"
compiler=("$project_dir/.venv/bin/python" -m ziglang c++)
"${compiler[@]}" -std=c++20 -O2 -Wno-nullability-completeness \
  -I"$project_dir/cpp/include" -I"$project_dir/cpp/src" -I"$headers" \
  -I"$liburing/include" \
  -I"$cuda_runtime/include" -I"$cuda_nvrtc/include" \
  "$project_dir/cpp/src/qwen_lm_head_main.cpp" \
  "$cuda_runtime/lib/libcudart.so.12" "$cuda_nvrtc/lib/libnvrtc.so.12" \
  "$cublas/lib/libcublas.so.12" /usr/lib/x86_64-linux-gnu/libcuda.so.535.309.01 \
  -Wl,-rpath,"$cuda_runtime/lib" -Wl,-rpath,"$cuda_nvrtc/lib" \
  -Wl,-rpath,"$cublas/lib" -ldl -o "$project_dir/build/cpp/solidattention-p1-3c0"
echo "built $project_dir/build/cpp/solidattention-p1-3c0"
