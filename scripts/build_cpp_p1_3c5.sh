#!/usr/bin/env bash
set -euo pipefail
p="$(cd "$(dirname "${BASH_SOURCE[0]}")/.."&&pwd)"; c=("$p/.venv/bin/python" -m ziglang c++)
"${c[@]}" -std=c++20 -O2 -I"$p/cpp/include" -I"$p/vendor/liburing/usr/include" "$p/cpp/src/merged_selection_main.cpp" "$p/cpp/src/selection.cpp" "$p/cpp/src/attention_reference.cpp" "$p/cpp/src/uring_reader.cpp" "$p/vendor/liburing/usr/lib/x86_64-linux-gnu/liburing.a" -o "$p/build/cpp/solidattention-p1-3c5"
