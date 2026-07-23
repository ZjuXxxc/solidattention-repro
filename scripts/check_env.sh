#!/usr/bin/env bash
set -u
echo "== OS / kernel (io_uring needs a modern kernel) =="
uname -a
echo "== GPU / driver =="
nvidia-smi || true
echo "== CUDA compiler (optional until compiling CUDA sources) =="
nvcc --version || echo "nvcc: absent"
echo "== Memory =="
free -h
echo "== NVMe =="
lsblk -o NAME,MODEL,TRAN,SIZE,ROTA,MOUNTPOINTS,FSTYPE
echo "== io_uring policy (0 means enabled) =="
cat /proc/sys/kernel/io_uring_disabled
echo "== Observability tools =="
for tool in iostat pidstat perf fio nsys nvtop; do command -v "$tool" || echo "$tool: absent"; done
