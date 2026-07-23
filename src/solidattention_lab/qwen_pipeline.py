"""All-layer real-I/O pipeline: overlap layer L GPU attention with L+1 SSD read."""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .qwen_kv_lab import cache_kv, cache_layer_count, capture_queries, read_blocks, repeat_prompt, write_store
from .trace import Event, write_trace


def choose_blocks(key: torch.Tensor, query: torch.Tensor, block_tokens: int,
                  budget_tokens: int) -> list[int]:
    kv_heads, tokens, head_dim = key.shape[1:]
    blocks = tokens // block_tokens
    representatives = key[0].view(kv_heads, blocks, block_tokens, head_dim).mean(2)
    grouped_q = query[0, :, 0].view(kv_heads, -1, head_dim).mean(1)
    scores = torch.einsum("hd,hbd->b", grouped_q.float(), representatives.float())
    target = budget_tokens // block_tokens
    chosen = {0, blocks - 1}
    for block in torch.argsort(scores, descending=True).cpu().tolist():
        if len(chosen) == min(target, blocks):
            break
        chosen.add(block)
    return sorted(chosen)


def gpu_attention(query: torch.Tensor, device: torch.Tensor, q_heads: int, kv_heads: int):
    key = device[:, 0].permute(1, 0, 2).unsqueeze(0)
    value = device[:, 1].permute(1, 0, 2).unsqueeze(0)
    groups = q_heads // kv_heads
    key = key.repeat_interleave(groups, dim=1)
    value = value.repeat_interleave(groups, dim=1)
    weights = torch.softmax(torch.matmul(query.float(), key.float().transpose(-1, -2)) /
                            math.sqrt(query.shape[-1]), dim=-1)
    return torch.matmul(weights, value.float())


def replay(store: Path, metadata: dict, queries: dict[int, torch.Tensor], selections: dict[int, list[int]],
           overlap: bool, cold_io: bool, q_heads: int, origin: float) -> tuple[list[Event], float]:
    events: list[Event] = []
    fd = os.open(store, os.O_RDONLY)
    if cold_io and hasattr(os, "posix_fadvise"):
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)

    def load(layer: int):
        start = time.perf_counter()
        host = read_blocks(fd, metadata["layers"][layer], selections[layer])
        end = time.perf_counter()
        events.append(Event("prefetch selected blocks", "qwen-pipeline", (start-origin)*1e6,
                            (end-start)*1e6, 3, "SSD read", {"layer": layer,
                            "blocks": len(selections[layer])}))
        return host

    started = time.perf_counter()
    outputs = []
    try:
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="kv-prefetch") as executor:
            future = executor.submit(load, 0)
            for layer, info in enumerate(metadata["layers"]):
                host_array = future.result()
                shape = (len(selections[layer]) * info["block_tokens"], 2,
                         info["kv_heads"], info["head_dim"])
                host = torch.from_numpy(host_array).view(torch.bfloat16).reshape(shape)
                copy_start = time.perf_counter()
                device = host.cuda(non_blocking=False)
                torch.cuda.synchronize()
                copy_end = time.perf_counter()
                events.append(Event("DRAM → VRAM", "qwen-pipeline", (copy_start-origin)*1e6,
                                    (copy_end-copy_start)*1e6, 3, "PCIe H2D",
                                    {"layer": layer, "bytes": host.nbytes}))

                # Issuing the next read before this layer's GPU work creates the overlap.
                if overlap and layer + 1 < len(metadata["layers"]):
                    future = executor.submit(load, layer + 1)
                compute_start = time.perf_counter()
                output = gpu_attention(queries[layer], device, q_heads, info["kv_heads"])
                torch.cuda.synchronize()
                compute_end = time.perf_counter()
                events.append(Event("SSD-backed sparse attention", "qwen-pipeline",
                                    (compute_start-origin)*1e6, (compute_end-compute_start)*1e6,
                                    3, "GPU compute", {"layer": layer}))
                outputs.append(float(output.norm().cpu()))
                del device, host, output
                if not overlap and layer + 1 < len(metadata["layers"]):
                    future = executor.submit(load, layer + 1)
    finally:
        os.close(fd)
    torch.cuda.synchronize()
    return events, time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("models/Qwen3-0.6B"))
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--block-tokens", type=int, default=32)
    parser.add_argument("--budget-tokens", type=int, default=512)
    parser.add_argument("--store", type=Path, default=Path("artifacts/qwen-pipeline-kv.bin"))
    parser.add_argument("--output-prefix", type=Path, default=Path("artifacts/qwen-pipeline"))
    parser.add_argument("--cold-io", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.tokens % args.block_tokens or args.budget_tokens < 2 * args.block_tokens:
        raise SystemExit("token count must align and budget must fit init/local blocks")

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, dtype=torch.bfloat16, attn_implementation="eager"
    ).cuda().eval()
    input_ids = repeat_prompt(tokenizer, args.tokens)
    cache, queries = capture_queries(model, input_ids)
    metadata = write_store(cache, args.store, args.block_tokens)
    selections = {layer: choose_blocks(cache_kv(cache, layer)[0], queries[layer],
                                       args.block_tokens, args.budget_tokens)
                  for layer in range(cache_layer_count(cache))}
    del cache
    torch.cuda.empty_cache()

    results = {}
    for name, overlap in (("serial", False), ("overlap", True)):
        origin = time.perf_counter()
        events, elapsed = replay(args.store, metadata, queries, selections, overlap,
                                 args.cold_io, model.config.num_attention_heads, origin)
        trace_path = Path(f"{args.output_prefix}-{name}.json")
        run = {"mode": name, "elapsed_ms": elapsed*1000, "tokens": args.tokens,
               "block_tokens": args.block_tokens, "budget_tokens": args.budget_tokens,
               "layers": len(metadata["layers"]), "cold_io_requested": args.cold_io}
        write_trace(trace_path, events, run)
        results[name] = run
    results["speedup"] = results["serial"]["elapsed_ms"] / results["overlap"]["elapsed_ms"]
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
