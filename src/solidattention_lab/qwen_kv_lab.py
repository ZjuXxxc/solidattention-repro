"""Real Qwen3 KV-cache SSD experiment inspired by SolidAttention.

The model produces real post-RoPE K/V tensors. They are persisted in a
token-major interleaved layout, selected in blocks, read with pread into DRAM,
copied to VRAM, and consumed by a real GPU attention calculation.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

from .trace import Event, write_trace


def cache_layer_count(cache) -> int:
    return len(cache.layers) if hasattr(cache, "layers") else len(cache.key_cache)


def cache_kv(cache, layer: int):
    if hasattr(cache, "layers"):
        return cache.layers[layer].keys, cache.layers[layer].values
    return cache.key_cache[layer], cache.value_cache[layer]


def timed(events: list[Event], origin: float, name: str, lane: str, layer: int, fn):
    start = time.perf_counter()
    result = fn()
    if lane in {"GPU compute", "PCIe H2D"}:
        torch.cuda.synchronize()
    end = time.perf_counter()
    events.append(Event(name, "qwen-real", (start-origin)*1e6, (end-start)*1e6,
                        2, lane, {"layer": layer}))
    return result


def repeat_prompt(tokenizer, target_tokens: int) -> torch.Tensor:
    unit = "SolidAttention把KV缓存放入SSD，并通过预取重叠I/O与GPU计算。"
    ids = tokenizer(unit, add_special_tokens=False).input_ids
    repeated = (ids * math.ceil(target_tokens / len(ids)))[:target_tokens]
    return torch.tensor([repeated], dtype=torch.long, device="cuda")


def capture_queries(model, input_ids: torch.Tensor):
    captured: dict[int, torch.Tensor] = {}
    handles = []
    for index, layer in enumerate(model.model.layers):
        attention = layer.self_attn

        def hook(module, args, kwargs, layer_index=index):
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            cos, sin = kwargs["position_embeddings"]
            shape = (*hidden.shape[:-1], -1, module.head_dim)
            query = module.q_norm(module.q_proj(hidden).view(shape)).transpose(1, 2)
            # A dummy key is sufficient; apply_rotary applies the same RoPE to Q.
            dummy = query[:, :module.config.num_key_value_heads]
            query, _ = apply_rotary_pos_emb(query, dummy, cos, sin)
            captured[layer_index] = query[:, :, -1:, :].detach()

        handles.append(attention.register_forward_pre_hook(hook, with_kwargs=True))
    cache = DynamicCache()
    with torch.inference_mode():
        model(input_ids=input_ids, past_key_values=cache, use_cache=True)
    for handle in handles:
        handle.remove()
    return cache, captured


def write_store(cache, path: Path, block_tokens: int) -> dict:
    """Write [token, K-or-V, kv_head, head_dim] BF16 bytes per layer."""
    path.parent.mkdir(parents=True, exist_ok=True)
    layers, offset = [], 0
    with path.open("wb", buffering=0) as stream:
        for layer in range(cache_layer_count(cache)):
            # DynamicCache layer API is intentionally handled below for version clarity.
            key_tensor, value_tensor = cache_kv(cache, layer)
            interleaved = torch.stack((key_tensor[0], value_tensor[0]), dim=2)
            interleaved = interleaved.permute(1, 2, 0, 3).contiguous().cpu()
            raw = interleaved.view(torch.uint8).numpy().tobytes()
            stream.write(raw)
            layers.append({"layer": layer, "offset": offset, "nbytes": len(raw),
                           "tokens": key_tensor.shape[2], "kv_heads": key_tensor.shape[1],
                           "head_dim": key_tensor.shape[3], "block_tokens": block_tokens})
            offset += len(raw)
        os.fsync(stream.fileno())
    cache_dtype = str(cache_kv(cache, 0)[0].dtype).removeprefix("torch.")
    return {"dtype": cache_dtype, "layout": "token,K_or_V,kv_head,head_dim",
            "layers": layers, "total_bytes": offset}


def read_blocks(fd: int, info: dict, block_ids: list[int]) -> np.ndarray:
    bytes_per_token = 2 * info["kv_heads"] * info["head_dim"] * 2
    chunks = []
    for block in block_ids:
        token_start = block * info["block_tokens"]
        count = min(info["block_tokens"], info["tokens"] - token_start)
        chunks.append(os.pread(fd, count * bytes_per_token,
                               info["offset"] + token_start * bytes_per_token))
    return np.frombuffer(b"".join(chunks), dtype=np.uint16).copy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("models/Qwen3-0.6B"))
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--layer", type=int, default=14)
    parser.add_argument("--block-tokens", type=int, default=32)
    parser.add_argument("--budget-tokens", type=int, default=512)
    parser.add_argument("--store", type=Path, default=Path("artifacts/qwen-kv.bin"))
    parser.add_argument("--trace", type=Path, default=Path("artifacts/qwen-real-trace.json"))
    parser.add_argument("--cold-io", action=argparse.BooleanOptionalAction, default=True,
                        help="ask Linux to evict the KV file from page cache before pread")
    args = parser.parse_args()
    if args.tokens % args.block_tokens:
        raise SystemExit("--tokens must be divisible by --block-tokens")

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, dtype=torch.bfloat16,
        attn_implementation="eager",
    ).cuda().eval()
    input_ids = repeat_prompt(tokenizer, args.tokens)
    torch.cuda.reset_peak_memory_stats()
    origin, events = time.perf_counter(), []
    cache, queries = timed(events, origin, "Qwen prefill + capture Q/K/V", "GPU compute",
                           -1, lambda: capture_queries(model, input_ids))
    metadata = timed(events, origin, "interleaved KV store", "SSD write", -1,
                     lambda: write_store(cache, args.store, args.block_tokens))

    info = metadata["layers"][args.layer]
    key, value = cache_kv(cache, args.layer)
    query = queries[args.layer]
    blocks = args.tokens // args.block_tokens
    # Mean post-RoPE key is the block representative. Group Q heads by KV head.
    representatives = key[0].view(info["kv_heads"], blocks, args.block_tokens, info["head_dim"]).mean(2)
    grouped_q = query[0, :, 0].view(info["kv_heads"], -1, info["head_dim"]).mean(1)
    scores = torch.einsum("hd,hbd->b", grouped_q.float(), representatives.float())
    selected_count = args.budget_tokens // args.block_tokens
    if selected_count < 2:
        raise SystemExit("--budget-tokens must fit at least init and local blocks")
    mandatory = {0, blocks-1}  # attention sink + local block
    ranked = torch.argsort(scores, descending=True).cpu().tolist()
    chosen = set(mandatory)
    for block in ranked:
        if len(chosen) == min(selected_count, blocks):
            break
        chosen.add(block)
    chosen = sorted(chosen)

    groups = model.config.num_attention_heads // model.config.num_key_value_heads

    def attention(q, k, v):
        k = k.repeat_interleave(groups, dim=1)
        v = v.repeat_interleave(groups, dim=1)
        weights = torch.softmax(torch.matmul(q.float(), k.float().transpose(-1, -2)) /
                                math.sqrt(q.shape[-1]), dim=-1)
        return torch.matmul(weights, v.float())

    dense = timed(events, origin, "dense attention reference", "GPU compute", args.layer,
                  lambda: attention(query, key, value)).cpu()
    # This is the actual tier transition: full KV leaves VRAM after persistence.
    del key, value, representatives, grouped_q, scores, cache
    torch.cuda.empty_cache()
    vram_after_offload = torch.cuda.memory_allocated()/2**20

    fd = os.open(args.store, os.O_RDONLY)
    try:
        if args.cold_io and hasattr(os, "posix_fadvise"):
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        raw = timed(events, origin, "pread selected KV blocks", "SSD read", args.layer,
                    lambda: read_blocks(fd, info, chosen))
    finally:
        os.close(fd)
    shape = (len(chosen)*args.block_tokens, 2, info["kv_heads"], info["head_dim"])
    stored_dtype = torch.float16 if metadata["dtype"] == "float16" else torch.bfloat16
    host = torch.from_numpy(raw).view(stored_dtype).reshape(shape)
    device = timed(events, origin, "DRAM → VRAM selected KV", "PCIe H2D", args.layer,
                   lambda: host.cuda(non_blocking=False))
    selected_k = device[:, 0].permute(1, 0, 2).unsqueeze(0)
    selected_v = device[:, 1].permute(1, 0, 2).unsqueeze(0)
    sparse = timed(events, origin, "SSD-backed sparse attention", "GPU compute", args.layer,
                   lambda: attention(query, selected_k, selected_v)).cpu()
    cosine = torch.nn.functional.cosine_similarity(dense.flatten(), sparse.flatten(), dim=0).item()
    report = {"model": str(args.model.resolve()), "tokens": args.tokens, "layer": args.layer,
              "block_tokens": args.block_tokens, "selected_blocks": len(chosen),
              "selected_tokens": len(chosen)*args.block_tokens, "selected_block_ids": chosen,
              "kv_store_mib": metadata["total_bytes"]/2**20,
              "full_layer_kv_mib": info["nbytes"]/2**20,
              "loaded_layer_kv_mib": host.nbytes/2**20,
              "dense_sparse_cosine": cosine,
              "vram_after_full_kv_offload_mib": vram_after_offload,
              "peak_vram_mib": torch.cuda.max_memory_allocated()/2**20,
              "cold_io_requested": args.cold_io,
              "warning": "research scaffold: layer attention replay, not end-to-end sparse decode"}
    write_trace(args.trace, events, report)
    args.store.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
