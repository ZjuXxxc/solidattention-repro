#!/usr/bin/env python3
"""Export one real Qwen3 layer and FP32 teacher intermediates for native parity."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from safetensors import safe_open


def rms_norm(value: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return value * torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + eps) * weight


def head_rms(value: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return value * torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + eps) * weight


def rope(value: torch.Tensor, positions: torch.Tensor, theta: float) -> torch.Tensor:
    head_dim = value.shape[-1]
    frequency = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    )
    angle = positions.float().unsqueeze(1) * frequency.unsqueeze(0)
    cosine = torch.cat((angle.cos(), angle.cos()), dim=-1).unsqueeze(1)
    sine = torch.cat((angle.sin(), angle.sin()), dim=-1).unsqueeze(1)
    half = head_dim // 2
    rotated = torch.cat((-value[..., half:], value[..., :half]), dim=-1)
    return value * cosine + rotated * sine


def save_tensor(directory: Path, name: str, value: torch.Tensor) -> dict[str, object]:
    contiguous = value.detach().float().contiguous().cpu()
    path = directory / f"{name}.f32"
    path.write_bytes(contiguous.numpy().tobytes())
    return {"file": path.name, "shape": list(contiguous.shape), "dtype": "float32"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("models/Qwen3-0.6B"))
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--position-start", type=int, default=384)
    parser.add_argument("--output", type=Path, default=Path("artifacts/qwen-layer0"))
    parser.add_argument("--sparse-export", action="store_true")
    args = parser.parse_args()

    config = json.loads((args.model / "config.json").read_text())
    shard = args.model / "model.safetensors"
    prefix = f"model.layers.{args.layer}."
    names = {
        "input_norm_weight": prefix + "input_layernorm.weight",
        "q_weight": prefix + "self_attn.q_proj.weight",
        "k_weight": prefix + "self_attn.k_proj.weight",
        "v_weight": prefix + "self_attn.v_proj.weight",
        "q_norm_weight": prefix + "self_attn.q_norm.weight",
        "k_norm_weight": prefix + "self_attn.k_norm.weight",
        "o_weight": prefix + "self_attn.o_proj.weight",
        "post_norm_weight": prefix + "post_attention_layernorm.weight",
        "gate_weight": prefix + "mlp.gate_proj.weight",
        "up_weight": prefix + "mlp.up_proj.weight",
        "down_weight": prefix + "mlp.down_proj.weight",
    }
    with safe_open(shard, framework="pt", device="cpu") as source:
        weights = {name: source.get_tensor(key).float() for name, key in names.items()}
        embeddings = source.get_slice("model.embed_tokens.weight")
        token_ids = torch.arange(1000, 1000 + args.tokens, dtype=torch.long)
        hidden = torch.stack([embeddings[token_id].float() for token_id in token_ids])

    eps = float(config["rms_norm_eps"])
    q_heads = int(config["num_attention_heads"])
    kv_heads = int(config["num_key_value_heads"])
    head_dim = int(config["head_dim"])
    positions = torch.arange(
        args.position_start, args.position_start + args.tokens, dtype=torch.long
    )
    normalized = rms_norm(hidden, weights["input_norm_weight"], eps)
    query = normalized @ weights["q_weight"].T
    key = normalized @ weights["k_weight"].T
    value = normalized @ weights["v_weight"].T
    query = head_rms(query.view(args.tokens, q_heads, head_dim),
                     weights["q_norm_weight"], eps)
    key = head_rms(key.view(args.tokens, kv_heads, head_dim),
                   weights["k_norm_weight"], eps)
    query = rope(query, positions, float(config["rope_theta"]))
    key = rope(key, positions, float(config["rope_theta"]))

    decode_query = query[-1]
    expanded_key = key.repeat_interleave(q_heads // kv_heads, dim=1)
    expanded_value = value.view(args.tokens, kv_heads, head_dim).repeat_interleave(
        q_heads // kv_heads, dim=1
    )
    logits = torch.einsum("hd,thd->ht", decode_query, expanded_key)
    probability = torch.softmax(logits / math.sqrt(head_dim), dim=-1)
    attended_heads = torch.einsum("ht,thd->hd", probability, expanded_value)
    attended = attended_heads.reshape(-1)
    attention_output = attended @ weights["o_weight"].T
    attention_residual = hidden[-1] + attention_output
    post_normalized = rms_norm(attention_residual, weights["post_norm_weight"], eps)
    gate = post_normalized @ weights["gate_weight"].T
    up = post_normalized @ weights["up_weight"].T
    activated = torch.nn.functional.silu(gate) * up
    mlp_output = activated @ weights["down_weight"].T
    layer_output = attention_residual + mlp_output

    sparse_tensors: dict[str, torch.Tensor] = {}
    selected_blocks: list[int] = []
    if args.sparse_export:
        if args.tokens != 512:
            raise SystemExit("--sparse-export currently requires --tokens 512")
        repository = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(repository / "src"))
        from solidattention_lab.selection import (  # noqa: PLC0415
            local_causal_token_scores,
            select_blocks,
        )

        query_by_head = query.permute(1, 0, 2).contiguous()
        key_by_head = key.permute(1, 0, 2).contiguous()
        scores = local_causal_token_scores(
            query_by_head, key_by_head, local_window=32, query_chunk=64
        )
        within = scores.view(q_heads, 16, 32).topk(4, dim=-1).indices
        block_base = torch.arange(16).view(1, 16, 1) * 32
        indices = within + block_base
        expanded_key = key_by_head.repeat_interleave(q_heads // kv_heads, dim=0)
        head_index = torch.arange(q_heads).view(q_heads, 1, 1)
        representatives = expanded_key[head_index, indices].mean(dim=2)
        selected_blocks = select_blocks(
            decode_query.view(1, q_heads, 1, head_dim),
            representatives,
            budget_blocks=4,
            init_blocks=1,
            local_blocks=1,
        )
        selected_tokens = torch.cat(
            [torch.arange(block * 32, (block + 1) * 32) for block in selected_blocks]
        )
        sparse_key = key[selected_tokens]
        sparse_value = value.view(args.tokens, kv_heads, head_dim)[selected_tokens]
        sparse_expanded_key = sparse_key.repeat_interleave(q_heads // kv_heads, dim=1)
        sparse_expanded_value = sparse_value.repeat_interleave(
            q_heads // kv_heads, dim=1
        )
        sparse_logits = torch.einsum("hd,thd->ht", decode_query, sparse_expanded_key)
        sparse_probability = torch.softmax(
            sparse_logits / math.sqrt(head_dim), dim=-1
        )
        sparse_attended = torch.einsum(
            "ht,thd->hd", sparse_probability, sparse_expanded_value
        ).reshape(-1)
        sparse_attention_output = sparse_attended @ weights["o_weight"].T
        sparse_attention_residual = hidden[-1] + sparse_attention_output
        sparse_post_normalized = rms_norm(
            sparse_attention_residual, weights["post_norm_weight"], eps
        )
        sparse_gate = sparse_post_normalized @ weights["gate_weight"].T
        sparse_up = sparse_post_normalized @ weights["up_weight"].T
        sparse_activated = torch.nn.functional.silu(sparse_gate) * sparse_up
        sparse_mlp_output = sparse_activated @ weights["down_weight"].T
        sparse_layer_output = sparse_attention_residual + sparse_mlp_output
        sparse_tensors = {
            "sparse_attended": sparse_attended,
            "sparse_attention_output": sparse_attention_output,
            "sparse_attention_residual": sparse_attention_residual,
            "sparse_post_normalized": sparse_post_normalized,
            "sparse_gate": sparse_gate,
            "sparse_up": sparse_up,
            "sparse_activated": sparse_activated,
            "sparse_mlp_output": sparse_mlp_output,
            "sparse_layer_output": sparse_layer_output,
        }
        interleaved = torch.stack(
            (key, value.view(args.tokens, kv_heads, head_dim)), dim=1
        ).to(torch.float16).contiguous()
        (args.output / "kv-store-fp16.bin").parent.mkdir(parents=True, exist_ok=True)
        (args.output / "kv-store-fp16.bin").write_bytes(
            interleaved.view(torch.uint8).numpy().tobytes()
        )

    args.output.mkdir(parents=True, exist_ok=True)
    tensors: dict[str, dict[str, object]] = {}
    for name, tensor in weights.items():
        tensors[name] = save_tensor(args.output, name, tensor)
    for name, tensor in ({
        "hidden": hidden,
        "normalized": normalized,
        "query": query,
        "key": key,
        "value": value.view(args.tokens, kv_heads, head_dim),
        "decode_query": decode_query,
        "decode_normalized": normalized[-1],
        "attention_probability": probability,
        "attended": attended,
        "attention_output": attention_output,
        "attention_residual": attention_residual,
        "post_normalized": post_normalized,
        "gate": gate,
        "up": up,
        "activated": activated,
        "mlp_output": mlp_output,
        "layer_output": layer_output,
    } | sparse_tensors).items():
        tensors[name] = save_tensor(args.output, name, tensor)
    manifest = {
        "version": "P1.2a-qwen-layer-export-v1",
        "model": "Qwen/Qwen3-0.6B",
        "layer": args.layer,
        "tokens": args.tokens,
        "position_start": args.position_start,
        "hidden_size": int(config["hidden_size"]),
        "intermediate_size": int(config["intermediate_size"]),
        "query_heads": q_heads,
        "kv_heads": kv_heads,
        "head_dim": head_dim,
        "rms_norm_eps": eps,
        "rope_theta": float(config["rope_theta"]),
        "source_dtype": str(config["torch_dtype"]),
        "export_dtype": "float32",
        "token_ids": [int(value) for value in token_ids],
        "sparse_export": args.sparse_export,
        "selected_blocks": selected_blocks,
        "kv_store": "kv-store-fp16.bin" if args.sparse_export else None,
        "kv_store_dtype": "float16" if args.sparse_export else None,
        "kv_store_layout": "token,K_or_V,kv_head,head_dim",
        "tensors": tensors,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"manifest={args.output / 'manifest.json'}")
    print(f"layer_output_norm={layer_output.norm().item():.9f}")
    print(f"attention_probability_sum_min={probability.sum(-1).min().item():.9f}")
    print(f"attention_probability_sum_max={probability.sum(-1).max().item():.9f}")


if __name__ == "__main__":
    main()
