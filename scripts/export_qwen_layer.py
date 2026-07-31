#!/usr/bin/env python3
"""Export one real Qwen3 layer and FP32 teacher intermediates for native parity."""

from __future__ import annotations

import argparse
import json
import math
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

    args.output.mkdir(parents=True, exist_ok=True)
    tensors: dict[str, dict[str, object]] = {}
    for name, tensor in weights.items():
        tensors[name] = save_tensor(args.output, name, tensor)
    for name, tensor in {
        "hidden": hidden,
        "normalized": normalized,
        "query": query,
        "key": key,
        "value": value.view(args.tokens, kv_heads, head_dim),
        "decode_query": decode_query,
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
    }.items():
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
        "tensors": tensors,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"manifest={args.output / 'manifest.json'}")
    print(f"layer_output_norm={layer_output.norm().item():.9f}")
    print(f"attention_probability_sum_min={probability.sum(-1).min().item():.9f}")
    print(f"attention_probability_sum_max={probability.sum(-1).max().item():.9f}")


if __name__ == "__main__":
    main()
