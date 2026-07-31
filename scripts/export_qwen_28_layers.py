#!/usr/bin/env python3
"""Stream 28 distinct Qwen3 layer bundles, shared FP16 KV, and sparse teachers."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM


def rms(value: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return value * torch.rsqrt(value.square().mean(-1, keepdim=True) + eps) * weight


def rope(value: torch.Tensor, positions: torch.Tensor, theta: float) -> torch.Tensor:
    width = value.shape[-1]
    inv = 1.0 / theta ** (torch.arange(0, width, 2, device=value.device).float() / width)
    angle = positions.float().unsqueeze(1) * inv.unsqueeze(0)
    cosine = torch.cat((angle.cos(), angle.cos()), -1).unsqueeze(1)
    sine = torch.cat((angle.sin(), angle.sin()), -1).unsqueeze(1)
    half = width // 2
    rotated = torch.cat((-value[..., half:], value[..., :half]), -1)
    return value * cosine + rotated * sine


def save(directory: Path, name: str, tensor: torch.Tensor) -> dict[str, object]:
    value = tensor.detach().float().contiguous().cpu()
    path = directory / f"{name}.f32"
    raw = value.numpy().tobytes()
    path.write_bytes(raw)
    return {
        "file": path.name,
        "shape": list(value.shape),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("models/Qwen3-0.6B"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/qwen-28-layers"))
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--max-layers", type=int, default=28)
    args = parser.parse_args()
    if args.tokens != 512:
        raise SystemExit("the P1.2d manifest currently fixes 512 prompt tokens")
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    from solidattention_lab.selection import local_causal_token_scores, select_blocks

    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    ).cuda().eval()
    token_ids = torch.arange(1000, 1000 + args.tokens, device="cuda").unsqueeze(0)
    with torch.inference_mode():
        output = model.model(
            input_ids=token_ids, use_cache=False, output_hidden_states=True,
            return_dict=True,
        )
    hidden_states = output.hidden_states
    config = model.config
    eps = float(config.rms_norm_eps)
    q_heads, kv_heads = config.num_attention_heads, config.num_key_value_heads
    head_dim = config.head_dim
    groups = q_heads // kv_heads
    positions = torch.arange(args.tokens, device="cuda")
    args.output.mkdir(parents=True, exist_ok=True)
    kv_path = args.output / "all-layers-kv-fp16.bin"
    manifest_layers = []
    offset = 0
    chain_hidden = hidden_states[0][0, -1].float()
    with kv_path.open("wb") as kv_stream, torch.inference_mode():
        for layer_index, layer in enumerate(model.model.layers[: args.max_layers]):
            directory = args.output / f"layer-{layer_index:02d}"
            directory.mkdir(exist_ok=True)
            hidden = hidden_states[layer_index][0].float()
            attention = layer.self_attn
            weights = {
                "input_norm_weight": layer.input_layernorm.weight.float(),
                "q_weight": attention.q_proj.weight.float(),
                "k_weight": attention.k_proj.weight.float(),
                "v_weight": attention.v_proj.weight.float(),
                "q_norm_weight": attention.q_norm.weight.float(),
                "k_norm_weight": attention.k_norm.weight.float(),
                "o_weight": attention.o_proj.weight.float(),
                "post_norm_weight": layer.post_attention_layernorm.weight.float(),
                "gate_weight": layer.mlp.gate_proj.weight.float(),
                "up_weight": layer.mlp.up_proj.weight.float(),
                "down_weight": layer.mlp.down_proj.weight.float(),
            }
            normalized = rms(hidden, weights["input_norm_weight"], eps)
            query = (normalized @ weights["q_weight"].T).view(args.tokens, q_heads, head_dim)
            key = (normalized @ weights["k_weight"].T).view(args.tokens, kv_heads, head_dim)
            value = (normalized @ weights["v_weight"].T).view(args.tokens, kv_heads, head_dim)
            query = rms(query, weights["q_norm_weight"], eps)
            key = rms(key, weights["k_norm_weight"], eps)
            query, key = rope(query, positions, config.rope_theta), rope(key, positions, config.rope_theta)
            # The SSD contract is FP16 KV. Selection and the sparse teacher must
            # consume the same quantized values as the native executor.
            stored_key = key.half().float()
            stored_value = value.half().float()
            scores = local_causal_token_scores(
                query.permute(1, 0, 2), stored_key.permute(1, 0, 2), 32, 64
            )
            within = scores.view(q_heads, 16, 32).topk(4, -1).indices
            indices = within + torch.arange(16, device="cuda").view(1, 16, 1) * 32
            expanded_key = stored_key.permute(1, 0, 2).repeat_interleave(groups, 0)
            representatives = expanded_key[
                torch.arange(q_heads, device="cuda").view(q_heads, 1, 1), indices
            ].mean(2)
            selected = select_blocks(
                query[-1].view(1, q_heads, 1, head_dim), representatives, 4, 1, 1
            )
            selected_tokens = torch.cat([
                torch.arange(block * 32, (block + 1) * 32, device="cuda")
                for block in selected
            ])
            sparse_key = stored_key[selected_tokens]
            sparse_value = stored_value[selected_tokens]
            expanded_sparse_key = sparse_key.repeat_interleave(groups, 1)
            expanded_sparse_value = sparse_value.repeat_interleave(groups, 1)
            probability = torch.softmax(
                torch.einsum("hd,thd->ht", query[-1], expanded_sparse_key) /
                math.sqrt(head_dim), -1,
            )
            attended = torch.einsum("ht,thd->hd", probability, expanded_sparse_value).reshape(-1)
            attention_output = attended @ weights["o_weight"].T
            attention_residual = hidden[-1] + attention_output
            post_normalized = rms(attention_residual, weights["post_norm_weight"], eps)
            gate = post_normalized @ weights["gate_weight"].T
            up = post_normalized @ weights["up_weight"].T
            activated = torch.nn.functional.silu(gate) * up
            mlp_output = activated @ weights["down_weight"].T
            sparse_output = attention_residual + mlp_output

            dense_expanded_key = key.repeat_interleave(groups, 1)
            dense_expanded_value = value.repeat_interleave(groups, 1)
            dense_probability = torch.softmax(
                torch.einsum("hd,thd->ht", query[-1], dense_expanded_key) /
                math.sqrt(head_dim), -1,
            )
            dense_attended = torch.einsum(
                "ht,thd->hd", dense_probability, dense_expanded_value
            ).reshape(-1)
            dense_attention_residual = hidden[-1] + dense_attended @ weights["o_weight"].T
            dense_post_normalized = rms(
                dense_attention_residual, weights["post_norm_weight"], eps
            )
            dense_gate = dense_post_normalized @ weights["gate_weight"].T
            dense_up = dense_post_normalized @ weights["up_weight"].T
            dense_mlp = (
                torch.nn.functional.silu(dense_gate) * dense_up
            ) @ weights["down_weight"].T
            dense_layer_output = dense_attention_residual + dense_mlp

            chain_input = chain_hidden
            chain_normalized = rms(chain_input, weights["input_norm_weight"], eps)
            chain_query = (chain_normalized @ weights["q_weight"].T).view(
                q_heads, head_dim
            )
            chain_query = rms(chain_query, weights["q_norm_weight"], eps)
            chain_query = rope(
                chain_query.unsqueeze(0), positions[-1:], config.rope_theta
            )[0]
            chain_selected = select_blocks(
                chain_query.view(1, q_heads, 1, head_dim), representatives, 4, 1, 1
            )
            chain_tokens = torch.cat([
                torch.arange(block * 32, (block + 1) * 32, device="cuda")
                for block in chain_selected
            ])
            chain_key = stored_key[chain_tokens].repeat_interleave(groups, 1)
            chain_value = stored_value[chain_tokens].repeat_interleave(groups, 1)
            chain_probability = torch.softmax(
                torch.einsum("hd,thd->ht", chain_query, chain_key) /
                math.sqrt(head_dim), -1,
            )
            chain_attended = torch.einsum(
                "ht,thd->hd", chain_probability, chain_value
            ).reshape(-1)
            chain_attention_output = chain_attended @ weights["o_weight"].T
            chain_attention_residual = chain_input + chain_attention_output
            chain_post_normalized = rms(
                chain_attention_residual, weights["post_norm_weight"], eps
            )
            chain_gate = chain_post_normalized @ weights["gate_weight"].T
            chain_up = chain_post_normalized @ weights["up_weight"].T
            chain_activated = torch.nn.functional.silu(chain_gate) * chain_up
            chain_mlp_output = chain_activated @ weights["down_weight"].T
            chain_output = chain_attention_residual + chain_mlp_output
            chain_hidden = chain_output

            tensors = {name: save(directory, name, tensor) for name, tensor in weights.items()}
            for name, tensor in {
                "hidden": hidden,
                "decode_normalized": normalized[-1],
                "decode_query": query[-1],
                "sparse_attended": attended,
                "sparse_attention_output": attention_output,
                "sparse_attention_residual": attention_residual,
                "sparse_post_normalized": post_normalized,
                "sparse_gate": gate,
                "sparse_up": up,
                "sparse_activated": activated,
                "sparse_mlp_output": mlp_output,
                "sparse_layer_output": sparse_output,
                "layer_output": dense_layer_output,
                "chain_input": chain_input,
                "chain_decode_normalized": chain_normalized,
                "chain_decode_query": chain_query,
                "chain_sparse_attended": chain_attended,
                "chain_sparse_attention_output": chain_attention_output,
                "chain_sparse_attention_residual": chain_attention_residual,
                "chain_sparse_post_normalized": chain_post_normalized,
                "chain_sparse_gate": chain_gate,
                "chain_sparse_up": chain_up,
                "chain_sparse_activated": chain_activated,
                "chain_sparse_mlp_output": chain_mlp_output,
                "chain_sparse_layer_output": chain_output,
            }.items():
                tensors[name] = save(directory, name, tensor)
            interleaved = torch.stack((stored_key, stored_value), 1).half().contiguous().cpu()
            raw = interleaved.view(torch.uint8).numpy().tobytes()
            kv_stream.write(raw)
            manifest_layers.append({
                "layer": layer_index,
                "directory": directory.name,
                "kv_offset": offset,
                "kv_bytes": len(raw),
                "selected_blocks": selected,
                "chain_selected_blocks": chain_selected,
                "sparse_teacher_output_sha256": tensors["sparse_layer_output"]["sha256"],
            })
            offset += len(raw)
            print(f"layer={layer_index} selected={selected} kv_offset={offset-len(raw)}")
            del weights, query, key, value, scores, representatives
            torch.cuda.empty_cache()
    manifest = {
        "version": "P1.2d-qwen-28-layer-stream-v1",
        "model": "Qwen/Qwen3-0.6B",
        "revision": "c1899de289a04d12100db370d81485cdf75e47ca",
        "tokens": args.tokens,
        "layers": manifest_layers,
        "kv_store": kv_path.name,
        "kv_dtype": "float16",
        "kv_layout": "layer,token,K_or_V,kv_head,head_dim",
        "weight_export_dtype": "float32",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"manifest={args.output / 'manifest.json'}")


if __name__ == "__main__":
    main()
