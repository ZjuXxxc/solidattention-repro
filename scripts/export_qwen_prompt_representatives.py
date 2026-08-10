#!/usr/bin/env python3
"""Rebuild real 512-token InfLLM representatives from exported Qwen tensors."""
from pathlib import Path
import json
import sys

import numpy as np
import torch

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))
from solidattention_lab.selection import local_causal_token_scores


def rms(value: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return value * torch.rsqrt(value.square().mean(-1, keepdim=True) + eps) * weight


def rope(value: torch.Tensor, positions: torch.Tensor, theta: float = 1_000_000.0) -> torch.Tensor:
    width = value.shape[-1]
    inv = 1.0 / theta ** (torch.arange(0, width, 2, device=value.device).float() / width)
    angle = positions.float().unsqueeze(1) * inv.unsqueeze(0)
    cosine = torch.cat((angle.cos(), angle.cos()), -1).unsqueeze(1)
    sine = torch.cat((angle.sin(), angle.sin()), -1).unsqueeze(1)
    half = width // 2
    return value * cosine + torch.cat((-value[..., half:], value[..., :half]), -1) * sine


fixture = root / "artifacts/qwen-28-layers"
output = fixture / "prompt-representatives.f32"
indices_output = fixture / "prompt-representative-indices.i32"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
kv = np.memmap(fixture / "all-layers-kv-fp16.bin", dtype=np.float16, mode="r", shape=(28, 512, 2, 8, 128))
representatives, all_indices = [], []
positions = torch.arange(512, device=device)
with torch.inference_mode():
    for layer in range(28):
        directory = fixture / f"layer-{layer:02d}"
        hidden = torch.from_numpy(np.fromfile(directory / "hidden.f32", np.float32).reshape(512, 1024)).to(device)
        norm = torch.from_numpy(np.fromfile(directory / "input_norm_weight.f32", np.float32)).to(device)
        q_weight = torch.from_numpy(np.fromfile(directory / "q_weight.f32", np.float32).reshape(2048, 1024)).to(device)
        q_norm = torch.from_numpy(np.fromfile(directory / "q_norm_weight.f32", np.float32)).to(device)
        query = (rms(hidden, norm) @ q_weight.T).view(512, 16, 128)
        query = rms(query, q_norm)
        query = rope(query, positions).permute(1, 0, 2)
        key = torch.from_numpy(np.array(kv[layer, :, 0], copy=True)).to(device).permute(1, 0, 2)
        scores = local_causal_token_scores(query, key, 32, 64)
        within = scores.view(16, 16, 32).topk(4, dim=-1).indices
        indices = within + torch.arange(16, device=device).view(1, 16, 1) * 32
        expanded = key.repeat_interleave(2, dim=0)
        selected = expanded[torch.arange(16, device=device).view(16, 1, 1), indices]
        representatives.append(selected.float().mean(2).cpu().numpy())
        all_indices.append(indices.cpu().numpy().astype(np.int32))
        print(f"layer={layer} representatives=16x16x128")
np.stack(representatives).astype(np.float32).tofile(output)
np.stack(all_indices).tofile(indices_output)
(fixture / "prompt-representatives.json").write_text(json.dumps({
    "method": "InfLLM local-causal query-head mass",
    "shape": [28, 16, 16, 128], "repr_topk": 4, "local_window": 32,
    "values": str(output.relative_to(root)), "indices": str(indices_output.relative_to(root)),
}, indent=2) + "\n")
