#!/usr/bin/env python3
import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM

root = Path(__file__).resolve().parents[1]
out = root / "artifacts/qwen-lm-head"
out.mkdir(parents=True, exist_ok=True)
model = AutoModelForCausalLM.from_pretrained(
    root / "models/Qwen3-0.6B", local_files_only=True,
    torch_dtype=torch.bfloat16).eval()
norm = model.model.norm.weight.detach().float().contiguous()
head = model.lm_head.weight.detach().float().contiguous()
(out / "final_norm.f32").write_bytes(norm.numpy().tobytes())
(out / "lm_head.f32").write_bytes(head.numpy().tobytes())
(out / "manifest.json").write_text(json.dumps({
    "version": "P1.3c.0-qwen-lm-head", "hidden": head.shape[1],
    "vocab": head.shape[0], "dtype": "float32",
    "tied_embeddings": bool(model.config.tie_word_embeddings),
}, indent=2) + "\n")
print(out)
