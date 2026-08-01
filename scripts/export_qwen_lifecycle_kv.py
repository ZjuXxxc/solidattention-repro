#!/usr/bin/env python3
import argparse, hashlib, json, os
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM

parser = argparse.ArgumentParser()
parser.add_argument("--tokens", type=int, default=544)
parser.add_argument("--model", type=Path, default=Path("models/Qwen3-0.6B"))
parser.add_argument("--output", type=Path, default=Path("artifacts/qwen-lifecycle-kv"))
args = parser.parse_args()
model = AutoModelForCausalLM.from_pretrained(
    args.model, local_files_only=True, torch_dtype=torch.bfloat16,
    attn_implementation="eager").cuda().eval()
ids = (torch.arange(args.tokens, device="cuda") % (model.config.vocab_size - 1000) + 1000).unsqueeze(0)
with torch.inference_mode():
    cache = model.model(input_ids=ids, use_cache=True, return_dict=True).past_key_values
args.output.mkdir(parents=True, exist_ok=True)
path = args.output / "layer-major-token-interleaved-kv-fp16.bin"
digest = hashlib.sha256()
with path.open("wb") as stream:
    for key, value in cache:
        packed = torch.stack((key[0].permute(1, 0, 2), value[0].permute(1, 0, 2)), 1)
        raw = packed.half().contiguous().cpu().numpy().tobytes()
        stream.write(raw); digest.update(raw)
(args.output / "manifest.json").write_text(json.dumps({
    "version": "P1.3b.1-real-qwen-projected-kv", "model": model.config._name_or_path,
    "tokens": args.tokens, "layers": len(cache), "kv_heads": model.config.num_key_value_heads,
    "head_dim": model.config.head_dim, "layout": "layer,token,K/V,kv_head,head_dim",
    "dtype": "float16", "bytes": path.stat().st_size, "sha256": digest.hexdigest(),
}, indent=2) + "\n")
print(path)
