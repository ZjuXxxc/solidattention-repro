"""Run a small, reproducible Qwen3 GPU baseline and report memory/timing."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("models/Qwen3-0.6B"))
    parser.add_argument("--prompt", default="用一句话解释什么是分层存储。")
    parser.add_argument("--new-tokens", type=int, default=32)
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA is required for this baseline"
    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    ).cuda().eval()
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False,
    )
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=args.new_tokens, do_sample=False)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    generated = output.shape[1] - inputs.input_ids.shape[1]
    report = {
        "model": str(args.model.resolve()), "input_tokens": inputs.input_ids.shape[1],
        "generated_tokens": generated, "elapsed_s": elapsed,
        "decode_tokens_per_s": generated / elapsed,
        "peak_vram_mib": torch.cuda.max_memory_allocated() / 2**20,
        "text": tokenizer.decode(output[0, inputs.input_ids.shape[1]:], skip_special_tokens=True),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
