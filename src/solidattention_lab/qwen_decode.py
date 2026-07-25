"""Continuous, free-running Qwen3 decode using sparse prompt KV loaded from SSD.

This is a Qwen3-specific research path. Prefill is dense. Every later token is
computed layer by layer; each layer selects prompt blocks using its current
post-RoPE query, reads them from the KV file, joins a resident decode tail, and
uses the resulting attention output to produce actual model logits.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, DynamicCache
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

from .qwen_kv_lab import cache_kv, cache_layer_count, read_blocks, write_store
from .io_uring_backend import IoUringDirectReader
from .selection import build_infllm_representatives, select_blocks
from .trace import Event, write_trace


def make_prompt(tokenizer, target_tokens: int) -> torch.Tensor:
    unit = "KV缓存可以在显存、内存和固态硬盘之间分层存储。"
    suffix = "请简短解释这种设计的优点。 /no_think"
    unit_ids = tokenizer(unit, add_special_tokens=False).input_ids
    suffix_ids = tokenizer(suffix, add_special_tokens=False).input_ids
    body = (unit_ids * math.ceil(max(0, target_tokens-len(suffix_ids))/len(unit_ids)))
    ids = (body + suffix_ids)[-target_tokens:]
    return torch.tensor([ids], dtype=torch.long, device="cuda")


def dense_decode(model, input_ids: torch.Tensor, new_tokens: int, capture_hidden: bool = False):
    cache = DynamicCache()
    generated, per_token_ms, step_logits = [], [], []
    decode_hidden = []
    torch.cuda.synchronize()
    with torch.inference_mode():
        output = model(
            input_ids=input_ids, past_key_values=cache, use_cache=True,
            output_hidden_states=capture_hidden,
        )
        token = output.logits[:, -1].argmax(-1, keepdim=True)
        generated.append(int(token))
        step_logits.append(output.logits[:, -1].float().cpu())
        for _ in range(new_tokens - 1):
            start = time.perf_counter()
            output = model(
                input_ids=token, past_key_values=cache, use_cache=True,
                output_hidden_states=capture_hidden,
            )
            if capture_hidden:
                decode_hidden.append(tuple(x.detach() for x in output.hidden_states))
            token = output.logits[:, -1].argmax(-1, keepdim=True)
            torch.cuda.synchronize()
            per_token_ms.append((time.perf_counter()-start)*1000)
            generated.append(int(token))
            step_logits.append(output.logits[:, -1].float().cpu())
    return generated, per_token_ms, step_logits, cache, decode_hidden


def audit_selection(
    dense_query, dense_key, chosen, prompt_tokens, block_tokens,
    init_blocks, local_blocks, budget_blocks,
):
    """Measure selected dense-attention mass and oracle dynamic-block recall."""
    groups = dense_query.shape[1] // dense_key.shape[1]
    expanded_key = dense_key.repeat_interleave(groups, dim=1)
    weights = torch.softmax(
        torch.matmul(dense_query.float(), expanded_key.float().transpose(-1, -2))
        / math.sqrt(dense_query.shape[-1]),
        dim=-1,
    )[0, :, 0]
    prompt_weights = weights[:, :prompt_tokens]
    block_mass_per_head = prompt_weights.view(
        prompt_weights.shape[0], prompt_tokens // block_tokens, block_tokens
    ).sum(-1)
    block_mass = block_mass_per_head.mean(0)
    mandatory = set(range(init_blocks))
    total_blocks = prompt_tokens // block_tokens
    mandatory.update(range(max(0, total_blocks - local_blocks), total_blocks))
    dynamic_count = max(0, budget_blocks - len(mandatory))
    candidates = [block for block in range(total_blocks) if block not in mandatory]
    oracle = sorted(candidates, key=lambda block: float(block_mass[block]), reverse=True)[
        :dynamic_count
    ]
    selected_tokens = torch.zeros(weights.shape[-1], dtype=torch.bool, device=weights.device)
    for block in chosen:
        selected_tokens[block * block_tokens:(block + 1) * block_tokens] = True
    selected_tokens[prompt_tokens:] = True
    selected_mass_per_head = weights[:, selected_tokens].sum(-1)
    chosen_dynamic = set(chosen) - mandatory
    return {
        "selected_attention_mass_mean": float(selected_mass_per_head.mean()),
        "selected_attention_mass_min_head": float(selected_mass_per_head.min()),
        "oracle_dynamic_block_recall": (
            len(chosen_dynamic & set(oracle)) / len(oracle) if oracle else 1.0
        ),
        "oracle_dynamic_blocks": oracle,
        "chosen_dynamic_blocks": sorted(chosen_dynamic),
    }


def build_representatives(cache, block_tokens: int):
    result = {}
    for layer in range(cache_layer_count(cache)):
        key, _ = cache_kv(cache, layer)
        heads, tokens, dim = key.shape[1:]
        blocks = tokens // block_tokens
        result[layer] = key[0].view(heads, blocks, block_tokens, dim).mean(2).contiguous()
    return result


def prefill_with_full_queries(model, input_ids):
    """Dense prefill plus post-RoPE queries used only to build V9 landmarks."""
    captured = {}
    handles = []
    for index, layer in enumerate(model.model.layers):
        def hook(module, args, kwargs, layer_index=index):
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            cos, sin = kwargs["position_embeddings"]
            shape = (*hidden.shape[:-1], -1, module.head_dim)
            query = module.q_norm(module.q_proj(hidden).view(shape)).transpose(1, 2)
            dummy = query[:, :module.config.num_key_value_heads]
            query, _ = apply_rotary_pos_emb(query, dummy, cos, sin)
            captured[layer_index] = query.detach()
        handles.append(layer.self_attn.register_forward_pre_hook(hook, with_kwargs=True))
    cache = DynamicCache()
    with torch.inference_mode():
        output = model(input_ids=input_ids, past_key_values=cache, use_cache=True)
    for handle in handles:
        handle.remove()
    return output, cache, captured


def build_attention_landmarks(cache, queries, block_tokens: int, observation_tokens: int = 32):
    """Select one real key/token per head and block by observed attention score."""
    result, chosen_tokens = {}, {}
    for layer in range(cache_layer_count(cache)):
        key = cache_kv(cache, layer)[0][0]  # [kv_head, token, dim]
        query = queries[layer][0]
        kv_heads, tokens, dim = key.shape
        groups = query.shape[0] // kv_heads
        grouped_query = query.view(kv_heads, groups, tokens, dim).mean(1)
        observed = grouped_query[:, -min(observation_tokens, tokens):].float()
        scores = torch.einsum("hod,htd->hot", observed, key.float()).amax(1)
        blocks = tokens // block_tokens
        within = scores.view(kv_heads, blocks, block_tokens).argmax(-1)
        block_base = torch.arange(blocks, device=key.device).unsqueeze(0)*block_tokens
        indices = within + block_base
        heads = torch.arange(kv_heads, device=key.device).unsqueeze(1).expand_as(indices)
        result[layer] = key[heads, indices].contiguous()
        chosen_tokens[layer] = indices.cpu().tolist()
    return result, chosen_tokens


def timed_event(events, origin, name, lane, token_index, layer, fn, sync=False, **extra):
    start = time.perf_counter()
    value = fn()
    if sync:
        torch.cuda.synchronize()
    end = time.perf_counter()
    events.append(Event(name, "qwen-decode", (start-origin)*1e6, (end-start)*1e6,
                        4, lane, {"token": token_index, "layer": layer, **extra}))
    return value


def read_block_map(fd, info, block_ids):
    """Read blocks separately so a wrong prediction can be overwritten in-place."""
    bytes_per_token = 2 * info["kv_heads"] * info["head_dim"] * 2
    result = {}
    for block in block_ids:
        token_start = block * info["block_tokens"]
        count = min(info["block_tokens"], info["tokens"] - token_start)
        result[block] = os.pread(fd, count*bytes_per_token,
                                 info["offset"] + token_start*bytes_per_token)
    return result


def stored_torch_dtype(metadata):
    return torch.float16 if metadata.get("dtype") == "float16" else torch.bfloat16


def sparse_step(model, token: torch.Tensor, position: int, fd: int, metadata: dict,
                representatives, tails, budget_blocks: int, events, origin, token_index: int,
                cold_io: bool):
    hidden = model.model.embed_tokens(token)
    position_ids = torch.tensor([[position]], device="cuda")
    cos, sin = model.model.rotary_emb(hidden, position_ids)
    q_heads = model.config.num_attention_heads
    kv_heads = model.config.num_key_value_heads
    groups = q_heads // kv_heads

    for layer_index, layer in enumerate(model.model.layers):
        residual = hidden
        normalized = layer.input_layernorm(hidden)
        attention = layer.self_attn
        shape = (*normalized.shape[:-1], -1, attention.head_dim)
        query = attention.q_norm(attention.q_proj(normalized).view(shape)).transpose(1, 2)
        key = attention.k_norm(attention.k_proj(normalized).view(shape)).transpose(1, 2)
        value = attention.v_proj(normalized).view(shape).transpose(1, 2)
        query, key = apply_rotary_pos_emb(query, key, cos, sin)
        chosen = select_blocks(query, representatives[layer_index], budget_blocks)
        info = metadata["layers"][layer_index]
        if cold_io and hasattr(os, "posix_fadvise"):
            os.posix_fadvise(fd, info["offset"], info["nbytes"], os.POSIX_FADV_DONTNEED)
        raw = timed_event(events, origin, "dynamic KV block read", "SSD read",
                          token_index, layer_index,
                          lambda: read_blocks(fd, info, chosen), blocks=len(chosen))
        shape_on_disk = (len(chosen)*info["block_tokens"], 2, kv_heads, attention.head_dim)
        host = torch.from_numpy(raw).view(stored_torch_dtype(metadata)).reshape(shape_on_disk)
        device = timed_event(events, origin, "DRAM → VRAM", "PCIe H2D", token_index,
                             layer_index, lambda: host.cuda(), sync=True, bytes=host.nbytes)
        prompt_key = device[:, 0].permute(1, 0, 2).unsqueeze(0)
        prompt_value = device[:, 1].permute(1, 0, 2).unsqueeze(0)
        if tails[layer_index] is None:
            all_key, all_value = torch.cat((prompt_key, key), dim=2), torch.cat((prompt_value, value), dim=2)
        else:
            tail_key, tail_value = tails[layer_index]
            all_key = torch.cat((prompt_key, tail_key, key), dim=2)
            all_value = torch.cat((prompt_value, tail_value, value), dim=2)
        tails[layer_index] = (key if tails[layer_index] is None else torch.cat((tails[layer_index][0], key), dim=2),
                              value if tails[layer_index] is None else torch.cat((tails[layer_index][1], value), dim=2))

        def attention_compute():
            expanded_key = all_key.repeat_interleave(groups, dim=1)
            expanded_value = all_value.repeat_interleave(groups, dim=1)
            weights = torch.softmax(torch.matmul(query.float(), expanded_key.float().transpose(-1, -2)) /
                                    math.sqrt(attention.head_dim), dim=-1)
            result = torch.matmul(weights, expanded_value.float()).to(normalized.dtype)
            result = result.transpose(1, 2).reshape(*normalized.shape[:-1], -1).contiguous()
            return attention.o_proj(result)

        attended = timed_event(events, origin, "dynamic sparse attention", "GPU compute",
                               token_index, layer_index, attention_compute, sync=True)
        hidden = residual + attended
        residual = hidden
        hidden = residual + layer.mlp(layer.post_attention_layernorm(hidden))
    hidden = model.model.norm(hidden)
    return model.lm_head(hidden[:, -1])


def sparse_step_history(model, token, position, fd, metadata, representatives, tails,
                        histories, budget_blocks, events, origin, token_index, cold_io,
                        executor, direct_reader=None):
    """Prefetch layer L from its previous-token selection; correct only misses."""
    hidden = model.model.embed_tokens(token)
    position_ids = torch.tensor([[position]], device="cuda")
    cos, sin = model.model.rotary_emb(hidden, position_ids)
    q_heads, kv_heads = model.config.num_attention_heads, model.config.num_key_value_heads
    groups = q_heads // kv_heads

    def submit_prefetch(layer_index):
        predicted = histories[layer_index]
        if predicted is None:
            return None
        info = metadata["layers"][layer_index]
        read_map = direct_reader.read_blocks if direct_reader else lambda i, b: read_block_map(fd, i, b)
        def work():
            start = time.perf_counter()
            blocks = read_map(info, predicted)
            end = time.perf_counter()
            events.append(Event("history speculative prefetch", "qwen-decode-v2",
                                (start-origin)*1e6, (end-start)*1e6, 4, "SSD read",
                                {"token": token_index, "layer": layer_index,
                                 "predicted_blocks": predicted}))
            return blocks
        return executor.submit(work)

    future = submit_prefetch(0)
    new_histories = [None] * len(histories)
    for layer_index, layer in enumerate(model.model.layers):
        residual = hidden
        normalized = layer.input_layernorm(hidden)
        attention = layer.self_attn
        shape = (*normalized.shape[:-1], -1, attention.head_dim)
        query = attention.q_norm(attention.q_proj(normalized).view(shape)).transpose(1, 2)
        key = attention.k_norm(attention.k_proj(normalized).view(shape)).transpose(1, 2)
        value = attention.v_proj(normalized).view(shape).transpose(1, 2)
        query, key = apply_rotary_pos_emb(query, key, cos, sin)
        actual = select_blocks(query, representatives[layer_index], budget_blocks)
        new_histories[layer_index] = actual
        info = metadata["layers"][layer_index]
        if direct_reader is None and cold_io and hasattr(os, "posix_fadvise"):
            os.posix_fadvise(fd, info["offset"], info["nbytes"], os.POSIX_FADV_DONTNEED)
        read_map = direct_reader.read_blocks if direct_reader else lambda i, b: read_block_map(fd, i, b)

        predicted = histories[layer_index]
        if future is None:
            loaded = timed_event(events, origin, "demand KV read", "SSD read", token_index,
                                 layer_index, lambda: read_map(info, actual),
                                 actual_blocks=actual)
            predicted = []
        else:
            loaded = future.result()
        missing = [block for block in actual if block not in loaded]
        wrong = [block for block in loaded if block not in actual]
        if missing:
            correction = timed_event(events, origin, "prefetch miss correction", "SSD read",
                                     token_index, layer_index,
                                     lambda: read_map(info, missing),
                                     missing_blocks=missing, overwrite_blocks=wrong)
            loaded.update(correction)
        events.append(Event("selection audit", "qwen-decode-v2", (time.perf_counter()-origin)*1e6,
                            0.01, 4, "Scheduler", {"token": token_index, "layer": layer_index,
                            "predicted": predicted or [], "actual": actual, "missing": missing,
                            "wrong": wrong, "hits": len(set(actual) & set(predicted or []))}))

        # Start L+1 I/O before L's H2D and GPU computation.
        future = submit_prefetch(layer_index+1) if layer_index+1 < len(histories) else None
        joined = b"".join(loaded[block] for block in actual)
        raw = torch.frombuffer(bytearray(joined), dtype=torch.uint16)
        disk_shape = (len(actual)*info["block_tokens"], 2, kv_heads, attention.head_dim)
        host = raw.view(stored_torch_dtype(metadata)).reshape(disk_shape)
        device = timed_event(events, origin, "DRAM → VRAM", "PCIe H2D", token_index,
                             layer_index, lambda: host.cuda(), sync=True, bytes=host.nbytes)
        prompt_key = device[:, 0].permute(1, 0, 2).unsqueeze(0)
        prompt_value = device[:, 1].permute(1, 0, 2).unsqueeze(0)
        if tails[layer_index] is None:
            all_key = torch.cat((prompt_key, key), dim=2)
            all_value = torch.cat((prompt_value, value), dim=2)
            tails[layer_index] = (key, value)
        else:
            tail_key, tail_value = tails[layer_index]
            all_key = torch.cat((prompt_key, tail_key, key), dim=2)
            all_value = torch.cat((prompt_value, tail_value, value), dim=2)
            tails[layer_index] = (torch.cat((tail_key, key), dim=2),
                                  torch.cat((tail_value, value), dim=2))

        def compute():
            expanded_key = all_key.repeat_interleave(groups, dim=1)
            expanded_value = all_value.repeat_interleave(groups, dim=1)
            weights = torch.softmax(torch.matmul(query.float(), expanded_key.float().transpose(-1, -2)) /
                                    math.sqrt(attention.head_dim), dim=-1)
            result = torch.matmul(weights, expanded_value.float()).to(normalized.dtype)
            result = result.transpose(1, 2).reshape(*normalized.shape[:-1], -1).contiguous()
            return attention.o_proj(result)
        attended = timed_event(events, origin, "dynamic sparse attention", "GPU compute",
                               token_index, layer_index, compute, sync=True)
        hidden = residual + attended
        residual = hidden
        hidden = residual + layer.mlp(layer.post_attention_layernorm(hidden))
    histories[:] = new_histories
    return model.lm_head(model.model.norm(hidden)[:, -1])


def sparse_step_pinned(model, token, position, metadata, representatives, tails, histories,
                       budget_blocks, events, origin, token_index, executor, reader,
                       host_buffers, device_buffers, copy_stream=None, buffer_ready=None,
                       cuda_timings=None, gpu_overwrite=False, fused_kv_weights=None,
                       init_blocks=1, local_blocks=1, audit_records=None,
                       dense_hidden_states=None, dense_cache=None, prompt_tokens=None):
    """V4: io_uring DMA into reusable pinned buffers and fixed VRAM slots."""
    hidden = model.model.embed_tokens(token)
    position_ids = torch.tensor([[position]], device="cuda")
    cos, sin = model.model.rotary_emb(hidden, position_ids)
    q_heads, kv_heads = model.config.num_attention_heads, model.config.num_key_value_heads
    groups = q_heads // kv_heads

    def submit(layer_index):
        predicted = histories[layer_index]
        if predicted is None:
            return None
        host = host_buffers[layer_index % 2]
        info = metadata["layers"][layer_index]
        def work():
            start = time.perf_counter()
            reader.read_blocks_into(info, predicted, host)
            end = time.perf_counter()
            events.append(Event("history prefetch → pinned buffer", "qwen-decode-v4",
                                (start-origin)*1e6, (end-start)*1e6, 4, "SSD read",
                                {"token": token_index, "layer": layer_index,
                                 "predicted_blocks": predicted, "buffer": layer_index % 2}))
            return predicted
        return executor.submit(work)

    future = submit(0)
    new_histories = [None]*len(histories)
    for layer_index, layer in enumerate(model.model.layers):
        residual = hidden
        normalized = layer.input_layernorm(hidden)
        attention = layer.self_attn
        shape = (*normalized.shape[:-1], -1, attention.head_dim)
        query = attention.q_norm(attention.q_proj(normalized).view(shape)).transpose(1, 2)
        if fused_kv_weights is None:
            key = attention.k_norm(attention.k_proj(normalized).view(shape)).transpose(1, 2)
            value = attention.v_proj(normalized).view(shape).transpose(1, 2)
        else:
            # One GEMM; first output plane is K, second is V. This tensor is
            # already token-major K/V-interleaved for the storage path.
            fused = F.linear(normalized, fused_kv_weights[layer_index]).view(
                *normalized.shape[:-1], 2, model.config.num_key_value_heads,
                attention.head_dim)
            key = attention.k_norm(fused[..., 0, :, :]).transpose(1, 2)
            value = fused[..., 1, :, :].transpose(1, 2)
        query, key = apply_rotary_pos_emb(query, key, cos, sin)
        info = metadata["layers"][layer_index]
        actual = select_blocks(query, representatives[layer_index], budget_blocks,
                               init_blocks, local_blocks)
        layer_audit = None
        if audit_records is not None:
            dense_input = dense_hidden_states[layer_index]
            dense_normalized = layer.input_layernorm(dense_input)
            dense_query = attention.q_norm(
                attention.q_proj(dense_normalized).view(shape)
            ).transpose(1, 2)
            dense_query, _ = apply_rotary_pos_emb(dense_query, key, cos, sin)
            dense_key = cache_kv(dense_cache, layer_index)[0][:, :, :position + 1]
            layer_audit = audit_selection(
                dense_query, dense_key, actual, prompt_tokens, info["block_tokens"],
                init_blocks, local_blocks, budget_blocks,
            )
        new_histories[layer_index] = actual
        host = host_buffers[layer_index % 2]
        predicted = histories[layer_index]
        if future is None:
            started = time.perf_counter()
            reader.read_blocks_into(info, actual, host)
            ended = time.perf_counter()
            events.append(Event("demand read → pinned buffer", "qwen-decode-v4",
                                (started-origin)*1e6, (ended-started)*1e6, 4, "SSD read",
                                {"token": token_index, "layer": layer_index,
                                 "buffer": layer_index % 2}))
            predicted = []
            slot_for = {block: slot for slot, block in enumerate(actual)}
        else:
            prefetched = future.result()
            slot_for = {block: slot for slot, block in enumerate(prefetched)}
        missing = [block for block in actual if block not in slot_for]
        wrong = [block for block in slot_for if block not in actual]
        device_u8 = device_buffers[layer_index % 2]
        copied_to_gpu = False
        if gpu_overwrite and predicted:
            # First transfer the prediction unchanged. Corrections use the other
            # pinned buffer and update only wrong GPU rows.
            scratch = host_buffers[1-(layer_index % 2)]
            copy_started = time.perf_counter()
            with torch.cuda.stream(copy_stream):
                if buffer_ready[layer_index % 2] is not None:
                    copy_stream.wait_event(buffer_ready[layer_index % 2])
                start_event, full_event = torch.cuda.Event(True), torch.cuda.Event(True)
                start_event.record(copy_stream)
                device_u8.copy_(host, non_blocking=True)
                full_event.record(copy_stream)
            correction_slots = [slot_for.pop(block) for block in wrong]
            if missing:
                read_started = time.perf_counter()
                reader.read_blocks_into(info, missing, scratch, list(range(len(missing))))
                read_ended = time.perf_counter()
                events.append(Event("miss read for GPU overwrite", "qwen-decode-v6",
                                    (read_started-origin)*1e6, (read_ended-read_started)*1e6,
                                    4, "SSD read", {"token": token_index, "layer": layer_index,
                                    "missing": missing, "wrong": wrong,
                                    "slots": correction_slots}))
                with torch.cuda.stream(copy_stream):
                    correction_start, correction_end = torch.cuda.Event(True), torch.cuda.Event(True)
                    correction_start.record(copy_stream)
                    for source, destination in enumerate(correction_slots):
                        device_u8[destination].copy_(scratch[source], non_blocking=True)
                    correction_end.record(copy_stream)
                correction_end.synchronize()  # scratch may become L+1's prefetch buffer
                for block, slot_index in zip(missing, correction_slots):
                    slot_for[block] = slot_index
                cuda_timings.append(("in-place miss H2D overwrite", "PCIe H2D", read_ended,
                                     correction_start, correction_end, token_index, layer_index,
                                     {"bytes": len(missing)*host.shape[1], "slots": correction_slots}))
            else:
                full_event.synchronize()
            ready_event = correction_end if missing else full_event
            torch.cuda.current_stream().wait_event(ready_event)
            cuda_timings.append(("predicted blocks H2D", "PCIe H2D", copy_started,
                                 start_event, full_event, token_index, layer_index,
                                 {"bytes": host.numel(), "buffer": layer_index % 2}))
            copied_to_gpu = True
        elif missing:
            correction_slots = [slot_for.pop(block) for block in wrong]
            started = time.perf_counter()
            reader.read_blocks_into(info, missing, host, correction_slots)
            ended = time.perf_counter()
            for block, slot in zip(missing, correction_slots):
                slot_for[block] = slot
            events.append(Event("miss direct overwrite pinned slot", "qwen-decode-v4",
                                (started-origin)*1e6, (ended-started)*1e6, 4, "SSD read",
                                {"token": token_index, "layer": layer_index,
                                 "missing": missing, "wrong": wrong,
                                 "slots": correction_slots}))
        events.append(Event("selection audit", "qwen-decode-v4", (time.perf_counter()-origin)*1e6,
                            .01, 4, "Scheduler", {"token": token_index, "layer": layer_index,
                            "predicted": predicted or [], "actual": actual, "missing": missing,
                            "wrong": wrong, "hits": len(set(actual)&set(predicted or []))}))
        future = submit(layer_index+1) if layer_index+1 < len(histories) else None

        # Attention is order-independent, so physical slot order need not match actual order.
        copy_start = time.perf_counter()
        if copied_to_gpu:
            pass
        elif copy_stream is None:
            device_u8.copy_(host, non_blocking=True)
            torch.cuda.synchronize()
            copy_end = time.perf_counter()
            events.append(Event("pinned DRAM → fixed VRAM", "qwen-decode-v4",
                                (copy_start-origin)*1e6, (copy_end-copy_start)*1e6, 4, "PCIe H2D",
                                {"token": token_index, "layer": layer_index,
                                 "bytes": host.numel(), "buffer": layer_index % 2}))
        else:
            slot = layer_index % 2
            with torch.cuda.stream(copy_stream):
                if buffer_ready[slot] is not None:
                    copy_stream.wait_event(buffer_ready[slot])
                start_event, end_event = torch.cuda.Event(True), torch.cuda.Event(True)
                start_event.record(copy_stream)
                device_u8.copy_(host, non_blocking=True)
                end_event.record(copy_stream)
            torch.cuda.current_stream().wait_event(end_event)
            cuda_timings.append(("async pinned H2D", "PCIe H2D", copy_start, start_event,
                                 end_event, token_index, layer_index,
                                 {"bytes": host.numel(), "buffer": slot}))
        slots = device_u8.view(stored_torch_dtype(metadata)).reshape(
            budget_blocks, info["block_tokens"], 2, kv_heads, attention.head_dim)
        prompt_key = slots[:, :, 0].reshape(-1, kv_heads, attention.head_dim).permute(1, 0, 2).unsqueeze(0)
        prompt_value = slots[:, :, 1].reshape(-1, kv_heads, attention.head_dim).permute(1, 0, 2).unsqueeze(0)
        if tails[layer_index] is None:
            all_key, all_value = torch.cat((prompt_key, key), 2), torch.cat((prompt_value, value), 2)
            tails[layer_index] = (key, value)
        else:
            tail_key, tail_value = tails[layer_index]
            all_key = torch.cat((prompt_key, tail_key, key), 2)
            all_value = torch.cat((prompt_value, tail_value, value), 2)
            tails[layer_index] = (torch.cat((tail_key, key), 2), torch.cat((tail_value, value), 2))

        def compute():
            expanded_key = all_key.repeat_interleave(groups, 1)
            expanded_value = all_value.repeat_interleave(groups, 1)
            weights = torch.softmax(torch.matmul(query.float(), expanded_key.float().transpose(-1,-2))/
                                    math.sqrt(attention.head_dim), dim=-1)
            result = torch.matmul(weights, expanded_value.float()).to(normalized.dtype)
            result = result.transpose(1,2).reshape(*normalized.shape[:-1],-1).contiguous()
            return attention.o_proj(result)
        compute_start = time.perf_counter()
        if copy_stream is None:
            attended = compute(); torch.cuda.synchronize(); compute_end = time.perf_counter()
            events.append(Event("dynamic sparse attention", "qwen-decode-v4",
                                (compute_start-origin)*1e6, (compute_end-compute_start)*1e6,
                                4, "GPU compute", {"token": token_index, "layer": layer_index}))
        else:
            start_event, end_event = torch.cuda.Event(True), torch.cuda.Event(True)
            start_event.record()
            attended = compute()
            end_event.record()
            buffer_ready[layer_index % 2] = end_event
            cuda_timings.append(("streamed sparse attention", "GPU compute", compute_start,
                                 start_event, end_event, token_index, layer_index, {}))
        hidden = residual + attended
        residual = hidden
        hidden = residual + layer.mlp(layer.post_attention_layernorm(hidden))
        if layer_audit is not None:
            dense_output = dense_hidden_states[layer_index + 1]
            layer_audit.update({
                "token": token_index,
                "layer": layer_index,
                "hidden_cosine": float(F.cosine_similarity(
                    hidden.float().flatten(), dense_output.float().flatten(), dim=0
                )),
            })
            audit_records.append(layer_audit)
    histories[:] = new_histories
    return model.lm_head(model.model.norm(hidden)[:, -1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("models/Qwen3-0.6B"))
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--new-tokens", type=int, default=16)
    parser.add_argument("--block-tokens", type=int, default=32)
    parser.add_argument("--budget-tokens", type=int, default=128)
    parser.add_argument("--store", type=Path, default=Path("artifacts/qwen-decode-kv.bin"))
    parser.add_argument("--trace", type=Path, default=Path("artifacts/qwen-decode-trace.json"))
    parser.add_argument("--cold-io", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prefetch-history", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--version", default=None,
                        help="experiment version label stored in the report")
    parser.add_argument("--io-backend", choices=("pread", "uring-direct"), default="pread")
    parser.add_argument("--fixed-buffers", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--cuda-streams", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--gpu-overwrite", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--async-writeback", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fused-kv-proj", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--writeback-file", type=Path,
                        default=Path("artifacts/qwen-decode-writeback.bin"))
    parser.add_argument("--paper-budget", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--attention-landmarks", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--landmark-observation", type=int, default=32)
    parser.add_argument(
        "--representative-method",
        choices=("mean-key", "legacy-tail-landmark", "infllm-local"),
        default=None,
    )
    parser.add_argument("--repr-topk", type=int, default=4)
    parser.add_argument("--repr-local-window", type=int, default=4096)
    parser.add_argument(
        "--teacher-forced-audit", action=argparse.BooleanOptionalAction, default=False,
        help="feed dense tokens and record per-layer hidden/selection quality",
    )
    parser.add_argument(
        "--init-tokens", type=int, default=None,
        help="explicit init allocation; the paper only fixes init+local combined",
    )
    parser.add_argument(
        "--local-tokens", type=int, default=None,
        help="explicit local allocation; the paper only fixes init+local combined",
    )
    args = parser.parse_args()
    if args.representative_method is None:
        args.representative_method = (
            "legacy-tail-landmark" if args.attention_landmarks else "mean-key"
        )
    if args.paper_budget:
        paper_budget = args.prompt_tokens//4 if args.prompt_tokens < 4096 else 1024
        args.budget_tokens = max(2*args.block_tokens,
                                 (paper_budget//args.block_tokens)*args.block_tokens)
    if args.prompt_tokens % args.block_tokens or args.budget_tokens < 2*args.block_tokens:
        raise SystemExit("prompt must align to blocks; budget must fit init/local blocks")
    mandatory_tokens = args.budget_tokens // 2 if args.paper_budget else 2 * args.block_tokens
    if args.init_tokens is None and args.local_tokens is None:
        # This is an explicit prototype default, not a parameter stated by the
        # SolidAttention paper. Keep one init block and give the remainder of
        # the mandatory half to the local window.
        args.init_tokens = args.block_tokens
        args.local_tokens = mandatory_tokens - args.init_tokens
    elif args.init_tokens is None or args.local_tokens is None:
        raise SystemExit("--init-tokens and --local-tokens must be supplied together")
    if (
        args.init_tokens % args.block_tokens
        or args.local_tokens % args.block_tokens
        or args.init_tokens + args.local_tokens != mandatory_tokens
    ):
        raise SystemExit(
            "init/local must align to blocks and sum to the mandatory budget half"
        )
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    is_quantized = getattr(config, "quantization_config", None) is not None
    activation_dtype = torch.float16 if is_quantized else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, torch_dtype=activation_dtype,
        attn_implementation="eager", device_map="cuda" if is_quantized else None,
        low_cpu_mem_usage=is_quantized,
    )
    if not is_quantized:
        model = model.cuda()
    model = model.eval()
    input_ids = make_prompt(tokenizer, args.prompt_tokens)

    torch.cuda.reset_peak_memory_stats()
    dense_ids, dense_ms, dense_logits, dense_cache, dense_hidden = dense_decode(
        model, input_ids, args.new_tokens, args.teacher_forced_audit
    )
    dense_peak_vram = torch.cuda.max_memory_allocated()/2**20
    # Re-prefill: the dense decode cache already includes generated tokens.
    needs_full_queries = args.representative_method in {
        "legacy-tail-landmark", "infllm-local"
    }
    if needs_full_queries:
        prefill, prefill_cache, full_queries = prefill_with_full_queries(model, input_ids)
    else:
        prefill_cache = DynamicCache()
        with torch.inference_mode():
            prefill = model(input_ids=input_ids, past_key_values=prefill_cache, use_cache=True)
        full_queries = None
    first_token = prefill.logits[:, -1].argmax(-1, keepdim=True)
    if args.representative_method == "legacy-tail-landmark":
        representatives, landmark_tokens = build_attention_landmarks(
            prefill_cache, full_queries, args.block_tokens, args.landmark_observation)
        del full_queries
    elif args.representative_method == "infllm-local":
        representatives, landmark_tokens = build_infllm_representatives(
            prefill_cache,
            full_queries,
            args.block_tokens,
            min(args.repr_local_window, args.prompt_tokens),
            args.repr_topk,
        )
        del full_queries
    else:
        representatives = build_representatives(prefill_cache, args.block_tokens)
        landmark_tokens = None
    metadata = write_store(prefill_cache, args.store, args.block_tokens)
    fused_kv_weights = None
    if args.fused_kv_proj:
        fused_kv_weights = []
        with torch.no_grad():
            for layer in model.model.layers:
                attention = layer.self_attn
                fused_kv_weights.append(torch.cat(
                    (attention.k_proj.weight, attention.v_proj.weight), dim=0
                ).contiguous().detach())
                # Prefill and dense reference are complete; retaining both original
                # modules would falsely inflate the V8 resident model footprint.
                attention.k_proj = None
                attention.v_proj = None
        del attention
        gc.collect()
    del prefill_cache
    if not args.teacher_forced_audit:
        del dense_cache
    torch.cuda.empty_cache()

    direct_reader = IoUringDirectReader(args.store) if args.io_backend == "uring-direct" else None
    fd = None if direct_reader else os.open(args.store, os.O_RDONLY)
    events, sparse_ids, sparse_ms, sparse_logits = [], [int(first_token)], [], [prefill.logits[:, -1].float().cpu()]
    tails = [None] * len(model.model.layers)
    token = first_token
    origin = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    histories = [None] * len(model.model.layers)
    budget_blocks = args.budget_tokens//args.block_tokens
    init_blocks = args.init_tokens // args.block_tokens
    local_blocks = args.local_tokens // args.block_tokens
    host_buffers = device_buffers = None
    if args.fixed_buffers:
        if direct_reader is None:
            raise SystemExit("--fixed-buffers requires --io-backend uring-direct")
        info0 = metadata["layers"][0]
        block_bytes = args.block_tokens*2*info0["kv_heads"]*info0["head_dim"]*2
        blocks = args.budget_tokens//args.block_tokens
        host_buffers = [torch.empty((blocks, block_bytes), dtype=torch.uint8, pin_memory=True)
                        for _ in range(2)]
        device_buffers = [torch.empty((blocks, block_bytes), dtype=torch.uint8, device="cuda")
                          for _ in range(2)]
    if args.cuda_streams and not args.fixed_buffers:
        raise SystemExit("--cuda-streams requires --fixed-buffers")
    if args.gpu_overwrite and not args.cuda_streams:
        raise SystemExit("--gpu-overwrite requires --cuda-streams")
    if args.teacher_forced_audit and not (args.prefetch_history and args.fixed_buffers):
        raise SystemExit(
            "--teacher-forced-audit currently requires history prefetch and fixed buffers"
        )
    copy_stream = torch.cuda.Stream() if args.cuda_streams else None
    buffer_ready = [None, None]
    cuda_timings = []
    writeback_fd = None
    write_buffers = None
    write_futures = []
    writeback_bytes = 0
    writeback_events = []
    teacher_audits = []
    writer_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="kv-writeback") \
        if args.async_writeback else None
    if args.async_writeback:
        # 8 × 4 KiB token KV = 32 KiB per layer; two banks avoid producer/writer races.
        args.writeback_file.parent.mkdir(parents=True, exist_ok=True)
        writeback_fd = os.open(args.writeback_file,
                               os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_DIRECT, 0o644)
        chunks = max(1, math.ceil((args.new_tokens-1)/8))
        os.posix_fallocate(writeback_fd, 0, chunks*len(model.model.layers)*32768)
        write_buffers = [torch.empty((len(model.model.layers), 32768), dtype=torch.uint8,
                                     pin_memory=True) for _ in range(2)]

    def stage_writeback(decode_step):
        nonlocal writeback_bytes
        slot = decode_step % 8
        chunk = decode_step // 8
        bank = chunk % 2
        if slot == 0 and chunk >= 2:
            write_futures[chunk-2].result()  # bank cannot be reused before its async write
        stream = copy_stream or torch.cuda.current_stream()
        stage_start = time.perf_counter()
        with torch.cuda.stream(stream):
            ready = torch.cuda.Event()
            for layer_index, (tail_key, tail_value) in enumerate(tails):
                token_kv = torch.stack((tail_key[0, :, -1], tail_value[0, :, -1]), dim=0)
                raw = token_kv.contiguous().view(torch.uint8).flatten()
                write_buffers[bank][layer_index, slot*4096:(slot+1)*4096].copy_(raw, non_blocking=True)
            ready.record(stream)
        events.append(Event("new KV → 32KiB pinned write buffer", "qwen-decode-v7",
                            (stage_start-origin)*1e6, (time.perf_counter()-stage_start)*1e6,
                            4, "PCIe D2H", {"token": decode_step+1, "slot": slot, "bank": bank,
                            "bytes": len(model.model.layers)*4096}))
        if slot != 7:
            return
        def write_job():
            ready.synchronize()
            started = time.perf_counter()
            base = chunk*len(model.model.layers)*32768
            total = 0
            for layer_index in range(len(model.model.layers)):
                view = memoryview(write_buffers[bank][layer_index].numpy())
                total += os.pwritev(writeback_fd, [view], base+layer_index*32768)
            ended = time.perf_counter()
            writeback_events.append(Event("async 32KiB KV writeback", "qwen-decode-v7",
                                          (started-origin)*1e6, (ended-started)*1e6,
                                          4, "SSD write", {"chunk": chunk, "bank": bank,
                                          "bytes": total, "writes": len(model.model.layers)}))
            return total
        write_futures.append(writer_executor.submit(write_job))
    try:
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="history-prefetch") as executor:
          with torch.inference_mode():
            for step in range(args.new_tokens-1):
                start = time.perf_counter()
                if args.prefetch_history:
                    if args.fixed_buffers:
                        logits = sparse_step_pinned(
                            model, token, args.prompt_tokens+step, metadata, representatives,
                            tails, histories, args.budget_tokens//args.block_tokens, events,
                            origin, step+1, executor, direct_reader, host_buffers, device_buffers,
                            copy_stream, buffer_ready, cuda_timings, args.gpu_overwrite,
                            fused_kv_weights, init_blocks, local_blocks,
                            teacher_audits if args.teacher_forced_audit else None,
                            dense_hidden[step] if args.teacher_forced_audit else None,
                            dense_cache if args.teacher_forced_audit else None,
                            args.prompt_tokens)
                    else:
                        logits = sparse_step_history(
                            model, token, args.prompt_tokens+step, fd, metadata, representatives,
                            tails, histories, args.budget_tokens//args.block_tokens, events, origin,
                            step+1, args.cold_io, executor, direct_reader)
                else:
                    logits = sparse_step(model, token, args.prompt_tokens+step, fd, metadata,
                                         representatives, tails, args.budget_tokens//args.block_tokens,
                                         events, origin, step+1, args.cold_io)
                predicted_token = logits.argmax(-1, keepdim=True)
                token = (
                    torch.tensor([[dense_ids[step + 1]]], device="cuda")
                    if args.teacher_forced_audit and step + 1 < len(dense_ids)
                    else predicted_token
                )
                if args.async_writeback:
                    stage_writeback(step)
                torch.cuda.synchronize()
                for name, lane, host_start, cuda_start, cuda_end, ti, li, extra in cuda_timings:
                    events.append(Event(name, "qwen-decode-v5", (host_start-origin)*1e6,
                                        cuda_start.elapsed_time(cuda_end)*1000, 4, lane,
                                        {"token": ti, "layer": li, **extra}))
                cuda_timings.clear()
                sparse_ms.append((time.perf_counter()-start)*1000)
                sparse_ids.append(int(predicted_token))
                sparse_logits.append(logits.float().cpu())
    finally:
        if writer_executor:
            writer_executor.shutdown(wait=True)
            writeback_bytes = sum(f.result() for f in write_futures)
            events.extend(writeback_events)
            os.fsync(writeback_fd)
            os.close(writeback_fd)
        if direct_reader:
            direct_reader.close()
        else:
            os.close(fd)

    prefix = 0
    for a, b in zip(dense_ids, sparse_ids):
        if a != b: break
        prefix += 1
    cosine = [torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
              for a, b in zip(dense_logits, sparse_logits)]
    def percentile(values, fraction):
        ordered = sorted(values)
        return ordered[min(len(ordered)-1, round((len(ordered)-1)*fraction))]
    io_bytes = sum(e.args.get("bytes", 0) for e in events if e.tid == "PCIe H2D")
    audits = [e for e in events if e.name == "selection audit"]
    predicted_total = sum(len(e.args["predicted"]) for e in audits)
    hit_total = sum(e.args["hits"] for e in audits)
    missing_total = sum(len(e.args["missing"]) for e in audits)
    wrong_total = sum(len(e.args["wrong"]) for e in audits)
    report = {
        "prompt_tokens": args.prompt_tokens, "new_tokens": args.new_tokens,
        "block_tokens": args.block_tokens, "budget_tokens": args.budget_tokens,
        "dense_mean_decode_ms": sum(dense_ms)/len(dense_ms),
        "sparse_mean_decode_ms": sum(sparse_ms)/len(sparse_ms),
        "dense_p50_decode_ms": statistics.median(dense_ms),
        "dense_p95_decode_ms": percentile(dense_ms, .95),
        "sparse_p50_decode_ms": statistics.median(sparse_ms),
        "sparse_p95_decode_ms": percentile(sparse_ms, .95),
        "dense_tokens_per_s": 1000/(sum(dense_ms)/len(dense_ms)),
        "sparse_tokens_per_s": 1000/(sum(sparse_ms)/len(sparse_ms)),
        "dense_peak_vram_mib": dense_peak_vram,
        "sparse_peak_vram_mib": torch.cuda.max_memory_allocated()/2**20,
        "sparse_h2d_mib_total": io_bytes/2**20,
        "exact_token_prefix": prefix,
        "mean_logits_cosine": sum(cosine)/len(cosine),
        "dense_token_ids": dense_ids, "sparse_token_ids": sparse_ids,
        "dense_text": tokenizer.decode(dense_ids, skip_special_tokens=True),
        "sparse_text": tokenizer.decode(sparse_ids, skip_special_tokens=True),
        "cold_io_requested": args.cold_io,
        "version": args.version or ("V2-history-prefetch" if args.prefetch_history else "V1-sync-sparse"),
        "history_prefetch": args.prefetch_history,
        "history_block_hit_rate": hit_total/predicted_total if predicted_total else None,
        "history_predicted_blocks": predicted_total,
        "history_hit_blocks": hit_total,
        "history_missing_blocks": missing_total,
        "history_wrong_blocks": wrong_total,
        "io_backend": args.io_backend,
        "fixed_buffers": args.fixed_buffers,
        "cuda_streams": args.cuda_streams,
        "gpu_overwrite": args.gpu_overwrite,
        "async_writeback": args.async_writeback,
        "fused_kv_projection": args.fused_kv_proj,
        "representative_method": args.representative_method,
        "landmark_observation_tokens": (
            args.landmark_observation
            if args.representative_method == "legacy-tail-landmark" else None
        ),
        "repr_topk": args.repr_topk if args.representative_method == "infllm-local" else None,
        "repr_local_window": (
            min(args.repr_local_window, args.prompt_tokens)
            if args.representative_method == "infllm-local" else None
        ),
        "paper_budget_policy": args.paper_budget,
        "budget_split_evidence": (
            "paper: init+local=half; explicit prototype split within that half"
            if args.paper_budget else "explicit experiment configuration"
        ),
        "init_tokens": args.init_tokens,
        "local_tokens": args.local_tokens,
        "init_blocks": init_blocks,
        "local_blocks": local_blocks,
        "dynamic_selected_blocks": budget_blocks-init_blocks-local_blocks,
        "fused_kv_weight_mib": (sum(x.numel()*x.element_size() for x in fused_kv_weights)/2**20)
                               if fused_kv_weights else 0,
        "writeback_bytes": writeback_bytes,
        "writeback_full_chunks": len(write_futures),
        "writeback_buffered_tail_tokens": ((args.new_tokens-1) % 8) if args.async_writeback else 0,
        "pinned_host_buffer_mib": (sum(x.numel() for x in host_buffers)/2**20) if host_buffers else 0,
        "fixed_vram_buffer_mib": (sum(x.numel() for x in device_buffers)/2**20) if device_buffers else 0,
        "scope": (
            "teacher-forced Qwen layer audit with dense tokens and SSD-sparse decode"
            if args.teacher_forced_audit
            else "actual free-running Qwen logits with dense prefill and SSD-sparse decode"
        ),
        "teacher_forced_audit": args.teacher_forced_audit,
        "teacher_forced_layer_records": teacher_audits,
        "teacher_forced_hidden_cosine_mean": (
            sum(x["hidden_cosine"] for x in teacher_audits) / len(teacher_audits)
            if teacher_audits else None
        ),
        "teacher_forced_selected_attention_mass_mean": (
            sum(x["selected_attention_mass_mean"] for x in teacher_audits)
            / len(teacher_audits) if teacher_audits else None
        ),
        "teacher_forced_oracle_block_recall_mean": (
            sum(x["oracle_dynamic_block_recall"] for x in teacher_audits)
            / len(teacher_audits) if teacher_audits else None
        ),
    }
    write_trace(args.trace, events, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
