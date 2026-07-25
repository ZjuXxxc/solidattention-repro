"""Block representatives and selection policies.

The InfLLM-compatible path deliberately keeps query-head granularity.  A GQA
key is expanded to its query heads before representative scoring, matching
InfLLM's ``from_group_kv``.  Token importance is accumulated from the
attention probability that the token actually receives while it is inside
each later query's causal local window.
"""
from __future__ import annotations

import math

import torch

from .qwen_kv_lab import cache_kv, cache_layer_count


def local_causal_token_scores(
    query: torch.Tensor,
    key: torch.Tensor,
    local_window: int,
    query_chunk: int = 64,
) -> torch.Tensor:
    """Return attention mass received by every key for every query head.

    Args:
        query: post-RoPE tensor ``[q_heads, tokens, head_dim]``.
        key: post-RoPE tensor ``[kv_heads, tokens, head_dim]``.
        local_window: causal sliding-window length used to measure importance.
        query_chunk: bounds the temporary ``q_heads × chunk × tokens`` matrix.
    """
    q_heads, tokens, dim = query.shape
    kv_heads = key.shape[0]
    if q_heads % kv_heads:
        raise ValueError("query heads must be divisible by KV heads")
    if key.shape[1:] != (tokens, dim):
        raise ValueError("query/key token and head dimensions must match")
    if local_window <= 0:
        raise ValueError("local_window must be positive")

    expanded_key = key.repeat_interleave(q_heads // kv_heads, dim=0)
    received = torch.zeros((q_heads, tokens), dtype=torch.float32, device=query.device)
    key_positions = torch.arange(tokens, device=query.device)
    scale = 1.0 / math.sqrt(dim)
    for start in range(0, tokens, query_chunk):
        end = min(tokens, start + query_chunk)
        query_positions = torch.arange(start, end, device=query.device)
        logits = torch.matmul(
            query[:, start:end].float(), expanded_key.float().transpose(-1, -2)
        ) * scale
        causal = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
        within_local = key_positions.unsqueeze(0) >= (
            query_positions.unsqueeze(1) - local_window + 1
        )
        logits.masked_fill_(~(causal & within_local).unsqueeze(0), float("-inf"))
        received.add_(torch.softmax(logits, dim=-1).sum(dim=1))
    return received


def build_infllm_representatives(
    cache,
    queries: dict[int, torch.Tensor],
    block_tokens: int,
    local_window: int,
    repr_topk: int = 4,
    query_chunk: int = 64,
):
    """Build InfLLM-style per-query-head block representatives.

    For each layer/head/block, choose the ``repr_topk`` real keys with the
    highest accumulated local causal attention mass.  Their mean is the block
    identifier, as in InfLLM's ``get_block_k`` + mean reduction.
    """
    representatives, chosen_tokens = {}, {}
    for layer in range(cache_layer_count(cache)):
        key = cache_kv(cache, layer)[0][0]
        query = queries[layer][0]
        q_heads, tokens, dim = query.shape
        if tokens % block_tokens:
            raise ValueError("token count must be divisible by block_tokens")
        topk = min(repr_topk, block_tokens)
        scores = local_causal_token_scores(query, key, local_window, query_chunk)
        blocks = tokens // block_tokens
        within = scores.view(q_heads, blocks, block_tokens).topk(topk, dim=-1).indices
        block_base = (
            torch.arange(blocks, device=key.device).view(1, blocks, 1) * block_tokens
        )
        indices = within + block_base
        expanded_key = key.repeat_interleave(q_heads // key.shape[0], dim=0)
        head_index = torch.arange(q_heads, device=key.device).view(q_heads, 1, 1)
        selected = expanded_key[head_index, indices]
        representatives[layer] = selected.mean(dim=2).contiguous()
        chosen_tokens[layer] = indices.cpu().tolist()
    return representatives, chosen_tokens


def select_blocks(
    query: torch.Tensor,
    representatives: torch.Tensor,
    budget_blocks: int,
    init_blocks: int = 1,
    local_blocks: int = 1,
) -> list[int]:
    """Select a shared block set after retaining head-level similarities."""
    blocks = representatives.shape[1]
    query_heads = query.shape[1]
    if representatives.shape[0] == query_heads:
        scoring_query = query[0, :, 0]
    else:
        kv_heads = representatives.shape[0]
        scoring_query = query[0, :, 0].view(kv_heads, -1, query.shape[-1]).mean(1)
    per_head = torch.einsum(
        "hd,hbd->hb", scoring_query.float(), representatives.float()
    )
    scores = per_head.mean(dim=0)
    chosen = set(range(min(init_blocks, blocks)))
    chosen.update(range(max(0, blocks - local_blocks), blocks))
    for block in torch.argsort(scores, descending=True).tolist():
        if len(chosen) == min(budget_blocks, blocks):
            break
        chosen.add(block)
    return sorted(chosen)
