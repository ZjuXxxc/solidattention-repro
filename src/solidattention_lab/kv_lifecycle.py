"""Decode KV lifecycle independent of a particular storage backend."""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SealedBlock:
    key: torch.Tensor
    value: torch.Tensor
    representative: torch.Tensor
    representative_indices: torch.Tensor


class LayerTail:
    """Resident local KV plus attention scores used to seal global blocks.

    Keys/values use ``[1, kv_heads, tokens, dim]``. Scores preserve query-head
    granularity as ``[q_heads, tokens]``. A block becomes sealable only when a
    complete older block can be removed while retaining ``local_tokens``.
    """

    def __init__(
        self, q_heads: int, kv_heads: int, head_dim: int, local_tokens: int,
        block_tokens: int, repr_topk: int,
    ):
        if q_heads % kv_heads:
            raise ValueError("q_heads must be divisible by kv_heads")
        if local_tokens < block_tokens or local_tokens % block_tokens:
            raise ValueError("local_tokens must be a positive block multiple")
        self.q_heads = q_heads
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.local_tokens = local_tokens
        self.block_tokens = block_tokens
        self.repr_topk = min(repr_topk, block_tokens)
        self.key = None
        self.value = None
        self.received_score = None

    @property
    def length(self) -> int:
        return 0 if self.key is None else self.key.shape[2]

    def append(self, key: torch.Tensor, value: torch.Tensor) -> None:
        if key.shape != value.shape or key.shape[:2] != (1, self.kv_heads):
            raise ValueError("invalid decode KV shape")
        if key.shape[3] != self.head_dim:
            raise ValueError("unexpected head dimension")
        added = key.shape[2]
        if self.key is None:
            self.key, self.value = key, value
            self.received_score = torch.zeros(
                (self.q_heads, added), dtype=torch.float32, device=key.device
            )
        else:
            self.key = torch.cat((self.key, key), dim=2)
            self.value = torch.cat((self.value, value), dim=2)
            self.received_score = torch.cat((
                self.received_score,
                torch.zeros(
                    (self.q_heads, added), dtype=torch.float32, device=key.device
                ),
            ), dim=1)

    def add_attention_mass(self, mass: torch.Tensor) -> None:
        """Accumulate the current query's normalized attention over local KV."""
        if mass.shape != (self.q_heads, self.length):
            raise ValueError(
                f"attention mass {tuple(mass.shape)} != {(self.q_heads, self.length)}"
            )
        self.received_score.add_(mass.float())

    def seed_attention_mass(self, mass: torch.Tensor) -> None:
        """Initialize scores captured before this tail object was created."""
        if mass.shape != (self.q_heads, self.length):
            raise ValueError("seed score shape does not match resident tail")
        self.received_score.copy_(mass.float())

    def seal_ready(self) -> list[SealedBlock]:
        sealed = []
        while self.length >= self.local_tokens + self.block_tokens:
            key = self.key[:, :, :self.block_tokens].contiguous()
            value = self.value[:, :, :self.block_tokens].contiguous()
            scores = self.received_score[:, :self.block_tokens]
            indices = scores.topk(self.repr_topk, dim=-1).indices
            expanded_key = key[0].repeat_interleave(
                self.q_heads // self.kv_heads, dim=0
            )
            heads = torch.arange(self.q_heads, device=key.device).unsqueeze(1)
            representative = expanded_key[heads, indices].mean(dim=1).contiguous()
            sealed.append(SealedBlock(key, value, representative, indices))
            self.key = self.key[:, :, self.block_tokens:].contiguous()
            self.value = self.value[:, :, self.block_tokens:].contiguous()
            self.received_score = self.received_score[:, self.block_tokens:].contiguous()
        return sealed
