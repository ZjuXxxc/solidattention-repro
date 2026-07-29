from types import SimpleNamespace

import torch

from solidattention_lab.qwen_kv_lab import write_store


def test_reserved_layer_segments_do_not_overlap(tmp_path):
    cache = SimpleNamespace(
        key_cache=[
            torch.arange(1 * 2 * 4 * 2, dtype=torch.bfloat16).view(1, 2, 4, 2),
            torch.full((1, 2, 4, 2), 7, dtype=torch.bfloat16),
        ],
        value_cache=[
            torch.full((1, 2, 4, 2), -1, dtype=torch.bfloat16),
            torch.full((1, 2, 4, 2), -2, dtype=torch.bfloat16),
        ],
    )
    path = tmp_path / "reserved-kv.bin"
    metadata = write_store(cache, path, block_tokens=2, capacity_tokens=8)
    first, second = metadata["layers"]
    bytes_per_token = 2 * 2 * 2 * 2
    assert second["offset"] - first["offset"] == 8 * bytes_per_token
    assert path.stat().st_size == 2 * 8 * bytes_per_token
    assert first["tokens"] == second["tokens"] == 4
    assert first["capacity_tokens"] == second["capacity_tokens"] == 8

