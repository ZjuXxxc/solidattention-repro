import torch

from solidattention_lab.kv_lifecycle import LayerTail


def make_tail():
    return LayerTail(
        q_heads=4, kv_heads=2, head_dim=2, local_tokens=32,
        block_tokens=32, repr_topk=4,
    )


def test_does_not_seal_tokens_still_in_local_window():
    tail = make_tail()
    key = torch.randn(1, 2, 63, 2)
    tail.append(key, key.clone())
    tail.add_attention_mass(torch.ones(4, 63))
    assert tail.seal_ready() == []
    assert tail.length == 63


def test_seals_oldest_block_and_keeps_fixed_local_window():
    tail = make_tail()
    key = torch.arange(1 * 2 * 64 * 2, dtype=torch.float32).view(1, 2, 64, 2)
    tail.append(key, -key)
    score = torch.zeros(4, 64)
    score[:, 7] = 10
    score[:, 9] = 9
    score[:, 11] = 8
    score[:, 13] = 7
    tail.add_attention_mass(score)
    sealed = tail.seal_ready()
    assert len(sealed) == 1
    assert tail.length == 32
    assert sealed[0].representative_indices.tolist() == [[7, 9, 11, 13]] * 4
    torch.testing.assert_close(sealed[0].key, key[:, :, :32])
    torch.testing.assert_close(tail.key, key[:, :, 32:])

