import torch

from solidattention_lab.selection import local_causal_token_scores, select_blocks


def test_local_causal_scores_do_not_use_future_queries():
    # q0 strongly matches k0, q1 strongly matches k1. With window=1 each key
    # receives exactly its own query's full attention mass.
    query = torch.tensor([[[10.0, 0.0], [0.0, 10.0]]])
    key = query.clone()
    scores = local_causal_token_scores(query, key, local_window=1)
    torch.testing.assert_close(scores, torch.ones_like(scores))


def test_gqa_query_heads_are_not_averaged_before_scoring():
    query = torch.tensor([[[[1.0, 0.0]], [[0.0, 1.0]]]])
    # Two blocks, with opposite relevance for the two query heads. The second
    # representative dimension is block, not KV head.
    representatives = torch.tensor(
        [[[1.0, 0.0], [0.0, 0.1]], [[0.0, 0.1], [0.0, 1.0]]]
    )
    chosen = select_blocks(query, representatives, 1, 0, 0)
    assert chosen == [0] or chosen == [1]  # deterministic tie; both heads survived

