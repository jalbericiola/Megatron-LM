# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

from megatron.core.models.hybrid import shared_prefix as shared_prefix_module
from megatron.core.models.hybrid.shared_prefix import SharedPrefixLayout, _validate_hybrid_stack
from megatron.core.models.hybrid.shared_prefix_fused import (
    _cp_local_kv_head_slice,
    _redo_cp_zigzag,
    _undo_cp_zigzag,
)
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.transformer_layer import TransformerLayer


class _FakeGroup:
    def __init__(self, size: int, rank: int = 0):
        self._size = size
        self._rank = rank

    def size(self):
        return self._size

    def rank(self):
        return self._rank


def _validation_stack(cp_size: int, *, layers=(), query_heads=4, kv_heads=1):
    config = SimpleNamespace(
        context_parallel_size=cp_size,
        sequence_parallel=False,
        recompute_granularity=None,
        fine_grained_activation_offloading=False,
        cuda_graph_impl="none",
        fp8=None,
        fp4=None,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        window_size=None,
        softmax_type="vanilla",
        num_moe_experts=None,
        qk_clip=False,
        log_max_attention_logit=False,
        num_attention_heads=query_heads,
        num_query_groups=kv_heads,
    )
    return SimpleNamespace(
        tp_group=_FakeGroup(1),
        pp_group=_FakeGroup(1),
        pg_collection=SimpleNamespace(cp=_FakeGroup(cp_size)),
        config=config,
        layers=list(layers),
    )


def _stub_attention_layer():
    attention = object.__new__(SelfAttention)
    torch.nn.Module.__init__(attention)
    attention.checkpoint_core_attention = False
    layer = object.__new__(TransformerLayer)
    torch.nn.Module.__init__(layer)
    layer.self_attention = attention
    return layer


def test_cp_zigzag_permutations_are_inverse():
    canonical = torch.arange(24)

    rank_major = _redo_cp_zigzag(canonical, cp_size=4)

    assert rank_major.tolist() == [
        0,
        1,
        2,
        21,
        22,
        23,
        3,
        4,
        5,
        18,
        19,
        20,
        6,
        7,
        8,
        15,
        16,
        17,
        9,
        10,
        11,
        12,
        13,
        14,
    ]
    assert torch.equal(_undo_cp_zigzag(rank_major, cp_size=4), canonical)


def test_layout_cp_indices_cover_standard_two_chunk_ownership():
    indices = [
        SharedPrefixLayout.cp_local_indices(24, 4, rank, "cpu").tolist() for rank in range(4)
    ]

    assert indices == [
        [0, 1, 2, 21, 22, 23],
        [3, 4, 5, 18, 19, 20],
        [6, 7, 8, 15, 16, 17],
        [9, 10, 11, 12, 13, 14],
    ]
    assert sorted(index for rank_indices in indices for index in rank_indices) == list(range(24))


def test_layout_cp_positions_preserve_branch_rope_and_make_trailing_pad_inert():
    layout = SharedPrefixLayout(prefix_len=3, completion_lens=[2, 4])
    global_positions = layout.padded_position_ids(12, "cpu")

    assert global_positions.tolist() == [0, 1, 2, 3, 4, 3, 4, 5, 6, 0, 0, 0]
    rank_zero = layout.cp_local_indices(12, 2, 0, "cpu")
    rank_one = layout.cp_local_indices(12, 2, 1, "cpu")
    assert global_positions.index_select(0, rank_zero).tolist() == [0, 1, 2, 0, 0, 0]
    assert global_positions.index_select(0, rank_one).tolist() == [3, 4, 3, 4, 5, 6]


@pytest.mark.parametrize(
    ("rank", "expected"), [(0, slice(0, 1)), (1, slice(0, 1)), (2, slice(1, 2)), (3, slice(1, 2))]
)
def test_nano_gqa_cp4_replicates_the_semantically_correct_kv_head(rank, expected):
    assert _cp_local_kv_head_slice(32, 2, 4, rank) == expected


def test_cp_gqa_fails_closed_when_a_rank_cuts_unequal_kv_groups():
    with pytest.raises(NotImplementedError, match="boundary cuts unequal"):
        _cp_local_kv_head_slice(12, 3, 2, 0)


def test_cp_validation_rejects_unequal_destination_kv_widths_before_forward():
    # Q12/KV2/CP3 produces destination widths 1,2,1. Every individual slice is valid,
    # but the equal-split all-to-all cannot represent the aggregate layout.
    layout = SharedPrefixLayout(prefix_len=3, completion_lens=[1, 2])
    stack = _validation_stack(
        cp_size=3, layers=[_stub_attention_layer()], query_heads=12, kv_heads=2
    )

    with pytest.raises(NotImplementedError, match="equal KV-head widths"):
        _validate_hybrid_stack(stack, torch.empty(2, 1, 8, dtype=torch.bfloat16), layout)


def test_cp_validation_accepts_minimal_trailing_padding():
    layout = SharedPrefixLayout(prefix_len=3, completion_lens=[4, 5])
    # total=12, already divisible by 2*CP; each rank owns six tokens.
    _validate_hybrid_stack(
        _validation_stack(cp_size=2), torch.empty(6, 1, 8, dtype=torch.bfloat16), layout
    )

    padded_layout = SharedPrefixLayout(prefix_len=3, completion_lens=[2, 4])  # total=9 -> 12
    _validate_hybrid_stack(
        _validation_stack(cp_size=2), torch.empty(6, 1, 8, dtype=torch.bfloat16), padded_layout
    )


def test_cp_validation_rejects_nonminimal_or_misaligned_physical_length():
    layout = SharedPrefixLayout(prefix_len=3, completion_lens=[2, 4])  # total=9
    stack = _validation_stack(cp_size=2)

    with pytest.raises(ValueError, match=r"divisible by 2 \* context parallel size"):
        _validate_hybrid_stack(stack, torch.empty(5, 1, 8, dtype=torch.bfloat16), layout)

    with pytest.raises(ValueError, match="minimal trailing pad"):
        _validate_hybrid_stack(stack, torch.empty(8, 1, 8, dtype=torch.bfloat16), layout)


def test_cp_validation_keeps_tp_and_sequence_parallel_fail_closed():
    layout = SharedPrefixLayout(prefix_len=3, completion_lens=[2, 4])
    hidden_states = torch.empty(6, 1, 8, dtype=torch.bfloat16)

    tp_stack = _validation_stack(cp_size=2)
    tp_stack.tp_group = _FakeGroup(4)
    with pytest.raises(NotImplementedError, match="TP1 only"):
        _validate_hybrid_stack(tp_stack, hidden_states, layout)

    sp_stack = _validation_stack(cp_size=2)
    sp_stack.config.sequence_parallel = True
    with pytest.raises(NotImplementedError, match="does not support sequence parallelism"):
        _validate_hybrid_stack(sp_stack, hidden_states, layout)


def test_cp_mamba_default_uses_optimized_state_fork(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(
        shared_prefix_module,
        "_forward_mamba_layer_shared_prefix_cp_state_fork",
        lambda *_args: sentinel,
    )

    assert shared_prefix_module._forward_mamba_layer_shared_prefix_cp(None, None, None) is sentinel
