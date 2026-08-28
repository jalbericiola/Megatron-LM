# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.models.hybrid import shared_prefix_fused


class _FakeGroup:
    def __init__(self, size: int, rank: int = 0) -> None:
        self._size = size
        self._rank = rank

    def size(self) -> int:
        return self._size

    def rank(self) -> int:
        return self._rank


@pytest.mark.parametrize(
    ("query_heads", "kv_heads", "cp_size", "expected_ranges", "expected_packed_heads"),
    [
        (32, 32, 4, ((0, 8), (8, 16), (16, 24), (24, 32)), tuple(range(32))),
        (32, 2, 4, ((0, 1), (0, 1), (1, 2), (1, 2)), (0, 0, 1, 1)),
        (4, 1, 2, ((0, 1), (0, 1)), (0, 0)),
    ],
    ids=("mha-q32-kv32-cp4", "nano-gqa-q32-kv2-cp4", "gqa-q4-kv1-cp2"),
)
def test_cp_packs_only_each_destinations_required_kv_heads(
    query_heads: int,
    kv_heads: int,
    cp_size: int,
    expected_ranges: tuple[tuple[int, int], ...],
    expected_packed_heads: tuple[int, ...],
) -> None:
    destination_slices = shared_prefix_fused._cp_kv_head_slices_for_destinations(
        query_heads, kv_heads, cp_size
    )
    source = torch.arange(kv_heads).reshape(1, 1, kv_heads, 1)

    packed = shared_prefix_fused._cp_pack_destination_head_slices(source, destination_slices)

    actual_ranges = tuple((head_slice.start, head_slice.stop) for head_slice in destination_slices)
    assert actual_ranges == expected_ranges
    assert tuple(packed.flatten().tolist()) == expected_packed_heads


@pytest.mark.parametrize(
    ("query_heads", "kv_heads", "cp_size", "expected_multiplicity"),
    [(32, 2, 4, (2.0, 2.0)), (4, 1, 2, (2.0,))],
    ids=("nano-overlapping-kv-heads", "single-overlapping-kv-head"),
)
def test_cp_overlapping_kv_slices_accumulate_gradient_multiplicity(
    query_heads: int, kv_heads: int, cp_size: int, expected_multiplicity: tuple[float, ...]
) -> None:
    source = torch.ones(1, 1, kv_heads, 1, requires_grad=True)
    destination_slices = shared_prefix_fused._cp_kv_head_slices_for_destinations(
        query_heads, kv_heads, cp_size
    )

    shared_prefix_fused._cp_pack_destination_head_slices(
        source, destination_slices
    ).sum().backward()

    assert source.grad is not None
    torch.testing.assert_close(
        source.grad.flatten(), torch.tensor(expected_multiplicity, dtype=source.dtype)
    )


def test_cp_kv_packing_fails_closed_for_unequal_destination_widths() -> None:
    # Q12/KV2/CP3 gives individually valid destination slices of widths 1, 2, 1,
    # which an equal-split all-to-all cannot express.
    with pytest.raises(NotImplementedError, match="equal KV-head widths"):
        shared_prefix_fused._cp_kv_head_slices_for_destinations(12, 2, 3)


def test_cp_kv_exchange_avoids_a_full_kv_activation_base(monkeypatch) -> None:
    cp_group = _FakeGroup(size=4)
    captured_shapes: list[tuple[int, ...]] = []

    def fake_all_to_all_sp2hp(input_: torch.Tensor, group=None) -> torch.Tensor:
        assert group is cp_group
        captured_shapes.append(tuple(input_.shape))
        return input_

    monkeypatch.setattr(shared_prefix_fused, "all_to_all_sp2hp", fake_all_to_all_sp2hp)
    source = torch.arange(2 * 2 * 3, dtype=torch.float32).reshape(2, 1, 2, 3)
    destination_slices = shared_prefix_fused._cp_kv_head_slices_for_destinations(32, 2, 4)

    exchanged = shared_prefix_fused._cp_sequence_to_head_parallel(
        source, cp_group, destination_head_slices=destination_slices
    )

    # Four one-head destination blocks are packed, rather than four copies of
    # both KV heads (the old eight-head input).
    assert captured_shapes == [(2, 1, 4 * 3)]
    assert exchanged.shape == (8, 1, 1, 3)
    assert exchanged.untyped_storage().nbytes() == exchanged.numel() * exchanged.element_size()


def test_cp_sequence_exchange_rejects_odd_local_sequence_before_collective(monkeypatch) -> None:
    cp_group = _FakeGroup(size=4)
    collective_called = False

    def unexpected_all_to_all(input_: torch.Tensor, group=None) -> torch.Tensor:
        nonlocal collective_called
        collective_called = True
        return input_

    monkeypatch.setattr(shared_prefix_fused, "all_to_all_sp2hp", unexpected_all_to_all)

    with pytest.raises(ValueError, match="even local sequence length"):
        shared_prefix_fused._cp_sequence_to_head_parallel(torch.randn(3, 1, 32, 4), cp_group)

    assert not collective_called


def test_cp_attention_rejects_invalid_qkv_before_collective(monkeypatch) -> None:
    cp_group = _FakeGroup(size=2)
    collective_called = False

    def unexpected_all_to_all(input_: torch.Tensor, group=None) -> torch.Tensor:
        nonlocal collective_called
        collective_called = True
        return input_

    monkeypatch.setattr(shared_prefix_fused, "all_to_all_sp2hp", unexpected_all_to_all)
    query = torch.randn(4, 1, 4, 8)
    key = torch.randn(4, 1, 1, 8)
    value = torch.randn(4, 1, 1, 8)
    invalid_inputs = (
        (query.squeeze(1), key, value, "rank-4"),
        (
            query.expand(-1, 2, -1, -1),
            key.expand(-1, 2, -1, -1),
            value.expand(-1, 2, -1, -1),
            "batch size 1",
        ),
        (query, key[..., :7], value, "head dimensions"),
        (query, key.double(), value.double(), "dtypes"),
        (query, key.to(device="meta"), value.to(device="meta"), "same device"),
    )

    for invalid_query, invalid_key, invalid_value, match in invalid_inputs:
        with pytest.raises(ValueError, match=match):
            shared_prefix_fused.flash_composed_forest_attention_cp(
                invalid_query, invalid_key, invalid_value, forest={}, cp_group=cp_group
            )

    assert not collective_called


def test_cp_head_to_sequence_inverse_restores_the_query_shape(monkeypatch) -> None:
    cp_group = _FakeGroup(size=4)

    def identity_all_to_all(input_: torch.Tensor, group=None) -> torch.Tensor:
        assert group is cp_group
        return input_

    monkeypatch.setattr(shared_prefix_fused, "all_to_all_sp2hp", identity_all_to_all)
    monkeypatch.setattr(shared_prefix_fused, "all_to_all_hp2sp", identity_all_to_all)
    query = torch.randn(2, 1, 32, 4)

    head_parallel = shared_prefix_fused._cp_sequence_to_head_parallel(query, cp_group)
    sequence_parallel = shared_prefix_fused._cp_head_to_sequence_parallel(head_parallel, cp_group)

    assert head_parallel.shape == (8, 1, 8, 4)
    assert sequence_parallel.shape == query.shape
