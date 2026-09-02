# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import ast
import importlib
import inspect
import textwrap

import pytest
import torch

from megatron.core.models.hybrid import shared_prefix_fused


def test_fused_merge_is_disabled_by_default(monkeypatch):
    """The production port keeps the known-corrupt Triton merge unreachable."""
    monkeypatch.delenv("NRL_SP_FUSED_MERGE", raising=False)

    module = importlib.reload(shared_prefix_fused)

    assert module._SP_FUSED_MERGE is False


def test_fused_merge_old_opt_in_fails_closed(monkeypatch):
    monkeypatch.setenv("NRL_SP_FUSED_MERGE", "1")

    with pytest.raises(RuntimeError, match="known nondeterministic"):
        shared_prefix_fused._resolve_fused_merge_setting()


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param(None, False, id="unset"),
        pytest.param("0", False, id="zero"),
        pytest.param("false", False, id="false"),
        pytest.param("1", True, id="one"),
        pytest.param("TRUE", True, id="true"),
    ],
)
def test_deterministic_backward_gate_is_explicit_and_default_off(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("NRL_SP_DETERMINISTIC_BACKWARD", raising=False)
    else:
        monkeypatch.setenv("NRL_SP_DETERMINISTIC_BACKWARD", value)

    assert shared_prefix_fused._resolve_deterministic_backward_setting() is expected


@pytest.mark.parametrize("value", ["", "yes", "2", " true "])
def test_deterministic_backward_gate_rejects_ambiguous_values(monkeypatch, value):
    monkeypatch.setenv("NRL_SP_DETERMINISTIC_BACKWARD", value)

    with pytest.raises(RuntimeError, match="must be one of"):
        shared_prefix_fused._resolve_deterministic_backward_setting()


def test_deterministic_backward_accessor_reports_import_time_setting(monkeypatch):
    monkeypatch.setattr(shared_prefix_fused, "_SP_DETERMINISTIC_BACKWARD", True)

    assert shared_prefix_fused.is_shared_prefix_deterministic_backward_enabled() is True


def test_both_flash_backward_paths_use_deterministic_gate():
    source = textwrap.dedent(inspect.getsource(shared_prefix_fused._ComposedForestAttn.backward))
    tree = ast.parse(source)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_flash_attn_varlen_backward"
    ]

    assert len(calls) == 2
    for call in calls:
        deterministic = call.args[20]
        assert isinstance(deterministic, ast.Name)
        assert deterministic.id == "_SP_DETERMINISTIC_BACKWARD"


def test_star_forest_builds_exact_cpu_attention_plan():
    """Planning is CPU-safe and preserves each completion's shared prompt ancestry."""
    forest = [(0, 3, [2, 4]), (9, 1, [3])]

    node_start, node_len, node_parent = shared_prefix_fused._forest_to_nodes(forest)

    assert node_start == [0, 3, 5, 9, 10]
    assert node_len == [3, 2, 4, 1, 3]
    assert node_parent == [-1, 0, 0, -1, 3]

    total, passes = shared_prefix_fused._forest_attention_plan(
        node_start, node_len, node_parent, torch.device("cpu")
    )

    assert total == 13
    assert len(passes) == 2

    q_idx, k_idx, cu_q, cu_k, max_q, max_k, causal = passes[0]
    assert q_idx is None
    assert k_idx is None
    assert cu_q.tolist() == [0, 3, 5, 9, 10, 13]
    assert cu_k.tolist() == [0, 3, 5, 9, 10, 13]
    assert max_q == 4
    assert max_k == 4
    assert causal is True

    q_idx, k_idx, cu_q, cu_k, max_q, max_k, causal = passes[1]
    assert q_idx.tolist() == [3, 4, 5, 6, 7, 8, 10, 11, 12]
    assert k_idx.tolist() == [0, 1, 2, 9]
    assert cu_q.tolist() == [0, 6, 9]
    assert cu_k.tolist() == [0, 3, 4]
    assert max_q == 6
    assert max_k == 3
    assert causal is False


def test_attention_plan_rejects_non_dfs_layout():
    with pytest.raises(ValueError, match="not DFS-preorder"):
        shared_prefix_fused._forest_attention_plan(
            node_start=[0, 1, 2, 3],
            node_len=[1, 1, 1, 1],
            node_parent=[-1, 0, -1, 0],
            device=torch.device("cpu"),
        )


def test_cached_chainfirst_plan_rejects_non_dfs_layout():
    with pytest.raises(ValueError, match="not DFS-preorder"):
        shared_prefix_fused._forest_attention_plan_cached(
            node_start=[0, 1, 2, 3],
            node_len=[1, 1, 1, 1],
            node_parent=[-1, 0, -1, 0],
            device=torch.device("cpu"),
        )


def test_plan_cache_reuses_cpu_plan():
    shared_prefix_fused._PLAN_CACHE.clear()
    args = ([0, 2, 5], [2, 3, 1], [-1, 0, 0], torch.device("cpu"))

    first = shared_prefix_fused._forest_attention_plan_cached(*args)
    second = shared_prefix_fused._forest_attention_plan_cached(*args)

    assert second is first
    assert first[0] == 6


@pytest.mark.internal
@pytest.mark.timeout(180)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="shared-prefix parity requires CUDA")
def test_fused_star_matches_dense_forward_and_exact_backward():
    """Compare the composed FlashAttention star with an explicit dense oracle."""
    pytest.importorskip("flash_attn")
    if not torch.cuda.is_bf16_supported():
        pytest.skip("shared-prefix parity requires CUDA bf16 support")

    prefix_len = 7
    completion_lens = [5, 3]
    total_len = prefix_len + sum(completion_lens)
    num_heads = 2
    head_dim = 16
    scale = head_dim**-0.5

    segment_ids = torch.zeros(total_len, dtype=torch.long, device="cuda")
    offset = prefix_len
    for segment_id, completion_len in enumerate(completion_lens, start=1):
        segment_ids[offset : offset + completion_len] = segment_id
        offset += completion_len
    positions = torch.arange(total_len, device="cuda")
    causal = positions[None, :] <= positions[:, None]
    allow = causal & ((segment_ids[None, :] == 0) | (segment_ids[:, None] == segment_ids[None, :]))

    torch.manual_seed(2026)
    base_tensors = [
        torch.randn(total_len, 1, num_heads, head_dim, dtype=torch.bfloat16, device="cuda")
        for _ in range(3)
    ]
    reference_q, reference_k, reference_v = [
        tensor.detach().clone().requires_grad_(True) for tensor in base_tensors
    ]
    fused_q, fused_k, fused_v = [
        tensor.detach().clone().requires_grad_(True) for tensor in base_tensors
    ]

    scores = (
        torch.einsum("thd,shd->hts", reference_q[:, 0].float(), reference_k[:, 0].float()) * scale
    )
    probabilities = torch.softmax(scores.masked_fill(~allow.unsqueeze(0), float("-inf")), dim=-1)
    reference_output = torch.einsum(
        "hts,shd->thd", probabilities, reference_v[:, 0].float()
    ).reshape(total_len, 1, num_heads * head_dim)
    fused_output = shared_prefix_fused.flash_composed_forest_attention(
        fused_q, fused_k, fused_v, [(0, prefix_len, completion_lens)], scale=scale
    )

    torch.testing.assert_close(fused_output.float(), reference_output, rtol=2e-2, atol=2e-2)

    upstream = torch.randn_like(fused_output)
    reference_output.backward(upstream.float())
    fused_output.backward(upstream)
    for fused_gradient, reference_gradient in zip(
        (fused_q.grad, fused_k.grad, fused_v.grad),
        (reference_q.grad, reference_k.grad, reference_v.grad),
    ):
        assert fused_gradient is not None
        assert reference_gradient is not None
        torch.testing.assert_close(
            fused_gradient.float(), reference_gradient.float(), rtol=3e-2, atol=3e-2
        )


def test_flash_attn_varlen_backward_contract():
    """Guard the private FlashAttention API used by the exact custom backward."""
    interface = pytest.importorskip("flash_attn.flash_attn_interface")
    backward = interface._flash_attn_varlen_backward
    if not inspect.isfunction(backward):
        backward = backward._init_fn

    assert list(inspect.signature(backward).parameters) == [
        "dout",
        "q",
        "k",
        "v",
        "out",
        "softmax_lse",
        "dq",
        "dk",
        "dv",
        "cu_seqlens_q",
        "cu_seqlens_k",
        "max_seqlen_q",
        "max_seqlen_k",
        "dropout_p",
        "softmax_scale",
        "causal",
        "window_size_left",
        "window_size_right",
        "softcap",
        "alibi_slopes",
        "deterministic",
        "rng_state",
        "zero_tensors",
    ]
