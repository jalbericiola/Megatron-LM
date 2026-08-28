# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from contextlib import nullcontext
from types import MethodType, SimpleNamespace

import pytest
import torch

from megatron.core import tensor_parallel
from megatron.core.models.hybrid import shared_prefix as shared_prefix_module
from megatron.core.models.hybrid.shared_prefix import (
    SharedPrefixLayout,
    _forward_mamba_layer_shared_prefix,
    forward_hybrid_stack_shared_prefix,
)
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.moe import moe_layer as moe_layer_module
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.moe.router import _expert_bias_token_counts
from megatron.core.transformer.transformer_layer import TransformerLayer


class _SizeOneGroup:
    def size(self):
        return 1

    def rank(self):
        return 0


def test_layout_multiplicities_cover_physical_branch_tails_and_not_topology_pad():
    layout = SharedPrefixLayout(prefix_len=3, completion_lens=[5, 5])

    multiplicities = layout.padded_token_multiplicities(16, "cpu")

    assert multiplicities.tolist() == [2, 2, 2] + [1] * 10 + [0, 0, 0]
    rank_zero = layout.cp_local_indices(16, 2, 0, "cpu")
    rank_one = layout.cp_local_indices(16, 2, 1, "cpu")
    assert multiplicities.index_select(0, rank_zero).tolist() == [2, 2, 2, 1, 1, 0, 0, 0]
    assert multiplicities.index_select(0, rank_one).tolist() == [1] * 8


def test_cp1_mamba_forward_preserves_explicit_topology_tail_and_backward(monkeypatch):
    layout = SharedPrefixLayout(
        prefix_len=3, completion_lens=[5, 5], logical_completion_lens=[2, 4], padding_multiple=8
    )
    observed_branch_shape = None

    def fork_prefix(_mixer, prefix, *, capture_state):
        assert capture_state
        return prefix * 2, None, prefix.new_zeros(1), prefix.new_zeros(1)

    def fork_branches(_mixer, branches, *, conv_context, ssm_initial_state):
        nonlocal observed_branch_shape
        assert conv_context is not None and ssm_initial_state is not None
        observed_branch_shape = branches.shape
        return branches * 3, None

    monkeypatch.setattr(shared_prefix_module, "_fork_mamba_segment", fork_prefix)
    monkeypatch.setattr(shared_prefix_module, "_fork_mamba_branches", fork_branches)
    layer = SimpleNamespace(
        config=SimpleNamespace(
            fp32_residual_connection=False, params_dtype=torch.float32, bias_dropout_fusion=False
        ),
        norm=torch.nn.Identity(),
        mixer=object(),
        bias_dropout_add_exec_handler=nullcontext,
        mamba_bda=lambda **_kwargs: (
            lambda output_with_bias, residual, _dropout: output_with_bias[0] + residual
        ),
        training=True,
        hidden_dropout=0.0,
    )
    hidden_states = torch.randn(16, 1, 4, requires_grad=True)

    output = _forward_mamba_layer_shared_prefix(layer, hidden_states, layout)

    assert output.shape == hidden_states.shape
    assert observed_branch_shape == (8, 2, 4)
    output.sum().backward()
    assert hidden_states.grad is not None
    assert torch.count_nonzero(hidden_states.grad[-3:]) == 12


def test_expert_bias_counts_apply_logical_multiplicity_and_padding_mask():
    routing_map = torch.tensor(
        [[1, 0, 1], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=torch.bool
    )
    multiplicities = torch.tensor([2, 2, 1, 1, 0], dtype=torch.float32)

    counts = _expert_bias_token_counts(
        routing_map,
        padding_mask=torch.tensor([False, False, False, True, False]),
        token_multiplicities=multiplicities,
    )

    torch.testing.assert_close(counts, torch.tensor([3.0, 3.0, 2.0]))
    torch.testing.assert_close(
        _expert_bias_token_counts(
            routing_map, padding_mask=torch.tensor([False, False, False, True, False])
        ),
        torch.tensor([3, 2, 1]),
    )
    with pytest.raises(ValueError, match="one value per routed token"):
        _expert_bias_token_counts(routing_map, token_multiplicities=torch.ones(4))


@pytest.mark.internal
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_real_checkpoint_updates_compiled_expert_bias_count_exactly_once():
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.moe.router import TopKRouter
    from tests.unit_tests.test_utilities import Utils

    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1, context_parallel_size=1
    )
    try:
        model_parallel_cuda_manual_seed(123)
        router = object.__new__(TopKRouter)
        torch.nn.Module.__init__(router)
        router.enable_expert_bias = True
        router.register_buffer(
            "local_tokens_per_expert", torch.zeros(3, dtype=torch.float32, device="cuda")
        )
        routing_map = torch.tensor(
            [[1, 0, 1], [0, 1, 0], [1, 1, 0]], dtype=torch.bool, device="cuda"
        )
        multiplicities = torch.tensor([2.0, 2.0, 1.0], dtype=torch.float32, device="cuda")

        def custom_forward(hidden_states):
            router._apply_expert_bias(routing_map, token_multiplicities=multiplicities)
            return hidden_states.square()

        hidden_states = torch.randn(3, device="cuda", requires_grad=True)
        output = tensor_parallel.checkpoint(custom_forward, False, hidden_states)
        torch.testing.assert_close(
            router.local_tokens_per_expert, torch.zeros_like(router.local_tokens_per_expert)
        )

        output.sum().backward()

        torch.testing.assert_close(
            router.local_tokens_per_expert, torch.tensor([3.0, 3.0, 2.0], device="cuda")
        )
    finally:
        Utils.destroy_model_parallel()


def test_selective_moe_checkpoint_closure_retains_multiplicity(monkeypatch):
    layer = object.__new__(MoELayer)
    torch.nn.Module.__init__(layer)
    layer.training = True
    layer.attn_tp_group = _SizeOneGroup()
    layer.config = SimpleNamespace(sequence_parallel=False, fp8=None, fp4=None)
    layer.moe_layer_recompute = True
    layer.fwd_execution_map = {"route", "expert_compute", "postprocess"}
    multiplicities = torch.tensor([2.0, 2.0, 1.0])
    layer._shared_prefix_token_multiplicities = multiplicities
    observed = []

    def shared_experts_compute(_self, _hidden_states):
        return None

    def route(_self, hidden_states, _padding_mask, token_multiplicities):
        observed.append(token_multiplicities.clone())
        probs = hidden_states.new_ones(hidden_states.shape[0], 1)
        routing_map = torch.ones(hidden_states.shape[0], 1, dtype=torch.bool)
        return probs, routing_map

    def preprocess(_self, hidden_states, probs, _routing_map):
        return hidden_states, probs

    def dispatch(_self, hidden_states, probs):
        return hidden_states, probs

    def routed_experts_compute(_self, hidden_states, _probs):
        return hidden_states, None

    def passthrough(_self, output, *_args):
        return output

    layer.shared_experts_compute = MethodType(shared_experts_compute, layer)
    layer.route = MethodType(route, layer)
    layer.preprocess = MethodType(preprocess, layer)
    layer.dispatch = MethodType(dispatch, layer)
    layer.routed_experts_compute = MethodType(routed_experts_compute, layer)
    layer.combine = MethodType(passthrough, layer)
    layer.postprocess = MethodType(passthrough, layer)

    def replay_after_scope_removed(run_function, _distribute, *args):
        del layer._shared_prefix_token_multiplicities
        return run_function(*args)

    monkeypatch.setattr(moe_layer_module.tensor_parallel, "checkpoint", replay_after_scope_removed)
    hidden_states = torch.randn(3, 1, 4, requires_grad=True)

    output, bias = layer(hidden_states)

    torch.testing.assert_close(output, hidden_states)
    assert bias is None
    assert len(observed) == 1
    torch.testing.assert_close(observed[0], multiplicities)


class _RecordingAttentionLayer(TransformerLayer):
    def __init__(self, events):
        torch.nn.Module.__init__(self)
        self.config = SimpleNamespace(cuda_graph_impl="none")
        attention = object.__new__(SelfAttention)
        torch.nn.Module.__init__(attention)
        self.self_attention = attention
        self.is_moe_layer = False
        self.events = events

    def forward(self, *, hidden_states, **_kwargs):
        self.events.append(tuple(self.self_attention._shared_prefix_forest))
        return hidden_states + 1, None


class _RecordingMoELayer(TransformerLayer):
    def __init__(self, events):
        torch.nn.Module.__init__(self)
        self.config = SimpleNamespace(cuda_graph_impl="none")
        self.self_attention = IdentityOp()
        self.is_moe_layer = True
        self.mlp = torch.nn.Module()
        self.events = events

    def forward(self, *, hidden_states, **_kwargs):
        self.events.append(self.mlp._shared_prefix_token_multiplicities.clone())
        return hidden_states + 1, None


def test_full_uniform_checkpoint_reinstalls_attention_and_moe_scopes(monkeypatch):
    attention_events = []
    moe_events = []
    attention_layer = _RecordingAttentionLayer(attention_events)
    moe_layer = _RecordingMoELayer(moe_events)
    stack = SimpleNamespace(
        layers=[attention_layer, moe_layer],
        pg_collection=SimpleNamespace(cp=_SizeOneGroup(), tp=_SizeOneGroup()),
        config=SimpleNamespace(
            sequence_parallel=False,
            recompute_granularity="full",
            recompute_num_layers=1,
            distribute_saved_activations=False,
            moe_router_enable_expert_bias=True,
        ),
        training=True,
        post_process=False,
        post_layer_norm=False,
    )
    monkeypatch.setattr(shared_prefix_module, "_validate_hybrid_stack", lambda *_args: None)
    checkpoint_ranges = []

    def checkpoint_with_replay(run_function, _distribute, hidden_states):
        first = run_function(hidden_states)
        replay = run_function(hidden_states)
        torch.testing.assert_close(first, replay)
        checkpoint_ranges.append(True)
        return first

    monkeypatch.setattr(shared_prefix_module.tensor_parallel, "checkpoint", checkpoint_with_replay)
    layout = SharedPrefixLayout(prefix_len=2, completion_lens=[2, 1])
    hidden_states = torch.zeros(layout.total_len, 1, 4, requires_grad=True)

    output = forward_hybrid_stack_shared_prefix(
        stack, hidden_states, layout, rotary_pos_emb=torch.zeros(layout.total_len, 1, 1, 1)
    )

    torch.testing.assert_close(output, hidden_states + 2)
    assert len(checkpoint_ranges) == 2
    assert attention_events == [tuple(layout.forest), tuple(layout.forest)]
    assert len(moe_events) == 2
    for event in moe_events:
        torch.testing.assert_close(event, torch.tensor([2, 2, 1, 1, 1], dtype=torch.float32))
    assert not hasattr(attention_layer.self_attention, "_shared_prefix_forest")
    assert not hasattr(moe_layer.mlp, "_shared_prefix_token_multiplicities")
