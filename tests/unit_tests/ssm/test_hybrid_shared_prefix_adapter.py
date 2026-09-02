# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import inspect
from types import SimpleNamespace

import pytest
import torch

from megatron.core.models.hybrid.hybrid_model import (
    HybridModel,
    _canonicalize_shared_prefix_cp_sequence,
    _hybrid_mtp_is_enabled,
    _iter_shared_prefix_mtp_branches,
    _pack_shared_prefix_mtp_branches,
    _reconstruct_shared_prefix_mtp_branches,
    _validate_shared_prefix_mtp_attention_backend,
    _validate_shared_prefix_mtp_pattern,
)
from megatron.core.models.hybrid.shared_prefix import (
    SHARED_PREFIX_CP_TP_SP_TRAINING_CAPABILITY,
    SHARED_PREFIX_CP_TRAINING_CAPABILITY,
    SHARED_PREFIX_EXPLICIT_PHYSICAL_PADDING_CAPABILITY,
    SHARED_PREFIX_FULL_RECOMPUTE_CAPABILITY,
    SHARED_PREFIX_MOE_EXPERT_BIAS_CAPABILITY,
    SHARED_PREFIX_MTP_DENSE_HEADS_CAPABILITY,
    SHARED_PREFIX_POSITIONLESS_ATTENTION_CAPABILITY,
    SHARED_PREFIX_TP_SP_TRAINING_CAPABILITY,
    SHARED_PREFIX_TRAINING_CAPABILITIES,
    SHARED_PREFIX_TRAINING_CAPABILITY,
    SharedPrefixLayout,
    _validate_hybrid_stack,
    forward_hybrid_stack_shared_prefix,
)
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.transformer_layer import TransformerLayer


def _relative_l2(reference, actual):
    reference = reference.double().flatten()
    actual = actual.double().flatten()
    denominator = reference.norm().clamp_min(torch.finfo(torch.float64).tiny)
    return ((reference - actual).norm() / denominator).item()


def _cosine(reference, actual):
    reference = reference.double().flatten()
    actual = actual.double().flatten()
    denominator = (reference.norm() * actual.norm()).clamp_min(torch.finfo(torch.float64).tiny)
    return (torch.dot(reference, actual) / denominator).item()


def _gradient_norms(reference, actual):
    reference = reference.double().flatten()
    actual = actual.double().flatten()
    return {
        "reference": reference.norm().item(),
        "actual": actual.norm().item(),
        "difference": (reference - actual).norm().item(),
        "max_abs_difference": (reference - actual).abs().max().item(),
    }


class _SizeOneGroup:
    def size(self):
        return 1


def _validation_stack(*, layers=(), **config_overrides):
    config = SimpleNamespace(
        tensor_model_parallel_size=1,
        context_parallel_size=1,
        sequence_parallel=False,
        recompute_granularity=None,
        recompute_method=None,
        recompute_num_layers=None,
        distribute_saved_activations=False,
        fine_grained_activation_offloading=False,
        cuda_graph_impl="none",
        fp8=None,
        fp4=None,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        window_size=None,
        softmax_type="vanilla",
        num_moe_experts=None,
        moe_router_force_load_balancing=False,
        moe_router_force_biased=None,
        moe_router_load_balancing_type="none",
        moe_aux_loss_coeff=0.0,
        moe_z_loss_coeff=None,
        moe_input_jitter_eps=None,
        moe_router_enable_expert_bias=False,
        moe_expert_capacity_factor=None,
        qk_clip=False,
        log_max_attention_logit=False,
    )
    for name, value in config_overrides.items():
        setattr(config, name, value)
    return SimpleNamespace(
        tp_group=_SizeOneGroup(),
        pp_group=_SizeOneGroup(),
        pg_collection=SimpleNamespace(tp=_SizeOneGroup(), cp=_SizeOneGroup()),
        config=config,
        layers=list(layers),
    )


def _stub_attention_layer():
    attention = object.__new__(SelfAttention)
    torch.nn.Module.__init__(attention)
    attention.checkpoint_core_attention = False
    attention.pg_collection = SimpleNamespace(tp=_SizeOneGroup(), cp=_SizeOneGroup())
    layer = object.__new__(TransformerLayer)
    torch.nn.Module.__init__(layer)
    layer.self_attention = attention
    return layer


def test_shared_prefix_layout_and_capability_are_explicit():
    layout = SharedPrefixLayout(prefix_len=3, completion_lens=[2, 4])

    assert SHARED_PREFIX_TRAINING_CAPABILITY == "hybrid_star_cp1_tp1_v1"
    assert SHARED_PREFIX_CP_TRAINING_CAPABILITY == "hybrid_star_cp_v1"
    assert (
        SHARED_PREFIX_EXPLICIT_PHYSICAL_PADDING_CAPABILITY
        == "hybrid_star_explicit_physical_padding_v1"
    )
    assert SHARED_PREFIX_MOE_EXPERT_BIAS_CAPABILITY == "hybrid_star_moe_expert_bias_v1"
    assert SHARED_PREFIX_FULL_RECOMPUTE_CAPABILITY == "hybrid_star_full_uniform_recompute_v1"
    assert SHARED_PREFIX_TP_SP_TRAINING_CAPABILITY == "hybrid_star_cp1_tp_sp_v1"
    assert SHARED_PREFIX_CP_TP_SP_TRAINING_CAPABILITY == "hybrid_star_cp_tp_sp_v1"
    assert SHARED_PREFIX_MTP_DENSE_HEADS_CAPABILITY == "hybrid_star_mtp_dense_heads_v1"
    assert SHARED_PREFIX_MTP_DENSE_HEADS_CAPABILITY in SHARED_PREFIX_TRAINING_CAPABILITIES
    assert (
        SHARED_PREFIX_POSITIONLESS_ATTENTION_CAPABILITY
        == "hybrid_star_positionless_attention_v1"
    )
    assert SHARED_PREFIX_POSITIONLESS_ATTENTION_CAPABILITY in SHARED_PREFIX_TRAINING_CAPABILITIES
    assert SHARED_PREFIX_TRAINING_CAPABILITIES == frozenset(
        {
            "hybrid_star_cp1_tp1_v1",
            "hybrid_star_cp_v1",
            "hybrid_star_explicit_physical_padding_v1",
            "hybrid_star_moe_expert_bias_v1",
            "hybrid_star_full_uniform_recompute_v1",
            "hybrid_star_cp1_tp_sp_v1",
            "hybrid_star_cp_tp_sp_v1",
            "hybrid_star_positionless_attention_v1",
            "hybrid_star_mtp_dense_heads_v1",
        }
    )
    assert layout.total_len == 9
    assert layout.forest == [(0, 3, [2, 4])]
    assert layout.completion_slices() == (slice(3, 5), slice(5, 9))
    assert layout.position_ids("cpu").tolist() == [0, 1, 2, 3, 4, 3, 4, 5, 6]
    model_parameter = inspect.signature(HybridModel.forward).parameters["shared_prefix_layout"]
    assert model_parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert model_parameter.default is None


def test_shared_prefix_mtp_narrows_only_the_predictor_mamba_scope():
    _validate_shared_prefix_mtp_pattern("*-")
    _validate_shared_prefix_mtp_pattern("*E")

    with pytest.raises(NotImplementedError, match="Mamba layer inside the MTP predictor"):
        _validate_shared_prefix_mtp_pattern("M*-")


def test_shared_prefix_mtp_rejects_only_the_local_attention_backend():
    """Only mcore's local attention can never run the packed THD MTP branches."""
    from megatron.core.transformer.enums import AttnBackend

    for backend in (AttnBackend.flash, AttnBackend.fused, AttnBackend.unfused, AttnBackend.auto):
        _validate_shared_prefix_mtp_attention_backend(backend)

    with pytest.raises(NotImplementedError, match="local DotProductAttention does not support"):
        _validate_shared_prefix_mtp_attention_backend(AttnBackend.local)


def test_hybrid_mtp_runtime_requires_a_positive_configured_depth():
    assert _hybrid_mtp_is_enabled(5, "*E", 5)
    assert not _hybrid_mtp_is_enabled(0, "*E", 5)
    assert not _hybrid_mtp_is_enabled(None, "*E", 5)
    assert not _hybrid_mtp_is_enabled(5, None, 0)


def test_shared_prefix_mtp_reconstructs_variable_physical_branches_and_gradients():
    layout = SharedPrefixLayout(
        prefix_len=2,
        completion_lens=[4, 6],
        logical_completion_lens=[2, 3],
        padding_multiple=2,
    )
    physical_len = layout.total_len + 2
    hidden_states = torch.arange(physical_len, dtype=torch.float32).reshape(-1, 1, 1)
    hidden_states.requires_grad_(True)
    input_ids = torch.arange(physical_len, dtype=torch.long).unsqueeze(0)
    loss_mask = torch.tensor(
        [[0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0]], dtype=torch.float32
    )

    branches = _reconstruct_shared_prefix_mtp_branches(
        hidden_states, input_ids, loss_mask, layout
    )

    assert [branch_ids.tolist() for _, branch_ids, _ in branches] == [
        [[0, 1, 2, 3, 4, 5]],
        [[0, 1, 6, 7, 8, 9, 10, 11]],
    ]
    assert [branch_mask.tolist() for _, _, branch_mask in branches] == [
        [[0, 0, 1, 1, 0, 0]],
        [[0, 0, 1, 1, 1, 0, 0, 0]],
    ]

    sum(branch_hidden.sum() for branch_hidden, _, _ in branches).backward()
    assert hidden_states.grad.flatten().tolist() == [
        2,
        2,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        0,
        0,
    ]


@pytest.mark.parametrize(
    "reconstruct",
    [_reconstruct_shared_prefix_mtp_branches, _pack_shared_prefix_mtp_branches],
    ids=["per_branch", "packed"],
)
def test_shared_prefix_mtp_reconstruction_rejects_loss_on_prompt_or_padding(reconstruct):
    layout = SharedPrefixLayout(
        prefix_len=2,
        completion_lens=[4, 4],
        logical_completion_lens=[2, 2],
        padding_multiple=2,
    )
    physical_len = layout.total_len + 2
    hidden_states = torch.zeros(physical_len, 1, 4)
    input_ids = torch.zeros(1, physical_len, dtype=torch.long)
    loss_mask = torch.zeros(1, physical_len)

    loss_mask[0, 0] = 1
    with pytest.raises(ValueError, match="exclude every prompt token"):
        reconstruct(hidden_states, input_ids, loss_mask, layout)

    loss_mask.zero_()
    loss_mask[0, layout.prefix_len + layout.logical_completion_lens[0]] = 1
    with pytest.raises(ValueError, match="ordinary per-sequence padding"):
        reconstruct(hidden_states, input_ids, loss_mask, layout)

    loss_mask.zero_()
    loss_mask[0, layout.total_len] = 1
    with pytest.raises(ValueError, match="topology-only padding"):
        reconstruct(hidden_states, input_ids, loss_mask, layout)

    # Loss on real completion tokens passes the single combined device-side check.
    loss_mask.zero_()
    loss_mask[0, layout.prefix_len] = 1
    reconstruct(hidden_states, input_ids, loss_mask, layout)


def test_shared_prefix_mtp_cp1_canonicalization_is_identity():
    layout = SharedPrefixLayout(prefix_len=2, completion_lens=[2, 2])
    local_tensor = torch.randn(layout.total_len, 1, 4, requires_grad=True)

    canonical = _canonicalize_shared_prefix_cp_sequence(
        local_tensor,
        layout,
        layout.total_len,
        _SizeOneGroup(),
        reduce_scatter_grad=True,
    )

    assert canonical is local_tensor


def test_shared_prefix_mtp_branch_iterator_composes_cp_ownership_without_dense_copy():
    layout = SharedPrefixLayout(
        prefix_len=2,
        completion_lens=[6, 6],
        logical_completion_lens=[3, 4],
        padding_multiple=8,
    )
    hidden_states = torch.arange(layout.total_len, dtype=torch.float32).reshape(-1, 1, 1)
    input_ids = torch.arange(layout.total_len, dtype=torch.long).unsqueeze(0)
    loss_mask = torch.tensor(
        [[0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0]], dtype=torch.float32
    )
    full_branches = _reconstruct_shared_prefix_mtp_branches(
        hidden_states, input_ids, loss_mask, layout
    )

    for cp_rank in (0, 1):
        local_branches = tuple(
            _iter_shared_prefix_mtp_branches(
                hidden_states,
                input_ids,
                loss_mask,
                layout,
                cp_size=2,
                cp_rank=cp_rank,
            )
        )
        for local, full in zip(local_branches, full_branches, strict=True):
            cp_indices = layout.cp_local_indices(full[0].shape[0], 2, cp_rank, "cpu")
            torch.testing.assert_close(local[0], full[0].index_select(0, cp_indices))
            torch.testing.assert_close(local[1], full[1].index_select(1, cp_indices))
            torch.testing.assert_close(local[2], full[2].index_select(1, cp_indices))


def test_shared_prefix_mtp_packing_matches_branch_reconstruction():
    """The single-call THD pack equals the branch-by-branch reconstruction, per CP rank."""
    layout = SharedPrefixLayout(
        prefix_len=2,
        completion_lens=[6, 6],
        logical_completion_lens=[3, 4],
        padding_multiple=8,
    )
    physical_len = layout.total_len + 2
    hidden_states = torch.arange(physical_len, dtype=torch.float32).reshape(-1, 1, 1)
    hidden_states.requires_grad_(True)
    input_ids = torch.arange(physical_len, dtype=torch.long).unsqueeze(0)
    loss_mask = torch.tensor(
        [[0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]], dtype=torch.float32
    )
    branch_lengths = [layout.prefix_len + length for length in layout.completion_lens]

    for cp_size, cp_rank in ((1, 0), (2, 0), (2, 1)):
        branches = tuple(
            _iter_shared_prefix_mtp_branches(
                hidden_states, input_ids, loss_mask, layout, cp_size=cp_size, cp_rank=cp_rank
            )
        )
        packed_hidden, packed_ids, packed_mask, packed_positions = (
            _pack_shared_prefix_mtp_branches(
                hidden_states, input_ids, loss_mask, layout, cp_size=cp_size, cp_rank=cp_rank
            )
        )
        assert packed_hidden.shape[0] == sum(branch_lengths) // cp_size
        torch.testing.assert_close(packed_hidden, torch.cat([b[0] for b in branches], dim=0))
        torch.testing.assert_close(packed_ids, torch.cat([b[1] for b in branches], dim=1))
        torch.testing.assert_close(packed_mask, torch.cat([b[2] for b in branches], dim=1))
        # Positions restart at zero per dense branch and follow the same zigzag
        # ownership as the tokens, so THD RoPE and roll_tensor see ordinary branches.
        expected_positions = torch.cat(
            [
                (
                    layout.cp_local_indices(branch_len, cp_size, cp_rank, "cpu")
                    if cp_size > 1
                    else torch.arange(branch_len, dtype=torch.long)
                )
                for branch_len in branch_lengths
            ]
        ).unsqueeze(0)
        torch.testing.assert_close(packed_positions, expected_positions)

    # CP1 spells out the branch-major contract: the prompt is repeated once per
    # branch and its backbone rows receive the summed gradient of every branch.
    packed_hidden, packed_ids, _, packed_positions = _pack_shared_prefix_mtp_branches(
        hidden_states, input_ids, loss_mask, layout
    )
    assert packed_ids.tolist() == [[0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 8, 9, 10, 11, 12, 13]]
    assert packed_positions.tolist() == [[*range(8), *range(8)]]
    packed_hidden.sum().backward()
    assert hidden_states.grad.flatten().tolist() == [2, 2, *([1] * 12), 0, 0]


@pytest.mark.parametrize(
    ("config_name", "config_value", "message"),
    [
        ("moe_router_force_load_balancing", True, "randomized forced MoE routing"),
        ("moe_router_force_biased", 0.1, "randomized forced MoE router bias"),
    ],
)
def test_shared_prefix_validation_rejects_randomized_moe_routing(
    config_name, config_value, message
):
    layout = SharedPrefixLayout(prefix_len=2, completion_lens=[2])
    stack = _validation_stack(num_moe_experts=2, **{config_name: config_value})
    hidden_states = torch.empty(layout.total_len, 1, 8, dtype=torch.bfloat16)

    with pytest.raises(NotImplementedError, match=message):
        _validate_hybrid_stack(stack, hidden_states, layout)


def test_shared_prefix_validation_rejects_fp32_before_flash_attention():
    layout = SharedPrefixLayout(prefix_len=2, completion_lens=[2])
    stack = _validation_stack(layers=[_stub_attention_layer()])
    hidden_states = torch.empty(layout.total_len, 1, 8, dtype=torch.float32)

    with pytest.raises(TypeError, match="requires fp16 or bf16 hidden states"):
        _validate_hybrid_stack(stack, hidden_states, layout)


def test_shared_prefix_validation_requires_explicit_physical_moe_layout():
    physical_layout = SharedPrefixLayout(
        prefix_len=2, completion_lens=[6, 6], logical_completion_lens=[2, 2], padding_multiple=8
    )
    implicit_layout = SharedPrefixLayout(prefix_len=2, completion_lens=[6, 6])
    stack = _validation_stack(num_moe_experts=2, moe_router_enable_expert_bias=True)
    hidden_states = torch.empty(16, 1, 8, dtype=torch.bfloat16)

    with pytest.raises(NotImplementedError, match="explicit physical branch padding"):
        _validate_hybrid_stack(
            stack,
            torch.empty(implicit_layout.total_len, 1, 8, dtype=torch.bfloat16),
            implicit_layout,
        )
    _validate_hybrid_stack(stack, hidden_states, physical_layout)


def test_shared_prefix_validation_accepts_only_uniform_full_recompute():
    layout = SharedPrefixLayout(prefix_len=2, completion_lens=[2])
    hidden_states = torch.empty(layout.total_len, 1, 8, dtype=torch.bfloat16)

    _validate_hybrid_stack(
        _validation_stack(
            recompute_granularity="full", recompute_method="uniform", recompute_num_layers=1
        ),
        hidden_states,
        layout,
    )
    with pytest.raises(NotImplementedError, match="only the uniform method"):
        _validate_hybrid_stack(
            _validation_stack(
                recompute_granularity="full", recompute_method="block", recompute_num_layers=1
            ),
            hidden_states,
            layout,
        )


@pytest.mark.parametrize("config_name", ["qk_clip", "log_max_attention_logit"])
def test_shared_prefix_validation_rejects_unavailable_attention_statistics(config_name):
    layout = SharedPrefixLayout(prefix_len=2, completion_lens=[2])
    stack = _validation_stack(layers=[_stub_attention_layer()], **{config_name: True})
    hidden_states = torch.empty(layout.total_len, 1, 8, dtype=torch.bfloat16)

    with pytest.raises(NotImplementedError, match="QK-clipping/max-logit statistics"):
        _validate_hybrid_stack(stack, hidden_states, layout)


@pytest.mark.internal
@pytest.mark.timeout(180)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="shared-prefix parity requires CUDA")
@pytest.mark.parametrize("position_embedding_type", ["rope", "none"])
def test_hybrid_shared_prefix_forward_and_gradient_parity(
    monkeypatch, position_embedding_type
):
    pytest.importorskip("causal_conv1d")
    pytest.importorskip("flash_attn")
    pytest.importorskip("mamba_ssm")
    if not torch.cuda.is_bf16_supported():
        pytest.skip("shared-prefix parity requires CUDA bf16 support")

    from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
    from megatron.core.models.hybrid.hybrid_block import HybridStack
    from megatron.core.models.hybrid.hybrid_layer_allocation import validate_segment_layers
    from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer import TransformerConfig
    from megatron.core.transformer.enums import AttnBackend
    from tests.unit_tests.test_utilities import Utils

    hidden_size = 256
    num_heads = 4
    layout = SharedPrefixLayout(prefix_len=24, completion_lens=[20, 28])
    layer_types = validate_segment_layers("M*-")
    monkeypatch.setenv("NVTE_APPLY_QK_LAYER_SCALING", "1")

    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1, context_parallel_size=1
    )
    try:
        model_parallel_cuda_manual_seed(123)
        config = TransformerConfig(
            hidden_size=hidden_size,
            num_layers=len(layer_types),
            num_attention_heads=num_heads,
            use_cpu_initialization=True,
            bf16=True,
            params_dtype=torch.bfloat16,
            use_mamba_mem_eff_path=False,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            attention_backend=AttnBackend.unfused,
            apply_rope_fusion=False,
            apply_query_key_layer_scaling=True,
        )
        process_groups = ProcessGroupCollection.use_mpu_process_groups(
            required_pgs=["tp", "pp", "cp", "dp_cp"]
        )
        stack = HybridStack(
            config,
            hybrid_stack_spec.submodules,
            layer_type_list=layer_types,
            pp_layer_offset=0,
            pg_collection=process_groups,
        ).cuda()
        attention_layers = [
            layer.self_attention
            for layer in stack.layers
            if isinstance(layer, TransformerLayer)
            and isinstance(layer.self_attention, SelfAttention)
        ]
        assert config.apply_query_key_layer_scaling
        assert [attention.layer_number for attention in attention_layers] == [2]
        rotary = (
            RotaryEmbedding(
                kv_channels=hidden_size // num_heads,
                rotary_percent=1.0,
                rotary_base=10000,
            ).cuda()
            if position_embedding_type == "rope"
            else None
        )

        torch.manual_seed(7)
        prefix = torch.randn(
            layout.prefix_len,
            1,
            hidden_size,
            dtype=torch.bfloat16,
            device="cuda",
            requires_grad=True,
        )
        completions = [
            torch.randn(
                length, 1, hidden_size, dtype=torch.bfloat16, device="cuda", requires_grad=True
            )
            for length in layout.completion_lens
        ]
        # A squared-output loss after the stack's final RMSNorm is nearly constant at
        # initialization. Reuse fixed cotangents so this compares a non-degenerate VJP.
        probe_generator = torch.Generator(device="cuda").manual_seed(19)
        gradient_probes = [
            torch.randn(
                completion.shape, dtype=torch.float32, device="cuda", generator=probe_generator
            )
            / completion.numel() ** 0.5
            for completion in completions
        ]

        def causal_mask(length):
            return ~torch.tril(torch.ones(1, 1, length, length, dtype=torch.bool, device="cuda"))

        def dense_outputs():
            outputs = []
            for completion in completions:
                length = layout.prefix_len + completion.shape[0]
                output = stack(
                    torch.cat([prefix, completion], dim=0),
                    attention_mask=causal_mask(length),
                    rotary_pos_emb=(rotary.get_emb(length) if rotary is not None else None),
                )
                outputs.append(output[layout.prefix_len :])
            return outputs

        def shared_outputs():
            packed = torch.cat([prefix, *completions], dim=0)
            position_ids = layout.position_ids(packed.device)
            rotary_pos_emb = (
                rotary.get_emb(int(position_ids.max().item()) + 1).index_select(
                    0, position_ids
                )
                if rotary is not None
                else None
            )
            output = forward_hybrid_stack_shared_prefix(
                stack,
                packed,
                layout,
                rotary_pos_emb=rotary_pos_emb,
                position_embedding_type=position_embedding_type,
            )
            return [output[completion_slice] for completion_slice in layout.completion_slices()]

        with torch.no_grad():
            dense = dense_outputs()
            shared = shared_outputs()
        forward_errors = [_relative_l2(d, s) for d, s in zip(dense, shared)]
        assert max(forward_errors) < 0.02
        for layer in stack.layers:
            if hasattr(layer, "self_attention"):
                assert not hasattr(layer.self_attention, "_shared_prefix_forest")

        watched_parameters = {
            name: parameter
            for name, parameter in stack.named_parameters()
            if "mixer.in_proj.weight" in name or "self_attention.linear_qkv.weight" in name
        }
        assert any("mixer.in_proj.weight" in name for name in watched_parameters)
        assert any("self_attention.linear_qkv.weight" in name for name in watched_parameters)

        def clear_gradients():
            prefix.grad = None
            for completion in completions:
                completion.grad = None
            stack.zero_grad(set_to_none=True)

        def probe_loss(outputs):
            return sum(
                (output.float() * probe).sum()
                for output, probe in zip(outputs, gradient_probes, strict=True)
            )

        clear_gradients()
        probe_loss(dense_outputs()).backward()
        dense_prefix_grad = prefix.grad.detach().clone()
        dense_completion_grads = [completion.grad.detach().clone() for completion in completions]
        dense_parameter_grads = {
            name: parameter.grad.detach().clone() for name, parameter in watched_parameters.items()
        }

        clear_gradients()
        probe_loss(shared_outputs()).backward()
        shared_prefix_grad = prefix.grad.detach().clone()
        shared_completion_grads = [completion.grad.detach().clone() for completion in completions]
        shared_parameter_grads = {
            name: parameter.grad.detach().clone() for name, parameter in watched_parameters.items()
        }

        prefix_gradient_error = _relative_l2(dense_prefix_grad, shared_prefix_grad)
        prefix_gradient_cosine = _cosine(dense_prefix_grad, shared_prefix_grad)
        completion_gradient_errors = [
            _relative_l2(dense_grad, shared_grad)
            for dense_grad, shared_grad in zip(dense_completion_grads, shared_completion_grads)
        ]
        completion_gradient_cosines = [
            _cosine(dense_grad, shared_grad)
            for dense_grad, shared_grad in zip(dense_completion_grads, shared_completion_grads)
        ]
        parameter_gradient_errors = {
            name: _relative_l2(dense_parameter_grads[name], shared_parameter_grads[name])
            for name in dense_parameter_grads
        }
        parameter_gradient_cosines = {
            name: _cosine(dense_parameter_grads[name], shared_parameter_grads[name])
            for name in dense_parameter_grads
        }
        gradient_norms = {
            "prefix": _gradient_norms(dense_prefix_grad, shared_prefix_grad),
            "completions": [
                _gradient_norms(dense_grad, shared_grad)
                for dense_grad, shared_grad in zip(dense_completion_grads, shared_completion_grads)
            ],
            "parameters": {
                name: _gradient_norms(dense_parameter_grads[name], shared_parameter_grads[name])
                for name in dense_parameter_grads
            },
        }
        print(
            "shared-prefix Hybrid parity: "
            f"forward_relative_l2={forward_errors}, "
            f"prefix_gradient_relative_l2={prefix_gradient_error}, "
            f"prefix_gradient_cosine={prefix_gradient_cosine}, "
            f"completion_gradient_relative_l2={completion_gradient_errors}, "
            f"completion_gradient_cosine={completion_gradient_cosines}, "
            f"parameter_gradient_relative_l2={parameter_gradient_errors}, "
            f"parameter_gradient_cosine={parameter_gradient_cosines}, "
            f"gradient_norms={gradient_norms}"
        )

        assert prefix_gradient_error < 0.02
        assert prefix_gradient_cosine > 0.999
        for gradient_error, gradient_cosine in zip(
            completion_gradient_errors, completion_gradient_cosines
        ):
            assert gradient_error < 0.02
            assert gradient_cosine > 0.999
        for name in dense_parameter_grads:
            assert parameter_gradient_errors[name] < 0.02
            assert parameter_gradient_cosines[name] > 0.999
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.internal
@pytest.mark.timeout(180)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="shared-prefix parity requires CUDA")
@pytest.mark.parametrize("position_embedding_type", ["rope", "none"])
def test_hybrid_model_explicit_shared_prefix_forward_matches_dense(
    position_embedding_type,
):
    pytest.importorskip("causal_conv1d")
    pytest.importorskip("flash_attn")
    pytest.importorskip("mamba_ssm")
    if not torch.cuda.is_bf16_supported():
        pytest.skip("shared-prefix parity requires CUDA bf16 support")

    from megatron.core.models.hybrid.hybrid_layer_allocation import validate_segment_layers
    from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer import TransformerConfig
    from megatron.core.transformer.enums import AttnBackend
    from tests.unit_tests.test_utilities import Utils

    hidden_size = 256
    num_heads = 4
    vocab_size = 512
    pattern = "M*-"
    layout = SharedPrefixLayout(prefix_len=24, completion_lens=[20, 28])

    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1, context_parallel_size=1
    )
    try:
        model_parallel_cuda_manual_seed(123)
        config = TransformerConfig(
            hidden_size=hidden_size,
            num_layers=len(validate_segment_layers(pattern)),
            num_attention_heads=num_heads,
            use_cpu_initialization=True,
            bf16=True,
            params_dtype=torch.bfloat16,
            use_mamba_mem_eff_path=False,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            attention_backend=AttnBackend.unfused,
            apply_rope_fusion=False,
        )
        process_groups = ProcessGroupCollection.use_mpu_process_groups(
            required_pgs=["tp", "pp", "cp", "embd", "dp_cp"]
        )
        model = HybridModel(
            config=config,
            hybrid_stack_spec=hybrid_stack_spec,
            vocab_size=vocab_size,
            max_sequence_length=layout.total_len,
            hybrid_layer_pattern=pattern,
            position_embedding_type=position_embedding_type,
            pre_process=True,
            post_process=True,
            parallel_output=False,
            pg_collection=process_groups,
        ).cuda()

        torch.manual_seed(11)
        prefix = torch.randint(0, vocab_size, (layout.prefix_len,), device="cuda")
        completions = [
            torch.randint(0, vocab_size, (length,), device="cuda")
            for length in layout.completion_lens
        ]

        def causal_mask(length):
            return ~torch.tril(torch.ones(1, 1, length, length, dtype=torch.bool, device="cuda"))

        with torch.no_grad():
            dense_completion_logits = []
            for completion in completions:
                tokens = torch.cat([prefix, completion]).unsqueeze(0)
                length = tokens.shape[1]
                logits = model(
                    tokens, torch.arange(length, device="cuda").unsqueeze(0), causal_mask(length)
                )
                dense_completion_logits.append(logits[0, layout.prefix_len :])

            packed_tokens = torch.cat([prefix, *completions]).unsqueeze(0)
            shared_logits = model(
                packed_tokens,
                layout.position_ids("cuda").unsqueeze(0),
                None,
                shared_prefix_layout=layout,
            )[0]

        for dense, completion_slice in zip(dense_completion_logits, layout.completion_slices()):
            shared = shared_logits[completion_slice]
            assert _relative_l2(dense, shared) < 0.05
            assert _cosine(dense, shared) > 0.99
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.internal
@pytest.mark.timeout(900)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="shared-prefix MTP parity requires CUDA")
@pytest.mark.parametrize("position_embedding_type", ["rope", "none"])
def test_hybrid_shared_prefix_mtp_head_gradient_parity(monkeypatch, position_embedding_type):
    """The single packed dense-head MTP call matches a conventional MTP batch."""
    pytest.importorskip("causal_conv1d")
    pytest.importorskip("flash_attn")
    pytest.importorskip("mamba_ssm")
    if not torch.cuda.is_bf16_supported():
        pytest.skip("shared-prefix MTP parity requires CUDA bf16 support")
    # The packed MTP path runs every dense branch as one THD sequence; let TE pick
    # its flash/fused THD kernels (AttnBackend.auto).  The unit-test conftest pins
    # NVTE_FLASH_ATTN=0 / NVTE_FUSED_ATTN=0, which AttnBackend.auto rejects.
    for env_name in ("NVTE_FLASH_ATTN", "NVTE_FUSED_ATTN", "NVTE_UNFUSED_ATTN"):
        monkeypatch.delenv(env_name, raising=False)

    from megatron.core.models.hybrid.hybrid_layer_allocation import validate_segment_layers
    from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer import TransformerConfig
    from megatron.core.transformer.enums import AttnBackend
    from tests.unit_tests.test_utilities import Utils

    prefix_len = 16
    logical_completion_len = 12
    physical_completion_len = 16
    dense_seq_len = prefix_len + physical_completion_len
    layout = SharedPrefixLayout(
        prefix_len=prefix_len,
        # K=3 makes the physical star 64 tokens, exactly two M=32 quanta.
        completion_lens=[physical_completion_len] * 3,
        logical_completion_lens=[logical_completion_len] * 3,
        padding_multiple=dense_seq_len,
    )
    # The target keeps Mamba in the shared backbone while its one physical MTP
    # predictor is attention/MLP based.  Do not accidentally test a different
    # architecture by copying the main Hybrid pattern into the MTP block.
    main_pattern = "M*-"
    mtp_pattern = "*-"
    mtp_depths = 5
    unified_pattern = "/".join([main_pattern, *([mtp_pattern] * mtp_depths)])

    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1, context_parallel_size=1
    )
    try:
        model_parallel_cuda_manual_seed(123)
        config = TransformerConfig(
            hidden_size=256,
            num_layers=len(validate_segment_layers(main_pattern)),
            mtp_num_layers=mtp_depths,
            mtp_use_repeated_layer=True,
            mtp_detach_heads=True,
            num_attention_heads=4,
            use_cpu_initialization=True,
            bf16=True,
            params_dtype=torch.bfloat16,
            use_mamba_mem_eff_path=False,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            attention_backend=AttnBackend.auto,
            apply_rope_fusion=False,
        )
        process_groups = ProcessGroupCollection.use_mpu_process_groups(
            required_pgs=["tp", "pp", "cp", "embd", "dp_cp"]
        )
        model = HybridModel(
            config=config,
            hybrid_stack_spec=hybrid_stack_spec,
            vocab_size=512,
            max_sequence_length=layout.total_len,
            hybrid_layer_pattern=unified_pattern,
            position_embedding_type=position_embedding_type,
            pre_process=True,
            post_process=True,
            parallel_output=False,
            pg_collection=process_groups,
        ).cuda()
        model.train()
        mtp_block_calls = []
        model.mtp.register_forward_pre_hook(
            lambda module, args, kwargs: mtp_block_calls.append(1), with_kwargs=True
        )

        torch.manual_seed(41)
        prefix = torch.randint(0, model.vocab_size, (prefix_len,), device="cuda")
        logical_completions = [
            torch.randint(0, model.vocab_size, (logical_completion_len,), device="cuda")
            for _ in range(3)
        ]
        physical_completions = [
            torch.cat(
                [
                    completion,
                    torch.zeros(
                        physical_completion_len - logical_completion_len,
                        device="cuda",
                        dtype=torch.long,
                    ),
                ]
            )
            for completion in logical_completions
        ]
        dense_tokens = torch.stack(
            [torch.cat([prefix, completion]) for completion in physical_completions]
        )
        dense_positions = torch.arange(dense_seq_len, device="cuda").expand_as(dense_tokens)
        dense_loss_mask = torch.zeros_like(dense_tokens, dtype=torch.float32)
        dense_loss_mask[:, prefix_len : prefix_len + logical_completion_len] = 1

        # Transformer Engine owns the causal mask for both the dense batch and the
        # packed THD MTP branches; an explicit boolean mask is not needed.
        dense_logits = model(dense_tokens, dense_positions, None, loss_mask=dense_loss_mask)
        (dense_logits.sum() * 0.0).backward()
        assert len(mtp_block_calls) == 1
        dense_mtp_grads = {
            name: parameter.grad.detach().clone()
            for name, parameter in model.mtp.named_parameters()
            if parameter.grad is not None
        }
        assert dense_mtp_grads
        mtp_block_calls.clear()

        model.zero_grad(set_to_none=True)
        star_tokens = torch.cat([prefix, *physical_completions]).unsqueeze(0)
        star_loss_mask = torch.cat(
            [
                torch.zeros(prefix_len, device="cuda"),
                *[
                    torch.cat(
                        [
                            torch.ones(logical_completion_len, device="cuda"),
                            torch.zeros(
                                physical_completion_len - logical_completion_len,
                                device="cuda",
                            ),
                        ]
                    )
                    for _ in physical_completions
                ],
            ]
        ).unsqueeze(0)
        shared_logits = model(
            star_tokens,
            layout.position_ids("cuda").unsqueeze(0),
            None,
            loss_mask=star_loss_mask,
            shared_prefix_layout=layout,
        )
        (shared_logits.sum() * 0.0).backward()
        # The shared-prefix path packs all three dense branches into one THD
        # sequence and runs the MTP block exactly once, not once per branch.
        assert len(mtp_block_calls) == 1
        shared_mtp_grads = {
            name: parameter.grad.detach()
            for name, parameter in model.mtp.named_parameters()
            if parameter.grad is not None
        }

        assert shared_mtp_grads.keys() == dense_mtp_grads.keys()
        for name in dense_mtp_grads:
            assert _relative_l2(dense_mtp_grads[name], shared_mtp_grads[name]) < 0.08, name
            assert _cosine(dense_mtp_grads[name], shared_mtp_grads[name]) > 0.995, name

        # Reference/current-policy logprob forwards put the model in eval mode
        # and do not materialize an MTP loss mask.  They need only main-model
        # logits, so the auxiliary heads must be skipped entirely.
        model.eval()
        with torch.no_grad():
            eval_logits = model(
                star_tokens,
                layout.position_ids("cuda").unsqueeze(0),
                None,
                shared_prefix_layout=layout,
            )
        assert eval_logits.shape == (1, layout.total_len, model.vocab_size)
    finally:
        Utils.destroy_model_parallel()
