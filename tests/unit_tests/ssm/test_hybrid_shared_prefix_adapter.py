# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import inspect
from types import SimpleNamespace

import pytest
import torch

from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.models.hybrid.shared_prefix import (
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
        context_parallel_size=1,
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
        tp_group=_SizeOneGroup(), pp_group=_SizeOneGroup(), config=config, layers=list(layers)
    )


def _stub_attention_layer():
    attention = object.__new__(SelfAttention)
    torch.nn.Module.__init__(attention)
    attention.checkpoint_core_attention = False
    layer = object.__new__(TransformerLayer)
    torch.nn.Module.__init__(layer)
    layer.self_attention = attention
    return layer


def test_shared_prefix_layout_and_capability_are_explicit():
    layout = SharedPrefixLayout(prefix_len=3, completion_lens=[2, 4])

    assert SHARED_PREFIX_TRAINING_CAPABILITY == "hybrid_star_cp1_tp1_v1"
    assert layout.total_len == 9
    assert layout.forest == [(0, 3, [2, 4])]
    assert layout.completion_slices() == (slice(3, 5), slice(5, 9))
    assert layout.position_ids("cpu").tolist() == [0, 1, 2, 3, 4, 3, 4, 5, 6]
    model_parameter = inspect.signature(HybridModel.forward).parameters["shared_prefix_layout"]
    assert model_parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert model_parameter.default is None


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
def test_hybrid_shared_prefix_forward_and_gradient_parity():
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
        rotary = RotaryEmbedding(
            kv_channels=hidden_size // num_heads, rotary_percent=1.0, rotary_base=10000
        ).cuda()

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
                completion.shape,
                dtype=torch.float32,
                device="cuda",
                generator=probe_generator,
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
                    rotary_pos_emb=rotary.get_emb(length),
                )
                outputs.append(output[layout.prefix_len :])
            return outputs

        def shared_outputs():
            packed = torch.cat([prefix, *completions], dim=0)
            position_ids = layout.position_ids(packed.device)
            rotary_pos_emb = rotary.get_emb(int(position_ids.max().item()) + 1).index_select(
                0, position_ids
            )
            output = forward_hybrid_stack_shared_prefix(
                stack, packed, layout, rotary_pos_emb=rotary_pos_emb
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
def test_hybrid_model_explicit_shared_prefix_forward_matches_dense():
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
            position_embedding_type="rope",
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
