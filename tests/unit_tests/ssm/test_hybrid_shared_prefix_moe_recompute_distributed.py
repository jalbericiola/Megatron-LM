# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Distributed parity for shared-prefix Hybrid MoE with full recomputation.

Run with::

    torchrun --nproc-per-node=4 -m pytest \
        tests/unit_tests/ssm/test_hybrid_shared_prefix_moe_recompute_distributed.py \
        -m internal -v

This is deliberately a validation gate, not a capability advertisement.  It uses a
real ``E*M`` Hybrid stack, all-to-all expert dispatch, CP2, EP4, BF16, ordinary THD
packing as the dense oracle, explicit per-branch physical padding in the star, and
full uniform activation recomputation.
"""

import os

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.models.hybrid.shared_prefix import SharedPrefixLayout
from megatron.core.models.hybrid.shared_prefix_fused import _undo_cp_zigzag
from tests.unit_tests.test_utilities import Utils

pytestmark = [
    pytest.mark.internal,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="distributed parity requires CUDA"),
    pytest.mark.skipif(
        int(os.environ.get("WORLD_SIZE", "1")) != 4,
        reason="MoE parity requires torchrun --nproc-per-node=4",
    ),
]


def _cp_local(tensor, cp_size, cp_rank):
    indices = SharedPrefixLayout.cp_local_indices(tensor.shape[0], cp_size, cp_rank, tensor.device)
    return tensor.index_select(0, indices)


def _all_gather_canonical(local_tensor, global_length, cp_group):
    gathered = [torch.empty_like(local_tensor) for _ in range(cp_group.size())]
    torch.distributed.all_gather(gathered, local_tensor, group=cp_group)
    rank_major = torch.cat(gathered, dim=0)
    assert rank_major.shape[0] == global_length
    return _undo_cp_zigzag(rank_major, cp_group.size())


def _all_gather_dense_sequences(local_tensor, physical_lengths, cp_group):
    sequences = []
    local_start = 0
    for physical_length in physical_lengths:
        local_length = physical_length // cp_group.size()
        local_sequence = local_tensor[local_start : local_start + local_length]
        sequences.append(_all_gather_canonical(local_sequence, physical_length, cp_group))
        local_start += local_length
    assert local_start == local_tensor.shape[0]
    return sequences


def _relative_l2(reference, actual):
    reference = reference.double().flatten()
    actual = actual.double().flatten()
    difference = (reference - actual).norm()
    reference_norm = reference.norm()
    if reference_norm == 0:
        return 0.0 if difference == 0 else float("inf")
    return (difference / reference_norm).item()


def _cosine(reference, actual):
    reference = reference.double().flatten()
    actual = actual.double().flatten()
    denominator = reference.norm() * actual.norm()
    if denominator == 0:
        return 1.0 if reference.norm() == actual.norm() == 0 else 0.0
    return (torch.dot(reference, actual) / denominator).item()


def _assert_vector_parity(reference, actual, *, rel_l2, cosine, label):
    assert torch.isfinite(reference).all(), f"{label}: non-finite reference"
    assert torch.isfinite(actual).all(), f"{label}: non-finite candidate"
    error = _relative_l2(reference, actual)
    alignment = _cosine(reference, actual)
    assert error < rel_l2, f"{label}: relative L2 {error} >= {rel_l2}"
    assert alignment > cosine, f"{label}: cosine {alignment} <= {cosine}"


def _snapshot_parameter_grads(stack):
    return {
        name: (
            torch.zeros_like(parameter)
            if parameter.grad is None
            else parameter.grad.detach().clone()
        )
        for name, parameter in stack.named_parameters()
        if parameter.requires_grad
    }


def _reduce_replicated_gradient(name, gradient, group):
    # Expert parameters are EP-sharded: the same local tensor position denotes a
    # different expert on each EP rank.  Every other parameter is replicated over
    # this TP1/PP1 topology and may be summed over TPxDPxCP for a canonical check.
    if ".mlp.experts." in name:
        return gradient.float()
    reduced = gradient.float()
    torch.distributed.all_reduce(reduced, group=group)
    return reduced


@pytest.mark.timeout(420)
def test_real_hybrid_moe_full_recompute_matches_dense_physical_packing():
    pytest.importorskip("causal_conv1d")
    pytest.importorskip("mamba_ssm")
    pytest.importorskip("flash_attn")
    if not torch.cuda.is_bf16_supported():
        pytest.skip("distributed parity requires BF16 support")

    from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
    from megatron.core.models.hybrid.hybrid_block import HybridStack
    from megatron.core.models.hybrid.hybrid_layer_allocation import validate_segment_layers
    from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
    from megatron.core.models.hybrid.shared_prefix import forward_hybrid_stack_shared_prefix
    from megatron.core.packed_seq_params import PackedSeqParams
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer import TransformerConfig
    from megatron.core.transformer.moe.moe_layer import MoELayer
    from megatron.core.transformer.moe.moe_utils import get_updated_expert_bias

    tp_size, cp_size, ep_size, etp_size = 1, 2, 4, 1
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=1,
        context_parallel_size=cp_size,
        expert_model_parallel_size=ep_size,
        expert_tensor_parallel_size=etp_size,
    )
    try:
        cp_group = parallel_state.get_context_parallel_group()
        cp_rank = cp_group.rank()
        torch.manual_seed(20260827)
        model_parallel_cuda_manual_seed(20260827)

        config = TransformerConfig(
            hidden_size=64,
            ffn_hidden_size=128,
            num_layers=3,
            num_attention_heads=4,
            num_query_groups=2,
            kv_channels=16,
            mamba_num_heads=8,
            mamba_num_groups=4,
            tensor_model_parallel_size=tp_size,
            context_parallel_size=cp_size,
            expert_model_parallel_size=ep_size,
            expert_tensor_parallel_size=etp_size,
            num_moe_experts=8,
            moe_router_topk=2,
            moe_router_dtype="fp32",
            moe_router_load_balancing_type="none",
            moe_router_score_function="sigmoid",
            moe_router_enable_expert_bias=True,
            moe_router_bias_update_rate=1.0e-3,
            moe_token_dispatcher_type="alltoall",
            moe_grouped_gemm=True,
            moe_aux_loss_coeff=0.0,
            moe_z_loss_coeff=None,
            moe_expert_capacity_factor=None,
            moe_input_jitter_eps=None,
            use_cpu_initialization=True,
            bf16=True,
            params_dtype=torch.bfloat16,
            use_mamba_mem_eff_path=True,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            recompute_granularity="full",
            recompute_method="uniform",
            recompute_num_layers=1,
            distribute_saved_activations=False,
            sequence_parallel=False,
            add_bias_linear=False,
        )
        process_groups = ProcessGroupCollection.use_mpu_process_groups()
        stack = HybridStack(
            config,
            hybrid_stack_spec.submodules,
            layer_type_list=validate_segment_layers("E*M"),
            pp_layer_offset=0,
            pg_collection=process_groups,
        ).cuda()
        stack.train()

        moe_layers = [layer.mlp for layer in stack.layers if getattr(layer, "is_moe_layer", False)]
        assert len(moe_layers) == 1 and isinstance(moe_layers[0], MoELayer)
        moe_layer = moe_layers[0]
        router = moe_layer.router
        # Keep routing deterministic and separated from the expected BF16 path
        # noise so this test measures multiplicity/recompute semantics exactly.
        with torch.no_grad():
            router.weight.zero_()
            router.expert_bias.copy_(
                torch.linspace(-0.04, 0.04, config.num_moe_experts, device="cuda")
            )
        router.weight.requires_grad_(False)
        initial_expert_bias = router.expert_bias.detach().clone()

        layout = SharedPrefixLayout(
            prefix_len=5,
            completion_lens=(11, 3),
            logical_completion_lens=(6, 3),
            padding_multiple=8,
        )
        physical_lengths = (16, 8)
        assert tuple(layout.prefix_len + length for length in layout.completion_lens) == (16, 8)
        physical_total = sum(physical_lengths)
        assert physical_total == 24 and layout.total_len == 19

        generator = torch.Generator(device="cuda").manual_seed(17)
        prefix = torch.randn(
            layout.prefix_len,
            1,
            config.hidden_size,
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        )
        logical_completions = [
            torch.randn(
                length,
                1,
                config.hidden_size,
                generator=generator,
                device="cuda",
                dtype=torch.bfloat16,
            )
            for length in layout.logical_completion_lens
        ]
        dense_global_sequences = [
            torch.cat(
                [
                    prefix,
                    completion,
                    completion.new_zeros(
                        physical_length - layout.prefix_len - completion.shape[0],
                        1,
                        config.hidden_size,
                    ),
                ],
                dim=0,
            )
            for completion, physical_length in zip(
                logical_completions, physical_lengths, strict=True
            )
        ]
        dense_local = (
            torch.cat(
                [_cp_local(value, cp_size, cp_rank) for value in dense_global_sequences], dim=0
            )
            .detach()
            .requires_grad_(True)
        )

        star_real = torch.cat(
            [
                prefix,
                logical_completions[0],
                logical_completions[0].new_zeros(5, 1, config.hidden_size),
                logical_completions[1],
            ],
            dim=0,
        )
        star_global = torch.cat(
            [
                star_real,
                star_real.new_zeros(physical_total - layout.total_len, 1, config.hidden_size),
            ],
            dim=0,
        )
        star_local = _cp_local(star_global, cp_size, cp_rank).detach().requires_grad_(True)
        assert dense_local.shape == star_local.shape == (12, 1, config.hidden_size)

        probe_generator = torch.Generator(device="cuda").manual_seed(29)
        probes = [
            torch.randn(
                completion.shape, generator=probe_generator, device="cuda", dtype=torch.float32
            )
            / completion.numel() ** 0.5
            for completion in logical_completions
        ]
        dense_global_probe_sequences = []
        for probe, physical_length in zip(probes, physical_lengths, strict=True):
            value = torch.zeros(
                physical_length, 1, config.hidden_size, device="cuda", dtype=torch.float32
            )
            value[layout.prefix_len : layout.prefix_len + probe.shape[0]] = probe
            dense_global_probe_sequences.append(value)
        dense_local_probe = torch.cat(
            [_cp_local(value, cp_size, cp_rank) for value in dense_global_probe_sequences], dim=0
        )
        star_global_probe = torch.zeros(
            physical_total, 1, config.hidden_size, device="cuda", dtype=torch.float32
        )
        star_start = layout.prefix_len
        for probe, physical_length in zip(probes, layout.completion_lens, strict=True):
            star_global_probe[star_start : star_start + probe.shape[0]] = probe
            star_start += physical_length
        star_local_probe = _cp_local(star_global_probe, cp_size, cp_rank)

        cu_seqlens = torch.tensor([0, 16, 24], device="cuda", dtype=torch.int32)
        dense_packed_seq_params = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            cu_seqlens_q_padded=cu_seqlens,
            cu_seqlens_kv_padded=cu_seqlens,
            max_seqlen_q=16,
            max_seqlen_kv=16,
            total_tokens=dense_local.shape[0],
        )
        rotary = RotaryEmbedding(
            kv_channels=config.kv_channels, rotary_percent=1.0, rotary_base=10000, cp_group=cp_group
        ).cuda()
        dense_rotary = rotary(16, packed_seq=True)
        star_positions = layout.padded_position_ids(physical_total, "cuda")
        star_indices = layout.cp_local_indices(
            physical_total, cp_size, cp_rank, star_positions.device
        )
        star_rotary = rotary.get_emb(16).index_select(
            0, star_positions.index_select(0, star_indices)
        )

        router.local_tokens_per_expert.zero_()
        dense_output = stack(
            hidden_states=dense_local,
            attention_mask=None,
            rotary_pos_emb=dense_rotary,
            packed_seq_params=dense_packed_seq_params,
            padding_mask=None,
        )
        assert torch.count_nonzero(router.local_tokens_per_expert) == 0
        (dense_output.float() * dense_local_probe).sum().backward()
        dense_counts_local = router.local_tokens_per_expert.detach().clone()
        dense_grads = _snapshot_parameter_grads(stack)
        dense_inputs = _all_gather_dense_sequences(
            dense_local.grad.detach(), physical_lengths, cp_group
        )
        dense_outputs = _all_gather_dense_sequences(
            dense_output.detach(), physical_lengths, cp_group
        )

        stack.zero_grad(set_to_none=True)
        router.local_tokens_per_expert.zero_()
        star_output = forward_hybrid_stack_shared_prefix(
            stack, star_local, layout, rotary_pos_emb=star_rotary
        )
        assert torch.count_nonzero(router.local_tokens_per_expert) == 0
        (star_output.float() * star_local_probe).sum().backward()
        star_counts_local = router.local_tokens_per_expert.detach().clone()
        star_grads = _snapshot_parameter_grads(stack)
        star_input = _all_gather_canonical(star_local.grad.detach(), physical_total, cp_group)
        star_output_global = _all_gather_canonical(star_output.detach(), physical_total, cp_group)

        completion_start = layout.prefix_len
        for branch, logical_length in enumerate(layout.logical_completion_lens):
            dense_slice = dense_outputs[branch][
                layout.prefix_len : layout.prefix_len + logical_length
            ]
            star_slice = star_output_global[completion_start : completion_start + logical_length]
            _assert_vector_parity(
                dense_slice, star_slice, rel_l2=0.03, cosine=0.995, label=f"branch {branch} forward"
            )
            dense_completion_grad = dense_inputs[branch][
                layout.prefix_len : layout.prefix_len + logical_length
            ]
            star_completion_grad = star_input[completion_start : completion_start + logical_length]
            _assert_vector_parity(
                dense_completion_grad,
                star_completion_grad,
                rel_l2=0.05,
                cosine=0.99,
                label=f"branch {branch} input gradient",
            )
            completion_start += layout.completion_lens[branch]

        dense_prefix_grad = sum(value[: layout.prefix_len] for value in dense_inputs)
        _assert_vector_parity(
            dense_prefix_grad,
            star_input[: layout.prefix_len],
            rel_l2=0.05,
            cosine=0.99,
            label="summed prompt input gradient",
        )
        for branch, (dense_input, logical_length, physical_length) in enumerate(
            zip(dense_inputs, layout.logical_completion_lens, physical_lengths, strict=True)
        ):
            dense_tail = dense_input[layout.prefix_len + logical_length : physical_length]
            assert (
                torch.count_nonzero(dense_tail) == 0
            ), f"dense branch {branch} physical padding received logical-loss gradient"
        star_interior_tail = star_input[
            layout.prefix_len
            + layout.logical_completion_lens[0] : layout.prefix_len
            + layout.completion_lens[0]
        ]
        assert star_interior_tail.shape[0] == 5
        assert (
            torch.count_nonzero(star_interior_tail) == 0
        ), "star interior per-branch padding received logical-loss gradient"
        star_topology_tail = star_input[layout.total_len : physical_total]
        assert star_topology_tail.shape[0] == 5
        assert (
            torch.count_nonzero(star_topology_tail) == 0
        ), "star final topology padding received logical-loss gradient"

        assert dense_grads.keys() == star_grads.keys()
        for name in dense_grads:
            dense_gradient = _reduce_replicated_gradient(
                name, dense_grads[name].clone(), router.tp_dp_cp_group
            )
            star_gradient = _reduce_replicated_gradient(
                name, star_grads[name].clone(), router.tp_dp_cp_group
            )
            _assert_vector_parity(
                dense_gradient,
                star_gradient,
                rel_l2=0.05,
                cosine=0.99,
                label=f"parameter gradient {name}",
            )

        dense_counts = dense_counts_local.clone()
        star_counts = star_counts_local.clone()
        torch.distributed.all_reduce(dense_counts, group=router.tp_dp_cp_group)
        torch.distributed.all_reduce(star_counts, group=router.tp_dp_cp_group)
        torch.testing.assert_close(star_counts, dense_counts, rtol=0.0, atol=0.0)
        expected_count = dense_local.shape[0] * router.tp_dp_cp_group.size()
        expected_counts = torch.zeros_like(dense_counts)
        selected_experts = torch.topk(initial_expert_bias, config.moe_router_topk).indices
        expected_counts[selected_experts] = expected_count
        torch.testing.assert_close(dense_counts, expected_counts, rtol=0.0, atol=0.0)

        dense_updated_bias = get_updated_expert_bias(
            dense_counts_local.clone(),
            initial_expert_bias.clone(),
            config.moe_router_bias_update_rate,
            tp_dp_cp_group=router.tp_dp_cp_group,
        )
        star_updated_bias = get_updated_expert_bias(
            star_counts_local.clone(),
            initial_expert_bias.clone(),
            config.moe_router_bias_update_rate,
            tp_dp_cp_group=router.tp_dp_cp_group,
        )
        torch.testing.assert_close(star_updated_bias, dense_updated_bias, rtol=0.0, atol=0.0)
        assert not torch.equal(dense_updated_bias, initial_expert_bias)

        if torch.distributed.get_rank() == 0:
            print(
                "shared-prefix MoE full-recompute parity passed: "
                f"TP{tp_size}/CP{cp_size}/EP{ep_size}/ETP{etp_size}, "
                f"physical={physical_lengths}, reduced_counts={dense_counts.tolist()}"
            )
    finally:
        Utils.destroy_model_parallel()
