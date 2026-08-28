# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""RLVR41-shaped distributed parity for the combined shared-prefix path.

Run on four 4-GPU nodes with::

    MCORE_SHARED_PREFIX_RLVR41_GATE=1 torchrun \
        --nnodes=4 --nproc-per-node=4 ... -m pytest -m internal -v \
        tests/unit_tests/ssm/test_hybrid_shared_prefix_rlvr41_distributed.py::\
test_rlvr41_tp4_cp4_ep16_shared_prefix_matches_dense_and_finalizes_expert_bias_once

This is an MCore model/gradient gate.  It deliberately does not advertise a
capability: the paired NeMo-RL gate owns selected-token log-probabilities,
fused loss, and deferred-FP32-logit integration.
"""

import os
from types import MethodType

import pytest
import torch
from torch.distributed.nn.functional import all_gather

from megatron.core import parallel_state
from megatron.core.models.hybrid.shared_prefix import SharedPrefixLayout
from megatron.core.models.hybrid.shared_prefix_fused import _undo_cp_zigzag

RLVR41_TP_SIZE = 4
RLVR41_CP_SIZE = 4
RLVR41_EP_SIZE = 16
RLVR41_ETP_SIZE = 1
RLVR41_WORLD_SIZE = 16
RLVR41_PADDING_MULTIPLE = 32
RLVR41_PATTERN = "E*M"
RLVR41_PREFIX_LEN = 5
RLVR41_LOGICAL_COMPLETION_LENS = tuple(range(1, 17))
RLVR41_PHYSICAL_COMPLETION_LENS = (RLVR41_PADDING_MULTIPLE - RLVR41_PREFIX_LEN,) * len(
    RLVR41_LOGICAL_COMPLETION_LENS
)
RLVR41_DENSE_PHYSICAL_LEN = RLVR41_PADDING_MULTIPLE
RLVR41_STAR_PHYSICAL_LEN = 448

BRINGUP_TP_SIZE = 2
BRINGUP_CP_SIZE = 2
BRINGUP_EP_SIZE = 4
BRINGUP_ETP_SIZE = 1
BRINGUP_WORLD_SIZE = 4
BRINGUP_PADDING_MULTIPLE = 16
BRINGUP_PREFIX_LEN = 3
BRINGUP_LOGICAL_COMPLETION_LENS = (2, 4, 5, 7)
BRINGUP_PHYSICAL_COMPLETION_LENS = (BRINGUP_PADDING_MULTIPLE - BRINGUP_PREFIX_LEN,) * len(
    BRINGUP_LOGICAL_COMPLETION_LENS
)
BRINGUP_DENSE_PHYSICAL_LEN = BRINGUP_PADDING_MULTIPLE
BRINGUP_STAR_PHYSICAL_LEN = 64

pytestmark = [
    pytest.mark.internal,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="RLVR41 parity requires CUDA"),
]


def rlvr41_shared_prefix_layout() -> SharedPrefixLayout:
    """Return the exact M=32, K=16 layout shared with the NeMo integration gate."""
    return SharedPrefixLayout(
        prefix_len=RLVR41_PREFIX_LEN,
        completion_lens=RLVR41_PHYSICAL_COMPLETION_LENS,
        logical_completion_lens=RLVR41_LOGICAL_COMPLETION_LENS,
        padding_multiple=RLVR41_PADDING_MULTIPLE,
    )


def build_combined_shared_prefix_model(
    pg_collection, *, tp_size, cp_size, ep_size, etp_size, num_moe_experts, max_sequence_length
):
    """Build the small E*M HybridModel used by the bring-up and RLVR41 gates."""
    from megatron.core.activations import squared_relu
    from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
    from megatron.core.models.hybrid.hybrid_model import HybridModel
    from megatron.core.transformer import TransformerConfig

    config = TransformerConfig(
        hidden_size=256,
        ffn_hidden_size=512,
        num_layers=len(RLVR41_PATTERN),
        num_attention_heads=32,
        num_query_groups=2,
        kv_channels=128,
        mamba_num_heads=64,
        mamba_num_groups=8,
        tensor_model_parallel_size=tp_size,
        context_parallel_size=cp_size,
        sequence_parallel=True,
        expert_model_parallel_size=ep_size,
        expert_tensor_parallel_size=etp_size,
        num_moe_experts=num_moe_experts,
        moe_router_topk=2,
        moe_router_dtype="fp32",
        moe_router_load_balancing_type="none",
        moe_router_score_function="sigmoid",
        moe_router_enable_expert_bias=True,
        moe_router_bias_update_rate=1.0e-3,
        moe_token_dispatcher_type="alltoall",
        # Bridge carries the selected all-to-all transport in this field even
        # though it is inert unless the dispatcher type is "flex".
        moe_flex_dispatcher_backend="alltoall",
        moe_enable_deepep=False,
        moe_grouped_gemm=True,
        moe_permute_fusion=True,
        moe_shared_expert_overlap=False,
        moe_per_layer_logging=True,
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
        apply_rope_fusion=True,
        activation_func=squared_relu,
        use_fused_weighted_squared_relu=True,
        recompute_granularity="full",
        recompute_method="uniform",
        recompute_num_layers=1,
        distribute_saved_activations=False,
        apply_query_key_layer_scaling=True,
        add_bias_linear=False,
    )
    model = HybridModel(
        config=config,
        hybrid_stack_spec=hybrid_stack_spec,
        vocab_size=512,
        max_sequence_length=max_sequence_length,
        hybrid_layer_pattern=RLVR41_PATTERN,
        position_embedding_type="rope",
        pre_process=True,
        post_process=True,
        parallel_output=True,
        pg_collection=pg_collection,
    ).cuda()
    return model


def build_rlvr41_shared_prefix_model(pg_collection):
    """Build the target TP4/CP4/EP16 model shared with the NeMo integration gate."""
    return build_combined_shared_prefix_model(
        pg_collection,
        tp_size=RLVR41_TP_SIZE,
        cp_size=RLVR41_CP_SIZE,
        ep_size=RLVR41_EP_SIZE,
        etp_size=RLVR41_ETP_SIZE,
        num_moe_experts=16,
        max_sequence_length=RLVR41_STAR_PHYSICAL_LEN,
    )


def freeze_rlvr41_router(model):
    """Freeze routing weights and install deterministic expert-bias decisions."""
    from megatron.core.transformer.moe.moe_layer import MoELayer

    moe_layers = [
        layer.mlp for layer in model.decoder.layers if getattr(layer, "is_moe_layer", False)
    ]
    if len(moe_layers) != 1 or not isinstance(moe_layers[0], MoELayer):
        raise AssertionError("RLVR41 fixture requires exactly one real MCore MoE layer")
    router = moe_layers[0].router
    with torch.no_grad():
        router.weight.zero_()
        router.expert_bias.copy_(
            torch.linspace(-0.04, 0.04, model.config.num_moe_experts, device="cuda")
        )
    router.weight.requires_grad_(False)
    return router


def _cp_then_tp_local(tensor, layout, cp_group, tp_group):
    cp_indices = layout.cp_local_indices(
        tensor.shape[0], cp_group.size(), cp_group.rank(), tensor.device
    )
    cp_local = tensor.index_select(0, cp_indices)
    if cp_local.shape[0] % tp_group.size():
        raise AssertionError("CP-local sequence does not divide over TP/SP")
    return torch.chunk(cp_local, tp_group.size(), dim=0)[tp_group.rank()].contiguous()


def _gather_canonical_with_grad(local_tensor, cp_group, tp_group):
    cp_local = torch.cat(all_gather(local_tensor, group=tp_group), dim=0)
    rank_major = torch.cat(all_gather(cp_local, group=cp_group), dim=0)
    return _undo_cp_zigzag(rank_major, cp_group.size())


def _gather_canonical(local_tensor, cp_group, tp_group):
    tp_values = [torch.empty_like(local_tensor) for _ in range(tp_group.size())]
    torch.distributed.all_gather(tp_values, local_tensor, group=tp_group)
    cp_local = torch.cat(tp_values, dim=0)
    cp_values = [torch.empty_like(cp_local) for _ in range(cp_group.size())]
    torch.distributed.all_gather(cp_values, cp_local, group=cp_group)
    return _undo_cp_zigzag(torch.cat(cp_values, dim=0), cp_group.size())


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


def _stats(reference, actual):
    return {"relative_l2": _relative_l2(reference, actual), "cosine": _cosine(reference, actual)}


def _assert_parity(reference, actual, *, rel_l2, cosine, label):
    assert torch.isfinite(reference).all(), f"{label}: non-finite reference"
    assert torch.isfinite(actual).all(), f"{label}: non-finite candidate"
    metrics = _stats(reference, actual)
    assert (
        metrics["relative_l2"] < rel_l2
    ), f"{label}: relative L2 {metrics['relative_l2']} >= {rel_l2}"
    assert metrics["cosine"] > cosine, f"{label}: cosine {metrics['cosine']} <= {cosine}"


def _install_finalize_probe(model):
    """Give a raw PP1 model the no-DP wrapper surface needed by the real finalizer."""
    from megatron.core.distributed import DistributedDataParallelConfig

    model.ddp_config = DistributedDataParallelConfig()
    model._shared_prefix_finish_grad_sync_calls = 0

    def finish_grad_sync(this, force_all_reduce=False):
        del force_all_reduce
        this._shared_prefix_finish_grad_sync_calls += 1

    model.finish_grad_sync = MethodType(finish_grad_sync, model)


def _snapshot_finalized_decoder_grads(model, cp_group):
    gradients = {}
    for name, parameter in model.decoder.named_parameters():
        if not parameter.requires_grad:
            continue
        gradient = (
            torch.zeros_like(parameter, dtype=torch.float32)
            if parameter.grad is None
            else parameter.grad.detach().float().clone()
        )
        # Expert tensors are EP-sharded.  All other local TP shards are replicated
        # over CP; sequence-parallel TP reductions have already run in
        # finalize_model_grads.
        if ".mlp.experts." not in name:
            torch.distributed.all_reduce(gradient, group=cp_group)
        gradients[name] = gradient
    return gradients


def _run_combined_parity(
    *,
    tp_size,
    cp_size,
    ep_size,
    etp_size,
    num_moe_experts,
    padding_multiple,
    prefix_len,
    logical_completion_lens,
    physical_completion_lens,
    dense_physical_len,
    star_physical_len,
):
    """Run one conjoined topology/model/gradient/finalization parity case."""
    gate_mode = os.environ.get("MCORE_SHARED_PREFIX_RLVR41_GATE") == "1"
    for dependency in ("causal_conv1d", "flash_attn", "mamba_ssm"):
        if gate_mode:
            __import__(dependency)
        else:
            pytest.importorskip(dependency)
    if not torch.cuda.is_bf16_supported():
        if gate_mode:
            raise AssertionError("RLVR41 gate requires BF16 support")
        pytest.skip("RLVR41 parity requires BF16 support")

    from megatron.core.distributed.finalize_model_grads import finalize_model_grads
    from megatron.core.models.hybrid.shared_prefix import forward_hybrid_stack_shared_prefix
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.moe.moe_utils import get_updated_expert_bias
    from tests.unit_tests.test_utilities import Utils

    previous_qk_layer_scaling = os.environ.get("NVTE_APPLY_QK_LAYER_SCALING")
    os.environ["NVTE_APPLY_QK_LAYER_SCALING"] = "1"
    # MCore's single-node test helper defaults process-group rank to LOCAL_RANK.
    # Multi-node torchrun needs the global RANK for rendezvous and LOCAL_RANK only
    # for CUDA ownership. Standard contiguous four-GPU nodes satisfy this mapping.
    world_size = int(os.environ["WORLD_SIZE"])
    global_rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    local_cuda_count = torch.cuda.device_count()
    assert world_size == max(tp_size * cp_size, ep_size * etp_size)
    assert global_rank % local_cuda_count == local_rank
    Utils.world_size = world_size
    Utils.rank = global_rank
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=1,
        context_parallel_size=cp_size,
        expert_model_parallel_size=ep_size,
        expert_tensor_parallel_size=etp_size,
    )
    try:
        process_groups = ProcessGroupCollection.use_mpu_process_groups()
        tp_group = process_groups.tp
        cp_group = process_groups.cp
        assert tp_group.size() == tp_size
        assert cp_group.size() == cp_size
        assert parallel_state.get_expert_model_parallel_world_size() == ep_size
        assert parallel_state.get_expert_tensor_parallel_world_size() == etp_size
        assert process_groups.tp_dp_cp.size() == max(tp_size * cp_size, ep_size * etp_size)

        torch.manual_seed(20260828)
        model_parallel_cuda_manual_seed(20260828)
        model = build_combined_shared_prefix_model(
            process_groups,
            tp_size=tp_size,
            cp_size=cp_size,
            ep_size=ep_size,
            etp_size=etp_size,
            num_moe_experts=num_moe_experts,
            max_sequence_length=star_physical_len,
        )
        model.train()
        assert model.config.apply_rope_fusion
        assert model.config.activation_func.__name__ == "squared_relu"
        assert model.config.use_fused_weighted_squared_relu
        assert not model.config.moe_enable_deepep
        assert model.config.moe_token_dispatcher_type == "alltoall"
        assert model.config.moe_flex_dispatcher_backend == "alltoall"
        assert not model.config.moe_shared_expert_overlap
        assert model.config.moe_per_layer_logging
        assert model.config.recompute_granularity == "full"
        assert model.config.recompute_method == "uniform"
        assert model.config.recompute_num_layers == 1
        router = freeze_rlvr41_router(model)
        _install_finalize_probe(model)
        initial_expert_bias = router.expert_bias.detach().clone()
        layout = SharedPrefixLayout(
            prefix_len=prefix_len,
            completion_lens=physical_completion_lens,
            logical_completion_lens=logical_completion_lens,
            padding_multiple=padding_multiple,
        )
        assert star_physical_len % padding_multiple == 0
        assert 0 <= star_physical_len - layout.total_len < padding_multiple

        generator = torch.Generator(device="cuda").manual_seed(17)
        prefix = torch.randn(
            layout.prefix_len,
            1,
            model.config.hidden_size,
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        )
        logical_completions = [
            torch.randn(
                length,
                1,
                model.config.hidden_size,
                generator=generator,
                device="cuda",
                dtype=torch.bfloat16,
            )
            for length in layout.logical_completion_lens
        ]
        dense_global = torch.cat(
            [
                torch.cat(
                    [
                        prefix,
                        completion,
                        completion.new_zeros(
                            dense_physical_len - layout.prefix_len - completion.shape[0],
                            1,
                            model.config.hidden_size,
                        ),
                    ],
                    dim=0,
                )
                for completion in logical_completions
            ],
            dim=1,
        )
        star_pieces = [prefix]
        for completion, physical_completion in zip(
            logical_completions, layout.completion_lens, strict=True
        ):
            star_pieces.extend(
                [
                    completion,
                    completion.new_zeros(
                        physical_completion - completion.shape[0], 1, model.config.hidden_size
                    ),
                ]
            )
        star_real = torch.cat(star_pieces, dim=0)
        star_global = torch.cat(
            [
                star_real,
                star_real.new_zeros(
                    star_physical_len - layout.total_len, 1, model.config.hidden_size
                ),
            ],
            dim=0,
        )
        dense_local = (
            _cp_then_tp_local(dense_global, layout, cp_group, tp_group)
            .detach()
            .requires_grad_(True)
        )
        star_local = (
            _cp_then_tp_local(star_global, layout, cp_group, tp_group).detach().requires_grad_(True)
        )
        assert dense_local.shape == (
            dense_physical_len // (tp_size * cp_size),
            len(layout.completion_lens),
            model.config.hidden_size,
        )
        assert star_local.shape == (
            star_physical_len // (tp_size * cp_size),
            1,
            model.config.hidden_size,
        )

        probe_generator = torch.Generator(device="cuda").manual_seed(29)
        probes = [
            torch.randn(
                completion.shape, generator=probe_generator, device="cuda", dtype=torch.float32
            )
            / completion.numel() ** 0.5
            for completion in logical_completions
        ]
        prompt_probes = [
            torch.randn(
                1,
                1,
                model.config.hidden_size,
                generator=probe_generator,
                device="cuda",
                dtype=torch.float32,
            )
            / model.config.hidden_size**0.5
            for _ in logical_completions
        ]

        def logical_outputs(global_output, shared):
            if not shared:
                return [
                    global_output[
                        layout.prefix_len : layout.prefix_len + logical_length, branch : branch + 1
                    ]
                    for branch, logical_length in enumerate(layout.logical_completion_lens)
                ]
            outputs = []
            for completion_slice, logical_length in zip(
                layout.completion_slices(), layout.logical_completion_lens, strict=True
            ):
                outputs.append(
                    global_output[completion_slice.start : completion_slice.start + logical_length]
                )
            return outputs

        def global_probe_loss(global_output, outputs, shared):
            loss = sum(
                (output.float() * probe).sum()
                for output, probe in zip(outputs, probes, strict=True)
            )
            if shared:
                loss = (
                    loss
                    + (
                        global_output[layout.prefix_len - 1 : layout.prefix_len].float()
                        * sum(prompt_probes)
                    ).sum()
                )
            else:
                loss = loss + sum(
                    (
                        global_output[
                            layout.prefix_len - 1 : layout.prefix_len, branch : branch + 1
                        ].float()
                        * probe
                    ).sum()
                    for branch, probe in enumerate(prompt_probes)
                )
            return loss * (1.0 if tp_group.rank() == 0 and cp_group.rank() == 0 else 0.0)

        dense_rotary = model.rotary_pos_emb(dense_physical_len, cp_group=cp_group)
        star_positions = layout.padded_position_ids(star_physical_len, "cuda")
        star_cp_indices = layout.cp_local_indices(
            star_physical_len, cp_group.size(), cp_group.rank(), "cuda"
        )
        star_rotary = model.rotary_pos_emb.get_emb(padding_multiple).index_select(
            0, star_positions.index_select(0, star_cp_indices)
        )

        def run(input_tensor, *, shared):
            model.zero_grad(set_to_none=True)
            model._shared_prefix_finish_grad_sync_calls = 0
            router.local_tokens_per_expert.zero_()
            with torch.no_grad():
                router.expert_bias.copy_(initial_expert_bias)
            if shared:
                local_output = forward_hybrid_stack_shared_prefix(
                    model.decoder, input_tensor, layout, rotary_pos_emb=star_rotary
                )
            else:
                local_output = model.decoder(
                    hidden_states=input_tensor,
                    attention_mask=None,
                    rotary_pos_emb=dense_rotary,
                    packed_seq_params=None,
                    padding_mask=None,
                )
            global_output = _gather_canonical_with_grad(local_output, cp_group, tp_group)
            outputs = logical_outputs(global_output, shared)
            assert (
                torch.count_nonzero(router.local_tokens_per_expert) == 0
            ), "full recomputation must defer expert-bias accounting to backward replay"
            global_probe_loss(global_output, outputs, shared).backward()
            assert router.weight.grad is None, "the frozen RLVR41 router received a gradient"

            counts_local = router.local_tokens_per_expert.detach().clone()
            counts = counts_local.clone()
            torch.distributed.all_reduce(counts, group=process_groups.tp_dp_cp)
            expected_updated_bias = get_updated_expert_bias(
                counts_local,
                initial_expert_bias.clone(),
                model.config.moe_router_bias_update_rate,
                tp_dp_cp_group=process_groups.tp_dp_cp,
            )
            finalize_model_grads([model], pg_collection=process_groups)
            assert model._shared_prefix_finish_grad_sync_calls == 1
            torch.testing.assert_close(
                router.expert_bias, expected_updated_bias, rtol=0.0, atol=0.0
            )
            assert torch.count_nonzero(router.local_tokens_per_expert) == 0
            gradients = _snapshot_finalized_decoder_grads(model, cp_group)
            input_gradient = _gather_canonical(input_tensor.grad.detach(), cp_group, tp_group)
            return outputs, input_gradient, gradients, counts, router.expert_bias.detach().clone()

        dense_outputs, dense_input_grad, dense_grads, dense_counts, dense_bias = run(
            dense_local, shared=False
        )
        star_outputs, star_input_grad, star_grads, star_counts, star_bias = run(
            star_local, shared=True
        )

        torch.testing.assert_close(star_counts, dense_counts, rtol=0.0, atol=0.0)
        expected_counts = torch.zeros_like(dense_counts)
        selected_experts = torch.topk(initial_expert_bias, model.config.moe_router_topk).indices
        expected_count = dense_physical_len * len(layout.completion_lens)
        expected_counts[selected_experts] = expected_count
        torch.testing.assert_close(dense_counts, expected_counts, rtol=0.0, atol=0.0)
        torch.testing.assert_close(star_bias, dense_bias, rtol=0.0, atol=0.0)
        assert not torch.equal(dense_bias, initial_expert_bias)

        for branch, (dense_output, star_output) in enumerate(
            zip(dense_outputs, star_outputs, strict=True)
        ):
            _assert_parity(
                dense_output,
                star_output,
                rel_l2=0.03,
                cosine=0.995,
                label=f"branch {branch} output",
            )

        dense_prefix_grad = dense_input_grad[: layout.prefix_len].sum(dim=1, keepdim=True)
        _assert_parity(
            dense_prefix_grad,
            star_input_grad[: layout.prefix_len],
            rel_l2=0.03,
            cosine=0.995,
            label="summed prompt input gradient",
        )
        for branch, (logical_length, completion_slice) in enumerate(
            zip(layout.logical_completion_lens, layout.completion_slices(), strict=True)
        ):
            dense_completion_grad = dense_input_grad[
                layout.prefix_len : layout.prefix_len + logical_length, branch : branch + 1
            ]
            star_completion_grad = star_input_grad[
                completion_slice.start : completion_slice.start + logical_length
            ]
            _assert_parity(
                dense_completion_grad,
                star_completion_grad,
                rel_l2=0.03,
                cosine=0.995,
                label=f"branch {branch} input gradient",
            )
            dense_tail = dense_input_grad[
                layout.prefix_len + logical_length : dense_physical_len, branch : branch + 1
            ]
            assert (
                torch.count_nonzero(dense_tail) == 0
            ), f"dense branch {branch} physical padding received logical-loss gradient"
            star_tail = star_input_grad[
                completion_slice.start + logical_length : completion_slice.stop
            ]
            assert (
                torch.count_nonzero(star_tail) == 0
            ), f"star branch {branch} physical padding received logical-loss gradient"
        star_topology_tail = star_input_grad[layout.total_len : star_physical_len]
        assert star_topology_tail.shape[0] == star_physical_len - layout.total_len
        assert (
            torch.count_nonzero(star_topology_tail) == 0
        ), "star final topology padding received logical-loss gradient"

        assert dense_grads.keys() == star_grads.keys()
        parameter_metrics = {
            name: _stats(dense_grads[name], star_grads[name]) for name in dense_grads
        }
        if tp_group.rank() == 0 and cp_group.rank() == 0:
            outliers = {
                name: metrics
                for name, metrics in parameter_metrics.items()
                if metrics["relative_l2"] >= 0.03 or metrics["cosine"] <= 0.995
            }
            print(
                "RLVR41 combined MCore parity metrics: "
                f"TP{tp_size}/CP{cp_size}/EP{ep_size}/ETP{etp_size}, "
                f"M={padding_multiple}, "
                f"K={len(layout.completion_lens)}, outliers={outliers}"
            )
        for name, metrics in parameter_metrics.items():
            assert metrics["relative_l2"] < 0.03, f"parameter gradient {name}: {metrics}"
            assert metrics["cosine"] > 0.995, f"parameter gradient {name}: {metrics}"
    finally:
        Utils.destroy_model_parallel()
        if previous_qk_layer_scaling is None:
            os.environ.pop("NVTE_APPLY_QK_LAYER_SCALING", None)
        else:
            os.environ["NVTE_APPLY_QK_LAYER_SCALING"] = previous_qk_layer_scaling


@pytest.mark.timeout(1200)
def test_combined_tp2_cp2_ep4_shared_prefix_matches_dense_and_finalizes_expert_bias_once():
    """Bring up every combined mechanism on one four-GPU node before WORLD16."""
    if int(os.environ.get("WORLD_SIZE", "1")) != BRINGUP_WORLD_SIZE:
        pytest.skip("combined bring-up requires torchrun world size 4")
    _run_combined_parity(
        tp_size=BRINGUP_TP_SIZE,
        cp_size=BRINGUP_CP_SIZE,
        ep_size=BRINGUP_EP_SIZE,
        etp_size=BRINGUP_ETP_SIZE,
        num_moe_experts=8,
        padding_multiple=BRINGUP_PADDING_MULTIPLE,
        prefix_len=BRINGUP_PREFIX_LEN,
        logical_completion_lens=BRINGUP_LOGICAL_COMPLETION_LENS,
        physical_completion_lens=BRINGUP_PHYSICAL_COMPLETION_LENS,
        dense_physical_len=BRINGUP_DENSE_PHYSICAL_LEN,
        star_physical_len=BRINGUP_STAR_PHYSICAL_LEN,
    )


@pytest.mark.timeout(1800)
def test_rlvr41_tp4_cp4_ep16_shared_prefix_matches_dense_and_finalizes_expert_bias_once():
    """Gate target MCore; NeMo separately gates scalar/fused-loss/deferred-FP32 integration."""
    if int(os.environ.get("WORLD_SIZE", "1")) != RLVR41_WORLD_SIZE:
        pytest.skip("RLVR41 target parity requires torchrun world size 16")
    _run_combined_parity(
        tp_size=RLVR41_TP_SIZE,
        cp_size=RLVR41_CP_SIZE,
        ep_size=RLVR41_EP_SIZE,
        etp_size=RLVR41_ETP_SIZE,
        num_moe_experts=16,
        padding_multiple=RLVR41_PADDING_MULTIPLE,
        prefix_len=RLVR41_PREFIX_LEN,
        logical_completion_lens=RLVR41_LOGICAL_COMPLETION_LENS,
        physical_completion_lens=RLVR41_PHYSICAL_COMPLETION_LENS,
        dense_physical_len=RLVR41_DENSE_PHYSICAL_LEN,
        star_physical_len=RLVR41_STAR_PHYSICAL_LEN,
    )
