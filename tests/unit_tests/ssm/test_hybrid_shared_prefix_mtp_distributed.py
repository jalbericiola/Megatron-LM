# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Distributed trainer-MTP parity for the production shared-prefix path.

Run the TP2/CP1 case on one 2-GPU node with::

    MCORE_SHARED_PREFIX_MTP_DISTRIBUTED_GATE=1 torchrun --nproc-per-node=2 \
        -m pytest -m internal -v \
        tests/unit_tests/ssm/test_hybrid_shared_prefix_mtp_distributed.py

Run the TP2/CP2 case on one 4-GPU node by changing ``--nproc-per-node`` to 4.
The flagship TP4/CP4/EP16 case uses four 4-GPU nodes and the ordinary multi-node
``torchrun`` arguments.  Nonmatching parameterizations are skipped in each
invocation.  Every invocation requires the production capability advertisement;
the test does not widen the runtime capability set.

The dense oracle repeats the prompt in a conventional three-row batch.  The
candidate stores the prompt once, gives every branch different tokens and a
different logical completion length, pads each dense branch to M=32, and adds
sixteen topology-only tokens to the physical star.  Thus an MTP shift across a
sibling boundary, a branch/label-order mismatch, or loss on either kind of
padding changes both the five per-depth losses and the repeated-head gradients.
"""

import os

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.models.hybrid.shared_prefix import (
    SHARED_PREFIX_MTP_DENSE_HEADS_CAPABILITY,
    SHARED_PREFIX_TRAINING_CAPABILITIES,
    SharedPrefixLayout,
)
from megatron.core.transformer.multi_token_prediction import MTPLossAutoScaler, MTPLossLoggingHelper
from tests.unit_tests.test_utilities import Utils

MTP_DEPTHS = 5
PADDING_MULTIPLE = 32
PREFIX_LEN = 8
LOGICAL_COMPLETION_LENS = (15, 19, 23)
PHYSICAL_COMPLETION_LENS = (24, 24, 24)
STAR_PHYSICAL_LEN = 96
VOCAB_SIZE = 256
NUM_MOE_EXPERTS = 128
GATE_ENV = "MCORE_SHARED_PREFIX_MTP_DISTRIBUTED_GATE"

pytestmark = [
    pytest.mark.internal,
    pytest.mark.skipif(
        not torch.cuda.is_available(), reason="distributed MTP parity requires CUDA"
    ),
]


def _require_exact_torchrun(tp_size, cp_size):
    required_world_size = tp_size * cp_size
    if os.environ.get(GATE_ENV) != "1":
        pytest.skip(f"set {GATE_ENV}=1 to run the distributed trainer-MTP gate")
    if int(os.environ.get("WORLD_SIZE", "1")) != required_world_size:
        pytest.skip(f"TP{tp_size}/CP{cp_size} requires world size {required_world_size}")
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    if local_world_size < 1 or required_world_size % local_world_size:
        raise AssertionError(
            f"world size {required_world_size} is incompatible with local world size "
            f"{local_world_size}"
        )


def _cp_local_batch(global_tensor, cp_size, cp_rank):
    """Slice a [batch, global-sequence] tensor in standard CP zigzag order."""
    indices = SharedPrefixLayout.cp_local_indices(
        global_tensor.shape[1], cp_size, cp_rank, global_tensor.device
    )
    return global_tensor.index_select(1, indices).contiguous()


def _build_inputs(layout, cp_size, cp_rank):
    prefix = torch.tensor([3, 5, 7, 9, 11, 13, 15, 17], dtype=torch.long, device="cuda")
    assert prefix.numel() == PREFIX_LEN
    logical_completions = [
        torch.arange(base, base + logical_len, dtype=torch.long, device="cuda")
        for base, logical_len in zip((40, 90, 140), layout.logical_completion_lens, strict=True)
    ]
    physical_completions = []
    for branch, (logical, physical_len) in enumerate(
        zip(logical_completions, layout.completion_lens, strict=True)
    ):
        # Nonzero, branch-specific pad IDs make an accidental loss-mask leak visible.
        padding = logical.new_full((physical_len - logical.shape[0],), 220 + branch)
        physical_completions.append(torch.cat((logical, padding)))

    dense_tokens_global = torch.stack(
        [torch.cat((prefix, completion)) for completion in physical_completions]
    )
    dense_positions_global = torch.arange(PADDING_MULTIPLE, device="cuda").expand_as(
        dense_tokens_global
    )
    dense_loss_mask_global = torch.zeros_like(dense_tokens_global, dtype=torch.float32)
    for branch, logical_len in enumerate(layout.logical_completion_lens):
        dense_loss_mask_global[branch, PREFIX_LEN : PREFIX_LEN + logical_len] = 1.0

    star_real = torch.cat((prefix, *physical_completions)).unsqueeze(0)
    topology_padding = STAR_PHYSICAL_LEN - layout.total_len
    assert topology_padding == 16
    # A distinctive token sentinel makes topology padding observable if it ever
    # leaks into reconstructed MTP labels.
    star_tokens_global = torch.cat(
        (star_real, star_real.new_full((1, topology_padding), VOCAB_SIZE - 1)), dim=1
    )
    star_positions_global = layout.padded_position_ids(STAR_PHYSICAL_LEN, "cuda").unsqueeze(0)
    star_loss_mask_global = torch.cat(
        (
            torch.zeros(PREFIX_LEN, device="cuda"),
            *[
                torch.cat(
                    (
                        torch.ones(logical_len, device="cuda"),
                        torch.zeros(physical_len - logical_len, device="cuda"),
                    )
                )
                for logical_len, physical_len in zip(
                    layout.logical_completion_lens, layout.completion_lens, strict=True
                )
            ],
            torch.zeros(topology_padding, device="cuda"),
        )
    ).unsqueeze(0)

    # These are the sentinels that make the parity comparison sensitive to
    # sibling-boundary shifts and branch/label ordering.
    assert len(set(layout.logical_completion_lens)) == len(layout.logical_completion_lens)
    assert len({int(completion[0]) for completion in logical_completions}) == len(
        logical_completions
    )
    assert star_loss_mask_global[:, layout.total_len :].count_nonzero().item() == 0

    return {
        "dense_tokens": _cp_local_batch(dense_tokens_global, cp_size, cp_rank),
        "dense_positions": _cp_local_batch(dense_positions_global, cp_size, cp_rank),
        "dense_loss_mask": _cp_local_batch(dense_loss_mask_global, cp_size, cp_rank),
        "star_tokens": _cp_local_batch(star_tokens_global, cp_size, cp_rank),
        "star_positions": _cp_local_batch(star_positions_global, cp_size, cp_rank),
        "star_loss_mask": _cp_local_batch(star_loss_mask_global, cp_size, cp_rank),
    }


def _build_model(process_groups, tp_size, cp_size, ep_size):
    from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
    from megatron.core.models.hybrid.hybrid_model import HybridModel
    from megatron.core.transformer import TransformerConfig
    from megatron.core.transformer.enums import AttnBackend

    main_pattern = "M"
    mtp_pattern = "*E"
    unified_pattern = "/".join([main_pattern, *([mtp_pattern] * MTP_DEPTHS)])
    config = TransformerConfig(
        hidden_size=128,
        ffn_hidden_size=256,
        num_layers=len(main_pattern),
        mtp_num_layers=MTP_DEPTHS,
        mtp_use_repeated_layer=True,
        mtp_detach_heads=True,
        mtp_loss_scaling_factor=1.0,
        num_attention_heads=8,
        num_query_groups=max(2, tp_size),
        kv_channels=16,
        # Mamba all-to-all CP requires one local head per TP×CP rank.  Keep
        # d_inner modest while making the synthetic model valid at TP4/CP4.
        mamba_head_dim=32,
        mamba_num_heads=16,
        mamba_num_groups=4,
        tensor_model_parallel_size=tp_size,
        context_parallel_size=cp_size,
        sequence_parallel=True,
        expert_model_parallel_size=ep_size,
        expert_tensor_parallel_size=1,
        num_moe_experts=NUM_MOE_EXPERTS,
        # Select one expert on every EP rank below, so every rank exercises
        # actual expert dispatch and gradient flow.
        moe_router_topk=ep_size,
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
        use_mamba_mem_eff_path=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        # CP attention requires a fused backend in Transformer Engine.  Let TE
        # select the supported implementation, matching the NeMo expanded-
        # sequence setup used by the target recipe.
        attention_backend=AttnBackend.auto,
        apply_rope_fusion=False,
        recompute_granularity="full",
        recompute_method="uniform",
        recompute_num_layers=1,
        distribute_saved_activations=False,
        cross_entropy_loss_fusion=False,
        add_bias_linear=False,
    )
    return HybridModel(
        config=config,
        hybrid_stack_spec=hybrid_stack_spec,
        vocab_size=VOCAB_SIZE,
        max_sequence_length=STAR_PHYSICAL_LEN,
        hybrid_layer_pattern=unified_pattern,
        position_embedding_type="none",
        pre_process=True,
        post_process=True,
        parallel_output=True,
        pg_collection=process_groups,
    ).cuda()


def _freeze_and_spread_mtp_routing(model, ep_size):
    from megatron.core.transformer.moe.router import TopKRouter

    routers = [module for module in model.mtp.modules() if isinstance(module, TopKRouter)]
    assert len(routers) == 1, "repeated *E MTP must own exactly one physical router"
    router = routers[0]
    assert model.config.num_moe_experts % ep_size == 0
    experts_per_ep_rank = model.config.num_moe_experts // ep_size
    selected_experts = torch.arange(ep_size, device="cuda") * experts_per_ep_rank
    with torch.no_grad():
        router.weight.zero_()
        router.expert_bias.fill_(-1.0)
        router.expert_bias[selected_experts] = torch.linspace(
            0.8, 0.5, ep_size, dtype=router.expert_bias.dtype, device="cuda"
        )
    router.weight.requires_grad_(False)
    router.frozen_expert_bias = True
    return router


def _snapshot_reduced_mtp_losses():
    tracker = MTPLossLoggingHelper.tracker
    assert "loss_values" in tracker
    losses = tracker["loss_values"].detach().float().clone()
    assert losses.shape == (MTP_DEPTHS,)
    avg_group = tracker.get("avg_group")
    if avg_group is not None and avg_group.size() > 1:
        torch.distributed.all_reduce(losses, op=torch.distributed.ReduceOp.AVG, group=avg_group)
    return losses


def _snapshot_mtp_parameter_grads(model):
    return {
        name: (
            torch.zeros_like(parameter, dtype=torch.float32)
            if parameter.grad is None
            else parameter.grad.detach().float().clone()
        )
        for name, parameter in model.mtp.named_parameters()
        if parameter.requires_grad
    }


def _run_mtp_backward(model, *, tokens, positions, loss_mask, layout=None):
    model.zero_grad(set_to_none=True)
    # ``clean_metrics_in_tracker`` intentionally retains its allocated depth;
    # clearing makes this gate independent of any earlier test's MTP depth.
    MTPLossLoggingHelper.tracker.clear()
    MTPLossAutoScaler.set_loss_scale(torch.ones((), dtype=torch.float32, device="cuda"))
    logits = model(tokens, positions, None, loss_mask=loss_mask, shared_prefix_layout=layout)
    losses = _snapshot_reduced_mtp_losses()
    # MTP's autograd attachment supplies the auxiliary gradient even though the
    # external RL loss contributes a zero main-logit gradient in this focused gate.
    (logits.float().sum() * 0.0).backward()
    gradients = _snapshot_mtp_parameter_grads(model)
    MTPLossLoggingHelper.tracker.clear()
    return losses, gradients


def _gradient_family(name):
    if ".mtp_model_layer.layers.0." in name:
        return "attention"
    if ".mtp_model_layer.layers.1." in name:
        return "moe"
    return "projection_and_norm"


def _assert_gradient_parity(reference, candidate):
    assert reference.keys() == candidate.keys()
    families = {}
    for name in reference:
        family = _gradient_family(name)
        families.setdefault(family, [[], []])
        families[family][0].append(reference[name].flatten())
        families[family][1].append(candidate[name].flatten())

    assert families.keys() == {"attention", "moe", "projection_and_norm"}
    for family, (reference_parts, candidate_parts) in families.items():
        reference_vector = torch.cat(reference_parts).double()
        candidate_vector = torch.cat(candidate_parts).double()
        assert torch.isfinite(reference_vector).all(), f"{family}: non-finite dense gradient"
        assert torch.isfinite(candidate_vector).all(), f"{family}: non-finite shared gradient"
        reference_norm = reference_vector.norm()
        candidate_norm = candidate_vector.norm()
        assert reference_norm > 0, f"{family}: dense gradient was not exercised"
        assert candidate_norm > 0, f"{family}: shared gradient was not exercised"
        relative_l2 = ((reference_vector - candidate_vector).norm() / reference_norm).item()
        cosine = torch.nn.functional.cosine_similarity(
            reference_vector.unsqueeze(0), candidate_vector.unsqueeze(0)
        ).item()
        assert relative_l2 < 0.18, f"{family}: relative L2 {relative_l2} >= 0.18"
        assert cosine > 0.97, f"{family}: cosine {cosine} <= 0.97"


@pytest.mark.parametrize(
    ("tp_size", "cp_size"),
    [
        pytest.param(2, 1, id="tp2-sp-cp1"),
        pytest.param(2, 2, id="tp2-sp-cp2"),
        pytest.param(4, 4, id="tp4-sp-cp4"),
    ],
)
@pytest.mark.timeout(1200)
def test_repeated_hybrid_mtp_dense_loss_and_gradients_match_shared_prefix(tp_size, cp_size):
    _require_exact_torchrun(tp_size, cp_size)
    # Grouped GEMM is supplied by Transformer Engine in the RL image rather
    # than by a standalone ``grouped_gemm`` Python package.
    for dependency in ("causal_conv1d", "flash_attn", "mamba_ssm"):
        __import__(dependency)
    assert torch.cuda.is_bf16_supported(), "distributed trainer-MTP gate requires CUDA BF16"

    from megatron.core.models.hybrid.hybrid_layer_allocation import validate_segment_layers
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.ssm.mamba_mixer import MambaMixer
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.attention import SelfAttention
    from megatron.core.transformer.moe.moe_layer import MoELayer

    ep_size = tp_size * cp_size
    world_size = int(os.environ["WORLD_SIZE"])
    global_rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    assert world_size == ep_size
    assert 0 <= local_rank < local_world_size
    assert global_rank % local_world_size == local_rank
    Utils.world_size = world_size
    Utils.rank = global_rank
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=1,
        context_parallel_size=cp_size,
        expert_model_parallel_size=ep_size,
        expert_tensor_parallel_size=1,
    )
    try:
        process_groups = ProcessGroupCollection.use_mpu_process_groups()
        assert process_groups.tp.size() == tp_size
        assert process_groups.cp.size() == cp_size
        assert parallel_state.get_expert_model_parallel_world_size() == ep_size

        torch.manual_seed(20260828)
        model_parallel_cuda_manual_seed(20260828)
        model = _build_model(process_groups, tp_size, cp_size, ep_size)
        model.train()
        assert model.position_embedding_type == "none"
        assert model.config.sequence_parallel
        assert model.config.recompute_granularity == "full"
        assert model.config.recompute_method == "uniform"
        assert model.config.recompute_num_layers == 1
        assert model.config.mtp_num_layers == MTP_DEPTHS
        assert model.config.mtp_use_repeated_layer
        assert model.config.tensor_model_parallel_size == tp_size
        assert model.config.context_parallel_size == cp_size
        assert model.config.expert_model_parallel_size == ep_size
        assert model.config.expert_tensor_parallel_size == 1
        assert model.config.num_moe_experts == NUM_MOE_EXPERTS
        assert model.config.moe_router_topk == ep_size
        assert len(model.mtp.layers) == 1
        assert validate_segment_layers("M") == model.decoder.layer_type_list
        assert any(isinstance(module, MambaMixer) for module in model.decoder.modules())
        assert any(isinstance(module, SelfAttention) for module in model.mtp.modules())
        assert any(isinstance(module, MoELayer) for module in model.mtp.modules())
        _freeze_and_spread_mtp_routing(model, ep_size)

        layout = SharedPrefixLayout(
            prefix_len=PREFIX_LEN,
            completion_lens=PHYSICAL_COMPLETION_LENS,
            logical_completion_lens=LOGICAL_COMPLETION_LENS,
            padding_multiple=PADDING_MULTIPLE,
        )
        assert layout.total_len == 80
        inputs = _build_inputs(layout, cp_size, process_groups.cp.rank())

        dense_losses, dense_gradients = _run_mtp_backward(
            model,
            tokens=inputs["dense_tokens"],
            positions=inputs["dense_positions"],
            loss_mask=inputs["dense_loss_mask"],
        )

        assert SHARED_PREFIX_MTP_DENSE_HEADS_CAPABILITY in SHARED_PREFIX_TRAINING_CAPABILITIES
        shared_losses, shared_gradients = _run_mtp_backward(
            model,
            tokens=inputs["star_tokens"],
            positions=inputs["star_positions"],
            loss_mask=inputs["star_loss_mask"],
            layout=layout,
        )

        assert torch.isfinite(dense_losses).all()
        assert torch.isfinite(shared_losses).all()
        assert torch.count_nonzero(dense_losses).item() == MTP_DEPTHS
        torch.testing.assert_close(shared_losses, dense_losses, rtol=0.03, atol=0.02)
        _assert_gradient_parity(dense_gradients, shared_gradients)
    finally:
        MTPLossLoggingHelper.tracker.clear()
        Utils.destroy_model_parallel()
