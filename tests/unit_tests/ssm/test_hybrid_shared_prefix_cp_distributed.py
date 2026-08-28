# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Exact forward/backward parity for context-parallel shared-prefix Hybrid layers.

Run with::

    uv run python -m torch.distributed.run --nproc-per-node=2 -m pytest \
        tests/unit_tests/ssm/test_hybrid_shared_prefix_cp_distributed.py -m internal -v

The four-rank gate exercises the Nemotron 3 Nano head geometry (Q32/KV2 and Mamba H64/G8)::

    uv run python -m torch.distributed.run --nproc-per-node=4 -m pytest \
        tests/unit_tests/ssm/test_hybrid_shared_prefix_cp_distributed.py -m internal -v
"""

import os
from contextlib import contextmanager
from unittest.mock import patch

import pytest
import torch
from torch.distributed.nn.functional import all_gather

from megatron.core import parallel_state
from megatron.core.models.hybrid.shared_prefix import SharedPrefixLayout
from megatron.core.models.hybrid.shared_prefix_fused import (
    _undo_cp_zigzag,
    flash_composed_forest_attention,
    flash_composed_forest_attention_cp,
)
from tests.unit_tests.test_utilities import Utils

pytestmark = [
    pytest.mark.internal,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CP parity requires CUDA"),
]


def _requires_world_size(size):
    return pytest.mark.skipif(
        int(os.environ.get("WORLD_SIZE", "1")) != size,
        reason=f"CP{size} parity requires torchrun --nproc_per_node={size}",
    )


def _gather_canonical_with_grad(local_tensor, cp_group, tp_group=None):
    if tp_group is not None and tp_group.size() > 1:
        local_tensor = torch.cat(all_gather(local_tensor, group=tp_group), dim=0)
    if cp_group.size() == 1:
        return local_tensor
    rank_major = torch.cat(all_gather(local_tensor, group=cp_group), dim=0)
    return _undo_cp_zigzag(rank_major, cp_group.size())


def _gather_canonical(local_tensor, cp_group, tp_group=None):
    if tp_group is not None and tp_group.size() > 1:
        gathered = [torch.empty_like(local_tensor) for _ in range(tp_group.size())]
        torch.distributed.all_gather(gathered, local_tensor, group=tp_group)
        local_tensor = torch.cat(gathered, dim=0)
    if cp_group.size() == 1:
        return local_tensor
    gathered = [torch.empty_like(local_tensor) for _ in range(cp_group.size())]
    torch.distributed.all_gather(gathered, local_tensor, group=cp_group)
    return _undo_cp_zigzag(torch.cat(gathered, dim=0), cp_group.size())


def _relative_l2(reference, actual):
    reference = reference.double().flatten()
    actual = actual.double().flatten()
    return ((reference - actual).norm() / reference.norm().clamp_min(1e-300)).item()


def _cosine(reference, actual):
    reference = reference.double().flatten()
    actual = actual.double().flatten()
    denominator = (reference.norm() * actual.norm()).clamp_min(1e-300)
    return (torch.dot(reference, actual) / denominator).item()


def _run_forest_attention_parity(cp_size, query_heads, kv_heads):
    pytest.importorskip("flash_attn")
    if not torch.cuda.is_bf16_supported():
        pytest.skip("CP parity requires CUDA bf16 support")

    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1, context_parallel_size=cp_size
    )
    try:
        cp_group = parallel_state.get_context_parallel_group()
        cp_rank = cp_group.rank()
        layout = SharedPrefixLayout(prefix_len=5, completion_lens=[4, 3])
        physical_len = ((layout.total_len + 2 * cp_size - 1) // (2 * cp_size)) * (2 * cp_size)
        head_dim = 16

        generator = torch.Generator(device="cuda").manual_seed(20260827)
        global_tensors = [
            torch.randn(
                physical_len,
                1,
                heads,
                head_dim,
                generator=generator,
                device="cuda",
                dtype=torch.bfloat16,
            )
            for heads in (query_heads, kv_heads, kv_heads)
        ]
        reference_q, reference_k, reference_v = [
            tensor.detach().clone().requires_grad_(True) for tensor in global_tensors
        ]
        local_indices = layout.cp_local_indices(physical_len, cp_group.size(), cp_rank, "cuda")
        local_q, local_k, local_v = [
            tensor.index_select(0, local_indices).detach().clone().requires_grad_(True)
            for tensor in global_tensors
        ]

        reference_output = flash_composed_forest_attention(
            reference_q, reference_k, reference_v, layout.forest
        )
        local_output = flash_composed_forest_attention_cp(
            local_q, local_k, local_v, layout.forest, cp_group=cp_group
        )
        expected_local_output = reference_output.index_select(0, local_indices)
        torch.testing.assert_close(
            local_output.float(), expected_local_output.float(), rtol=2e-2, atol=2e-2
        )

        upstream = torch.randn(
            reference_output.shape, generator=generator, device="cuda", dtype=reference_output.dtype
        )
        reference_output.backward(upstream)
        local_output.backward(upstream.index_select(0, local_indices))
        for local_tensor, reference_tensor in zip(
            (local_q, local_k, local_v), (reference_q, reference_k, reference_v)
        ):
            expected_local_gradient = reference_tensor.grad.index_select(0, local_indices)
            torch.testing.assert_close(
                local_tensor.grad.float(), expected_local_gradient.float(), rtol=3e-2, atol=3e-2
            )
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.timeout(180)
@_requires_world_size(2)
def test_cp2_forest_attention_matches_global_forward_and_backward():
    _run_forest_attention_parity(cp_size=2, query_heads=4, kv_heads=1)


@pytest.mark.timeout(180)
@_requires_world_size(4)
def test_cp4_nano_gqa_forest_attention_matches_global_forward_and_backward():
    _run_forest_attention_parity(cp_size=4, query_heads=32, kv_heads=2)


@pytest.mark.timeout(180)
@_requires_world_size(4)
def test_cp4_mha_forest_attention_matches_global_forward_and_backward():
    _run_forest_attention_parity(cp_size=4, query_heads=32, kv_heads=32)


def _run_hybrid_shared_prefix_parity(
    pattern,
    *,
    cp_size,
    hidden_size,
    query_heads,
    kv_heads,
    kv_channels=None,
    mamba_heads=None,
    mamba_groups=8,
    layout=None,
    data_seed=7,
    dense_oracle="separate",
    run_state_fork_candidate=True,
    tp_size=1,
    sequence_parallel=False,
):
    pytest.importorskip("causal_conv1d")
    pytest.importorskip("mamba_ssm")
    if not torch.cuda.is_bf16_supported():
        pytest.skip("CP parity requires CUDA bf16 support")

    previous_qk_layer_scaling = os.environ.get("NVTE_APPLY_QK_LAYER_SCALING")
    os.environ["NVTE_APPLY_QK_LAYER_SCALING"] = "1"

    from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
    from megatron.core.models.hybrid import shared_prefix as shared_prefix_module
    from megatron.core.models.hybrid.hybrid_block import HybridStack
    from megatron.core.models.hybrid.hybrid_layer_allocation import validate_segment_layers
    from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
    from megatron.core.models.hybrid.shared_prefix import (
        _forward_mamba_layer_shared_prefix_cp_replay,
        _forward_mamba_layer_shared_prefix_cp_state_fork,
        _scan_mamba_projected_segment,
        forward_hybrid_stack_shared_prefix,
    )
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.ssm import mamba_mixer as mamba_mixer_module
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer import TransformerConfig
    from megatron.core.typed_torch import apply_module

    Utils.initialize_model_parallel(
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=1,
        context_parallel_size=cp_size,
    )
    try:
        tp_group = parallel_state.get_tensor_model_parallel_group()
        tp_rank = tp_group.rank()
        cp_group = parallel_state.get_context_parallel_group()
        cp_rank = cp_group.rank()
        if (tp_size > 1) != sequence_parallel:
            raise ValueError("TP/SP parity requires TP1/SP=false or TP>1/SP=true")
        # The default total of 31 requires exactly one trailing physical token for
        # both CP2 and CP4. Diagnostics may also pass a no-pad layout explicitly.
        if layout is None:
            layout = SharedPrefixLayout(prefix_len=12, completion_lens=[8, 11])
        if dense_oracle not in {"separate", "matched-batch"}:
            raise ValueError(f"unsupported dense oracle: {dense_oracle}")
        layer_types = validate_segment_layers(pattern)
        torch.manual_seed(123 + data_seed)
        model_parallel_cuda_manual_seed(123 + data_seed)
        config = TransformerConfig(
            hidden_size=hidden_size,
            num_layers=len(pattern),
            num_attention_heads=query_heads,
            num_query_groups=kv_heads,
            kv_channels=kv_channels,
            mamba_num_heads=mamba_heads,
            mamba_num_groups=mamba_groups,
            use_cpu_initialization=True,
            bf16=True,
            params_dtype=torch.bfloat16,
            use_mamba_mem_eff_path=False,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            tensor_model_parallel_size=tp_size,
            sequence_parallel=sequence_parallel,
            context_parallel_size=cp_size,
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
        if "*" in pattern:
            attention_layer_numbers = [
                layer.self_attention.layer_number
                for layer in stack.layers
                if hasattr(layer, "self_attention")
            ]
            assert config.apply_query_key_layer_scaling
            assert attention_layer_numbers == [pattern.index("*") + 1]
            assert attention_layer_numbers[0] > 1
        # CPU initialization is rank-local. Explicitly synchronize the CP-replicated
        # model so changing a token's zigzag owner between dense and packed layouts
        # cannot change the projection it receives.
        cp_source_rank = torch.distributed.get_global_rank(cp_group, 0)
        for parameter in stack.parameters():
            torch.distributed.broadcast(parameter.data, src=cp_source_rank, group=cp_group)
        rotary = RotaryEmbedding(
            kv_channels=config.kv_channels, rotary_percent=1.0, rotary_base=10000, cp_group=cp_group
        ).cuda()

        generator = torch.Generator(device="cuda").manual_seed(data_seed)
        global_prefix = torch.randn(
            layout.prefix_len,
            1,
            hidden_size,
            generator=generator,
            dtype=torch.bfloat16,
            device="cuda",
        )
        global_completions = [
            torch.randn(
                length, 1, hidden_size, generator=generator, dtype=torch.bfloat16, device="cuda"
            )
            for length in layout.completion_lens
        ]
        probe_generator = torch.Generator(device="cuda").manual_seed(data_seed + 12)
        probes = [
            torch.randn(
                completion.shape, generator=probe_generator, dtype=torch.float32, device="cuda"
            )
            / completion.numel() ** 0.5
            for completion in global_completions
        ]

        def localize(global_tensor):
            if cp_size > 1:
                indices = layout.cp_local_indices(
                    global_tensor.shape[0], cp_group.size(), cp_rank, global_tensor.device
                )
                local_tensor = global_tensor.index_select(0, indices)
            else:
                local_tensor = global_tensor
            if tp_size > 1:
                if local_tensor.shape[0] % tp_size:
                    raise ValueError("CP-local sequence must be divisible by TP size")
                local_tensor = torch.chunk(local_tensor, tp_size, dim=0)[tp_rank]
            return local_tensor.detach().clone().requires_grad_(True)

        def pad_for_cp(global_tensor):
            if tp_size > 1:
                multiple = 2 * tp_size * cp_size
            elif cp_size > 1:
                multiple = 2 * cp_size
            else:
                return global_tensor
            padding = (-global_tensor.shape[0]) % multiple
            if not padding:
                return global_tensor
            return torch.cat(
                [global_tensor, global_tensor.new_zeros(padding, *global_tensor.shape[1:])], dim=0
            )

        def probe_loss(outputs):
            loss = sum(
                (output.float() * probe).sum()
                for output, probe in zip(outputs, probes, strict=True)
            )
            # One global objective drives the collective autograd graph. Every rank
            # still participates in backward, but only rank zero seeds the VJP.
            return loss * (1.0 if cp_rank == 0 and tp_rank == 0 else 0.0)

        scan_impl = mamba_mixer_module.mamba_chunk_scan_combined
        assert shared_prefix_module.mamba_chunk_scan_combined is scan_impl

        @contextmanager
        def capture_mamba_scans(records):
            def record_scan(*args, **kwargs):
                result = scan_impl(*args, **kwargs)
                scan_output = result[0] if isinstance(result, tuple) else result
                scan_output.retain_grad()
                records.append((args[0], scan_output))
                return result

            with (
                patch.object(mamba_mixer_module, "mamba_chunk_scan_combined", record_scan),
                patch.object(shared_prefix_module, "mamba_chunk_scan_combined", record_scan),
            ):
                yield

        def install_layer_captures():
            records = [[] for _ in stack.layers]
            attention_records = [[] for _ in stack.layers]
            handles = []
            for layer_index, layer in enumerate(stack.layers):

                def capture_output(_module, _inputs, output, index=layer_index):
                    value = output[0] if isinstance(output, tuple) else output
                    value.retain_grad()
                    records[index].append(value)

                handles.append(layer.register_forward_hook(capture_output))

                attention = getattr(layer, "self_attention", None)
                if attention is not None:

                    def capture_attention_output(_module, _inputs, output, index=layer_index):
                        value = output[0] if isinstance(output, tuple) else output
                        value.retain_grad()
                        attention_records[index].append(value)

                    handles.append(attention.register_forward_hook(capture_attention_output))
            return records, attention_records, handles

        def remove_handles(handles):
            for handle in handles:
                handle.remove()

        def gather_layer_records(records):
            outputs = [
                [_gather_canonical(value.detach(), cp_group, tp_group) for value in layer_records]
                for layer_records in records
            ]
            gradients = [
                [
                    _gather_canonical(value.grad.detach(), cp_group, tp_group)
                    for value in layer_records
                ]
                for layer_records in records
            ]
            return outputs, gradients

        def assemble_scan_snapshot(scan_records):
            call_shapes = [(tuple(x.shape), tuple(y.grad.shape)) for x, y in scan_records]
            assembled = []
            for x, y in scan_records:
                if y.grad is None:
                    raise RuntimeError("captured Mamba scan output did not receive a gradient")
                local_x = x.detach().contiguous()
                local_dy = y.grad.detach().contiguous()
                if cp_size == 1:
                    assembled.append((local_x, local_dy))
                    continue
                x_by_rank = [torch.empty_like(local_x) for _ in range(cp_size)]
                dy_by_rank = [torch.empty_like(local_dy) for _ in range(cp_size)]
                torch.distributed.all_gather(x_by_rank, local_x, group=cp_group)
                torch.distributed.all_gather(dy_by_rank, local_dy, group=cp_group)
                assembled.append((torch.cat(x_by_rank, dim=2), torch.cat(dy_by_rank, dim=2)))

            branch_count = len(layout.completion_lens)
            max_logical_length = layout.prefix_len + max(layout.completion_lens)

            def logical_branch(value, branch_index):
                logical_length = layout.prefix_len + layout.completion_lens[branch_index]
                if value.shape[0] != 1 or value.shape[1] < logical_length:
                    raise RuntimeError(
                        "captured Mamba branch scan does not cover its logical branch span"
                    )
                value = value[:, :logical_length]
                if logical_length == max_logical_length:
                    return value
                return torch.cat(
                    [
                        value,
                        value.new_zeros(1, max_logical_length - logical_length, *value.shape[2:]),
                    ],
                    dim=1,
                )

            if len(assembled) == 1:
                x, dy = assembled[0]
                if x.shape[0] == branch_count:
                    x = torch.cat(
                        [
                            logical_branch(x[index : index + 1], index)
                            for index in range(branch_count)
                        ]
                    )
                    dy = torch.cat(
                        [
                            logical_branch(dy[index : index + 1], index)
                            for index in range(branch_count)
                        ]
                    )
                return {"x": x, "dy": dy, "calls": assembled, "call_shapes": call_shapes}
            if len(assembled) == branch_count and all(x.shape[0] == 1 for x, _ in assembled):
                return {
                    "x": torch.cat(
                        [logical_branch(x, index) for index, (x, _) in enumerate(assembled)], dim=0
                    ),
                    "dy": torch.cat(
                        [logical_branch(dy, index) for index, (_, dy) in enumerate(assembled)],
                        dim=0,
                    ),
                    "calls": assembled,
                    "call_shapes": call_shapes,
                }
            return {"x": None, "dy": None, "calls": assembled, "call_shapes": call_shapes}

        def reconstruct_d_gradient(scan_records):
            mixer = stack.layers[0].mixer
            local_gradient = None
            local_absolute_sum = None
            for x, y in scan_records:
                if y.grad is None:
                    raise RuntimeError("captured Mamba scan output did not receive a gradient")
                product = y.grad.double() * x.double()
                contribution = product.sum(dim=(0, 1))
                absolute_sum = product.abs().sum(dim=(0, 1))
                if not mixer.D_has_hdim:
                    contribution = contribution.sum(dim=-1)
                    absolute_sum = absolute_sum.sum(dim=-1)
                contribution = contribution.reshape(-1)
                absolute_sum = absolute_sum.reshape(-1)
                local_gradient = (
                    contribution if local_gradient is None else local_gradient + contribution
                )
                local_absolute_sum = (
                    absolute_sum
                    if local_absolute_sum is None
                    else local_absolute_sum + absolute_sum
                )
            if local_gradient is None:
                raise RuntimeError("Mamba scan diagnostics captured no calls")

            local_full_gradient = torch.zeros_like(mixer.D, dtype=torch.float64)
            local_full_absolute_sum = torch.zeros_like(mixer.D, dtype=torch.float64)
            if cp_size == 1:
                local_full_gradient.copy_(local_gradient)
                local_full_absolute_sum.copy_(local_absolute_sum)
            else:
                start = cp_rank * local_gradient.numel()
                local_full_gradient[start : start + local_gradient.numel()] = local_gradient
                local_full_absolute_sum[start : start + local_gradient.numel()] = local_absolute_sum
            full_gradient = local_full_gradient.clone()
            full_absolute_sum = local_full_absolute_sum.clone()
            if cp_size > 1:
                torch.distributed.all_reduce(full_gradient, group=cp_group)
                torch.distributed.all_reduce(full_absolute_sum, group=cp_group)
            return (full_gradient, full_absolute_sum, local_full_gradient, local_full_absolute_sum)

        global_dense_branches = [
            torch.cat([global_prefix, completion], dim=0) for completion in global_completions
        ]
        if dense_oracle == "matched-batch":
            dense_physical_len = max(branch.shape[0] for branch in global_dense_branches)
            if cp_size > 1 or tp_size > 1:
                multiple = 2 * cp_size * tp_size
                dense_physical_len = ((dense_physical_len + multiple - 1) // multiple) * multiple
            global_dense_batch = torch.cat(
                [
                    torch.cat(
                        [
                            branch,
                            branch.new_zeros(
                                dense_physical_len - branch.shape[0], *branch.shape[1:]
                            ),
                        ],
                        dim=0,
                    )
                    for branch in global_dense_branches
                ],
                dim=1,
            )
            dense_inputs = [localize(global_dense_batch)]
        else:
            dense_inputs = [localize(pad_for_cp(branch)) for branch in global_dense_branches]
        global_packed = pad_for_cp(torch.cat([global_prefix, *global_completions], dim=0))
        shared_input = localize(global_packed)
        shared_positions = layout.padded_position_ids(global_packed.shape[0], global_packed.device)
        shared_local_indices = (
            torch.arange(global_packed.shape[0], device=global_packed.device)
            if cp_size == 1
            else layout.cp_local_indices(
                global_packed.shape[0], cp_size, cp_rank, global_packed.device
            )
        )
        shared_rotary = rotary.get_emb(int(shared_positions.max().item()) + 1).index_select(
            0, shared_positions.index_select(0, shared_local_indices)
        )

        if pattern == "M" and dense_oracle == "separate" and tp_size == 1:
            # Diagnose every collective/state boundary before comparing the full layer. This
            # characterizes the pinned dependency's explicit prompt-state handoff and independently
            # verifies the uninterrupted replay baseline.
            layer = stack.layers[0]
            mixer = layer.mixer

            def canonical_projection(local_input):
                normalized = apply_module(layer.norm)(
                    local_input.to(dtype=layer.config.params_dtype)
                )
                projected, _ = mixer.in_proj(normalized)
                return mixer.cp.pre_conv_ssm(projected)

            with torch.no_grad():
                dense_projected = [canonical_projection(value) for value in dense_inputs]
                shared_projected = canonical_projection(shared_input)
                shared_prefix_y, _, shared_conv_state, shared_ssm_state = (
                    _scan_mamba_projected_segment(
                        mixer, shared_projected[: layout.prefix_len], capture_state=True
                    )
                )
                assert shared_ssm_state is not None
                assert shared_conv_state is not None
                stage_dtypes = {
                    "projected": str(shared_projected.dtype),
                    "conv_state": str(shared_conv_state.dtype),
                    "ssm_state": str(shared_ssm_state.dtype),
                }
                stage_errors = []
                for branch_index, dense_value in enumerate(dense_projected):
                    completion_slice = layout.completion_slices()[branch_index]
                    completion_len = layout.completion_lens[branch_index]
                    real_len = layout.prefix_len + completion_len
                    replay_real = torch.cat(
                        [shared_projected[: layout.prefix_len], shared_projected[completion_slice]],
                        dim=0,
                    )
                    replay_value = torch.cat(
                        [
                            replay_real,
                            replay_real.new_zeros(
                                dense_value.shape[0] - real_len, *replay_real.shape[1:]
                            ),
                        ],
                        dim=0,
                    )
                    projection_error = max(
                        _relative_l2(
                            dense_value[: layout.prefix_len], shared_projected[: layout.prefix_len]
                        ),
                        _relative_l2(
                            dense_value[layout.prefix_len : real_len],
                            shared_projected[completion_slice],
                        ),
                    )
                    full_y, full_z, _, _ = _scan_mamba_projected_segment(mixer, dense_value)
                    replay_y, replay_z, _, _ = _scan_mamba_projected_segment(mixer, replay_value)
                    prefix_y, prefix_z, conv_state, ssm_state = _scan_mamba_projected_segment(
                        mixer, dense_value[: layout.prefix_len], capture_state=True
                    )
                    branch_y, branch_z, _, _ = _scan_mamba_projected_segment(
                        mixer,
                        dense_value[layout.prefix_len : real_len],
                        conv_context=conv_state,
                        ssm_initial_state=ssm_state,
                    )
                    split_y = torch.cat([prefix_y, branch_y], dim=0)
                    split_z = torch.cat([prefix_z, branch_z], dim=0)
                    split_scan_error = max(
                        _relative_l2(full_y[:real_len], split_y),
                        _relative_l2(full_z[:real_len], split_z),
                    )
                    replay_scan_error = max(
                        _relative_l2(full_y[:real_len], replay_y[:real_len]),
                        _relative_l2(full_z[:real_len], replay_z[:real_len]),
                    )
                    full_local_y = mixer.cp.post_conv_ssm(full_y)
                    full_local_z = mixer.cp.post_conv_ssm(full_z)
                    replay_local_y = mixer.cp.post_conv_ssm(replay_y)
                    replay_local_z = mixer.cp.post_conv_ssm(replay_z)
                    local_indices = (
                        torch.arange(dense_value.shape[0], device=dense_value.device)
                        if cp_size == 1
                        else layout.cp_local_indices(
                            dense_value.shape[0], cp_size, cp_rank, dense_value.device
                        )
                    )
                    local_real = local_indices < real_len
                    post_error = max(
                        _relative_l2(full_local_y[local_real], replay_local_y[local_real]),
                        _relative_l2(full_local_z[local_real], replay_local_z[local_real]),
                    )
                    full_output, _ = mixer.out_proj(mixer.norm(full_local_y, full_local_z))
                    replay_output, _ = mixer.out_proj(mixer.norm(replay_local_y, replay_local_z))
                    output_error = _relative_l2(full_output[local_real], replay_output[local_real])
                    state_error = max(
                        _relative_l2(conv_state, shared_conv_state),
                        _relative_l2(ssm_state, shared_ssm_state),
                        _relative_l2(prefix_y, shared_prefix_y),
                    )
                    stage_errors.append(
                        {
                            "projection": projection_error,
                            "prefix_state": state_error,
                            "split_state_boundary": split_scan_error,
                            "replay_scan": replay_scan_error,
                            "post_conv": post_error,
                            "output_projection": output_error,
                        }
                    )
                if cp_rank == 0 and tp_rank == 0:
                    print(f"CP{cp_size} Mamba stage dtypes: {stage_dtypes}")
                    print(f"CP{cp_size} Mamba stage parity: {stage_errors}")
                for errors in stage_errors:
                    assert errors["projection"] < 0.02
                    assert errors["split_state_boundary"] < 0.02
                    assert errors["replay_scan"] < 0.02
                    assert errors["post_conv"] < 0.02
                    assert errors["output_projection"] < 0.02

        dense_outputs = []
        dense_scan_records = []
        dense_layer_records, dense_attention_records, dense_handles = install_layer_captures()
        try:
            with capture_mamba_scans(dense_scan_records):
                if dense_oracle == "matched-batch":
                    local_input = dense_inputs[0]
                    dense_rotary = rotary(
                        local_input.shape[0] * cp_size * tp_size, cp_group=cp_group
                    )
                    local_output = stack(
                        hidden_states=local_input, attention_mask=None, rotary_pos_emb=dense_rotary
                    )
                    global_output = _gather_canonical_with_grad(local_output, cp_group, tp_group)
                    dense_outputs.extend(
                        global_output[
                            layout.prefix_len : layout.prefix_len + completion_len,
                            branch_index : branch_index + 1,
                        ]
                        for branch_index, completion_len in enumerate(layout.completion_lens)
                    )
                else:
                    for local_input, completion_len in zip(
                        dense_inputs, layout.completion_lens, strict=True
                    ):
                        dense_rotary = rotary(
                            local_input.shape[0] * cp_size * tp_size, cp_group=cp_group
                        )
                        local_output = stack(
                            hidden_states=local_input,
                            attention_mask=None,
                            rotary_pos_emb=dense_rotary,
                        )
                        global_output = _gather_canonical_with_grad(
                            local_output, cp_group, tp_group
                        )
                        dense_outputs.append(
                            global_output[layout.prefix_len : layout.prefix_len + completion_len]
                        )
        finally:
            remove_handles(dense_handles)
        probe_loss(dense_outputs).backward()
        dense_layer_outputs, dense_layer_grads = gather_layer_records(dense_layer_records)
        dense_attention_outputs, dense_attention_grads = gather_layer_records(
            dense_attention_records
        )
        if dense_oracle == "matched-batch":

            def split_matched_batch_records(records, label):
                split_records = []
                for values in records:
                    if not values:
                        split_records.append([])
                        continue
                    if len(values) != 1:
                        raise RuntimeError(f"matched dense {label} expected one batched record")
                    value = values[0]
                    split_records.append(
                        [
                            value[:, branch_index : branch_index + 1]
                            for branch_index in range(len(probes))
                        ]
                    )
                return split_records

            dense_layer_outputs = split_matched_batch_records(dense_layer_outputs, "layer outputs")
            dense_layer_grads = split_matched_batch_records(dense_layer_grads, "layer gradients")
            dense_attention_outputs = split_matched_batch_records(
                dense_attention_outputs, "attention outputs"
            )
            dense_attention_grads = split_matched_batch_records(
                dense_attention_grads, "attention gradients"
            )
        dense_scan_snapshot = assemble_scan_snapshot(dense_scan_records)
        (
            dense_d_formula,
            dense_d_absolute_sum,
            dense_local_d_formula,
            dense_local_d_absolute_sum,
        ) = reconstruct_d_gradient(dense_scan_records)
        if dense_oracle == "matched-batch":
            global_dense_input_grad = _gather_canonical(dense_inputs[0].grad, cp_group, tp_group)
            dense_input_grads = [
                global_dense_input_grad[:, branch_index : branch_index + 1]
                for branch_index in range(len(probes))
            ]
        else:
            dense_input_grads = [
                _gather_canonical(local_input.grad, cp_group, tp_group)
                for local_input in dense_inputs
            ]
        dense_parameter_grads = {}
        dense_local_parameter_grads = {}
        for name, parameter in stack.named_parameters():
            if parameter.grad is None:
                continue
            local_gradient = parameter.grad.detach().clone()
            dense_local_parameter_grads[name] = local_gradient
            gradient = local_gradient.clone()
            if tp_size > 1 and getattr(parameter, "sequence_parallel", False):
                # Match finalize_model_grads: replicated parameters see only their
                # local SP token slice during backward and require a TP SUM.
                torch.distributed.all_reduce(gradient, group=tp_group)
            torch.distributed.all_reduce(gradient, group=cp_group)
            dense_parameter_grads[name] = gradient

        dense_prefix_grad = sum(gradient[: layout.prefix_len] for gradient in dense_input_grads)

        def run_shared(local_input, mamba_impl=None):
            stack.zero_grad(set_to_none=True)
            scan_records = []
            layer_records, attention_records, handles = install_layer_captures()
            mamba_attribute = (
                "_forward_mamba_layer_shared_prefix"
                if cp_size == 1 and tp_size == 1
                else "_forward_mamba_layer_shared_prefix_cp"
            )
            mamba_implementation = mamba_impl or getattr(shared_prefix_module, mamba_attribute)
            mamba_layer_indices = {id(layer): index for index, layer in enumerate(stack.layers)}

            def capture_shared_mamba(layer, *args, **kwargs):
                output = mamba_implementation(layer, *args, **kwargs)
                output.retain_grad()
                layer_records[mamba_layer_indices[id(layer)]].append(output)
                return output

            try:
                with (
                    patch.object(shared_prefix_module, mamba_attribute, capture_shared_mamba),
                    capture_mamba_scans(scan_records),
                ):
                    local_output = forward_hybrid_stack_shared_prefix(
                        stack, local_input, layout, rotary_pos_emb=shared_rotary
                    )
            finally:
                remove_handles(handles)
            global_output = _gather_canonical_with_grad(local_output, cp_group, tp_group)
            outputs = [
                global_output[completion_slice] for completion_slice in layout.completion_slices()
            ]
            probe_loss(outputs).backward()
            layer_outputs, layer_grads = gather_layer_records(layer_records)
            attention_outputs, attention_grads = gather_layer_records(attention_records)
            scan_snapshot = assemble_scan_snapshot(scan_records)
            d_formula, d_absolute_sum, local_d_formula, local_d_absolute_sum = (
                reconstruct_d_gradient(scan_records)
            )
            input_grad = _gather_canonical(local_input.grad, cp_group, tp_group)
            parameter_grads = {}
            local_parameter_grads = {}
            for name, parameter in stack.named_parameters():
                if parameter.grad is None:
                    continue
                local_gradient = parameter.grad.detach().clone()
                local_parameter_grads[name] = local_gradient
                gradient = local_gradient.clone()
                if tp_size > 1 and getattr(parameter, "sequence_parallel", False):
                    torch.distributed.all_reduce(gradient, group=tp_group)
                torch.distributed.all_reduce(gradient, group=cp_group)
                parameter_grads[name] = gradient
            return (
                outputs,
                input_grad,
                parameter_grads,
                local_parameter_grads,
                layer_outputs,
                layer_grads,
                attention_outputs,
                attention_grads,
                scan_snapshot,
                d_formula,
                d_absolute_sum,
                local_d_formula,
                local_d_absolute_sum,
            )

        def parameter_stats(reference, actual):
            reference = reference.double()
            actual = actual.double()
            difference = reference - actual
            return {
                "relative_l2": _relative_l2(reference, actual),
                "reference_norm": reference.norm().item(),
                "actual_norm": actual.norm().item(),
                "absolute_l2": difference.norm().item(),
                "absolute_max": difference.abs().max().item(),
                "cosine": _cosine(reference, actual),
            }

        def parity_metrics(
            label,
            outputs,
            input_grad,
            parameter_grads,
            local_parameter_grads,
            layer_outputs,
            layer_grads,
            attention_outputs,
            attention_grads,
            scan_snapshot,
            d_formula,
            d_absolute_sum,
            local_d_formula,
            local_d_absolute_sum,
        ):
            forward_errors = [
                _relative_l2(dense_output, output)
                for dense_output, output in zip(dense_outputs, outputs, strict=True)
            ]
            prefix_error = _relative_l2(dense_prefix_grad, input_grad[: layout.prefix_len])
            prefix_cosine = _cosine(dense_prefix_grad, input_grad[: layout.prefix_len])
            completion_errors = []
            completion_cosines = []
            for dense_gradient, completion_slice in zip(
                dense_input_grads, layout.completion_slices(), strict=True
            ):
                completion_len = completion_slice.stop - completion_slice.start
                dense_completion_grad = dense_gradient[
                    layout.prefix_len : layout.prefix_len + completion_len
                ]
                shared_completion_grad = input_grad[completion_slice]
                completion_errors.append(
                    _relative_l2(dense_completion_grad, shared_completion_grad)
                )
                completion_cosines.append(_cosine(dense_completion_grad, shared_completion_grad))

            assert dense_parameter_grads.keys() == parameter_grads.keys()
            parameters = {
                name: parameter_stats(dense_parameter_grads[name], parameter_grads[name])
                for name in dense_parameter_grads
            }
            worst_parameter = max(parameters, key=lambda name: parameters[name]["relative_l2"])
            outliers = {
                name: stats
                for name, stats in parameters.items()
                if stats["relative_l2"] >= 0.03 or stats["cosine"] <= 0.995
            }
            summary = {
                "forward_rel_l2_max": max(forward_errors),
                "prefix_input_grad_rel_l2": prefix_error,
                "completion_input_grad_rel_l2_max": max(completion_errors),
                "parameter_grad_rel_l2_max": parameters[worst_parameter]["relative_l2"],
                "parameter_grad_worst": worst_parameter,
            }
            boundary_diagnostics = {}
            for layer_index, (dense_outputs_at_layer, dense_grads_at_layer) in enumerate(
                zip(dense_layer_outputs, dense_layer_grads, strict=True)
            ):
                if len(layer_outputs[layer_index]) != 1 or len(layer_grads[layer_index]) != 1:
                    raise RuntimeError("shared-prefix layer diagnostics expected one packed record")
                shared_output_at_layer = layer_outputs[layer_index][0]
                shared_grad_at_layer = layer_grads[layer_index][0]
                completion_output_stats = []
                completion_grad_stats = []
                prefix_output_stats = []
                for branch_index, completion_slice in enumerate(layout.completion_slices()):
                    completion_len = layout.completion_lens[branch_index]
                    dense_output_at_layer = dense_outputs_at_layer[branch_index]
                    dense_grad_at_layer = dense_grads_at_layer[branch_index]
                    prefix_output_stats.append(
                        parameter_stats(
                            dense_output_at_layer[: layout.prefix_len],
                            shared_output_at_layer[: layout.prefix_len],
                        )
                    )
                    completion_output_stats.append(
                        parameter_stats(
                            dense_output_at_layer[
                                layout.prefix_len : layout.prefix_len + completion_len
                            ],
                            shared_output_at_layer[completion_slice],
                        )
                    )
                    completion_grad_stats.append(
                        parameter_stats(
                            dense_grad_at_layer[
                                layout.prefix_len : layout.prefix_len + completion_len
                            ],
                            shared_grad_at_layer[completion_slice],
                        )
                    )
                dense_prefix_grad_at_layer = sum(
                    value[: layout.prefix_len] for value in dense_grads_at_layer
                )
                boundary_diagnostics[f"layer_{layer_index}"] = {
                    "prefix_output": prefix_output_stats,
                    "completion_output": completion_output_stats,
                    "prefix_incoming_grad": parameter_stats(
                        dense_prefix_grad_at_layer, shared_grad_at_layer[: layout.prefix_len]
                    ),
                    "completion_incoming_grad": completion_grad_stats,
                }

            attention_diagnostics = {}
            for layer_index, (dense_outputs_at_attention, dense_grads_at_attention) in enumerate(
                zip(dense_attention_outputs, dense_attention_grads, strict=True)
            ):
                if not dense_outputs_at_attention:
                    continue
                if (
                    len(attention_outputs[layer_index]) != 1
                    or len(attention_grads[layer_index]) != 1
                ):
                    raise RuntimeError(
                        "shared-prefix attention diagnostics expected one packed record"
                    )
                shared_output_at_attention = attention_outputs[layer_index][0]
                shared_grad_at_attention = attention_grads[layer_index][0]
                completion_output_stats = []
                completion_grad_stats = []
                prefix_output_stats = []
                for branch_index, completion_slice in enumerate(layout.completion_slices()):
                    completion_len = layout.completion_lens[branch_index]
                    dense_output_at_attention = dense_outputs_at_attention[branch_index]
                    dense_grad_at_attention = dense_grads_at_attention[branch_index]
                    prefix_output_stats.append(
                        parameter_stats(
                            dense_output_at_attention[: layout.prefix_len],
                            shared_output_at_attention[: layout.prefix_len],
                        )
                    )
                    completion_output_stats.append(
                        parameter_stats(
                            dense_output_at_attention[
                                layout.prefix_len : layout.prefix_len + completion_len
                            ],
                            shared_output_at_attention[completion_slice],
                        )
                    )
                    completion_grad_stats.append(
                        parameter_stats(
                            dense_grad_at_attention[
                                layout.prefix_len : layout.prefix_len + completion_len
                            ],
                            shared_grad_at_attention[completion_slice],
                        )
                    )
                dense_prefix_grad_at_attention = sum(
                    value[: layout.prefix_len] for value in dense_grads_at_attention
                )
                attention_diagnostics[f"layer_{layer_index}"] = {
                    "prefix_output": prefix_output_stats,
                    "completion_output": completion_output_stats,
                    "prefix_incoming_grad": parameter_stats(
                        dense_prefix_grad_at_attention,
                        shared_grad_at_attention[: layout.prefix_len],
                    ),
                    "completion_incoming_grad": completion_grad_stats,
                }

            scan_diagnostics = {
                "dense_call_shapes": dense_scan_snapshot["call_shapes"],
                "shared_call_shapes": scan_snapshot["call_shapes"],
            }
            if dense_scan_snapshot["x"] is not None and scan_snapshot["x"] is not None:
                dense_scan_x = dense_scan_snapshot["x"]
                dense_scan_dy = dense_scan_snapshot["dy"]
                shared_scan_x = scan_snapshot["x"]
                shared_scan_dy = scan_snapshot["dy"]
                if dense_scan_x.shape != shared_scan_x.shape:
                    raise RuntimeError(
                        "canonical dense/shared replay scan shapes differ: "
                        f"{dense_scan_x.shape} != {shared_scan_x.shape}"
                    )
                if dense_scan_x.shape[0] != len(layout.completion_lens):
                    raise RuntimeError("canonical replay scan batch does not match branch count")
                prefix_dy_dense = dense_scan_dy[:, : layout.prefix_len].sum(dim=0)
                prefix_dy_shared = shared_scan_dy[:, : layout.prefix_len].sum(dim=0)
                prefix_product_dense = (
                    dense_scan_dy[:, : layout.prefix_len] * dense_scan_x[:, : layout.prefix_len]
                ).sum(dim=0)
                prefix_product_shared = (
                    shared_scan_dy[:, : layout.prefix_len] * shared_scan_x[:, : layout.prefix_len]
                ).sum(dim=0)
                scan_diagnostics.update(
                    {
                        "prefix_x": [
                            parameter_stats(
                                dense_scan_x[branch_index, : layout.prefix_len],
                                shared_scan_x[branch_index, : layout.prefix_len],
                            )
                            for branch_index in range(len(layout.completion_lens))
                        ],
                        "prefix_dy_summed_across_branches": parameter_stats(
                            prefix_dy_dense, prefix_dy_shared
                        ),
                        "prefix_dy_x_summed_across_branches": parameter_stats(
                            prefix_product_dense, prefix_product_shared
                        ),
                        "completion_x": [
                            parameter_stats(
                                dense_scan_x[
                                    branch_index,
                                    layout.prefix_len : layout.prefix_len + completion_len,
                                ],
                                shared_scan_x[
                                    branch_index,
                                    layout.prefix_len : layout.prefix_len + completion_len,
                                ],
                            )
                            for branch_index, completion_len in enumerate(layout.completion_lens)
                        ],
                        "completion_dy": [
                            parameter_stats(
                                dense_scan_dy[
                                    branch_index,
                                    layout.prefix_len : layout.prefix_len + completion_len,
                                ],
                                shared_scan_dy[
                                    branch_index,
                                    layout.prefix_len : layout.prefix_len + completion_len,
                                ],
                            )
                            for branch_index, completion_len in enumerate(layout.completion_lens)
                        ],
                    }
                )

            def logical_scan_segments(snapshot):
                """Return prefix and logical completion scan tensors in branch order."""
                if snapshot["x"] is not None:
                    scan_x = snapshot["x"]
                    scan_dy = snapshot["dy"]
                    return (
                        (scan_x[:, : layout.prefix_len], scan_dy[:, : layout.prefix_len]),
                        [
                            (
                                scan_x[
                                    branch_index : branch_index + 1,
                                    layout.prefix_len : layout.prefix_len + completion_len,
                                ],
                                scan_dy[
                                    branch_index : branch_index + 1,
                                    layout.prefix_len : layout.prefix_len + completion_len,
                                ],
                            )
                            for branch_index, completion_len in enumerate(layout.completion_lens)
                        ],
                    )

                calls = snapshot["calls"]
                if (
                    len(calls) != 2
                    or calls[0][0].shape[0] != 1
                    or calls[0][0].shape[1] != layout.prefix_len
                    or calls[1][0].shape[0] != len(layout.completion_lens)
                ):
                    raise RuntimeError(
                        "state-fork scan diagnostics require one prefix and one branch call"
                    )
                prefix = calls[0]
                branch_x, branch_dy = calls[1]
                completions = [
                    (
                        branch_x[branch_index : branch_index + 1, :completion_len],
                        branch_dy[branch_index : branch_index + 1, :completion_len],
                    )
                    for branch_index, completion_len in enumerate(layout.completion_lens)
                ]
                return prefix, completions

            def d_path_error_diagnostics(reference_snapshot, actual_snapshot, denominator):
                """Attribute each D-formula delta to the captured scan ``dy*x`` path."""
                mixer = stack.layers[0].mixer
                reference_prefix, reference_completions = logical_scan_segments(reference_snapshot)
                actual_prefix, actual_completions = logical_scan_segments(actual_snapshot)
                path_absolute = torch.zeros_like(denominator, dtype=torch.float64)
                dy_absolute = torch.zeros_like(denominator, dtype=torch.float64)
                x_absolute = torch.zeros_like(denominator, dtype=torch.float64)
                reference_x_spread = 0.0

                def reduce_to_d(value):
                    reduced = value.sum(dim=(0, 1))
                    if not mixer.D_has_hdim:
                        reduced = reduced.sum(dim=-1)
                    return reduced.reshape_as(denominator)

                def accumulate(reference_x, reference_dy, actual_x, actual_dy):
                    nonlocal path_absolute, dy_absolute, x_absolute, reference_x_spread
                    if reference_x.shape[0] != actual_x.shape[0]:
                        if actual_x.shape[0] != 1:
                            raise RuntimeError("only the state-fork prefix may change scan batch")
                        reference_x_spread = max(
                            reference_x_spread, (reference_x - reference_x[:1]).abs().max().item()
                        )
                        reference_product = (reference_dy.double() * reference_x.double()).sum(
                            dim=0, keepdim=True
                        )
                        reference_dy = reference_dy.sum(dim=0, keepdim=True)
                        reference_x = reference_x[:1]
                    else:
                        reference_product = reference_dy.double() * reference_x.double()
                    actual_product = actual_dy.double() * actual_x.double()
                    path_absolute += reduce_to_d((reference_product - actual_product).abs())
                    dy_absolute += reduce_to_d(
                        ((reference_dy.double() - actual_dy.double()) * reference_x.double()).abs()
                    )
                    x_absolute += reduce_to_d(
                        (actual_dy.double() * (reference_x.double() - actual_x.double())).abs()
                    )

                accumulate(*reference_prefix, *actual_prefix)
                for reference_completion, actual_completion in zip(
                    reference_completions, actual_completions, strict=True
                ):
                    accumulate(*reference_completion, *actual_completion)

                conditioned_path = path_absolute / denominator.double().clamp_min(1e-300)
                conditioned_dy = dy_absolute / denominator.double().clamp_min(1e-300)
                conditioned_x = x_absolute / denominator.double().clamp_min(1e-300)
                return {
                    "max_dy_x_path_conditioned": conditioned_path.max().item(),
                    "max_dy_only_conditioned": conditioned_dy.max().item(),
                    "max_x_only_conditioned": conditioned_x.max().item(),
                    "reference_prefix_x_spread_max": reference_x_spread,
                    "path_absolute": path_absolute,
                    "conditioned_path": conditioned_path,
                    "conditioned_dy": conditioned_dy,
                    "conditioned_x": conditioned_x,
                }

            d_name = next(name for name in dense_parameter_grads if name.endswith(".mixer.D"))
            dense_d_by_rank = [
                torch.empty_like(dense_local_parameter_grads[d_name]) for _ in range(cp_size)
            ]
            shared_d_by_rank = [
                torch.empty_like(local_parameter_grads[d_name]) for _ in range(cp_size)
            ]
            dense_d_formula_by_rank = [
                torch.empty_like(dense_local_d_formula) for _ in range(cp_size)
            ]
            shared_d_formula_by_rank = [torch.empty_like(local_d_formula) for _ in range(cp_size)]
            dense_d_absolute_sum_by_rank = [
                torch.empty_like(dense_local_d_absolute_sum) for _ in range(cp_size)
            ]
            shared_d_absolute_sum_by_rank = [
                torch.empty_like(local_d_absolute_sum) for _ in range(cp_size)
            ]
            torch.distributed.all_gather(
                dense_d_by_rank, dense_local_parameter_grads[d_name], group=cp_group
            )
            torch.distributed.all_gather(
                shared_d_by_rank, local_parameter_grads[d_name], group=cp_group
            )
            torch.distributed.all_gather(
                dense_d_formula_by_rank, dense_local_d_formula, group=cp_group
            )
            torch.distributed.all_gather(shared_d_formula_by_rank, local_d_formula, group=cp_group)
            torch.distributed.all_gather(
                dense_d_absolute_sum_by_rank, dense_local_d_absolute_sum, group=cp_group
            )
            torch.distributed.all_gather(
                shared_d_absolute_sum_by_rank, local_d_absolute_sum, group=cp_group
            )

            def max_conditioned_error(reference, actual, absolute_sum):
                return (
                    (
                        (reference.double() - actual.double()).abs()
                        / absolute_sum.double().clamp_min(1e-300)
                    )
                    .max()
                    .item()
                )

            d_formula_diagnostics = {
                "all_finite": all(
                    torch.isfinite(value).all().item()
                    for value in (
                        dense_parameter_grads[d_name],
                        parameter_grads[d_name],
                        dense_d_formula,
                        d_formula,
                        dense_d_absolute_sum,
                        d_absolute_sum,
                    )
                ),
                "dense_formula_vs_autograd": parameter_stats(
                    dense_parameter_grads[d_name], dense_d_formula
                ),
                "shared_formula_vs_autograd": parameter_stats(parameter_grads[d_name], d_formula),
                "dense_vs_shared_formula": parameter_stats(dense_d_formula, d_formula),
                "dense_sum_abs_dy_x": dense_d_absolute_sum.sum().item(),
                "shared_sum_abs_dy_x": d_absolute_sum.sum().item(),
                "dense_vs_shared_formula_l2_over_sum_abs_dy_x": (
                    (dense_d_formula - d_formula).norm().item()
                    / dense_d_absolute_sum.sum().clamp_min(1e-300).item()
                ),
                "dense_vs_shared_formula_max_elementwise_conditioned": max_conditioned_error(
                    dense_d_formula, d_formula, dense_d_absolute_sum
                ),
                "dense_formula_vs_autograd_max_elementwise_conditioned": max_conditioned_error(
                    dense_d_formula, dense_parameter_grads[d_name], dense_d_absolute_sum
                ),
                "shared_formula_vs_autograd_max_elementwise_conditioned": max_conditioned_error(
                    d_formula, parameter_grads[d_name], d_absolute_sum
                ),
                "local_autograd_dense_vs_shared": [
                    parameter_stats(dense_local, shared_local)
                    for dense_local, shared_local in zip(
                        dense_d_by_rank, shared_d_by_rank, strict=True
                    )
                ],
                "local_dense_formula_vs_autograd": [
                    {
                        "stats": parameter_stats(autograd, formula),
                        "max_elementwise_conditioned": max_conditioned_error(
                            formula, autograd, absolute_sum
                        ),
                    }
                    for autograd, formula, absolute_sum in zip(
                        dense_d_by_rank,
                        dense_d_formula_by_rank,
                        dense_d_absolute_sum_by_rank,
                        strict=True,
                    )
                ],
                "local_shared_formula_vs_autograd": [
                    {
                        "stats": parameter_stats(autograd, formula),
                        "max_elementwise_conditioned": max_conditioned_error(
                            formula, autograd, absolute_sum
                        ),
                    }
                    for autograd, formula, absolute_sum in zip(
                        shared_d_by_rank,
                        shared_d_formula_by_rank,
                        shared_d_absolute_sum_by_rank,
                        strict=True,
                    )
                ],
            }
            path_diagnostics = d_path_error_diagnostics(
                dense_scan_snapshot, scan_snapshot, dense_d_absolute_sum
            )
            formula_conditioned = (
                dense_d_formula.double() - d_formula.double()
            ).abs() / dense_d_absolute_sum.double().clamp_min(1e-300)
            worst_index = int(formula_conditioned.flatten().argmax().item())
            d_formula_diagnostics["dy_x_path"] = {
                key: value
                for key, value in path_diagnostics.items()
                if not isinstance(value, torch.Tensor)
            }
            d_formula_diagnostics["worst_conditioned_element"] = {
                "flat_index": worst_index,
                "dense_formula": dense_d_formula.flatten()[worst_index].item(),
                "shared_formula": d_formula.flatten()[worst_index].item(),
                "dense_sum_abs_dy_x": dense_d_absolute_sum.flatten()[worst_index].item(),
                "formula_conditioned": formula_conditioned.flatten()[worst_index].item(),
                "dy_x_path_conditioned": path_diagnostics["conditioned_path"]
                .flatten()[worst_index]
                .item(),
                "dy_only_conditioned": path_diagnostics["conditioned_dy"]
                .flatten()[worst_index]
                .item(),
                "x_only_conditioned": path_diagnostics["conditioned_x"]
                .flatten()[worst_index]
                .item(),
            }
            if cp_rank == 0 and tp_rank == 0:
                topology = f"TP{tp_size}/CP{cp_size}/SP{int(sequence_parallel)}"
                print(f"{topology} {pattern} {label} parity: {summary}")
                print(f"{topology} {pattern} {label} parameter outliers: {outliers}")
                print(
                    f"{topology} {pattern} {label} layer-boundary diagnostics: "
                    f"{boundary_diagnostics}"
                )
                print(
                    f"{topology} {pattern} {label} attention diagnostics: "
                    f"{attention_diagnostics}"
                )
                print(f"{topology} {pattern} {label} scan diagnostics: {scan_diagnostics}")
                print(
                    f"{topology} {pattern} {label} D-formula diagnostics: "
                    f"{d_formula_diagnostics}"
                )

            return {
                "summary": summary,
                "forward_errors": forward_errors,
                "prefix_error": prefix_error,
                "prefix_cosine": prefix_cosine,
                "completion_errors": completion_errors,
                "completion_cosines": completion_cosines,
                "parameters": parameters,
                "boundary_diagnostics": boundary_diagnostics,
                "attention_diagnostics": attention_diagnostics,
                "scan_diagnostics": scan_diagnostics,
                "d_formula_diagnostics": d_formula_diagnostics,
            }

        def assert_parity(label, metrics):
            assert metrics["d_formula_diagnostics"]["all_finite"], label
            assert max(metrics["forward_errors"]) < 0.02, label
            assert metrics["prefix_error"] < 0.03, label
            assert metrics["prefix_cosine"] > 0.995, label
            assert max(metrics["completion_errors"]) < 0.03, label
            assert min(metrics["completion_cosines"]) > 0.995, label
            for name, stats in metrics["parameters"].items():
                if name.endswith(".mixer.D"):
                    d_diagnostics = metrics["d_formula_diagnostics"]
                    # D is a cancellation-heavy direct-skip reduction: dividing by its
                    # near-zero final norm amplifies harmless BF16 path differences. Gate
                    # the actual FP32 kernel VJP against an independent FP64 formula. For
                    # two independently rounded BF16 paths, each may contribute one unit
                    # roundoff (u=eps/2), so the pairwise condition-aware bound is 2u=eps.
                    # The optimized state fork is additionally held to a single-u bound on
                    # the complete captured dy*x path; replay has a repeated-prefix
                    # cotangent representation, so only its final formula is comparable.
                    assert (
                        d_diagnostics["dense_formula_vs_autograd_max_elementwise_conditioned"]
                        < torch.finfo(torch.float32).eps
                    ), label
                    assert (
                        d_diagnostics["shared_formula_vs_autograd_max_elementwise_conditioned"]
                        < torch.finfo(torch.float32).eps
                    ), label
                    assert (
                        d_diagnostics["dense_vs_shared_formula_max_elementwise_conditioned"]
                        < torch.finfo(torch.bfloat16).eps
                    ), label
                    path_error = d_diagnostics["dy_x_path"]["max_dy_x_path_conditioned"]
                    assert (
                        d_diagnostics["dense_vs_shared_formula_max_elementwise_conditioned"]
                        <= path_error + 16 * torch.finfo(torch.float64).eps
                    ), label
                    if "state-fork" in label:
                        assert path_error < torch.finfo(torch.bfloat16).eps / 2, label
                    assert stats["cosine"] > 0.995, f"{label}: {name}"
                    continue
                assert stats["relative_l2"] < 0.03, f"{label}: {name}"
                assert stats["cosine"] > 0.995, f"{label}: {name}"

        fork_metrics = None
        fork_parameter_grads = None
        uses_parallel_mamba_adapter = cp_size > 1 or tp_size > 1
        if uses_parallel_mamba_adapter and run_state_fork_candidate:
            (
                fork_outputs,
                fork_input_grad,
                fork_parameter_grads,
                fork_local_parameter_grads,
                fork_layer_outputs,
                fork_layer_grads,
                fork_attention_outputs,
                fork_attention_grads,
                fork_scan_snapshot,
                fork_d_formula,
                fork_d_absolute_sum,
                fork_local_d_formula,
                fork_local_d_absolute_sum,
            ) = run_shared(shared_input, _forward_mamba_layer_shared_prefix_cp_state_fork)
            fork_label = (
                f"state-fork-vs-{dense_oracle}-seed{data_seed}" f"-logical{layout.total_len}"
            )
            fork_metrics = parity_metrics(
                fork_label,
                fork_outputs,
                fork_input_grad,
                fork_parameter_grads,
                fork_local_parameter_grads,
                fork_layer_outputs,
                fork_layer_grads,
                fork_attention_outputs,
                fork_attention_grads,
                fork_scan_snapshot,
                fork_d_formula,
                fork_d_absolute_sum,
                fork_local_d_formula,
                fork_local_d_absolute_sum,
            )

        production_input = localize(global_packed)
        (
            production_outputs,
            production_input_grad,
            production_parameter_grads,
            production_local_parameter_grads,
            production_layer_outputs,
            production_layer_grads,
            production_attention_outputs,
            production_attention_grads,
            production_scan_snapshot,
            production_d_formula,
            production_d_absolute_sum,
            production_local_d_formula,
            production_local_d_absolute_sum,
        ) = run_shared(production_input)
        production_label = "tp1-cp1" if not uses_parallel_mamba_adapter else "state-fork-default"
        if dense_oracle != "separate" or not run_state_fork_candidate:
            production_label = (
                f"{production_label}-vs-{dense_oracle}-seed{data_seed}"
                f"-logical{layout.total_len}"
            )
        production_metrics = parity_metrics(
            production_label,
            production_outputs,
            production_input_grad,
            production_parameter_grads,
            production_local_parameter_grads,
            production_layer_outputs,
            production_layer_grads,
            production_attention_outputs,
            production_attention_grads,
            production_scan_snapshot,
            production_d_formula,
            production_d_absolute_sum,
            production_local_d_formula,
            production_local_d_absolute_sum,
        )

        fallback_metrics = None
        fallback_parameter_grads = None
        if uses_parallel_mamba_adapter and run_state_fork_candidate:
            fallback_input = localize(global_packed)
            (
                fallback_outputs,
                fallback_input_grad,
                fallback_parameter_grads,
                fallback_local_parameter_grads,
                fallback_layer_outputs,
                fallback_layer_grads,
                fallback_attention_outputs,
                fallback_attention_grads,
                fallback_scan_snapshot,
                fallback_d_formula,
                fallback_d_absolute_sum,
                fallback_local_d_formula,
                fallback_local_d_absolute_sum,
            ) = run_shared(fallback_input, _forward_mamba_layer_shared_prefix_cp_replay)
            fallback_label = (
                f"replay-fallback-vs-{dense_oracle}-seed{data_seed}" f"-logical{layout.total_len}"
            )
            fallback_metrics = parity_metrics(
                fallback_label,
                fallback_outputs,
                fallback_input_grad,
                fallback_parameter_grads,
                fallback_local_parameter_grads,
                fallback_layer_outputs,
                fallback_layer_grads,
                fallback_attention_outputs,
                fallback_attention_grads,
                fallback_scan_snapshot,
                fallback_d_formula,
                fallback_d_absolute_sum,
                fallback_local_d_formula,
                fallback_local_d_absolute_sum,
            )

        if fork_metrics is not None:
            outlier_names = {
                name
                for metrics in (fork_metrics, production_metrics, fallback_metrics)
                for name, stats in metrics["parameters"].items()
                if stats["relative_l2"] >= 0.03 or stats["cosine"] <= 0.995
            }
            default_fallback_parameter_stats = {
                name: parameter_stats(
                    production_parameter_grads[name], fallback_parameter_grads[name]
                )
                for name in sorted(outlier_names)
            }
            if cp_rank == 0 and tp_rank == 0:
                print(
                    f"TP{tp_size}/CP{cp_size} {pattern} state-fork default vs replay fallback "
                    f"parameter outliers: {default_fallback_parameter_stats}"
                )

        # The optimized state fork is the production default and always gates this suite.
        # Representative cases also retain the uninterrupted replay path as an explicit
        # correctness oracle/fallback under the same unchanged thresholds.
        assert_parity(production_label, production_metrics)
        if fallback_metrics is not None:
            assert_parity(fallback_label, fallback_metrics)
        if (
            fork_metrics is not None
            and os.environ.get("MCORE_REQUIRE_SHARED_PREFIX_STATE_FORK_PARITY") == "1"
        ):
            assert_parity(fork_label, fork_metrics)
    finally:
        Utils.destroy_model_parallel()
        if previous_qk_layer_scaling is None:
            os.environ.pop("NVTE_APPLY_QK_LAYER_SCALING", None)
        else:
            os.environ["NVTE_APPLY_QK_LAYER_SCALING"] = previous_qk_layer_scaling


@pytest.mark.timeout(300)
@_requires_world_size(1)
def test_cp1_mixed_hybrid_shared_prefix_all_parameter_baseline():
    _run_hybrid_shared_prefix_parity("M*", cp_size=1, hidden_size=256, query_heads=4, kv_heads=1)


@pytest.mark.timeout(600)
@_requires_world_size(2)
def test_tp2_cp1_sp_mixed_shared_prefix_matches_dense_forward_and_backward():
    _run_hybrid_shared_prefix_parity(
        "M*",
        tp_size=2,
        sequence_parallel=True,
        cp_size=1,
        hidden_size=256,
        query_heads=32,
        kv_heads=2,
        kv_channels=128,
        mamba_heads=64,
        mamba_groups=8,
        layout=SharedPrefixLayout(prefix_len=4, completion_lens=[3, 4]),
        dense_oracle="matched-batch",
    )


@pytest.mark.parametrize("cp_size", [pytest.param(1, id="cp1"), pytest.param(2, id="cp2")])
@pytest.mark.timeout(900)
def test_tp2_sp_hybrid_model_returns_full_cp_local_tp_vocab_shard(cp_size):
    pytest.importorskip("causal_conv1d")
    pytest.importorskip("flash_attn")
    pytest.importorskip("mamba_ssm")
    if not torch.cuda.is_bf16_supported():
        pytest.skip("TP/SP parity requires CUDA bf16 support")

    from megatron.core.models.hybrid.hybrid_layer_allocation import validate_segment_layers
    from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
    from megatron.core.models.hybrid.hybrid_model import HybridModel
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer import TransformerConfig

    tp_size = 2
    required_world_size = tp_size * cp_size
    if int(os.environ.get("WORLD_SIZE", "1")) != required_world_size:
        pytest.skip(f"TP2/CP{cp_size} requires torchrun --nproc_per_node={required_world_size}")
    hidden_size = 256
    vocab_size = 512
    pattern = "M*"
    layout = SharedPrefixLayout(prefix_len=4, completion_lens=[3, 4])
    physical_multiple = 2 * tp_size * cp_size
    physical_len = (
        (layout.total_len + physical_multiple - 1) // physical_multiple
    ) * physical_multiple

    Utils.initialize_model_parallel(
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=1,
        context_parallel_size=cp_size,
    )
    try:
        previous_qk_layer_scaling = os.environ.get("NVTE_APPLY_QK_LAYER_SCALING")
        os.environ["NVTE_APPLY_QK_LAYER_SCALING"] = "1"
        torch.manual_seed(20260827)
        model_parallel_cuda_manual_seed(20260827)
        config = TransformerConfig(
            hidden_size=hidden_size,
            num_layers=len(validate_segment_layers(pattern)),
            num_attention_heads=4,
            num_query_groups=1,
            tensor_model_parallel_size=tp_size,
            context_parallel_size=cp_size,
            sequence_parallel=True,
            use_cpu_initialization=True,
            bf16=True,
            params_dtype=torch.bfloat16,
            use_mamba_mem_eff_path=False,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            apply_query_key_layer_scaling=True,
        )
        process_groups = ProcessGroupCollection.use_mpu_process_groups(
            required_pgs=["tp", "pp", "cp", "embd", "dp_cp"]
        )
        model = HybridModel(
            config=config,
            hybrid_stack_spec=hybrid_stack_spec,
            vocab_size=vocab_size,
            max_sequence_length=physical_len,
            hybrid_layer_pattern=pattern,
            position_embedding_type="rope",
            pre_process=True,
            post_process=True,
            parallel_output=True,
            pg_collection=process_groups,
        ).cuda()
        cp_group = process_groups.cp
        cp_rank = cp_group.rank()

        def cp_localize(global_tensor):
            if cp_size == 1:
                return global_tensor
            indices = layout.cp_local_indices(
                global_tensor.shape[0], cp_size, cp_rank, global_tensor.device
            )
            return global_tensor.index_select(0, indices)

        global_tokens = torch.arange(physical_len, device="cuda") % vocab_size
        tokens = cp_localize(global_tokens).unsqueeze(0)
        position_ids = cp_localize(layout.padded_position_ids(physical_len, "cuda")).unsqueeze(0)
        dense_completion_logits = []
        with torch.no_grad():
            for completion_slice in layout.completion_slices():
                completion_len = completion_slice.stop - completion_slice.start
                branch_tokens = torch.cat(
                    [global_tokens[: layout.prefix_len], global_tokens[completion_slice]], dim=0
                )
                dense_len = 8
                branch_tokens = torch.cat(
                    [branch_tokens, branch_tokens.new_zeros(dense_len - branch_tokens.shape[0])]
                ).unsqueeze(0)
                dense_position_ids = torch.arange(dense_len, device="cuda")
                branch_tokens = cp_localize(branch_tokens[0]).unsqueeze(0)
                dense_position_ids = cp_localize(dense_position_ids).unsqueeze(0)
                dense_logits = model(branch_tokens, dense_position_ids, None)
                dense_logits = _gather_canonical(dense_logits.transpose(0, 1), cp_group).transpose(
                    0, 1
                )
                dense_completion_logits.append(
                    dense_logits[0, layout.prefix_len : layout.prefix_len + completion_len]
                )

        logits = model(tokens, position_ids, None, shared_prefix_layout=layout)

        assert logits.shape == (1, physical_len // cp_size, vocab_size // tp_size)
        assert logits.shape[-1] == model.output_layer.output_size_per_partition
        assert torch.isfinite(logits).all()
        global_shared_logits = _gather_canonical(
            logits.detach().transpose(0, 1), cp_group
        ).transpose(0, 1)
        for dense_logits, completion_slice in zip(
            dense_completion_logits, layout.completion_slices(), strict=True
        ):
            shared_logits = global_shared_logits[0, completion_slice]
            assert _relative_l2(dense_logits, shared_logits) < 0.05
            assert _cosine(dense_logits, shared_logits) > 0.99

        logits.float().square().mean().backward()
        assert model.output_layer.weight.grad is not None
        assert model.decoder.layers[0].mixer.in_proj.weight.grad is not None
    finally:
        Utils.destroy_model_parallel()
        if previous_qk_layer_scaling is None:
            os.environ.pop("NVTE_APPLY_QK_LAYER_SCALING", None)
        else:
            os.environ["NVTE_APPLY_QK_LAYER_SCALING"] = previous_qk_layer_scaling


@pytest.mark.parametrize("pattern", [pytest.param("M", id="mamba"), pytest.param("M*", id="mixed")])
@pytest.mark.timeout(300)
@_requires_world_size(2)
def test_cp2_hybrid_shared_prefix_matches_dense_branches_forward_and_backward(pattern):
    _run_hybrid_shared_prefix_parity(pattern, cp_size=2, hidden_size=256, query_heads=4, kv_heads=1)


@pytest.mark.timeout(900)
@_requires_world_size(4)
def test_tp2_cp2_sp_mixed_shared_prefix_matches_dense_forward_and_backward():
    _run_hybrid_shared_prefix_parity(
        "M*",
        tp_size=2,
        sequence_parallel=True,
        cp_size=2,
        hidden_size=256,
        query_heads=32,
        kv_heads=2,
        kv_channels=128,
        mamba_heads=64,
        mamba_groups=8,
        layout=SharedPrefixLayout(prefix_len=4, completion_lens=[3, 4]),
        dense_oracle="matched-batch",
    )


@pytest.mark.parametrize(
    "completion_lens", [pytest.param([8, 11], id="pad1"), pytest.param([8, 12], id="no-pad")]
)
@pytest.mark.parametrize("data_seed", [7, 11, 23], ids=lambda seed: f"seed-{seed}")
@pytest.mark.timeout(600)
@_requires_world_size(2)
def test_cp2_mixed_replay_matches_matched_batch_dense_oracle(completion_lens, data_seed):
    _run_hybrid_shared_prefix_parity(
        "M*",
        cp_size=2,
        hidden_size=256,
        query_heads=4,
        kv_heads=1,
        layout=SharedPrefixLayout(prefix_len=12, completion_lens=completion_lens),
        data_seed=data_seed,
        dense_oracle="matched-batch",
        run_state_fork_candidate=False,
    )


@pytest.mark.parametrize(
    "completion_lens", [pytest.param([8, 11], id="pad1"), pytest.param([8, 12], id="no-pad")]
)
@pytest.mark.parametrize("data_seed", [7, 11, 23], ids=lambda seed: f"seed-{seed}")
@pytest.mark.timeout(600)
@_requires_world_size(2)
def test_cp2_mixed_replay_matches_separate_branch_dense_oracle_matrix(completion_lens, data_seed):
    _run_hybrid_shared_prefix_parity(
        "M*",
        cp_size=2,
        hidden_size=256,
        query_heads=4,
        kv_heads=1,
        layout=SharedPrefixLayout(prefix_len=12, completion_lens=completion_lens),
        data_seed=data_seed,
        dense_oracle="separate",
        run_state_fork_candidate=False,
    )


@pytest.mark.parametrize("pattern", [pytest.param("M", id="mamba"), pytest.param("M*", id="mixed")])
@pytest.mark.timeout(600)
@_requires_world_size(4)
def test_cp4_nano_geometry_hybrid_shared_prefix_matches_dense_branches_forward_and_backward(
    pattern,
):
    _run_hybrid_shared_prefix_parity(
        pattern,
        cp_size=4,
        hidden_size=256,
        query_heads=32,
        kv_heads=2,
        kv_channels=128,
        mamba_heads=64,
        mamba_groups=8,
    )


@pytest.mark.parametrize(
    "completion_lens", [pytest.param([8, 11], id="pad1"), pytest.param([8, 12], id="no-pad")]
)
@pytest.mark.parametrize("data_seed", [7, 11, 23], ids=lambda seed: f"seed-{seed}")
@pytest.mark.timeout(900)
@_requires_world_size(4)
def test_cp4_nano_mixed_replay_matches_matched_batch_dense_oracle(completion_lens, data_seed):
    _run_hybrid_shared_prefix_parity(
        "M*",
        cp_size=4,
        hidden_size=256,
        query_heads=32,
        kv_heads=2,
        kv_channels=128,
        mamba_heads=64,
        mamba_groups=8,
        layout=SharedPrefixLayout(prefix_len=12, completion_lens=completion_lens),
        data_seed=data_seed,
        dense_oracle="matched-batch",
        run_state_fork_candidate=True,
    )
