# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Explicit shared-prefix adapter for a HybridStack.

This module deliberately does not alter :class:`HybridStack`'s normal ``forward`` path. Call
``forward_hybrid_stack_shared_prefix`` with one packed star ``[P, C_1, ..., C_G]`` to opt in.
Mamba layers scan the prefix once and fork differentiable convolution/SSM state into the completion
branches, including CP>1 through sequence-to-head collectives. An exact uninterrupted-replay path
is retained as an explicit parity oracle and fallback. Attention uses the exact-backward fused forest
kernel.
Unsupported topology or model features fail before executing a partial forward.

The implementation is the narrow production slice of the shared-prefix work developed in
Megatron-RL/Megatron-LM commit ``0bf30804f`` plus the fused kernel port ``5b7173f7``. CP1 and CP>1
are advertised as distinct capabilities so integrations can negotiate against the validated
topology. CP>1 uses the model's standard zigzag sequence shards and sequence-to-head all-to-alls.
Validated production capabilities cover TP1 and TP>1 with sequence parallelism, explicit physical
padding, MoE expert-bias accounting, and full uniform activation recomputation. Topology and feature
tokens remain distinct so integrations can require the exact supported conjunction.
"""

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from megatron.core import tensor_parallel
from megatron.core.ssm.mamba_layer import MambaLayer
from megatron.core.ssm.mamba_mixer import (
    MAMBA_HAS_STATE_DTYPE,
    MambaMixer,
    causal_conv1d_fn,
    mamba_chunk_scan_combined,
)
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.transformer_layer import TransformerLayer
from megatron.core.typed_torch import apply_module
from megatron.core.utils import make_viewless_tensor


@dataclass(frozen=True)
class SharedPrefixLayout:
    """One exact-prompt star packed as ``[prefix, completion_1, ..., completion_G]``."""

    prefix_len: int
    completion_lens: Sequence[int]
    logical_completion_lens: Sequence[int] | None = None
    padding_multiple: int | None = None

    def __post_init__(self) -> None:
        prefix_len = int(self.prefix_len)
        completion_lens = tuple(int(length) for length in self.completion_lens)
        if prefix_len < 1:
            raise ValueError("shared-prefix layout requires a non-empty prefix")
        if not completion_lens or any(length < 1 for length in completion_lens):
            raise ValueError("shared-prefix layout requires one or more non-empty completions")
        object.__setattr__(self, "prefix_len", prefix_len)
        object.__setattr__(self, "completion_lens", completion_lens)
        if self.logical_completion_lens is not None:
            logical_completion_lens = tuple(int(length) for length in self.logical_completion_lens)
            if len(logical_completion_lens) != len(completion_lens) or any(
                logical < 1 or logical > physical
                for logical, physical in zip(logical_completion_lens, completion_lens, strict=True)
            ):
                raise ValueError(
                    "logical completion lengths must be positive, match the physical "
                    "branch count, and not exceed physical completion lengths"
                )
            object.__setattr__(self, "logical_completion_lens", logical_completion_lens)
        if (self.logical_completion_lens is None) != (self.padding_multiple is None):
            raise ValueError(
                "logical completion lengths and padding_multiple must be provided together"
            )
        if self.padding_multiple is not None:
            if isinstance(self.padding_multiple, bool) or not isinstance(
                self.padding_multiple, int
            ):
                raise ValueError("shared-prefix padding_multiple must be an integer")
            if self.padding_multiple < 1:
                raise ValueError("shared-prefix padding_multiple must be positive")

    @property
    def total_len(self) -> int:
        return self.prefix_len + sum(self.completion_lens)

    @property
    def forest(self) -> list[tuple[int, int, list[int]]]:
        return [(0, self.prefix_len, list(self.completion_lens))]

    def completion_slices(self) -> tuple[slice, ...]:
        slices = []
        start = self.prefix_len
        for length in self.completion_lens:
            slices.append(slice(start, start + length))
            start += length
        return tuple(slices)

    def position_ids(self, device: torch.device | str) -> Tensor:
        """Prefix-continued RoPE positions for the packed star."""
        pieces = [torch.arange(self.prefix_len, device=device, dtype=torch.long)]
        pieces.extend(
            torch.arange(self.prefix_len, self.prefix_len + length, device=device, dtype=torch.long)
            for length in self.completion_lens
        )
        return torch.cat(pieces)

    def padded_position_ids(self, physical_len: int, device: torch.device | str) -> Tensor:
        """Return global star positions plus inert positions for trailing CP padding."""
        physical_len = int(physical_len)
        if physical_len < self.total_len:
            raise ValueError(
                f"physical length {physical_len} is shorter than layout {self.total_len}"
            )
        positions = self.position_ids(device)
        if physical_len == self.total_len:
            return positions
        return torch.cat(
            [positions, torch.zeros(physical_len - self.total_len, device=device, dtype=torch.long)]
        )

    def padded_token_multiplicities(self, physical_len: int, device: torch.device | str) -> Tensor:
        """Dense-baseline multiplicity for each physical shared-prefix token.

        Prompt tokens occur once per completion in a conventional rollout batch.
        Branch tokens, including ordinary per-sequence padding, have unit
        multiplicity. Trailing topology-only padding is inert.
        """
        physical_len = int(physical_len)
        if physical_len < self.total_len:
            raise ValueError(
                f"physical length {physical_len} is shorter than layout {self.total_len}"
            )
        multiplicities = torch.cat(
            [
                torch.full(
                    (self.prefix_len,),
                    len(self.completion_lens),
                    device=device,
                    dtype=torch.float32,
                ),
                torch.ones(sum(self.completion_lens), device=device, dtype=torch.float32),
            ]
        )
        if physical_len == self.total_len:
            return multiplicities
        return torch.cat(
            [
                multiplicities,
                torch.zeros(physical_len - self.total_len, device=device, dtype=torch.float32),
            ]
        )

    @staticmethod
    def cp_local_indices(
        physical_len: int, cp_size: int, cp_rank: int, device: torch.device | str
    ) -> Tensor:
        """Global indices owned by one rank under standard two-chunk CP zigzag."""
        physical_len, cp_size, cp_rank = int(physical_len), int(cp_size), int(cp_rank)
        if cp_size < 1 or not 0 <= cp_rank < cp_size:
            raise ValueError(f"invalid CP geometry: {cp_size=}, {cp_rank=}")
        if physical_len % (2 * cp_size):
            raise ValueError(
                f"physical length {physical_len} must be divisible by 2 * CP size {cp_size}"
            )
        chunk = physical_len // (2 * cp_size)
        front = torch.arange(cp_rank * chunk, (cp_rank + 1) * chunk, device=device)
        back_chunk = 2 * cp_size - cp_rank - 1
        back = torch.arange(back_chunk * chunk, (back_chunk + 1) * chunk, device=device)
        return torch.cat([front, back]).to(torch.long)


def _mamba_state_dtype_kwargs(mixer: MambaMixer) -> dict:
    if not MAMBA_HAS_STATE_DTYPE:
        return {}
    return {"state_dtype": mixer.mamba_training_ssm_states_dtype}


def _validate_mamba_fork(mixer: MambaMixer) -> None:
    tp_size = mixer.pg_collection.tp.size()
    if tp_size > 1 and not mixer.config.sequence_parallel:
        raise NotImplementedError("shared-prefix Mamba TP>1 requires sequence parallelism")
    if tp_size == 1 and mixer.config.sequence_parallel:
        raise NotImplementedError("shared-prefix Mamba sequence parallelism requires TP>1")
    if not mixer.rmsnorm:
        raise NotImplementedError("shared-prefix Mamba state forking requires gated RMSNorm")
    if causal_conv1d_fn is None or mamba_chunk_scan_combined is None:
        raise RuntimeError("shared-prefix Mamba state forking requires causal-conv1d and mamba-ssm")


def _prefix_conv_context(xbc: Tensor, width: int) -> Tensor:
    """Last ``width`` pre-convolution columns, including the causal left-zero padding."""
    if width == 0:
        return xbc[:, :, :0]
    padded = F.pad(xbc, (max(0, width - xbc.shape[-1]), 0))
    return padded[:, :, -width:].clone()


def _scan_mamba_projected_segment(
    mixer: MambaMixer,
    projected: Tensor,
    *,
    conv_context: Tensor | None = None,
    ssm_initial_state: Tensor | None = None,
    capture_state: bool = False,
) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None]:
    """Scan already CP-canonical, head-sharded Mamba projections for one segment batch."""
    cp = mixer.cp
    if projected.ndim != 3:
        raise ValueError(
            "shared-prefix projected Mamba segments must have shape [sequence,batch,d]"
        )
    num_groups = cp.ngroups_local_tpcp
    num_heads = cp.nheads_local_tpcp
    d_inner = cp.d_inner_local_tpcp
    branch_count = projected.shape[1]

    projected = rearrange(projected, "l b d -> b l d").contiguous()
    z, xbc, dt = torch.split(
        projected, [d_inner, d_inner + 2 * num_groups * mixer.d_state, num_heads], dim=-1
    )
    A = -torch.exp(cp.get_A_log().float())

    xbc = rearrange(xbc, "b l d -> b d l").contiguous()
    next_conv_context = _prefix_conv_context(xbc, mixer.d_conv - 1) if capture_state else None
    if conv_context is not None:
        if (
            conv_context.ndim != 3
            or conv_context.shape[1] != xbc.shape[1]
            or conv_context.shape[2] != mixer.d_conv - 1
        ):
            raise ValueError("prefix convolution state is incompatible with the branch")
        if conv_context.shape[0] not in (1, branch_count):
            raise ValueError("prefix convolution state batch is incompatible with the branch")
        repeated_context = conv_context.to(xbc.dtype).expand(branch_count, -1, -1)
        conv_input = torch.cat([repeated_context, xbc], dim=-1)
        conv_output = causal_conv1d_fn(
            conv_input,
            rearrange(cp.get_conv1d_weight(), "d 1 w -> d w"),
            cp.get_conv1d_bias(),
            activation=mixer.activation,
        )[:, :, repeated_context.shape[-1] :]
    else:
        conv_output = causal_conv1d_fn(
            xbc,
            rearrange(cp.get_conv1d_weight(), "d 1 w -> d w"),
            cp.get_conv1d_bias(),
            activation=mixer.activation,
        )
    xbc = rearrange(conv_output, "b d l -> b l d").contiguous()

    x, B, C = torch.split(
        xbc, [d_inner, num_groups * mixer.d_state, num_groups * mixer.d_state], dim=-1
    )
    x = rearrange(x, "b l (h p) -> b l h p", p=mixer.headdim).contiguous()
    B = rearrange(B, "b l (g n) -> b l g n", n=mixer.d_state).contiguous()
    C = rearrange(C, "b l (g n) -> b l g n", n=mixer.d_state).contiguous()
    z = rearrange(z, "b l (h p) -> b l h p", p=mixer.headdim).contiguous()

    initial_states = ssm_initial_state
    if initial_states is not None:
        if initial_states.shape[0] not in (1, branch_count):
            raise ValueError("prefix SSM state batch is incompatible with the branch")
        initial_states = initial_states.expand(branch_count, *initial_states.shape[1:]).contiguous()
    scan = mamba_chunk_scan_combined(
        x,
        dt.contiguous(),
        A,
        B,
        C,
        mixer.chunk_size,
        D=(
            rearrange(cp.get_D().float(), "(h p) -> h p", p=mixer.headdim)
            if mixer.D_has_hdim
            else cp.get_D()
        ),
        z=None,
        dt_bias=cp.get_dt_bias().float(),
        dt_softplus=True,
        initial_states=initial_states,
        return_final_states=capture_state,
        **_mamba_state_dtype_kwargs(mixer),
    )
    if capture_state:
        y, final_state = scan
    else:
        y, final_state = scan, None
    y = rearrange(y, "b l h p -> l b (h p)").contiguous()
    z = rearrange(z, "b l h p -> l b (h p)").contiguous()
    return y, z, next_conv_context, final_state


def _fork_mamba_segment(
    mixer: MambaMixer,
    hidden_states: Tensor,
    *,
    conv_context: Tensor | None = None,
    ssm_initial_state: Tensor | None = None,
    capture_state: bool = False,
) -> tuple[Tensor, Tensor | None, Tensor | None, Tensor | None]:
    """Scan one segment, optionally capturing a differentiable Mamba end state."""
    _validate_mamba_fork(mixer)
    if mixer.pg_collection.tp.size() != 1:
        raise RuntimeError(
            "TP sequence-sharded Mamba segments require the collective parallel adapter path"
        )
    if mixer.cp.cp_size != 1:
        raise RuntimeError("CP-sharded Mamba segments require the collective CP adapter path")
    if hidden_states.ndim != 3 or hidden_states.shape[1] != 1:
        raise ValueError("shared-prefix Mamba segments must have shape [sequence, 1, hidden]")

    cp = mixer.cp
    num_groups = cp.ngroups_local_tpcp
    num_heads = cp.nheads_local_tpcp
    d_inner = cp.d_inner_local_tpcp

    projected, _ = mixer.in_proj(hidden_states)
    projected = cp.pre_conv_ssm(projected)
    projected = rearrange(projected, "l b d -> b l d").contiguous()
    z, xbc, dt = torch.split(
        projected, [d_inner, d_inner + 2 * num_groups * mixer.d_state, num_heads], dim=-1
    )
    A = -torch.exp(cp.get_A_log().float())

    xbc = rearrange(xbc, "b l d -> b d l").contiguous()
    next_conv_context = _prefix_conv_context(xbc, mixer.d_conv - 1) if capture_state else None
    if conv_context is not None:
        if conv_context.shape[:2] != xbc.shape[:2]:
            raise ValueError("prefix convolution state is incompatible with the branch")
        conv_input = torch.cat([conv_context.to(xbc.dtype), xbc], dim=-1)
        conv_output = causal_conv1d_fn(
            conv_input,
            rearrange(cp.get_conv1d_weight(), "d 1 w -> d w"),
            cp.get_conv1d_bias(),
            activation=mixer.activation,
        )[:, :, conv_context.shape[-1] :]
    else:
        conv_output = causal_conv1d_fn(
            xbc,
            rearrange(cp.get_conv1d_weight(), "d 1 w -> d w"),
            cp.get_conv1d_bias(),
            activation=mixer.activation,
        )
    xbc = rearrange(conv_output, "b d l -> b l d").contiguous()

    x, B, C = torch.split(
        xbc, [d_inner, num_groups * mixer.d_state, num_groups * mixer.d_state], dim=-1
    )
    x = rearrange(x, "b l (h p) -> b l h p", p=mixer.headdim).contiguous()
    B = rearrange(B, "b l (g n) -> b l g n", n=mixer.d_state).contiguous()
    C = rearrange(C, "b l (g n) -> b l g n", n=mixer.d_state).contiguous()
    z = rearrange(z, "b l (h p) -> b l h p", p=mixer.headdim).contiguous()

    scan = mamba_chunk_scan_combined(
        x,
        dt.contiguous(),
        A,
        B,
        C,
        mixer.chunk_size,
        D=(
            rearrange(cp.get_D().float(), "(h p) -> h p", p=mixer.headdim)
            if mixer.D_has_hdim
            else cp.get_D()
        ),
        z=None,
        dt_bias=cp.get_dt_bias().float(),
        dt_softplus=True,
        initial_states=ssm_initial_state,
        return_final_states=capture_state,
        **_mamba_state_dtype_kwargs(mixer),
    )
    if capture_state:
        y, final_state = scan
    else:
        y, final_state = scan, None

    y = rearrange(y, "b l h p -> l b (h p)").contiguous()
    y = cp.post_conv_ssm(y)
    z = rearrange(z, "b l h p -> l b (h p)").contiguous()
    z = cp.post_conv_ssm(z)
    y = mixer.norm(y, z)
    output, output_bias = mixer.out_proj(y)
    return output, output_bias, next_conv_context, final_state


def _fork_mamba_branches(
    mixer: MambaMixer, branches: Tensor, *, conv_context: Tensor, ssm_initial_state: Tensor
) -> tuple[Tensor, Tensor | None]:
    """Scan a right-padded batch of completion branches from one prefix state."""
    _validate_mamba_fork(mixer)
    if mixer.pg_collection.tp.size() != 1:
        raise RuntimeError(
            "TP sequence-sharded Mamba branches require the collective parallel adapter path"
        )
    if mixer.cp.cp_size != 1:
        raise RuntimeError("CP-sharded Mamba branches require the collective CP adapter path")
    if branches.ndim != 3:
        raise ValueError(
            "shared-prefix Mamba branches must have shape [sequence, branches, hidden]"
        )

    cp = mixer.cp
    num_groups = cp.ngroups_local_tpcp
    num_heads = cp.nheads_local_tpcp
    d_inner = cp.d_inner_local_tpcp
    branch_count = branches.shape[1]

    projected, _ = mixer.in_proj(branches)
    projected = cp.pre_conv_ssm(projected)
    projected = rearrange(projected, "l b d -> b l d").contiguous()
    z, xbc, dt = torch.split(
        projected, [d_inner, d_inner + 2 * num_groups * mixer.d_state, num_heads], dim=-1
    )
    A = -torch.exp(cp.get_A_log().float())

    xbc = rearrange(xbc, "b l d -> b d l").contiguous()
    repeated_context = conv_context.to(xbc.dtype).expand(branch_count, -1, -1)
    conv_input = torch.cat([repeated_context, xbc], dim=-1)
    conv_output = causal_conv1d_fn(
        conv_input,
        rearrange(cp.get_conv1d_weight(), "d 1 w -> d w"),
        cp.get_conv1d_bias(),
        activation=mixer.activation,
    )[:, :, repeated_context.shape[-1] :]
    xbc = rearrange(conv_output, "b d l -> b l d").contiguous()

    x, B, C = torch.split(
        xbc, [d_inner, num_groups * mixer.d_state, num_groups * mixer.d_state], dim=-1
    )
    x = rearrange(x, "b l (h p) -> b l h p", p=mixer.headdim).contiguous()
    B = rearrange(B, "b l (g n) -> b l g n", n=mixer.d_state).contiguous()
    C = rearrange(C, "b l (g n) -> b l g n", n=mixer.d_state).contiguous()
    z = rearrange(z, "b l (h p) -> b l h p", p=mixer.headdim).contiguous()
    initial_states = ssm_initial_state.expand(
        branch_count, *ssm_initial_state.shape[1:]
    ).contiguous()

    y = mamba_chunk_scan_combined(
        x,
        dt.contiguous(),
        A,
        B,
        C,
        mixer.chunk_size,
        D=(
            rearrange(cp.get_D().float(), "(h p) -> h p", p=mixer.headdim)
            if mixer.D_has_hdim
            else cp.get_D()
        ),
        z=None,
        dt_bias=cp.get_dt_bias().float(),
        dt_softplus=True,
        initial_states=initial_states,
        return_final_states=False,
        **_mamba_state_dtype_kwargs(mixer),
    )
    y = rearrange(y, "b l h p -> l b (h p)").contiguous()
    y = cp.post_conv_ssm(y)
    z = rearrange(z, "b l h p -> l b (h p)").contiguous()
    z = cp.post_conv_ssm(z)
    y = mixer.norm(y, z)
    return mixer.out_proj(y)


def _forward_mamba_layer_shared_prefix(
    layer: MambaLayer, hidden_states: Tensor, layout: SharedPrefixLayout
) -> Tensor:
    residual = hidden_states.float() if layer.config.fp32_residual_connection else hidden_states
    normalized = apply_module(layer.norm)(hidden_states.to(dtype=layer.config.params_dtype))

    prefix_output, output_bias, conv_context, final_state = _fork_mamba_segment(
        layer.mixer, normalized[: layout.prefix_len], capture_state=True
    )
    physical_completion_lens = list(layout.completion_lens)
    physical_completion_lens[-1] += hidden_states.shape[0] - layout.total_len
    max_completion_len = max(physical_completion_lens)
    branches = normalized.new_zeros(
        max_completion_len, len(physical_completion_lens), normalized.shape[-1]
    )
    start = layout.prefix_len
    for branch_index, completion_len in enumerate(physical_completion_lens):
        branches[:completion_len, branch_index] = normalized[start : start + completion_len, 0]
        start += completion_len
    if start != hidden_states.shape[0]:
        raise RuntimeError("shared-prefix branch spans do not cover physical sequence")
    branch_output, _ = _fork_mamba_branches(
        layer.mixer, branches, conv_context=conv_context, ssm_initial_state=final_state
    )
    packed_output = torch.cat(
        [prefix_output]
        + [
            branch_output[:length, index : index + 1]
            for index, length in enumerate(physical_completion_lens)
        ],
        dim=0,
    )

    with layer.bias_dropout_add_exec_handler():
        return layer.mamba_bda(training=layer.training, fused=layer.config.bias_dropout_fusion)(
            (packed_output, output_bias), residual, layer.hidden_dropout
        )


def _forward_mamba_layer_shared_prefix_cp_impl(
    layer: MambaLayer, hidden_states: Tensor, layout: SharedPrefixLayout, *, replay_prefix: bool
) -> Tensor:
    """Run a TP/SP/CP Mamba star with state-fork or uninterrupted-replay prefix handling.

    The input projection first gathers sequence-parallel TP shards, then the CP adapter converts
    local zigzag sequence shards into one canonical global sequence with TP/CP-local channels.
    State-fork scans the prefix once and expands its differentiable state; replay is retained as a
    parity baseline and repeats the projected prefix per branch. The packed result returns through
    the inverse CP transform and TP output-projection reduce-scatter.
    """
    mixer = layer.mixer
    _validate_mamba_fork(mixer)
    cp = mixer.cp

    residual = hidden_states.float() if layer.config.fp32_residual_connection else hidden_states
    normalized = apply_module(layer.norm)(hidden_states.to(dtype=layer.config.params_dtype))
    projected, _ = mixer.in_proj(normalized)
    # Input SP gather: [physical/(TP*C), 1, hidden] -> [physical/C, 1, projection/TP].
    # CP A2A: [physical/C, 1, projection/TP] -> [physical, 1, projection/(TP*C)].
    projected = cp.pre_conv_ssm(projected)
    physical_len = projected.shape[0]
    if physical_len < layout.total_len:
        raise ValueError(
            f"parallel physical length {physical_len} is shorter than shared layout "
            f"{layout.total_len}"
        )

    physical_completion_lens = list(layout.completion_lens)
    # The topology validator owns physical alignment. Padding is placed after the final completion;
    # scanning it as that branch's causal tail cannot affect any real-token output.
    physical_completion_lens[-1] += physical_len - layout.total_len
    branch_prefix_len = layout.prefix_len if replay_prefix else 0
    max_branch_len = branch_prefix_len + max(physical_completion_lens)
    branches = projected.new_zeros(
        max_branch_len, len(physical_completion_lens), projected.shape[-1]
    )
    start = layout.prefix_len
    for branch_index, completion_len in enumerate(physical_completion_lens):
        if replay_prefix:
            branches[: layout.prefix_len, branch_index] = projected[: layout.prefix_len, 0]
        branches[branch_prefix_len : branch_prefix_len + completion_len, branch_index] = projected[
            start : start + completion_len, 0
        ]
        start += completion_len
    if start != physical_len:
        raise RuntimeError("shared-prefix CP branch spans do not cover physical sequence")

    if replay_prefix:
        branch_y, branch_z, _, _ = _scan_mamba_projected_segment(mixer, branches)
        prefix_y = branch_y[: layout.prefix_len, :1]
        prefix_z = branch_z[: layout.prefix_len, :1]
    else:
        prefix_y, prefix_z, conv_context, final_state = _scan_mamba_projected_segment(
            mixer, projected[: layout.prefix_len], capture_state=True
        )
        if conv_context is None or final_state is None:
            raise RuntimeError("shared-prefix CP Mamba scan did not return a differentiable state")
        branch_y, branch_z, _, _ = _scan_mamba_projected_segment(
            mixer, branches, conv_context=conv_context, ssm_initial_state=final_state
        )
    packed_y = torch.cat(
        [prefix_y]
        + [
            branch_y[branch_prefix_len : branch_prefix_len + length, index : index + 1]
            for index, length in enumerate(physical_completion_lens)
        ],
        dim=0,
    )
    packed_z = torch.cat(
        [prefix_z]
        + [
            branch_z[branch_prefix_len : branch_prefix_len + length, index : index + 1]
            for index, length in enumerate(physical_completion_lens)
        ],
        dim=0,
    )
    # Canonical channel shards -> local zigzag sequence shards with full channels.
    packed_y = cp.post_conv_ssm(packed_y)
    packed_z = cp.post_conv_ssm(packed_z)
    packed_y = mixer.norm(packed_y, packed_z)
    packed_output, output_bias = mixer.out_proj(packed_y)

    with layer.bias_dropout_add_exec_handler():
        return layer.mamba_bda(training=layer.training, fused=layer.config.bias_dropout_fusion)(
            (packed_output, output_bias), residual, layer.hidden_dropout
        )


def _forward_mamba_layer_shared_prefix_cp_state_fork(
    layer: MambaLayer, hidden_states: Tensor, layout: SharedPrefixLayout
) -> Tensor:
    """Optimized CP Mamba candidate: scan the prefix once and fork differentiable state."""
    return _forward_mamba_layer_shared_prefix_cp_impl(
        layer, hidden_states, layout, replay_prefix=False
    )


def _forward_mamba_layer_shared_prefix_cp_replay(
    layer: MambaLayer, hidden_states: Tensor, layout: SharedPrefixLayout
) -> Tensor:
    """Exact CP Mamba fallback: replay each prefix without an explicit state boundary."""
    return _forward_mamba_layer_shared_prefix_cp_impl(
        layer, hidden_states, layout, replay_prefix=True
    )


def _forward_mamba_layer_shared_prefix_cp(
    layer: MambaLayer, hidden_states: Tensor, layout: SharedPrefixLayout
) -> Tensor:
    """Production CP Mamba path using the optimized differentiable state fork."""
    return _forward_mamba_layer_shared_prefix_cp_state_fork(layer, hidden_states, layout)


def _has_nonzero_config_value(value) -> bool:
    """Return whether a scalar or per-layer configuration contains a nonzero value."""
    if value is None:
        return False
    if isinstance(value, (list, tuple)):
        return any(float(item) != 0.0 for item in value)
    return float(value) != 0.0


def _validate_shared_prefix_physical_length(
    layout: SharedPrefixLayout,
    physical_len: int,
    *,
    tp_size: int,
    cp_size: int,
    sequence_parallel: bool,
) -> None:
    """Validate the global star length against its negotiated topology/padding contract."""
    physical_len = int(physical_len)
    if tp_size > 1:
        topology_multiple = 2 * tp_size * cp_size
    elif cp_size > 1:
        topology_multiple = 2 * cp_size
    else:
        topology_multiple = 1
    padding = physical_len - layout.total_len

    if layout.padding_multiple is not None:
        padding_multiple = layout.padding_multiple
        if padding_multiple % topology_multiple:
            raise ValueError(
                "shared-prefix padding_multiple must be divisible by the topology quantum: "
                f"M={padding_multiple}, Q={topology_multiple}, TP={tp_size}, CP={cp_size}"
            )
        for branch, (physical_completion, logical_completion) in enumerate(
            zip(layout.completion_lens, layout.logical_completion_lens, strict=True)
        ):
            if (layout.prefix_len + physical_completion) % padding_multiple or not (
                0 <= physical_completion - logical_completion < padding_multiple
            ):
                raise ValueError(
                    "shared-prefix physical completion span must use the minimal per-branch "
                    "padding to padding_multiple: "
                    f"branch={branch}, prefix={layout.prefix_len}, "
                    f"logical={logical_completion}, physical={physical_completion}, "
                    f"M={padding_multiple}"
                )
        if physical_len % padding_multiple:
            raise ValueError(
                "shared-prefix physical length must be divisible by padding_multiple: "
                f"physical={physical_len}, M={padding_multiple}"
            )
        if not 0 <= padding < padding_multiple:
            raise ValueError(
                "shared-prefix input must use the minimal trailing pad to padding_multiple: "
                f"physical={physical_len}, layout={layout.total_len}, M={padding_multiple}"
            )
        return

    if tp_size > 1:
        if physical_len % topology_multiple:
            raise ValueError(
                "shared-prefix TP/SP physical length must be divisible by 2 * tensor parallel "
                "size * context parallel size"
            )
        if not 0 <= padding < topology_multiple:
            raise ValueError(
                "shared-prefix TP/SP input must use the minimal trailing pad to a 2*TP*CP "
                f"multiple: physical={physical_len}, layout={layout.total_len}, "
                f"TP={tp_size}, CP={cp_size}"
            )
    elif cp_size == 1:
        if physical_len != layout.total_len:
            raise ValueError(
                f"packed sequence length {physical_len} does not match layout {layout.total_len}"
            )
    else:
        if physical_len % topology_multiple:
            raise ValueError(
                "shared-prefix CP physical length must be divisible by 2 * context parallel size"
            )
        if not 0 <= padding < topology_multiple:
            raise ValueError(
                "shared-prefix CP input must use the minimal trailing pad to a 2*CP multiple: "
                f"physical={physical_len}, layout={layout.total_len}, CP={cp_size}"
            )


def _validate_hybrid_stack(stack, hidden_states: Tensor, layout: SharedPrefixLayout) -> None:
    tp_size = stack.tp_group.size()
    sequence_parallel = bool(stack.config.sequence_parallel)
    if tp_size > 1 and not sequence_parallel:
        raise NotImplementedError("shared-prefix Hybrid TP>1 requires sequence parallelism")
    if tp_size == 1 and sequence_parallel:
        raise NotImplementedError("shared-prefix Hybrid sequence parallelism requires TP>1")
    if stack.config.tensor_model_parallel_size != tp_size:
        raise RuntimeError(
            "shared-prefix Hybrid tensor-parallel config does not match its process group"
        )
    if stack.pg_collection.tp.size() != tp_size:
        raise RuntimeError("shared-prefix Hybrid tensor-parallel groups disagree on their size")
    if stack.pp_group.size() != 1:
        raise NotImplementedError("shared-prefix Hybrid adapter currently supports PP1 only")
    cp_group = stack.pg_collection.cp
    cp_size = cp_group.size()
    if stack.config.context_parallel_size != cp_size:
        raise RuntimeError(
            "shared-prefix Hybrid context-parallel config does not match its process group"
        )
    if stack.config.recompute_granularity == "full":
        if stack.config.recompute_method != "uniform":
            raise NotImplementedError(
                "shared-prefix Hybrid full recomputation currently supports only the uniform method"
            )
        if not isinstance(stack.config.recompute_num_layers, int) or (
            stack.config.recompute_num_layers < 1
        ):
            raise ValueError(
                "shared-prefix Hybrid uniform recomputation requires recompute_num_layers >= 1"
            )
    if stack.config.fine_grained_activation_offloading:
        raise NotImplementedError(
            "shared-prefix Hybrid adapter does not support fine-grained activation offloading"
        )
    if stack.config.cuda_graph_impl != "none":
        raise NotImplementedError("shared-prefix Hybrid adapter does not support CUDA graphs")
    if stack.config.fp8 or stack.config.fp4:
        raise NotImplementedError("shared-prefix Hybrid adapter currently supports fp16/bf16 only")
    if hidden_states.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("fused shared-prefix attention requires fp16 or bf16 hidden states")
    if hidden_states.ndim != 3 or hidden_states.shape[1] != 1:
        raise ValueError(
            "shared-prefix Hybrid input must have shape [sequence/(TP*CP), 1, hidden] "
            "when sequence parallelism is enabled"
        )
    sequence_shards = tp_size if sequence_parallel else 1
    physical_len = hidden_states.shape[0] * cp_size * sequence_shards
    _validate_shared_prefix_physical_length(
        layout, physical_len, tp_size=tp_size, cp_size=cp_size, sequence_parallel=sequence_parallel
    )
    if stack.config.attention_dropout != 0.0 or stack.config.hidden_dropout != 0.0:
        raise NotImplementedError("shared-prefix Hybrid adapter currently requires zero dropout")
    if stack.config.window_size not in (None, (-1, -1)):
        raise NotImplementedError(
            "shared-prefix Hybrid adapter does not support sliding-window attention"
        )
    if stack.config.softmax_type != "vanilla":
        raise NotImplementedError("shared-prefix fused attention supports only vanilla softmax")

    num_moe_experts = stack.config.num_moe_experts
    if num_moe_experts is not None and num_moe_experts > 0:
        if stack.config.moe_router_force_load_balancing:
            raise NotImplementedError(
                "shared-prefix Hybrid adapter does not support randomized forced MoE routing"
            )
        if stack.config.moe_router_force_biased is not None:
            raise NotImplementedError(
                "shared-prefix Hybrid adapter does not support randomized forced MoE router bias"
            )
        load_balancing = stack.config.moe_router_load_balancing_type
        load_balancing_types = (
            [load_balancing] if isinstance(load_balancing, str) else load_balancing
        )
        if any(item != "none" for item in load_balancing_types):
            raise NotImplementedError(
                "shared-prefix Hybrid adapter requires MoE router load balancing type 'none'"
            )
        if _has_nonzero_config_value(stack.config.moe_aux_loss_coeff):
            raise NotImplementedError(
                "shared-prefix Hybrid adapter does not support MoE auxiliary router loss"
            )
        if _has_nonzero_config_value(stack.config.moe_z_loss_coeff):
            raise NotImplementedError(
                "shared-prefix Hybrid adapter does not support MoE router z-loss"
            )
        if _has_nonzero_config_value(stack.config.moe_input_jitter_eps):
            raise NotImplementedError(
                "shared-prefix Hybrid adapter does not support MoE input jitter"
            )
        if stack.config.moe_router_enable_expert_bias and layout.logical_completion_lens is None:
            raise NotImplementedError(
                "shared-prefix MoE expert-bias accounting requires explicit physical "
                "branch padding and logical completion lengths"
            )
        if stack.config.moe_expert_capacity_factor is not None:
            raise NotImplementedError(
                "shared-prefix Hybrid adapter does not support MoE expert capacity or token dropping"
            )
        if getattr(stack.config, "mlp_chunks_for_training", 1) != 1:
            raise NotImplementedError(
                "shared-prefix Hybrid MoE adapter does not support training MLP chunking"
            )

    for layer in stack.layers:
        if isinstance(layer, MambaLayer):
            if not isinstance(layer.mixer, MambaMixer):
                raise NotImplementedError(
                    "shared-prefix state forking only supports MambaMixer-backed Mamba layers"
                )
            _validate_mamba_fork(layer.mixer)
            if layer.mixer.pg_collection.tp.size() != tp_size:
                raise RuntimeError(
                    "shared-prefix Mamba TP helper does not match the Hybrid stack TP group"
                )
            if layer.mixer.cp.cp_size != cp_size:
                raise RuntimeError(
                    "shared-prefix Mamba CP helper does not match the Hybrid stack CP group"
                )
        elif isinstance(layer, TransformerLayer):
            if not isinstance(layer.self_attention, (IdentityOp, SelfAttention)):
                raise NotImplementedError(
                    "shared-prefix Hybrid adapter supports standard self-attention and MLP/MoE layers"
                )
            if (
                isinstance(layer.self_attention, SelfAttention)
                and layer.self_attention.checkpoint_core_attention
            ):
                raise NotImplementedError(
                    "shared-prefix fused attention does not yet support selective core-attention recomputation"
                )
            if isinstance(layer.self_attention, SelfAttention) and (
                stack.config.qk_clip or stack.config.log_max_attention_logit
            ):
                raise NotImplementedError(
                    "shared-prefix fused attention does not yet produce QK-clipping/max-logit statistics"
                )
            if isinstance(layer.self_attention, SelfAttention):
                if layer.self_attention.pg_collection.tp.size() != tp_size:
                    raise RuntimeError(
                        "shared-prefix attention TP helper does not match the Hybrid stack TP group"
                    )
                if layer.self_attention.pg_collection.cp.size() != cp_size:
                    raise RuntimeError(
                        "shared-prefix attention CP helper does not match the Hybrid stack CP group"
                    )
            if cp_size > 1 and isinstance(layer.self_attention, SelfAttention):
                from megatron.core.models.hybrid.shared_prefix_fused import (
                    _cp_kv_head_slices_for_destinations,
                )

                if stack.config.num_attention_heads % tp_size:
                    raise NotImplementedError(
                        "shared-prefix attention requires query heads divisible by TP size"
                    )
                query_heads = stack.config.num_attention_heads // tp_size
                # This is the actual local K/V tensor width. When global KV heads are fewer than
                # TP ranks, SelfAttention replicates one KV head across the relevant TP ranks.
                kv_heads = layer.self_attention.num_query_groups_per_partition
                _cp_kv_head_slices_for_destinations(query_heads, kv_heads, cp_size)
            if stack.config.moe_router_enable_expert_bias and getattr(layer, "is_moe_layer", False):
                from megatron.core.transformer.moe.moe_layer import MoELayer
                from megatron.core.transformer.moe.router import TopKRouter

                if not isinstance(layer.mlp, MoELayer) or not isinstance(
                    layer.mlp.router, TopKRouter
                ):
                    raise NotImplementedError(
                        "shared-prefix expert-bias accounting requires an MCore "
                        "MoELayer with TopKRouter"
                    )
        else:
            raise NotImplementedError(
                f"shared-prefix state forking is not implemented for {type(layer).__name__}"
            )


def forward_hybrid_stack_shared_prefix(
    stack,
    hidden_states: Tensor,
    layout: SharedPrefixLayout,
    *,
    rotary_pos_emb: Tensor | tuple[Tensor, Tensor] | None = None,
    position_embedding_type: str = "rope",
) -> Tensor:
    """Explicit exact-prompt star forward for a supported ``HybridStack`` topology.

    Normal ``HybridStack.forward`` and ``Attention.forward`` behavior is unchanged unless this
    function installs its scoped private forest descriptor. The descriptor is always removed in a
    ``finally`` block, including when a layer raises.
    """
    _validate_hybrid_stack(stack, hidden_states, layout)
    has_attention = any(
        isinstance(layer, TransformerLayer) and isinstance(layer.self_attention, SelfAttention)
        for layer in stack.layers
    )
    if position_embedding_type not in ("rope", "none"):
        raise NotImplementedError(
            "shared-prefix attention supports only RoPE or positionless Hybrid models"
        )
    if has_attention and position_embedding_type == "rope" and rotary_pos_emb is None:
        raise ValueError("position-aware rotary_pos_emb is required for shared-prefix attention")
    if position_embedding_type == "none" and rotary_pos_emb is not None:
        raise ValueError("positionless shared-prefix attention must not receive rotary_pos_emb")

    cp_group = stack.pg_collection.cp
    tp_group = stack.pg_collection.tp
    tp_size = tp_group.size()
    sequence_shards = tp_size if stack.config.sequence_parallel else 1
    physical_len = hidden_states.shape[0] * cp_group.size() * sequence_shards
    token_multiplicities = None
    if stack.config.moe_router_enable_expert_bias:
        token_multiplicities = layout.padded_token_multiplicities(
            physical_len, hidden_states.device
        )
        if cp_group.size() > 1:
            token_multiplicities = token_multiplicities.index_select(
                0,
                layout.cp_local_indices(
                    physical_len, cp_group.size(), cp_group.rank(), hidden_states.device
                ),
            )
        if tp_size > 1:
            if token_multiplicities.numel() % tp_size:
                raise RuntimeError(
                    "shared-prefix CP-local token multiplicities must divide evenly over TP"
                )
            token_multiplicities = torch.chunk(token_multiplicities, tp_size, dim=0)[
                tp_group.rank()
            ].contiguous()
        if token_multiplicities.numel() != hidden_states.shape[0]:
            raise RuntimeError(
                "shared-prefix token multiplicity ownership does not match the local hidden rows"
            )

    def forward_layer_range(hidden_states: Tensor, start: int, end: int) -> Tensor:
        for layer in stack.layers[start:end]:
            moe_layer = (
                layer.mlp
                if stack.config.moe_router_enable_expert_bias
                and getattr(layer, "is_moe_layer", False)
                else None
            )
            if moe_layer is not None:
                if getattr(moe_layer, "_shared_prefix_token_multiplicities", None) is not None:
                    raise RuntimeError("nested shared-prefix MoE dispatch is not supported")
                moe_layer._shared_prefix_token_multiplicities = token_multiplicities
            try:
                if isinstance(layer, MambaLayer):
                    if cp_group.size() > 1 or tp_size > 1:
                        hidden_states = _forward_mamba_layer_shared_prefix_cp(
                            layer, hidden_states, layout
                        )
                    else:
                        hidden_states = _forward_mamba_layer_shared_prefix(
                            layer, hidden_states, layout
                        )
                elif isinstance(layer.self_attention, IdentityOp):
                    hidden_states = layer(hidden_states=hidden_states, attention_mask=None)
                else:
                    attention = layer.self_attention
                    if getattr(attention, "_shared_prefix_forest", None) is not None:
                        raise RuntimeError(
                            "nested shared-prefix attention dispatch is not supported"
                        )
                    attention._shared_prefix_forest = layout.forest
                    try:
                        hidden_states = layer(
                            hidden_states=hidden_states,
                            attention_mask=None,
                            rotary_pos_emb=rotary_pos_emb,
                        )
                    finally:
                        del attention._shared_prefix_forest
            finally:
                if moe_layer is not None:
                    del moe_layer._shared_prefix_token_multiplicities

            if isinstance(hidden_states, tuple):
                hidden_states = hidden_states[0]
        return hidden_states

    if stack.config.recompute_granularity == "full" and stack.training:
        chunk_size = stack.config.recompute_num_layers
        for start in range(0, len(stack.layers), chunk_size):
            end = min(start + chunk_size, len(stack.layers))

            def custom_forward(hidden_states: Tensor, start=start, end=end) -> Tensor:
                # Scopes live inside the callable so backward replay reconstructs them.
                return forward_layer_range(hidden_states, start, end)

            hidden_states = tensor_parallel.checkpoint(
                custom_forward, stack.config.distribute_saved_activations, hidden_states
            )
    else:
        hidden_states = forward_layer_range(hidden_states, 0, len(stack.layers))

    if stack.post_process and stack.post_layer_norm:
        hidden_states = stack.final_norm(hidden_states)
    return make_viewless_tensor(
        inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True
    )


# The legacy scalar remains the validated CP1 negotiation surface. CP-aware integrations must
# negotiate against the collection so installing the CP track cannot regress existing CP1 runs.
# ``SHARED_PREFIX_CP_TRAINING_CAPABILITY`` is advertised independently so integrations can retain
# the CP1 fast path while requiring the validated CP>1 Hybrid forward/backward contract.
SHARED_PREFIX_TRAINING_CAPABILITY = "hybrid_star_cp1_tp1_v1"
SHARED_PREFIX_CP_TRAINING_CAPABILITY = "hybrid_star_cp_v1"
SHARED_PREFIX_EXPLICIT_PHYSICAL_PADDING_CAPABILITY = "hybrid_star_explicit_physical_padding_v1"
SHARED_PREFIX_MOE_EXPERT_BIAS_CAPABILITY = "hybrid_star_moe_expert_bias_v1"
SHARED_PREFIX_FULL_RECOMPUTE_CAPABILITY = "hybrid_star_full_uniform_recompute_v1"
SHARED_PREFIX_TP_SP_TRAINING_CAPABILITY = "hybrid_star_cp1_tp_sp_v1"
SHARED_PREFIX_CP_TP_SP_TRAINING_CAPABILITY = "hybrid_star_cp_tp_sp_v1"
# Validated target-model feature: Nemotron-H attention is positionless while
# Mamba remains state-positioned by sequence order.
SHARED_PREFIX_POSITIONLESS_ATTENTION_CAPABILITY = "hybrid_star_positionless_attention_v1"
# Validated MTP predictor feature: reconstruct dense attention/MLP or attention/MoE
# heads from the shared-prefix physical layout on the supported distributed TP/CP
# topologies.
SHARED_PREFIX_MTP_DENSE_HEADS_CAPABILITY = "hybrid_star_mtp_dense_heads_v1"
# Topology and feature capabilities are independent so integrations can negotiate their exact
# validated conjunction without inferring support from a broader aggregate token.
SHARED_PREFIX_TRAINING_CAPABILITIES = frozenset(
    {
        SHARED_PREFIX_TRAINING_CAPABILITY,
        SHARED_PREFIX_CP_TRAINING_CAPABILITY,
        SHARED_PREFIX_EXPLICIT_PHYSICAL_PADDING_CAPABILITY,
        SHARED_PREFIX_MOE_EXPERT_BIAS_CAPABILITY,
        SHARED_PREFIX_FULL_RECOMPUTE_CAPABILITY,
        SHARED_PREFIX_TP_SP_TRAINING_CAPABILITY,
        SHARED_PREFIX_CP_TP_SP_TRAINING_CAPABILITY,
        SHARED_PREFIX_POSITIONLESS_ATTENTION_CAPABILITY,
        SHARED_PREFIX_MTP_DENSE_HEADS_CAPABILITY,
    }
)
