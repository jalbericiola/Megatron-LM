# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Explicit CP1/TP1 shared-prefix adapter for a HybridStack.

This module deliberately does not alter :class:`HybridStack`'s normal ``forward`` path. Call
``forward_hybrid_stack_shared_prefix`` with one packed star ``[P, C_1, ..., C_G]`` to opt in. The
adapter scans each Mamba prefix once and forks its differentiable convolution/SSM state into the
completion branches; attention uses the exact-backward fused forest kernel. Unsupported topology
or model features fail before executing a partial forward.

The implementation is the narrow production slice of the shared-prefix work developed in
Megatron-RL/Megatron-LM commit ``0bf30804f`` plus the fused kernel port ``5b7173f7``. TP sequence
parallelism and CP>1 require the later state-routing work and are intentionally not advertised by
this adapter.
"""

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

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

    def __post_init__(self) -> None:
        prefix_len = int(self.prefix_len)
        completion_lens = tuple(int(length) for length in self.completion_lens)
        if prefix_len < 1:
            raise ValueError("shared-prefix layout requires a non-empty prefix")
        if not completion_lens or any(length < 1 for length in completion_lens):
            raise ValueError("shared-prefix layout requires one or more non-empty completions")
        object.__setattr__(self, "prefix_len", prefix_len)
        object.__setattr__(self, "completion_lens", completion_lens)

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


def _mamba_state_dtype_kwargs(mixer: MambaMixer) -> dict:
    if not MAMBA_HAS_STATE_DTYPE:
        return {}
    return {"state_dtype": mixer.mamba_training_ssm_states_dtype}


def _validate_mamba_fork(mixer: MambaMixer) -> None:
    if mixer.pg_collection.tp.size() != 1 or mixer.cp.cp_size != 1:
        raise NotImplementedError("shared-prefix Mamba state forking currently supports TP1/CP1")
    if mixer.config.sequence_parallel:
        raise NotImplementedError(
            "shared-prefix Mamba state forking does not support sequence parallelism"
        )
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
    max_completion_len = max(layout.completion_lens)
    branches = normalized.new_zeros(
        max_completion_len, len(layout.completion_lens), normalized.shape[-1]
    )
    for branch_index, completion_slice in enumerate(layout.completion_slices()):
        branch = normalized[completion_slice, 0]
        branches[: branch.shape[0], branch_index] = branch
    branch_output, _ = _fork_mamba_branches(
        layer.mixer, branches, conv_context=conv_context, ssm_initial_state=final_state
    )
    packed_output = torch.cat(
        [prefix_output]
        + [
            branch_output[:length, index : index + 1]
            for index, length in enumerate(layout.completion_lens)
        ],
        dim=0,
    )

    with layer.bias_dropout_add_exec_handler():
        return layer.mamba_bda(training=layer.training, fused=layer.config.bias_dropout_fusion)(
            (packed_output, output_bias), residual, layer.hidden_dropout
        )


def _has_nonzero_config_value(value) -> bool:
    """Return whether a scalar or per-layer configuration contains a nonzero value."""
    if value is None:
        return False
    if isinstance(value, (list, tuple)):
        return any(float(item) != 0.0 for item in value)
    return float(value) != 0.0


def _validate_hybrid_stack(stack, hidden_states: Tensor, layout: SharedPrefixLayout) -> None:
    if stack.tp_group.size() != 1:
        raise NotImplementedError("shared-prefix Hybrid adapter currently supports TP1 only")
    if stack.pp_group.size() != 1:
        raise NotImplementedError("shared-prefix Hybrid adapter currently supports PP1 only")
    if stack.config.context_parallel_size != 1:
        raise NotImplementedError("shared-prefix Hybrid adapter currently supports CP1 only")
    if stack.config.sequence_parallel:
        raise NotImplementedError(
            "shared-prefix Hybrid adapter does not support sequence parallelism"
        )
    if stack.config.recompute_granularity == "full":
        raise NotImplementedError(
            "shared-prefix Hybrid adapter does not yet preserve full-layer recomputation"
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
        raise ValueError("shared-prefix Hybrid input must have shape [sequence, 1, hidden]")
    if hidden_states.shape[0] != layout.total_len:
        raise ValueError(
            f"packed sequence length {hidden_states.shape[0]} does not match layout {layout.total_len}"
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
        if stack.config.moe_router_enable_expert_bias:
            raise NotImplementedError(
                "shared-prefix Hybrid adapter does not yet correct MoE expert-bias token counts"
            )
        if stack.config.moe_expert_capacity_factor is not None:
            raise NotImplementedError(
                "shared-prefix Hybrid adapter does not support MoE expert capacity or token dropping"
            )

    for layer in stack.layers:
        if isinstance(layer, MambaLayer):
            if not isinstance(layer.mixer, MambaMixer):
                raise NotImplementedError(
                    "shared-prefix state forking only supports MambaMixer-backed Mamba layers"
                )
            _validate_mamba_fork(layer.mixer)
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
) -> Tensor:
    """Explicit exact-prompt star forward for a supported CP1/TP1 ``HybridStack``.

    Normal ``HybridStack.forward`` and ``Attention.forward`` behavior is unchanged unless this
    function installs its scoped private forest descriptor. The descriptor is always removed in a
    ``finally`` block, including when a layer raises.
    """
    _validate_hybrid_stack(stack, hidden_states, layout)
    has_attention = any(
        isinstance(layer, TransformerLayer) and isinstance(layer.self_attention, SelfAttention)
        for layer in stack.layers
    )
    if has_attention and rotary_pos_emb is None:
        raise ValueError("position-aware rotary_pos_emb is required for shared-prefix attention")

    for layer in stack.layers:
        if isinstance(layer, MambaLayer):
            hidden_states = _forward_mamba_layer_shared_prefix(layer, hidden_states, layout)
        elif isinstance(layer.self_attention, IdentityOp):
            hidden_states = layer(hidden_states=hidden_states, attention_mask=None)
        else:
            attention = layer.self_attention
            if getattr(attention, "_shared_prefix_forest", None) is not None:
                raise RuntimeError("nested shared-prefix attention dispatch is not supported")
            attention._shared_prefix_forest = layout.forest
            try:
                hidden_states = layer(
                    hidden_states=hidden_states, attention_mask=None, rotary_pos_emb=rotary_pos_emb
                )
            finally:
                del attention._shared_prefix_forest

        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]

    if stack.post_process and stack.post_layer_norm:
        hidden_states = stack.final_norm(hidden_states)
    return make_viewless_tensor(
        inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True
    )


# Outer integrations must exact-match this value. It intentionally does not advertise TP>1/CP>1.
SHARED_PREFIX_TRAINING_CAPABILITY = "hybrid_star_cp1_tp1_v1"
