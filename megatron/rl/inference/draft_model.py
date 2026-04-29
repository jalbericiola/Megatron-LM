# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Early-exit draft model for speculative rollout generation.

EarlyExitGPTModel wraps a full GPTModel and exits the transformer stack after
the first ``exit_layer`` blocks.  All parameters are *shared* with the full
model — no new parameters are created and no explicit weight synchronisation is
ever needed.
"""

from __future__ import annotations

import inspect
from functools import lru_cache
from typing import Optional

import torch
import torch.nn as nn

from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.typed_torch import apply_module
from megatron.core.utils import WrappedTensor, make_viewless_tensor


@lru_cache(maxsize=None)
def _layer_forward_param_names(layer_cls: type) -> frozenset[str]:
    """Names of kwargs the layer's ``forward`` accepts.

    Cached per class because ``inspect.signature`` is slow.  Returns a sentinel
    ``frozenset({'**'})`` when the layer accepts arbitrary kwargs (``**kwargs``)
    so the caller can pass everything.
    """
    sig = inspect.signature(layer_cls.forward)
    has_var_keyword = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    if has_var_keyword:
        return frozenset({"**"})
    return frozenset(
        p.name for p in sig.parameters.values()
        if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    )


def _call_layer(layer: nn.Module, kwargs: dict) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Call one layer of a possibly-heterogeneous decoder stack.

    Hybrid Mamba/Attention/MoE models put different layer classes in
    ``decoder.layers``:

    * ``TransformerBlock`` layer accepts ``context``, ``context_mask``,
      ``rotary_pos_cos``, ``rotary_pos_sin``, ``rotary_pos_cos_sin``,
      ``sequence_len_offset``, ``padding_mask`` and returns
      ``(hidden_states, context)``.
    * ``MambaLayer`` accepts only a small subset (``attention_mask``,
      ``inference_context``, ``rotary_pos_emb``, ``packed_seq_params``)
      and returns ``hidden_states`` (a Tensor, not a tuple).

    This helper:
      1. Drops kwargs the layer doesn't accept (so MambaLayer doesn't
         TypeError on ``context``).
      2. Normalises the return shape to ``(hidden_states, context_or_None)``.
    """
    accepted = _layer_forward_param_names(type(layer))
    if "**" in accepted:
        filtered = kwargs
    else:
        filtered = {k: v for k, v in kwargs.items() if k in accepted}
    out = layer(**filtered)
    if isinstance(out, tuple):
        # TransformerBlock layer: (hidden_states, context)
        return out[0], out[1] if len(out) > 1 else None
    return out, None


# Attributes the inference engine looks up via get_attr_wrapped_model().  Any
# new attribute that the engine starts requiring must be added here so the
# draft model continues to look like a GPTModel from the engine's perspective.
_FORWARDED_SCALAR_ATTRS = (
    "max_sequence_length",
    "model_type",
    "xattn_needed",
    "pg_collection",
    "share_embeddings_and_output_weights",
    # Required by megatron.core.inference.text_generation_controllers.
    # text_generation_controller.TextGenerationController.__init__:
    "vocab_size",
    # Various engine paths look these up too; harmless to forward
    # `if hasattr(full_model, ...)` (the constructor guards each).
    "padded_vocab_size",
    "seq_length",
)


class EarlyExitGPTModel(nn.Module):
    """GPT model that exits after the first ``exit_layer`` transformer blocks.

    Parameters are shared with ``full_model`` — no separate copy exists and no
    weight-sync step is ever needed.  The draft always reflects the live weights
    of the full model.

    Designed for *inference only* (speculative rollout generation).  Exposes the
    same ``forward()`` interface as :class:`GPTModel` so it can be passed
    directly to ``MegatronLocal.launch()``.

    Args:
        full_model: The reference GPTModel whose components are reused.
        exit_layer: Number of transformer blocks to run.  Must satisfy
            ``1 ≤ exit_layer < len(full_model.decoder.layers)``.
    """

    def __init__(self, full_model: GPTModel, exit_layer: int) -> None:
        super().__init__()
        total = len(full_model.decoder.layers)
        if not (1 <= exit_layer < total):
            raise ValueError(
                f"exit_layer must satisfy 1 ≤ exit_layer < {total}; got {exit_layer}"
            )

        self.exit_layer = exit_layer

        # Forward scalar / non-Module attributes that the inference engine and
        # config builders read via get_attr_wrapped_model().
        self.config = full_model.config
        self.pre_process = full_model.pre_process
        self.post_process = full_model.post_process
        self.position_embedding_type = full_model.position_embedding_type
        self.parallel_output = full_model.parallel_output
        for attr in _FORWARDED_SCALAR_ATTRS:
            if hasattr(full_model, attr):
                setattr(self, attr, getattr(full_model, attr))

        # Hold the full model in a list so nn.Module.__setattr__ does NOT
        # register it as a child — that would double-count its parameters.
        self._full_model_holder: list[GPTModel] = [full_model]

    # ------------------------------------------------------------------
    # Plumbing
    # ------------------------------------------------------------------

    @property
    def _full_model(self) -> GPTModel:
        return self._full_model_holder[0]

    @property
    def decoder(self):
        # Property — bypasses nn.Module.__setattr__ child registration.
        # Required by MambaInferenceStateConfig.from_model() and any other
        # consumer that walks the decoder via get_attr_wrapped_model().
        return self._full_model.decoder

    def train(self, mode: bool = True) -> "EarlyExitGPTModel":
        super().train(mode)
        self._full_model.train(mode)
        return self

    def eval(self) -> "EarlyExitGPTModel":
        return self.train(False)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        decoder_input: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        inference_context: Optional[BaseInferenceContext] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        extra_block_kwargs: Optional[dict] = None,
        runtime_gather_output: Optional[bool] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Run the first ``exit_layer`` transformer blocks and return logits in [b, s, h].

        ``labels`` is accepted for interface compatibility but ignored — the
        draft model is never trained directly.
        """
        fm = self._full_model

        # Embeddings + rotary positional encodings.
        preproc = fm._preprocess(
            input_ids=input_ids,
            position_ids=position_ids,
            decoder_input=decoder_input,
            inference_context=inference_context,
            packed_seq_params=packed_seq_params,
        )
        (
            decoder_input_tensor,
            rotary_pos_emb,
            rotary_pos_cos,
            rotary_pos_sin,
            sequence_len_offset,
            padding_mask,
        ) = preproc[:6]
        rotary_pos_cos_sin = preproc[6] if len(preproc) == 7 else None

        # _preprocess wraps decoder_input in a WrappedTensor during inference
        # so that TransformerBlock can drop its caller's reference for early
        # GC.  We iterate layers manually, so unwrap explicitly.
        if isinstance(decoder_input_tensor, WrappedTensor):
            decoder_input_tensor = decoder_input_tensor.unwrap()

        # NOTE: We iterate layers directly instead of calling fm.decoder(...)
        # to skip the full TransformerBlock — that block does activation
        # checkpointing and other train-time bookkeeping we don't need at
        # inference.  Trade-off: any inference-specific TransformerBlock
        # bookkeeping (e.g. CUDA-graph capture) is also skipped.  Revisit if
        # CUDA graphs become required for the draft path.
        #
        # In hybrid Mamba/Attention/MoE stacks the layers in
        # ``decoder.layers`` are heterogeneous and accept disjoint kwarg
        # sets; ``_call_layer()`` filters per-layer so we don't
        # ``TypeError`` on layers that don't accept ``context`` /
        # ``rotary_pos_cos`` / etc.
        hidden_states = decoder_input_tensor
        context = None
        extra = extra_block_kwargs or {}
        for layer in fm.decoder.layers[: self.exit_layer]:
            kwargs = dict(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                context=context,
                context_mask=None,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                rotary_pos_cos_sin=rotary_pos_cos_sin,
                inference_context=inference_context,
                packed_seq_params=packed_seq_params,
                sequence_len_offset=sequence_len_offset,
                padding_mask=padding_mask,
                **extra,
            )
            hidden_states, context = _call_layer(layer, kwargs)

        # Final layer norm — shared with the full model.
        if fm.decoder.final_layernorm is not None:
            hidden_states = apply_module(fm.decoder.final_layernorm)(hidden_states)
            hidden_states = make_viewless_tensor(
                inp=hidden_states, requires_grad=False, keep_graph=False
            )

        if not fm.post_process:
            return hidden_states

        # Tied input/output embeddings: pass the shared weight through to
        # ColumnParallelLinear.  Matches GPTModel._postprocess.
        output_weight = None
        if getattr(fm, "share_embeddings_and_output_weights", False):
            output_weight = fm.shared_embedding_or_output_weight()

        gather = (
            runtime_gather_output
            if runtime_gather_output is not None
            else not fm.parallel_output
        )
        logits, _ = fm.output_layer(
            hidden_states, weight=output_weight, runtime_gather_output=gather
        )

        # MuP logit scaling, when configured on the full model.
        logits = fm._scale_logits(logits)

        # Logits arrive as [s, b, h]; downstream consumers expect [b, s, h].
        return logits.transpose(0, 1).contiguous()


# ---------------------------------------------------------------------------
# Weight synchronisation helper for non-weight-sharing drafts
# ---------------------------------------------------------------------------


def sync_draft_weights(
    draft_model: nn.Module,
    full_model: GPTModel,
    num_layers: int,
) -> None:
    """Copy the first ``num_layers`` transformer blocks from full to draft.

    No-op for :class:`EarlyExitGPTModel` (parameters are shared by reference).
    Provided for API symmetry with draft variants that hold separate
    parameters (e.g. distilled or quantised drafts).
    """
    if isinstance(draft_model, EarlyExitGPTModel):
        return

    with torch.no_grad():
        src_params = dict(
            full_model.decoder.layers[:num_layers].named_parameters()
        )
        dst_params = dict(
            draft_model.decoder.layers[:num_layers].named_parameters()
        )
        for name, src in src_params.items():
            if name in dst_params:
                dst_params[name].copy_(src)
