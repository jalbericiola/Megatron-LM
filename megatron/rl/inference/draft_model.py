# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Early-exit draft model and utilities for speculative rollout generation.

EarlyExitGPTModel wraps a full GPTModel and exits the transformer stack after
the first `exit_layer` blocks.  All parameters are *shared* with the full model
— no new parameters are created and no explicit weight synchronisation is ever
needed.

DraftDistillationConfig enables an optional brief knowledge-distillation warm-up
before RL training to tighten the distribution gap between the early-exit outputs
and the full model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.typed_torch import apply_module
from megatron.core.utils import make_viewless_tensor


# ---------------------------------------------------------------------------
# EarlyExitGPTModel
# ---------------------------------------------------------------------------


class EarlyExitGPTModel(nn.Module):
    """GPT model that exits after the first `exit_layer` transformer blocks.

    Parameters are shared with ``full_model`` — no separate copy exists and no
    weight-sync step is ever needed.  The draft always reflects the live weights
    of the full model.

    The class is designed for *inference only* (speculative rollout generation).
    It exposes the same ``forward()`` interface as :class:`GPTModel` so that it
    can be passed directly to ``MegatronLocal.launch()``.

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

        # Expose config properties expected by GPTInferenceWrapper.
        self.config = full_model.config
        self.pre_process = full_model.pre_process
        self.post_process = full_model.post_process
        self.position_embedding_type = full_model.position_embedding_type
        self.parallel_output = full_model.parallel_output

        # Store the full model in a plain Python list so that nn.Module does
        # NOT register it as a child, preventing double-counting of its
        # parameters in the optimizer.  Lists are not traversed by
        # nn.Module.__setattr__, making this the canonical way to hold a
        # non-child module reference.
        self._full_model_holder: list[GPTModel] = [full_model]

    # ------------------------------------------------------------------
    # Convenience property
    # ------------------------------------------------------------------

    @property
    def _full_model(self) -> GPTModel:
        return self._full_model_holder[0]

    # ------------------------------------------------------------------
    # train / eval propagation
    # ------------------------------------------------------------------

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
        """Run the first ``exit_layer`` transformer blocks and return logits.

        ``labels`` is accepted for interface compatibility but ignored — the
        draft model is never trained directly.
        """
        fm = self._full_model

        # ---- Preprocessing: embeddings + rotary positional encodings ----
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

        # ---- Partial decoder: only the first `exit_layer` blocks ----
        hidden_states = decoder_input_tensor
        context = None
        extra = extra_block_kwargs or {}
        for layer in fm.decoder.layers[: self.exit_layer]:
            hidden_states, context = layer(
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

        # ---- Final layer norm (shared with the full model) ----
        if fm.decoder.final_layernorm is not None:
            hidden_states = apply_module(fm.decoder.final_layernorm)(hidden_states)
            hidden_states = make_viewless_tensor(
                inp=hidden_states, requires_grad=False, keep_graph=False
            )

        # ---- Output projection → logits ----
        if fm.post_process:
            # runtime_gather_output controls TP gather; mirror GPTModel behaviour.
            gather = (
                runtime_gather_output
                if runtime_gather_output is not None
                else not fm.parallel_output
            )
            logits, _ = fm.output_layer(hidden_states, runtime_gather_output=gather)
            return logits

        return hidden_states


# ---------------------------------------------------------------------------
# Weight synchronisation (no-op for EarlyExitGPTModel)
# ---------------------------------------------------------------------------


def sync_draft_weights(
    draft_model: nn.Module,
    full_model: GPTModel,
    num_layers: int,
) -> None:
    """Copy the first ``num_layers`` transformer blocks from the full model to the draft.

    For :class:`EarlyExitGPTModel` this is a **no-op** because all parameters
    are already shared by reference — the draft always reflects the live weights
    of the full model.  The function is provided for API symmetry with draft
    models that keep separate parameter copies (e.g. a distilled or quantised
    draft).

    Args:
        draft_model: The draft model to update.
        full_model: The full model providing the source weights.
        num_layers: How many transformer layers to copy.
    """
    if isinstance(draft_model, EarlyExitGPTModel):
        # Parameters are shared — nothing to do.
        return

    # Generic path: explicit tensor copy.
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


# ---------------------------------------------------------------------------
# Optional knowledge-distillation warm-up config
# ---------------------------------------------------------------------------


@dataclass
class DraftDistillationConfig:
    """Configuration for an optional KD warm-up of the early-exit draft model.

    When ``num_warmup_steps > 0``, the draft is briefly trained to minimise
    KL divergence between its output distribution and the full model's
    output distribution on a representative dataset.  This is useful when the
    early-exit distribution diverges significantly from the full model (common
    when ``exit_layer`` is a small fraction of the total depth).

    Args:
        num_warmup_steps: Number of KD gradient steps.  Set to 0 to skip.
        temperature: Softmax temperature applied to both distributions
            (higher → softer, more informative targets).
        kd_weight: Relative weight of the KD loss.
    """

    num_warmup_steps: int = 0
    temperature: float = 1.0
    kd_weight: float = 1.0

    @property
    def enabled(self) -> bool:
        """Return True when distillation warm-up is active."""
        return self.num_warmup_steps > 0
