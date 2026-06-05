# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Runtime context for shared-prefix ("tree") two-pass forward through a HybridStack.

A GRPO group of G completions shares one prompt P. Instead of running the model on the G
duplicated ``[P + C_i]`` sequences (re-scanning P every time), a shared-prefix forward runs:

  * PASS 1 ("capture"): scan the shared prefix P once. Each STATEFUL layer (Mamba; later
    attention) writes its prefix end-state into this context, keyed by ``layer_number``.
  * PASS 2 ("inject"): scan the completion(s) with each stateful layer's state FORKED from the
    captured prefix end-state -- equivalent (fwd + bwd) to the dense ``[P + C_i]`` forward.

This object is threaded through ``forward`` like ``inference_context``: layers consult it to
decide whether to capture, inject, or run normally. It is a pure-Python state holder (no model
imports) so it can be shared between ``megatron/core/ssm`` (MambaLayer) and
``megatron/core/models/hybrid`` (HybridStack) without an import cycle.
"""

from typing import Dict, Tuple

import torch

OFF = "off"
CAPTURE = "capture"
INJECT = "inject"


class SharedPrefixContext:
    """State threaded through a HybridStack shared-prefix two-pass forward.

    The captured per-layer states are kept in the autograd graph (NOT detached) so that, in
    PASS 2, a completion's gradient flows back through the injected state into PASS 1's prefix
    scan -- giving the shared prompt the summed gradient of all its completions, exactly as the
    dense duplicated forward would.
    """

    def __init__(self, mode: str = OFF) -> None:
        self.mode = mode
        # layer_number -> (conv_ctx, ssm_final) captured in PASS 1 by each MambaLayer
        self._mamba_state: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}

    @property
    def capturing(self) -> bool:
        return self.mode == CAPTURE

    @property
    def injecting(self) -> bool:
        return self.mode == INJECT

    @property
    def active(self) -> bool:
        return self.mode != OFF

    def capture_mamba(self, layer_number: int, conv_ctx, ssm_final) -> None:
        self._mamba_state[layer_number] = (conv_ctx, ssm_final)

    def mamba_state(self, layer_number: int):
        return self._mamba_state[layer_number]

    def reset(self, mode: str = OFF) -> None:
        self.mode = mode
        self._mamba_state.clear()
