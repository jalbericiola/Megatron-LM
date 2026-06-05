# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Runtime context for shared-prefix ("tree") packing through a HybridStack.

A GRPO group of G completions shares one prompt P. Instead of running the model on the G
duplicated ``[P + C_i]`` sequences (re-scanning / re-attending P every time), the stack runs a
SINGLE forward over the packed sequence ``[P, C_1, ..., C_G]`` (P stored once):

  * ATTENTION layers honor the *tree mask* (each token attends causally to the prefix and to its
    own completion branch, never to a sibling branch) with position-aware RoPE (each completion
    continues from the prefix's positions). Mask + RoPE express the sharing -- no module surgery.
  * MAMBA (and other recurrent) layers cannot express branch isolation with a mask -- a scan is
    sequential -- so they fork INTERNALLY: scan P once, then scan each completion C_i from P's
    captured conv + SSM end-state (``MambaMixer.fork_segment``), and write the per-segment outputs
    back into the packed positions.

``SharedPrefixContext`` carries the packed layout (prefix length + per-completion lengths) so a
stateful layer can slice the packed activation into its P / C_i segments. It is threaded through
``forward`` like ``inference_context`` and is a pure-Python holder (no model imports) so it can be
shared between ``megatron/core/ssm`` (MambaLayer) and ``megatron/core/models/hybrid`` (HybridStack)
without an import cycle.
"""

from typing import Iterator, List, Tuple


class SharedPrefixContext:
    """Packed shared-prefix layout for a single group ``[P, C_1, ..., C_G]``.

    A layer that consumes this slices the packed activation ``(total_len, 1, D)`` into the prefix
    ``[0:Lp]`` and each completion ``[start_i:start_i+Lc_i]``; gradients flow back through the
    forked state into the prefix, so the shared prompt receives the summed gradient of all its
    completions (loss/gradient-preserving).
    """

    def __init__(self, prefix_len: int, completion_lens: List[int]) -> None:
        self.prefix_len = int(prefix_len)
        self.completion_lens = [int(x) for x in completion_lens]
        assert self.prefix_len >= 1, "shared prefix must be non-empty"

    @property
    def num_completions(self) -> int:
        return len(self.completion_lens)

    @property
    def total_len(self) -> int:
        return self.prefix_len + sum(self.completion_lens)

    def segments(self) -> Iterator[Tuple[int, int, bool]]:
        """Yield ``(start, length, is_prefix)`` for the prefix then each completion, in packed
        order."""
        yield 0, self.prefix_len, True
        cursor = self.prefix_len
        for lc in self.completion_lens:
            yield cursor, lc, False
            cursor += lc
