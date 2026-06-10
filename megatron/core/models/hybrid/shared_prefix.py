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

from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple

import torch

try:
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention

    HAVE_FLEX_ATTENTION = True
except ImportError:  # torch < 2.5
    create_block_mask = None
    flex_attention = None
    HAVE_FLEX_ATTENTION = False

# Lazily torch.compile'd flex_attention. Compilation is keyed on tensor shapes, so distinct packed
# lengths recompile (and cache) -- acceptable since bin shapes recur within a run. The compiled
# kernel skips fully-masked (sibling-branch) blocks, which is the whole speedup over the dense
# [T,T]-mask SDPA path (measured ~11x faster, ~5x faster than the un-shared baseline @ Lp5247/G16).
_COMPILED_FLEX = None


def _get_compiled_flex():
    global _COMPILED_FLEX
    if _COMPILED_FLEX is None:
        _COMPILED_FLEX = torch.compile(flex_attention)
    return _COMPILED_FLEX


def build_tree_segment_ids(prefix_len: int, completion_lens: List[int], device) -> torch.Tensor:
    """``[total_len]`` long: 0 for prefix tokens, ``i+1`` for completion ``i``'s tokens."""
    total = int(prefix_len) + sum(int(x) for x in completion_lens)
    seg = torch.zeros(total, dtype=torch.long, device=device)
    cursor = int(prefix_len)
    for i, lc in enumerate(completion_lens):
        seg[cursor : cursor + int(lc)] = i + 1
        cursor += int(lc)
    return seg


def build_tree_block_mask(prefix_len: int, completion_lens: List[int], device):
    """FlexAttention ``BlockMask`` for the shared-prefix tree: a query attends a key iff the key is
    causally before it AND (the key is in the prefix OR in the same completion branch). Sibling
    branches never attend to each other. Returns ``None`` if FlexAttention is unavailable.

    This is the kernel realization of ``shared_prefix_packing.dense_tree_mask`` -- built here from
    just (prefix_len, completion_lens) so ``megatron.core`` needs no ``megatron.rl`` import.
    """
    if not HAVE_FLEX_ATTENTION:
        return None
    seg = build_tree_segment_ids(prefix_len, completion_lens, device)
    total = int(seg.numel())

    def mask_mod(b, h, q_idx, kv_idx):
        causal = kv_idx <= q_idx
        k_is_prefix = seg[kv_idx] == 0
        same_branch = seg[kv_idx] == seg[q_idx]
        return causal & (k_is_prefix | same_branch)

    return create_block_mask(mask_mod, B=1, H=None, Q_LEN=total, KV_LEN=total, device=device)


def flex_tree_attention(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, block_mask, scale=None
) -> torch.Tensor:
    """Run the tree-masked attention via (compiled) FlexAttention.

    ``query/key/value`` are in Megatron core-attention layout ``[sq, b, n_heads, head_dim]``
    (key/value may have fewer heads for GQA). Returns the context in ``[sq, b, n_heads*head_dim]``,
    matching what ``core_attention`` returns to ``linear_proj``.
    """
    q = query.permute(1, 2, 0, 3)  # [b, np, sq, hn]
    k = key.permute(1, 2, 0, 3)  # [b, ng, sk, hn]
    v = value.permute(1, 2, 0, 3)
    enable_gqa = q.shape[1] != k.shape[1]
    out = _get_compiled_flex()(q, k, v, block_mask=block_mask, enable_gqa=enable_gqa, scale=scale)
    sq, b = query.shape[0], query.shape[1]
    return out.permute(2, 0, 1, 3).reshape(sq, b, -1).contiguous()  # [sq, b, np*hn]


@dataclass
class SharedPrefixParams:
    """Model-forward input describing one packed shared-prefix group ``[P, C_1, ..., C_G]``.

    Analogous to ``PackedSeqParams`` for THD packing: the RL/data layer builds it (lengths + the
    tree mask + prefix-continued position_ids from ``shared_prefix_packing``), threads it into
    ``HybridModel.forward``, which routes to ``HybridStack.forward_shared_prefix`` (instead of the
    dense decoder) and makes RoPE position-aware from ``position_ids``.
    """

    prefix_len: int
    completion_lens: List[int]
    # tree attention mask (Megatron convention, True == masked): prefix causal + each completion
    # attends the prefix and its own branch only. None is allowed for Mamba/MLP-only stacks.
    attention_mask: Optional[torch.Tensor] = None
    # prefix-continued positions [total_len] (P -> 0..Lp-1, each C_i -> Lp..Lp+Lc_i-1) for
    # position-aware RoPE; None falls back to packed-index RoPE (only correct without attention).
    position_ids: Optional[torch.Tensor] = None

    @property
    def total_len(self) -> int:
        return self.prefix_len + sum(self.completion_lens)


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
