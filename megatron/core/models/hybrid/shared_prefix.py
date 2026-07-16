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
#
# mode="max-autotune-no-cudagraphs": the DEFAULT compile picks a poor BACKWARD kernel (bwd ~13x the
# fwd; flex bwd 2x SLOWER than flash -> shared-prefix net-loses for TRAINING). Autotuning the fwd+bwd
# Triton configs cuts the backward ~2.8x (60->21ms @ Lp5247/G16), so flex bwd (21ms) now BEATS flash
# (37ms) and attention fwd+bwd flips from 0.79x to ~2x. First call autotunes (slow, cached);
# no-cudagraphs avoids capture conflicts with the RL trainer's own cuda graphs + variable bin shapes.
_COMPILED_FLEX = None


def _get_compiled_flex():
    global _COMPILED_FLEX
    if _COMPILED_FLEX is None:
        _COMPILED_FLEX = torch.compile(flex_attention, mode="max-autotune-no-cudagraphs")
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


def build_local_tree_block_mask(q_global_pos: torch.Tensor, seg_full: torch.Tensor, device,
                                is_pad_full: Optional[torch.Tensor] = None):
    """Tree ``BlockMask`` for a CP rank's LOCAL queries against the FULL (gathered) key sequence.

    ``q_global_pos[i]`` is the global packed position of local query ``i``; ``seg_full[j]`` is the
    segment id (0=prefix, b+1=branch b) of full-key ``j``. A local query attends a key iff the key is
    causally before it (by GLOBAL position) AND (key is prefix OR same branch). Q_LEN=n_local,
    KV_LEN=T -- a non-square mask, so each CP rank attends the whole prefix + its branch keys while
    computing only its own query rows. Returns None if FlexAttention is unavailable.
    """
    if not HAVE_FLEX_ATTENTION:
        return None
    n_local, T = int(q_global_pos.numel()), int(seg_full.numel())
    seg_q = seg_full[q_global_pos]                                # [n_local]

    def mask_mod(b, h, q_idx, kv_idx):
        # create_block_mask rounds Q_LEN/KV_LEN up to the block size and evaluates mask_mod at
        # PADDED indices, so clamp before gathering (avoid OOB) and mask the padded region out.
        in_range = (q_idx < n_local) & (kv_idx < T)
        qi = torch.clamp(q_idx, max=n_local - 1)
        ki = torch.clamp(kv_idx, max=T - 1)
        causal = kv_idx <= q_global_pos[qi]
        same = (seg_full[ki] == 0) | (seg_full[ki] == seg_q[qi])
        keep = in_range & causal & same
        if is_pad_full is not None:
            keep = keep & ~is_pad_full[ki]                  # never attend an end-pad key
        return keep

    return create_block_mask(mask_mod, B=1, H=None, Q_LEN=n_local, KV_LEN=T, device=device)


def local_tree_attention(
    query_local: torch.Tensor, key_full: torch.Tensor, value_full: torch.Tensor,
    block_mask, scale=None,
) -> torch.Tensor:
    """CP shared-prefix attention: a rank's LOCAL queries attend the FULL (gathered) K/V via the
    tree mask. ``query_local`` ``[n_local, b, nq, hd]`` (the rank's shard, post-RoPE); ``key/value_full``
    ``[T, b, nk, hd]`` (all CP ranks' K/V gathered to global order, post-RoPE; nk<=nq for GQA).
    Returns ``[n_local, b, nq*hd]``. The K/V are gathered (cheap under GQA) so each rank reuses the
    validated dense FlexAttention kernel locally -- no distributed-attention kernel needed; the
    gather's autograd reduce-scatters the K/V gradient back across CP."""
    q = query_local.permute(1, 2, 0, 3)                          # [b, nq, n_local, hd]
    k = key_full.permute(1, 2, 0, 3)                             # [b, nk, T, hd]
    v = value_full.permute(1, 2, 0, 3)
    enable_gqa = q.shape[1] != k.shape[1]
    out = _get_compiled_flex()(q, k, v, block_mask=block_mask, enable_gqa=enable_gqa, scale=scale)
    n_local, b = query_local.shape[0], query_local.shape[1]
    return out.permute(2, 0, 1, 3).reshape(n_local, b, -1).contiguous()


class CPSharedPrefixLayout:
    """Per-CP-rank layout for the segment-local-zigzag shared-prefix transport (Phase D).

    The packed ``[P, C_1, ..., C_G]`` is sharded by zigzag-splitting EACH segment independently
    across the CP group; rank r's local sequence is the concatenation of its zigzag shard of every
    segment. Built from the GLOBAL (prefix_len, completion_lens) + cp_size/cp_rank, it exposes:
      * ``local_global_pos`` [T_local]: the global packed position of each local token (drives the
        local-query tree mask and the RoPE slice);
      * ``seg_full`` [T]: segment id (0=prefix, b+1=branch b) of each global position;
      * ``gather_kv``: differentiable all-gather of the rank-local K/V to the FULL global-order
        sequence (un-zigzagging each segment), so attention runs ``local_tree_attention``.
    Each segment length must be a multiple of ``2*cp_size`` (the zigzag chunking).
    """

    def __init__(self, prefix_len, completion_lens, cp_size, cp_rank, device,
                 real_prefix_len=None, real_completion_lens=None):
        self.cp_size, self.cp_rank, self.device = cp_size, cp_rank, device
        self.global_seg_lens = [int(prefix_len)] + [int(c) for c in completion_lens]
        assert all(L % (2 * cp_size) == 0 for L in self.global_seg_lens), (
            f"each segment must be a multiple of 2*cp ({2 * cp_size}); got {self.global_seg_lens}")
        # per-segment REAL lengths (<= padded global len); positions beyond them are end-pads,
        # excluded as attention keys (the Mamba fork excludes them via fork_segment real_len).
        real_lens = [int(real_prefix_len) if real_prefix_len is not None else int(prefix_len)] + (
            [int(c) for c in real_completion_lens] if real_completion_lens is not None
            else [int(c) for c in completion_lens])
        ispad = []
        for L, rl in zip(self.global_seg_lens, real_lens):
            ispad += [0] * rl + [1] * (L - rl)
        self.is_pad_full = torch.tensor(ispad, dtype=torch.bool, device=device)
        self.any_pad = bool(self.is_pad_full.any().item())
        self.local_seg_lens = [L // cp_size for L in self.global_seg_lens]
        self.total_local = sum(self.local_seg_lens)
        self.total_global = sum(self.global_seg_lens)
        lgp, gstart = [], 0
        for L in self.global_seg_lens:                       # zigzag positions of rank cp_rank
            cs = L // (2 * cp_size)
            r = cp_rank
            lgp += list(range(gstart + r * cs, gstart + (r + 1) * cs))
            lgp += list(range(gstart + (2 * cp_size - 1 - r) * cs, gstart + (2 * cp_size - r) * cs))
            gstart += L
        self.local_global_pos = torch.tensor(lgp, dtype=torch.long, device=device)
        seg = []
        for s, L in enumerate(self.global_seg_lens):
            seg += [s] * L
        self.seg_full = torch.tensor(seg, dtype=torch.long, device=device)

    def _gather(self, x_local, cp_group):
        from torch.distributed.nn.functional import all_gather as diff_all_gather

        cp = self.cp_size
        gathered = diff_all_gather(x_local.contiguous(), group=cp_group)   # list cp x [T_local,...]
        out_segs, loff = [], 0
        for Ll in self.local_seg_lens:
            half = Ll // 2
            slots = [None] * (2 * cp)
            for r in range(cp):
                piece = gathered[r][loff:loff + Ll]
                slots[r], slots[2 * cp - 1 - r] = piece[:half], piece[half:]
            out_segs.append(torch.cat(slots, dim=0))
            loff += Ll
        return torch.cat(out_segs, dim=0)                                  # [T_global,...]

    def gather_kv(self, key_local, value_local, cp_group):
        """Differentiable all-gather + per-segment un-zigzag of K/V to FULL global order."""
        return self._gather(key_local, cp_group), self._gather(value_local, cp_group)

    def local_block_mask(self):
        return build_local_tree_block_mask(
            self.local_global_pos, self.seg_full, self.device,
            is_pad_full=self.is_pad_full if self.any_pad else None)


def two_term_tree_attention(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
    prefix_len: int, completion_lens: List[int], scale=None,
) -> torch.Tensor:
    """CP-compatible shared-prefix attention as TWO flash terms + an online-softmax (LSE) merge,
    instead of one FlexAttention tree BlockMask. Numerically equals ``flex_tree_attention`` but uses
    only dense/causal flash kernels, which (unlike FlexAttention) compose with context parallelism:
    the prefix K/V are gathered full once, and each completion's queries (the rank-local shard under
    CP) attend (1) the full prefix non-causally and (2) their own branch causally; the two disjoint
    key sets are merged exactly via their log-sum-exps.

    For each completion token q (prefix-continued position): the tree softmax is over {all prefix
    keys} ∪ {own-branch keys causally <= q}. These sets are disjoint, so
        out = (o1·e^{l1} + o2·e^{l2}) / (e^{l1} + e^{l2})
    where (o1,l1)=cross-attend(prefix), (o2,l2)=causal-self(own completion), and l* are flash's LSEs
    (log Σ e^{scaled score}). Prefix tokens are plain causal self-attention within the prefix.

    ``query/key/value``: Megatron core-attention layout ``[sq, b, n_heads, head_dim]`` (b==1 for a
    packed group; key/value may have fewer heads for GQA -- flash handles GQA natively). Returns the
    context ``[sq, b, n_heads*head_dim]``, matching ``flex_tree_attention`` / ``core_attention``.

    NOTE: q/k must already carry RoPE (prefix at positions 0..Lp-1, each completion prefix-continued)
    -- applied upstream, exactly as for the flex path. This function is RoPE-agnostic.
    """
    from flash_attn import flash_attn_func, flash_attn_varlen_func

    T, b, nq, hd = query.shape
    assert b == 1, "two_term_tree_attention expects a packed group with batch=1"
    Lp = int(prefix_len)
    q, k, v = query[:, 0], key[:, 0], value[:, 0]                 # [T, n*, hd]
    kp, vp = k[:Lp], v[:Lp]                                       # prefix K/V (full)

    # prefix region: causal self-attention within the prefix
    op = flash_attn_func(q[:Lp][None], kp[None], vp[None], softmax_scale=scale, causal=True)[0]

    parts = [op]
    if completion_lens:
        qc, kc, vc = q[Lp:], k[Lp:], v[Lp:]                      # [Lc_sum, n*, hd]
        # term1: every completion query attends the FULL prefix (non-causal; all prefix precedes it)
        o1, l1, _ = flash_attn_func(qc[None], kp[None], vp[None], softmax_scale=scale,
                                    causal=False, return_attn_probs=True)
        o1, l1 = o1[0], l1[0]                                     # o1 [Lc,nq,hd], l1 [nq,Lc]
        # term2: per-completion causal self-attention (varlen block-diagonal over the G branches)
        cu = torch.zeros(len(completion_lens) + 1, device=q.device, dtype=torch.int32)
        cu[1:] = torch.tensor(completion_lens, device=q.device, dtype=torch.int32).cumsum(0)
        maxl = max(completion_lens)
        o2, l2, _ = flash_attn_varlen_func(qc, kc, vc, cu, cu, maxl, maxl, softmax_scale=scale,
                                           causal=True, return_attn_probs=True)
        # online-softmax merge of the two disjoint key sets (LSE in fp32 for stability)
        l1t = l1.transpose(0, 1).unsqueeze(-1).float()           # [Lc,nq,1]
        l2t = l2.transpose(0, 1).unsqueeze(-1).float()
        m = torch.maximum(l1t, l2t)
        w1, w2 = (l1t - m).exp(), (l2t - m).exp()
        oc = (o1.float() * w1 + o2.float() * w2) / (w1 + w2)     # [Lc,nq,hd]
        parts.append(oc.to(query.dtype))

    return torch.cat(parts, dim=0).reshape(T, 1, nq * hd).contiguous()   # [sq, b, nq*hd]


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
    # Under CP, prefix_len/completion_lens are the PADDED (multiple-of-2*cp) segment lengths; these
    # carry the REAL (unpadded) lengths so the Mamba fork captures at the real prefix end and the
    # attention excludes end-pad keys. None -> no padding (real == padded).
    real_prefix_len: Optional[int] = None
    real_completion_lens: Optional[List[int]] = None

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

    def __init__(self, prefix_len: int, completion_lens: List[int],
                 real_prefix_len: Optional[int] = None) -> None:
        self.prefix_len = int(prefix_len)
        self.completion_lens = [int(x) for x in completion_lens]
        # FULL real prefix length (head-parallel coords) used as fork_segment(real_len=...) when the
        # prefix is end-padded to a multiple of 2*cp under CP; defaults to prefix_len (no padding).
        self.real_prefix_len = int(real_prefix_len) if real_prefix_len is not None else self.prefix_len
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
