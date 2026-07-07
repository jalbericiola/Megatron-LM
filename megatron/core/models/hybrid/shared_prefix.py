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

import logging
import os
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

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


# --- Lightweight shared-prefix attention profiling -------------------------------------------
# Attributes the SP attention cost into mask-build / flex-kernel and exposes a *recompile proxy*
# (count of DISTINCT packed lengths -- torch.compile is shape-keyed, so each new length is a fresh
# compile). Call counts + unique-shape tracking are free (no sync) and always on. Per-call CUDA-
# event timing forces a sync, so it is gated behind NRL_SP_PROFILE=1. ``maybe_log_sp_profile``
# emits a summary line every ``NRL_SP_PROFILE_EVERY`` flex calls (default 200).
_SP_PROFILE_TIMING = os.environ.get("NRL_SP_PROFILE", "0") not in ("0", "", "false", "False")
_SP_PROFILE_EVERY = int(os.environ.get("NRL_SP_PROFILE_EVERY", "200"))
_SP_PROFILE = {
    "flex_calls": 0,
    "mask_builds": 0,
    "flex_shapes": set(),  # distinct packed lengths -> recompile proxy
    "mask_ms": 0.0,        # only populated when _SP_PROFILE_TIMING
    "flex_ms": 0.0,
}


def sp_profile_summary() -> dict:
    """Snapshot of shared-prefix attention counters (safe to call any time)."""
    p = _SP_PROFILE
    calls = p["flex_calls"]
    n_shapes = len(p["flex_shapes"])
    return {
        "shared_prefix/flex_calls": calls,
        "shared_prefix/mask_builds": p["mask_builds"],
        "shared_prefix/unique_flex_shapes": n_shapes,
        # ~1.0 means almost every call is a new shape (recompile-bound); ~0 means shapes recur.
        "shared_prefix/recompile_proxy": (n_shapes / calls) if calls else 0.0,
        "shared_prefix/mask_ms_total": p["mask_ms"],
        "shared_prefix/flex_ms_total": p["flex_ms"],
        "shared_prefix/timing_enabled": _SP_PROFILE_TIMING,
    }


def maybe_log_sp_profile() -> None:
    if _SP_PROFILE["flex_calls"] % _SP_PROFILE_EVERY == 0 and _SP_PROFILE["flex_calls"] > 0:
        logger.info("[shared_prefix profile] %s", sp_profile_summary())


def build_tree_segment_ids(
    prefix_len: int, completion_lens: List[int], device, padded_len: Optional[int] = None
) -> torch.Tensor:
    """``[padded_len or total_len]`` long: 0 for prefix tokens, ``i+1`` for completion ``i``'s
    tokens, and 0 for any trailing pad positions ``[total_len:padded_len]``.

    Trailing pad positions (used to make the packed length divisible by the tensor-parallel size
    for sequence parallelism) keep segment 0: under the causal rule no real token ever attends them
    (they sit after every real token), and pad queries attend only the prefix -- their outputs are
    discarded (never in ``comp_positions``).
    """
    total = int(prefix_len) + sum(int(x) for x in completion_lens)
    n = int(padded_len) if padded_len is not None else total
    seg = torch.zeros(n, dtype=torch.long, device=device)
    cursor = int(prefix_len)
    for i, lc in enumerate(completion_lens):
        seg[cursor : cursor + int(lc)] = i + 1
        cursor += int(lc)
    return seg


def build_tree_block_mask(
    prefix_len: int, completion_lens: List[int], device, padded_len: Optional[int] = None
):
    """FlexAttention ``BlockMask`` for the shared-prefix tree: a query attends a key iff the key is
    causally before it AND (the key is in the prefix OR in the same completion branch). Sibling
    branches never attend to each other. Returns ``None`` if FlexAttention is unavailable.

    ``padded_len`` extends the mask to a tensor-parallel-divisible packed length (for sequence
    parallelism); the trailing pad positions are causally harmless (see ``build_tree_segment_ids``).

    This is the kernel realization of ``shared_prefix_packing.dense_tree_mask`` -- built here from
    just (prefix_len, completion_lens) so ``megatron.core`` needs no ``megatron.rl`` import.
    """
    if not HAVE_FLEX_ATTENTION:
        return None
    seg = build_tree_segment_ids(prefix_len, completion_lens, device, padded_len=padded_len)
    total = int(seg.numel())

    def mask_mod(b, h, q_idx, kv_idx):
        causal = kv_idx <= q_idx
        k_is_prefix = seg[kv_idx] == 0
        same_branch = seg[kv_idx] == seg[q_idx]
        return causal & (k_is_prefix | same_branch)

    _SP_PROFILE["mask_builds"] += 1
    if _SP_PROFILE_TIMING and device is not None and str(device).startswith("cuda"):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        bm = create_block_mask(mask_mod, B=1, H=None, Q_LEN=total, KV_LEN=total, device=device)
        end.record()
        torch.cuda.synchronize()
        _SP_PROFILE["mask_ms"] += start.elapsed_time(end)
        return bm
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
    sq, b = query.shape[0], query.shape[1]

    _SP_PROFILE["flex_calls"] += 1
    _SP_PROFILE["flex_shapes"].add(int(sq))  # distinct packed lengths -> recompile proxy
    if _SP_PROFILE_TIMING and query.is_cuda:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = _get_compiled_flex()(q, k, v, block_mask=block_mask, enable_gqa=enable_gqa, scale=scale)
        end.record()
        torch.cuda.synchronize()
        _SP_PROFILE["flex_ms"] += start.elapsed_time(end)
    else:
        out = _get_compiled_flex()(q, k, v, block_mask=block_mask, enable_gqa=enable_gqa, scale=scale)
    maybe_log_sp_profile()
    return out.permute(2, 0, 1, 3).reshape(sq, b, -1).contiguous()  # [sq, b, np*hn]


# --- Backend selection: FlexAttention (default) vs flash-composed -----------------------------
# FlexAttention's *forward* is at parity with flash, but its generated-Triton *backward* is
# ~4x slower than flash's hand-tuned CUDA backward at equal FLOPs (worse with GQA), making the
# shared-prefix training step backward-bound. ``flash_composed`` rebuilds the same tree attention
# from flash kernels + an online-softmax merge (see ``flash_composed_tree_attention``), reaching
# ~flash-class fwd AND bwd. Switch with NRL_SP_ATTENTION_BACKEND=flash_composed (default: flex).
# Dense attention path only; the hybrid/Mamba path always uses flex (no ``_sp_layout`` set).
_SP_ATTENTION_BACKEND = os.environ.get("NRL_SP_ATTENTION_BACKEND", "flex").lower()
_SP_FALLBACK_WARNED = False


def sp_attention_backend() -> str:
    """Selected shared-prefix attention backend: ``"flex"`` or ``"flash_composed"``."""
    return _SP_ATTENTION_BACKEND


def flash_composed_tree_attention(query, key, value, prefix_len, completion_lens, scale=None):
    """Tree attention rebuilt from flash kernels (no FlexAttention BlockMask).

    A completion token attends {all prefix} ∪ {its own branch, causally}. That union splits into
    two flash-doable attentions whose softmaxes are merged by log-sum-exp (LSE):
      (a) completions -> prefix     : NON-causal flash (prefix wholly precedes the completions)
      (b) completions -> own branch : block-diagonal causal flash_varlen over [C_1..C_G]
    Prefix rows come from a causal flash over [P]. Trailing pad positions (TP-divisibility) get
    zero outputs -- they are never read downstream (not in ``comp_positions``). Flash handles GQA
    natively for fwd AND bwd, so this is ~flash-class where FlexAttention's backward is not.

    ``query`` is ``[sq, b, np, hn]`` and ``key``/``value`` ``[sq, b, ng, hn]`` (core-attention
    layout); ``b`` must be 1 (a single packed sequence). Returns ``[sq, b, np*hn]`` to match
    ``flex_tree_attention`` / the static ``core_attention`` output. Same default scale
    (1/sqrt(hn)) as the flex path when ``scale is None``.
    """
    from flash_attn import flash_attn_func, flash_attn_varlen_func

    sq, b, np_, hn = query.shape
    assert b == 1, "shared-prefix packing uses a single packed sequence (b == 1)"
    P = int(prefix_len)
    Cs = [int(c) for c in completion_lens]
    ncomp = sum(Cs)
    total = P + ncomp
    assert sq >= total, f"packed length {sq} < Lp+sum(Lc) {total}"

    q = query[:, 0]
    k = key[:, 0]
    v = value[:, 0]  # [sq, n, hn]
    qf, kf, vf = q[:P], k[:P], v[:P]
    qc, kc, vc = q[P:total], k[P:total], v[P:total]

    # (a) completions attend the full prefix (non-causal cross-attention).
    oa, lse_a, _ = flash_attn_func(
        qc.unsqueeze(0), kf.unsqueeze(0), vf.unsqueeze(0),
        softmax_scale=scale, causal=False, return_attn_probs=True,
    )
    oa = oa.squeeze(0)        # [ncomp, np, hn]
    lse_a = lse_a.squeeze(0)  # [np, ncomp]

    # (b) completions attend their own branch, causally (block-diagonal varlen).
    cu = [0]
    for c in Cs:
        cu.append(cu[-1] + c)
    cu = torch.tensor(cu, dtype=torch.int32, device=q.device)
    max_c = max(Cs)
    ob, lse_b, _ = flash_attn_varlen_func(
        qc, kc, vc, cu, cu, max_c, max_c,
        softmax_scale=scale, causal=True, return_attn_probs=True,
    )                          # ob [ncomp, np, hn], lse_b [np, ncomp]

    # online-softmax (LSE) merge -> exact softmax over the union of attended keys.
    m = torch.maximum(lse_a, lse_b)
    wa = torch.exp(lse_a - m)
    wb = torch.exp(lse_b - m)
    denom = wa + wb
    wa = (wa / denom).transpose(0, 1).unsqueeze(-1)  # [ncomp, np, 1]
    wb = (wb / denom).transpose(0, 1).unsqueeze(-1)
    # LSE (hence wa/wb) is fp32; recast the merged output back to the input dtype so the packed
    # output matches the flex path and downstream Linear layers (bf16) do not see an fp32 input.
    o_comp = (oa * wa + ob * wb).to(query.dtype)       # [ncomp, np, hn]

    # prefix rows: causal self-attention over [P].
    cu_p = torch.tensor([0, P], dtype=torch.int32, device=q.device)
    o_pre = flash_attn_varlen_func(
        qf, kf, vf, cu_p, cu_p, P, P, softmax_scale=scale, causal=True,
    )                          # [P, np, hn]

    out = torch.cat([o_pre, o_comp], dim=0)            # [total, np, hn]
    if sq > total:  # trailing pad positions; outputs discarded downstream.
        out = torch.cat([out, out.new_zeros(sq - total, np_, hn)], dim=0)
    return out.reshape(sq, 1, np_ * hn).contiguous()   # [sq, b, np*hn]


# --- Optimization: plan caching + fused (Triton) LSE merge ------------------------------------
# The fused kernel's overhead vs raw flash is NOT attention math; it is (a) rebuilding the pass
# plan (pure-Python node loops + Python-int index lists -> H2D copies) on EVERY call -- the same
# bin layout recurs across all ~50 layers of a step -- and (b) the eager online-softmax merge,
# which upcasts every pass output to fp32 and round-trips full [total, np, hn] tensors through
# memory several times per forward. (a) is fixed by an LRU plan cache keyed on the node arrays;
# (b) by a single Triton kernel that reads each pass's output/LSE once and writes the merged
# output (+ final LSE) once, fp32 math in-register, bf16 out. NRL_SP_FUSED_MERGE=0 restores the
# eager merge (fallback also automatic if Triton is unavailable).
try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:
    HAVE_TRITON = False

_SP_FUSED_MERGE = os.environ.get("NRL_SP_FUSED_MERGE", "1") not in ("0", "", "false", "False")

_PLAN_CACHE: dict = {}
_PLAN_CACHE_MAX = 128


if HAVE_TRITON:

    @triton.jit
    def _sp_merge_fwd_kernel(
        o0, o1, o2, o3, o4, o5, o6,          # pass outputs, [rows_p, np, HN] (dtype of q)
        l0, l1, l2, l3, l4, l5, l6,          # pass LSEs, fp32 [np, rows_p]
        i0, i1, i2, i3, i4, i5, i6,          # int32 [total]: token -> row in pass (or -1)
        r0, r1, r2, r3, r4, r5, r6,          # rows_p per pass (for LSE stride)
        out_ptr,                              # merged output [total, np, HN] (dtype of q)
        lsef_ptr,                             # final LSE fp32 [np, total]
        n_passes, total, np_: tl.constexpr, HN: tl.constexpr,
    ):
        t = tl.program_id(0)
        h = tl.program_id(1)
        offs = tl.arange(0, HN)
        m = float("-inf")
        s = 0.0
        acc = tl.zeros([HN], dtype=tl.float32)
        for p in tl.static_range(7):
            if p < n_passes:
                if p == 0:
                    idx_ptr, o_ptr, l_ptr, rows = i0, o0, l0, r0
                elif p == 1:
                    idx_ptr, o_ptr, l_ptr, rows = i1, o1, l1, r1
                elif p == 2:
                    idx_ptr, o_ptr, l_ptr, rows = i2, o2, l2, r2
                elif p == 3:
                    idx_ptr, o_ptr, l_ptr, rows = i3, o3, l3, r3
                elif p == 4:
                    idx_ptr, o_ptr, l_ptr, rows = i4, o4, l4, r4
                elif p == 5:
                    idx_ptr, o_ptr, l_ptr, rows = i5, o5, l5, r5
                else:
                    idx_ptr, o_ptr, l_ptr, rows = i6, o6, l6, r6
                r = tl.load(idx_ptr + t)
                if r >= 0:
                    lse = tl.load(l_ptr + h * rows + r)
                    o = tl.load(o_ptr + (r * np_ + h) * HN + offs).to(tl.float32)
                    m_new = tl.maximum(m, lse)
                    scale_old = tl.exp(m - m_new)
                    w = tl.exp(lse - m_new)
                    acc = acc * scale_old + o * w
                    s = s * scale_old + w
                    m = m_new
        out = acc / s
        tl.store(out_ptr + (t * np_ + h) * HN + offs, out.to(out_ptr.dtype.element_ty))
        tl.store(lsef_ptr + h * total + t, m + tl.log(s))


if HAVE_TRITON:

    @triton.jit
    def _sp_scale_gather_kernel(
        do_ptr,        # [total, np, HN] upstream grad (q dtype)
        lse_ptr,       # fp32 [np, rows] this pass's LSE
        lsef_ptr,      # fp32 [np, total] merged LSE
        qidx_ptr,      # int64 [rows] pass-row -> token (identity pass passes arange)
        dox_ptr,       # out [rows, np, HN] (q dtype): w * do[qidx]
        rows, total, np_: tl.constexpr, HN: tl.constexpr,
    ):
        r = tl.program_id(0)
        h = tl.program_id(1)
        offs = tl.arange(0, HN)
        t = tl.load(qidx_ptr + r)
        w = tl.exp(tl.load(lse_ptr + h * rows + r) - tl.load(lsef_ptr + h * total + t))
        do = tl.load(do_ptr + (t * np_ + h) * HN + offs).to(tl.float32)
        tl.store(dox_ptr + (r * np_ + h) * HN + offs, (do * w).to(dox_ptr.dtype.element_ty))

    @triton.jit
    def _sp_scatter_accum_kernel(
        dst_ptr,       # fp32 [total, n, HN] accumulator
        src_ptr,       # [rows, n, HN] pass grad (q dtype)
        idx_ptr,       # int64 [rows] pass-row -> token
        n_: tl.constexpr, HN: tl.constexpr,
    ):
        r = tl.program_id(0)
        h = tl.program_id(1)
        offs = tl.arange(0, HN)
        t = tl.load(idx_ptr + r)
        cur = tl.load(dst_ptr + (t * n_ + h) * HN + offs)
        add = tl.load(src_ptr + (r * n_ + h) * HN + offs).to(tl.float32)
        tl.store(dst_ptr + (t * n_ + h) * HN + offs, cur + add)


if HAVE_TRITON:

    @triton.jit
    def _sp_gather_kv_kernel(
        k_ptr, v_ptr,      # [total, ng, HN] sources (q dtype)
        idx_ptr,           # int64 [rows] pass-row -> token
        kx_ptr, vx_ptr,    # [rows, ng, HN] destinations
        ng: tl.constexpr, HN: tl.constexpr,
    ):
        r = tl.program_id(0)
        h = tl.program_id(1)
        offs = tl.arange(0, HN)
        t = tl.load(idx_ptr + r)
        tl.store(kx_ptr + (r * ng + h) * HN + offs, tl.load(k_ptr + (t * ng + h) * HN + offs))
        tl.store(vx_ptr + (r * ng + h) * HN + offs, tl.load(v_ptr + (t * ng + h) * HN + offs))


# Round-3: overlap the independent per-pass flash calls on side CUDA streams (they only join at
# the LSE merge / the gradient scatters), and gather K+V through one fused kernel into
# plan-cached workspace buffers (no per-call allocations, one index read for both tensors).
# NRL_SP_STREAMS=0 disables the stream overlap (kernels still fused).
_SP_STREAMS = os.environ.get("NRL_SP_STREAMS", "1") not in ("0", "", "false", "False")
_SP_STREAM_POOL: List = []
_SP_STREAM_POOL_N = 4


def _sp_streams():
    if not _SP_STREAMS or not torch.cuda.is_available():
        return None
    if not _SP_STREAM_POOL:
        _SP_STREAM_POOL.extend(torch.cuda.Stream() for _ in range(_SP_STREAM_POOL_N))
    return _SP_STREAM_POOL


def _gather_kv(k, v, k_idx):
    """Fused K+V gather (one index read, both tensors) via Triton; falls back to two
    index_selects without it."""
    if HAVE_TRITON and _SP_FUSED_MERGE:
        rows = k_idx.numel()
        ng, hn = k.shape[1], k.shape[2]
        kx = torch.empty(rows, ng, hn, dtype=k.dtype, device=k.device)
        vx = torch.empty(rows, ng, hn, dtype=v.dtype, device=v.device)
        _sp_gather_kv_kernel[(rows, ng)](k, v, k_idx, kx, vx, ng=ng, HN=hn)
        return kx, vx
    return k.index_select(0, k_idx), v.index_select(0, k_idx)


def _plan_key(node_start, node_len, node_parent, device):
    return (tuple(int(x) for x in node_start), tuple(int(x) for x in node_len),
            tuple(int(x) for x in node_parent), str(device))


def _forest_attention_plan_cached(node_start, node_len, node_parent, device):
    """Cached ``(total, passes, inv_maps)`` for a bin layout. The same layout is reused by every
    attention layer of the step (and often across steps), so the Python plan construction and the
    token->pass-row inverse maps (for the fused merge) are built once. ``inv_maps[p]`` is an int32
    ``[total]`` tensor mapping token -> its row in pass ``p`` (-1 if the token is not a query of
    that pass; pass 0 -- the self pass -- is the identity)."""
    key = _plan_key(node_start, node_len, node_parent, device)
    hit = _PLAN_CACHE.get(key)
    if hit is not None:
        return hit
    total, passes = _forest_attention_plan(node_start, node_len, node_parent, device)
    inv_maps, qidx64 = [], []
    identity = torch.arange(total, dtype=torch.long, device=device)
    for (q_idx, *_rest) in passes:
        if q_idx is None:
            inv = torch.arange(total, dtype=torch.int32, device=device)
            qidx64.append(identity)
        else:
            inv = torch.full((total,), -1, dtype=torch.int32, device=device)
            inv[q_idx] = torch.arange(q_idx.numel(), dtype=torch.int32, device=device)
            qidx64.append(q_idx)
        inv_maps.append(inv)
    if len(_PLAN_CACHE) >= _PLAN_CACHE_MAX:
        _PLAN_CACHE.pop(next(iter(_PLAN_CACHE)))
    _PLAN_CACHE[key] = (total, passes, inv_maps, qidx64)
    return _PLAN_CACHE[key]


def _merge_passes_triton(passes, inv_maps, outs, lses, total, np_, hn, dtype, device):
    """One-kernel online-softmax merge across passes. Returns (o_merged [total, np, hn] in
    ``dtype``, lse_final fp32 [np, total])."""
    MAXP = 7
    n = len(passes)
    assert n <= MAXP, f"fused merge supports <= {MAXP} passes (got {n}); deepen tl.static_range"
    dummy_o = outs[0]
    dummy_l = lses[0]
    dummy_i = inv_maps[0]
    o_args, l_args, i_args, r_args = [], [], [], []
    for p in range(MAXP):
        if p < n:
            o_args.append(outs[p].contiguous())
            l_args.append(lses[p].contiguous())
            i_args.append(inv_maps[p])
            r_args.append(lses[p].shape[1])
        else:
            o_args.append(dummy_o)
            l_args.append(dummy_l)
            i_args.append(dummy_i)
            r_args.append(1)
    out = torch.empty(total, np_, hn, dtype=dtype, device=device)
    lse_final = torch.empty(np_, total, dtype=torch.float32, device=device)
    _sp_merge_fwd_kernel[(total, np_)](
        *o_args, *l_args, *i_args, *r_args, out, lse_final,
        n, total, np_=np_, HN=hn,
    )
    return out, lse_final


def _forest_attention_plan(node_start, node_len, node_parent, device):
    """Decompose a forest/tree into the flash passes the composed attention runs.

    Returns ``(total, passes)`` where ``passes`` is a list of
    ``(q_idx, k_idx, cu_q, cu_k, max_q, max_k, causal)``:
      * one SELF pass -- every node attends its own span causally (block-diagonal varlen over all
        tokens), and
      * one CROSS pass per ancestor depth level L -- tokens strictly below depth L attend their
        level-L ancestor span, non-causally (DFS contiguity makes each ancestor's descendant tokens
        a contiguous run).
    A token attends ``{own node, causal} ∪ {each ancestor node, full}`` -- the union of the passes it
    appears in as a query. ``max_depth + 1`` passes total, independent of group count: depth-1 forest
    (stars) ⇒ self + 1 cross; arbitrary-depth trees ⇒ ``depth + 1``. ``node_*`` give the structure
    (parents precede children; a subtree is a contiguous DFS run).
    """
    ns = [int(x) for x in node_start]
    nl = [int(x) for x in node_len]
    par = [int(x) for x in node_parent]
    N = len(ns)
    total = max((ns[i] + nl[i] for i in range(N)), default=0)

    # PRECONDITION: node arrays must be DFS-preorder (each node's subtree is the contiguous run
    # immediately after it). ``subtree_end`` below relies on this -- a non-DFS layout would silently
    # attend the WRONG ancestor spans (branches lose interior-ancestor context -> corrupt logprobs).
    # PackedTreeLayout only checks contiguity + parent<i, which do NOT imply DFS; validate here (O(N),
    # negligible vs attention) so a mis-ordered layout fails loudly instead of training on garbage.
    _stack: List[int] = []
    for i in range(N):
        p = par[i]
        if p == -1:
            _stack = [i]
            continue
        while _stack and _stack[-1] != p:
            _stack.pop()
        if not _stack or _stack[-1] != p:
            raise ValueError(
                f"forest/tree layout is not DFS-preorder at node {i} (parent {p}): the fused tree "
                "attention requires each node's subtree to be the contiguous run after it. "
                "Emit nodes in DFS preorder (parent, then each child's full subtree)."
            )
        _stack.append(i)

    depth = [0] * N
    for i in range(N):
        depth[i] = 0 if par[i] == -1 else depth[par[i]] + 1
    d_max = max(depth, default=-1)
    subtree_end = [ns[i] + nl[i] for i in range(N)]
    for i in range(N):
        j = i + 1
        while j < N and depth[j] > depth[i]:
            subtree_end[i] = ns[j] + nl[j]
            j += 1

    def _i32(x):
        return torch.tensor(x, dtype=torch.int32, device=device)

    def _i64(x):
        return torch.tensor(x, dtype=torch.long, device=device)

    passes = []
    # self pass: one block-diagonal causal varlen over every node's own span. Its q/k indices are the
    # identity over [0, total), so they are left as ``None`` -- the kernel then uses q/k/v directly and
    # skips a full-tensor gather/scatter every forward and backward (a real cost on big packed bins).
    cu = [0]
    for i in range(N):
        cu.append(cu[-1] + nl[i])
    mnl = max(nl, default=0)
    passes.append((None, None, _i32(cu), _i32(cu), mnl, mnl, True))

    # cross passes: one per ancestor depth level.
    for L in range(d_max):
        qpos: List[int] = []
        kpos: List[int] = []
        cuq, cuk = [0], [0]
        for a in range(N):
            if depth[a] != L:
                continue
            qs, qe = ns[a] + nl[a], subtree_end[a]  # strict descendants of a (contiguous, DFS)
            if qe <= qs:
                continue
            qpos.extend(range(qs, qe))
            kpos.extend(range(ns[a], ns[a] + nl[a]))
            cuq.append(cuq[-1] + (qe - qs))
            cuk.append(cuk[-1] + nl[a])
        if not qpos:
            continue
        mxq = max(cuq[i + 1] - cuq[i] for i in range(len(cuq) - 1))
        mxk = max(cuk[i + 1] - cuk[i] for i in range(len(cuk) - 1))
        passes.append((_i64(qpos), _i64(kpos), _i32(cuq), _i32(cuk), mxq, mxk, False))
    return total, passes


class _ComposedForestAttn(torch.autograd.Function):
    """Composed forest/tree attention with an EXACT backward.

    The forward runs the ``_forest_attention_plan`` passes with flash and merges them by online
    softmax (LSE) -- the union-softmax identity, so the forward is exact. The backward is the
    delicate part: each pass's flash output is a sub-attention, and the *naive* autograd-through-flash
    drops the inter-pass normalizer-coupling term (flash exposes no gradient through its LSE), giving
    ~15% wrong q/k grads. We fix it by calling the low-level ``_flash_attn_varlen_backward`` per pass
    with ``dout = w_pass * do`` AND substituting the MERGED output ``o`` for the pass's own output:
    flash uses ``out`` only to form the row-delta ``D = rowsum(dout ∘ out)``, which is exactly the
    softmax ``G`` term, so feeding the merged ``o`` injects the global normalizer that was missing.
    The result is the exact union-softmax score gradient ``P_ij (v_j·do − o·do)`` for every pass --
    dq, dk, dv all correct -- with no custom kernel (see TREE_PACKING_DESIGN.md §5)."""

    @staticmethod
    def forward(ctx, q, k, v, node_start, node_len, node_parent, scale):
        # q: [total, np, hn]; k, v: [total, ng, hn]; scale already resolved to a float.
        from flash_attn import flash_attn_varlen_func

        total, passes, inv_maps, qidx64 = _forest_attention_plan_cached(
            node_start, node_len, node_parent, q.device
        )
        np_, hn = q.shape[1], q.shape[2]
        streams = _sp_streams()
        outs, lses = [None] * len(passes), [None] * len(passes)
        if streams is not None and len(passes) > 1:
            # passes are independent until the merge: fan them out on side streams so the small
            # cross passes hide under the big self pass. Their outputs are consumed back on the
            # current stream after the join events (record_stream keeps the allocator honest).
            cur = torch.cuda.current_stream()
            ev_in = torch.cuda.Event()
            ev_in.record(cur)
            join = []
            for i, (q_idx, k_idx, cu_q, cu_k, mxq, mxk, causal) in enumerate(passes):
                st = streams[i % len(streams)]
                st.wait_event(ev_in)
                with torch.cuda.stream(st):
                    qx = q if q_idx is None else q.index_select(0, q_idx)
                    if k_idx is None:
                        kx, vx = k, v
                    else:
                        kx, vx = _gather_kv(k, v, k_idx)
                    o, lse, _ = flash_attn_varlen_func(
                        qx, kx, vx, cu_q, cu_k, mxq, mxk,
                        softmax_scale=scale, causal=causal, return_attn_probs=True,
                    )
                    o.record_stream(cur)
                    lse.record_stream(cur)
                    ev = torch.cuda.Event()
                    ev.record(st)
                    join.append(ev)
                outs[i], lses[i] = o, lse
            for ev in join:
                cur.wait_event(ev)
        else:
            for i, (q_idx, k_idx, cu_q, cu_k, mxq, mxk, causal) in enumerate(passes):
                qx = q if q_idx is None else q.index_select(0, q_idx)  # None => identity (self pass)
                if k_idx is None:
                    kx, vx = k, v
                else:
                    kx, vx = _gather_kv(k, v, k_idx)
                o, lse, _ = flash_attn_varlen_func(
                    qx, kx, vx, cu_q, cu_k, mxq, mxk,
                    softmax_scale=scale, causal=causal, return_attn_probs=True,
                )  # o [Σq, np, hn], lse [np, Σq]
                outs[i], lses[i] = o, lse

        if _SP_FUSED_MERGE and HAVE_TRITON and len(passes) <= 7:
            # single-kernel online-softmax merge: reads each pass output/LSE once, writes the
            # merged output + final LSE once (fp32 in-register), replacing the eager fp32
            # upcast/mul/index_add round-trips below.
            o_merged, lse_final = _merge_passes_triton(
                passes, inv_maps, outs, lses, total, np_, hn, q.dtype, q.device
            )
        else:
            # merged LSE per (head, token): logsumexp over every pass the token queries in.
            lse_final = torch.full((np_, total), float("-inf"), device=q.device, dtype=torch.float32)
            for (q_idx, *_), lse in zip(passes, lses):
                if q_idx is None:
                    lse_final = torch.logaddexp(lse_final, lse.float())
                else:
                    lse_final[:, q_idx] = torch.logaddexp(lse_final[:, q_idx], lse.float())
            # merged output: sum_pass w_pass * o_pass, w_pass = exp(lse_pass - lse_final).
            o_merged = torch.zeros(total, np_, hn, device=q.device, dtype=torch.float32)
            for (q_idx, *_), o, lse in zip(passes, outs, lses):
                lf = lse_final if q_idx is None else lse_final.index_select(1, q_idx)
                contrib = torch.exp(lse.float() - lf).transpose(0, 1).unsqueeze(-1) * o.float()
                if q_idx is None:
                    o_merged = o_merged + contrib
                else:
                    o_merged.index_add_(0, q_idx, contrib)
            o_merged = o_merged.to(q.dtype)

        ctx.save_for_backward(q, k, v, o_merged)
        ctx.passes = passes
        ctx.lses = lses
        ctx.lse_final = lse_final
        ctx.scale = scale
        ctx.qidx64 = qidx64
        return o_merged

    @staticmethod
    def backward(ctx, do):
        from flash_attn.flash_attn_interface import _flash_attn_varlen_backward

        q, k, v, o_merged = ctx.saved_tensors
        lse_final, scale = ctx.lse_final, ctx.scale
        do = do.contiguous()
        total, np_, hn = q.shape[0], q.shape[1], q.shape[2]
        use_triton = _SP_FUSED_MERGE and HAVE_TRITON and getattr(ctx, "qidx64", None) is not None

        if use_triton:
            # Fused backward glue: (a) per-pass dout scaling w*do fused with the query gather in
            # one kernel (the eager path materialized an fp32 exp/mul chain + an index_select per
            # pass); (b) the self pass (always pass 0, identity indices over all tokens) INITIALIZES
            # the fp32 accumulators instead of zeros+add; cross passes scatter-accumulate through a
            # cast-fused kernel (no per-pass .float() temporaries). The flash calls are unchanged
            # -- the exact-backward trick (merged o substituted for the pass output) is preserved.
            ng = k.shape[1]
            streams = _sp_streams()
            cur = torch.cuda.current_stream()
            results = [None] * len(ctx.passes)
            join = [None] * len(ctx.passes)
            ev_in = None
            if streams is not None and len(ctx.passes) > 1:
                ev_in = torch.cuda.Event()
                ev_in.record(cur)
            for i, ((q_idx, k_idx, cu_q, cu_k, mxq, mxk, causal), lse) in enumerate(
                zip(ctx.passes, ctx.lses)
            ):
                st = None if ev_in is None else streams[i % len(streams)]
                stream_ctx = torch.cuda.stream(st) if st is not None else nullcontext()
                if st is not None:
                    st.wait_event(ev_in)
                with stream_ctx:
                    qidx = ctx.qidx64[i]
                    rows = qidx.numel()
                    qx = q if q_idx is None else q.index_select(0, q_idx)
                    if k_idx is None:
                        kx, vx = k, v
                    else:
                        kx, vx = _gather_kv(k, v, k_idx)
                    ox = o_merged if q_idx is None else o_merged.index_select(0, q_idx)
                    dox = torch.empty(rows, np_, hn, dtype=q.dtype, device=q.device)
                    _sp_scale_gather_kernel[(rows, np_)](
                        do, lse.contiguous(), lse_final, qidx, dox, rows, total, np_=np_, HN=hn,
                    )
                    dqx, dkx, dvx = torch.empty_like(qx), torch.empty_like(kx), torch.empty_like(vx)
                    _flash_attn_varlen_backward(
                        dox, qx, kx, vx, ox, lse, dqx, dkx, dvx, cu_q, cu_k, mxq, mxk,
                        0.0, scale, causal, -1, -1, 0.0, None, False, None, False,
                    )
                    results[i] = (dqx, dkx, dvx, qidx, k_idx, rows)
                    if st is not None:
                        dqx.record_stream(cur)
                        dkx.record_stream(cur)
                        dvx.record_stream(cur)
                        ev = torch.cuda.Event()
                        ev.record(st)
                        join[i] = ev
            # accumulate in pass order on the current stream (scatters are read-modify-write on
            # the shared accumulators, so they stay ordered; the flash backwards above overlap).
            dq = dk = dv = None
            for i, res in enumerate(results):
                if join[i] is not None:
                    cur.wait_event(join[i])
                dqx, dkx, dvx, qidx, k_idx, rows = res
                if i == 0:
                    # self pass: identity over all tokens -> direct init, no zeros/scatter.
                    dq = dqx.float()
                    dk = dkx.float()
                    dv = dvx.float()
                else:
                    _sp_scatter_accum_kernel[(rows, np_)](dq, dqx, qidx, n_=np_, HN=hn)
                    _sp_scatter_accum_kernel[(k_idx.numel(), ng)](dk, dkx, k_idx, n_=ng, HN=hn)
                    _sp_scatter_accum_kernel[(k_idx.numel(), ng)](dv, dvx, k_idx, n_=ng, HN=hn)
            return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), None, None, None, None

        dq = torch.zeros(q.shape, device=q.device, dtype=torch.float32)
        dk = torch.zeros(k.shape, device=k.device, dtype=torch.float32)
        dv = torch.zeros(v.shape, device=v.device, dtype=torch.float32)
        for (q_idx, k_idx, cu_q, cu_k, mxq, mxk, causal), lse in zip(ctx.passes, ctx.lses):
            qx = q if q_idx is None else q.index_select(0, q_idx)  # q_idx None => identity (self pass)
            kx = k if k_idx is None else k.index_select(0, k_idx)
            vx = v if k_idx is None else v.index_select(0, k_idx)
            ox = o_merged if q_idx is None else o_merged.index_select(0, q_idx)  # MERGED output -> exact
            lf = lse_final if q_idx is None else lse_final.index_select(1, q_idx)
            dox_full = do if q_idx is None else do.index_select(0, q_idx)
            dox = (torch.exp(lse.float() - lf).transpose(0, 1).unsqueeze(-1) * dox_full).to(q.dtype)
            dqx, dkx, dvx = torch.empty_like(qx), torch.empty_like(kx), torch.empty_like(vx)
            _flash_attn_varlen_backward(
                dox, qx, kx, vx, ox, lse, dqx, dkx, dvx, cu_q, cu_k, mxq, mxk,
                0.0, scale, causal, -1, -1, 0.0, None, False, None, False,
            )
            if q_idx is None:
                dq += dqx.float()
            else:
                dq.index_add_(0, q_idx, dqx.float())
            if k_idx is None:
                dk += dkx.float()
                dv += dvx.float()
            else:
                dk.index_add_(0, k_idx, dkx.float())
                dv.index_add_(0, k_idx, dvx.float())
        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), None, None, None, None


def flash_composed_forest_attention_fused(query, key, value, node_start, node_len, node_parent, scale=None):
    """Level-decomposed forest/tree attention, fused to ``max_depth + 1`` flash passes per bin.

    A token attends ``{its own node, causally} ∪ {each ancestor node, fully}``. Decomposed by ancestor
    depth level (see :func:`_forest_attention_plan`) and merged by online softmax, so cost is
    independent of group count (depth-1 stars ⇒ 1 self + 1 cross pass) -- vs the per-group loop's
    ``~3 * #groups`` launches. Forward AND backward are exact via :class:`_ComposedForestAttn`.

    ``query`` is ``[sq, b, np, hn]`` and ``key``/``value`` ``[sq, b, ng, hn]`` (b == 1). Trailing pad
    positions (beyond the last node) get zero outputs and are never read downstream. Returns
    ``[sq, b, np*hn]``; same default scale (1/sqrt(hn)) as the flex/loop paths.
    """
    sq, b, np_, hn = query.shape
    assert b == 1, "shared-prefix packing uses a single packed sequence (b == 1)"
    total = max((int(s) + int(l) for s, l in zip(node_start, node_len)), default=0)
    scale = scale if scale is not None else hn ** -0.5
    out = _ComposedForestAttn.apply(
        query[:total, 0], key[:total, 0], value[:total, 0], node_start, node_len, node_parent, scale
    )  # [total, np, hn]
    if sq > total:
        out = torch.cat([out, out.new_zeros(sq - total, np_, hn)], dim=0)
    return out.reshape(sq, 1, np_ * hn).contiguous()


def _forest_to_nodes(forest):
    """Expand a depth-1 ``forest`` list into flat node arrays (PackedTreeLayout structure).

    ``forest`` is ``[(token_offset, prefix_len, completion_lens), ...]``. Each group becomes a root
    node (the prefix span, parent -1) followed by one child node per completion. Node order is DFS
    (root then its children), as the fused kernel requires.
    """
    node_start, node_len, node_parent = [], [], []
    for off, prefix_len, completion_lens in forest:
        root = len(node_start)
        node_start.append(int(off)); node_len.append(int(prefix_len)); node_parent.append(-1)
        pos = int(off) + int(prefix_len)
        for c in completion_lens:
            node_start.append(pos); node_len.append(int(c)); node_parent.append(root)
            pos += int(c)
    return node_start, node_len, node_parent


def flash_composed_forest_attention(query, key, value, forest, scale=None):
    """Forest (multi-group / Case-1) shared-prefix attention -- fused, level-decomposed.

    ``forest`` is a list of ``(token_offset, prefix_len, completion_lens)``, one per group packed
    into this bin. Expanded to flat node arrays and dispatched to
    :func:`flash_composed_forest_attention_fused`, which fuses ALL groups into ``max_depth + 1``
    flash calls (depth-1 forest -> 1 self + 1 cross + merge) regardless of group count -- vs the
    old per-group loop's ``~3 * #groups`` launches, which scaled badly with many small groups.
    See :func:`flash_composed_forest_attention_loop` for the reference per-group form (kept for
    parity testing).
    """
    node_start, node_len, node_parent = _forest_to_nodes(forest)
    return flash_composed_forest_attention_fused(
        query, key, value, node_start, node_len, node_parent, scale=scale
    )


def flash_composed_forest_attention_loop(query, key, value, forest, scale=None):
    """Reference per-group forest attention (correctness baseline for the fused kernel).

    Groups are block-diagonal (a token never attends another group), so this just applies the
    validated single-group :func:`flash_composed_tree_attention` to each group's contiguous slice.
    Correct but launches ``~3 * #groups`` kernels; superseded by the fused path in production.
    """
    sq, b, np_, hn = query.shape
    assert b == 1, "shared-prefix packing uses a single packed sequence (b == 1)"
    out = query.new_zeros(sq, np_ * hn)
    for off, prefix_len, completion_lens in forest:
        total = int(prefix_len) + sum(int(c) for c in completion_lens)
        o = flash_composed_tree_attention(
            query[off : off + total], key[off : off + total], value[off : off + total],
            prefix_len, completion_lens, scale=scale,
        )  # [total, 1, np*hn]
        out[off : off + total] = o.reshape(total, np_ * hn)
    return out.reshape(sq, 1, np_ * hn).contiguous()


def run_shared_prefix_attention(
    query, key, value, *, block_mask, layout=None, forest=None, node_layout=None, scale=None
):
    """Dispatch shared-prefix attention to the selected backend.

    ``node_layout`` (a ``(node_start, node_len, node_parent)`` triple) selects the depth-general
    tree path (Case 2): an arbitrary-depth packed tree fed straight to the fused kernel.
    ``forest`` (a list of per-group ``(offset, prefix_len, completion_lens)``) selects the
    multi-group depth-1 Case-1 path; ``layout`` (a single ``(prefix_len, completion_lens)``) is the
    one-group path. All require ``flash_composed``; the hybrid path supplies only ``block_mask``
    (no layout/forest/node_layout) and always takes the flex path (warned once).
    """
    global _SP_FALLBACK_WARNED
    if sp_attention_backend() == "flash_composed":
        if node_layout is not None:
            return flash_composed_forest_attention_fused(query, key, value, *node_layout, scale=scale)
        if forest is not None:
            return flash_composed_forest_attention(query, key, value, forest, scale=scale)
        if layout is not None:
            return flash_composed_tree_attention(query, key, value, layout[0], layout[1], scale=scale)
        if not _SP_FALLBACK_WARNED:
            logger.warning(
                "NRL_SP_ATTENTION_BACKEND=flash_composed but no shared-prefix layout was "
                "provided (e.g. hybrid/Mamba path); falling back to FlexAttention."
            )
            _SP_FALLBACK_WARNED = True
    return flex_tree_attention(query, key, value, block_mask, scale=scale)


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
    # prefix-continued positions [packed_len] (P -> 0..Lp-1, each C_i -> Lp..Lp+Lc_i-1, trailing
    # pad -> 0) for position-aware RoPE; None falls back to packed-index RoPE.
    position_ids: Optional[torch.Tensor] = None
    # Full packed length fed to the model (>= total_len, padded to a tensor-parallel multiple for
    # sequence parallelism). The tree BlockMask is built over this length. None == total_len (no
    # padding). Needed explicitly because under sequence parallelism the decoder sees a
    # sequence-sharded activation (length packed_len // TP), not the full packed sequence.
    packed_len: Optional[int] = None
    # Case-1 multi-group "forest" bin: list of per-group ``(token_offset, prefix_len,
    # completion_lens)``. When set, the forward uses the forest attention (groups are
    # block-diagonal) instead of the single-group ``prefix_len``/``completion_lens`` above.
    forest: Optional[List[tuple]] = None
    # Case-2 arbitrary-depth tree bin: a ``(node_start, node_len, node_parent)`` triple (the
    # PackedTreeLayout node arrays) fed straight to the depth-general fused kernel. Supersedes
    # ``forest`` (a depth-1 forest is just a tree of stars); when set it takes precedence.
    node_layout: Optional[tuple] = None

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
