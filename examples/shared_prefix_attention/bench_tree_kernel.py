# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Self-contained benchmark for the fused shared-prefix (tree) attention kernel.

Goal: give a faithful, standalone harness to drive down the per-token cost of
``flash_composed_forest_attention_fused`` (in
``megatron/core/models/hybrid/shared_prefix.py``). Depends only on that module +
``flash_attn`` + ``torch`` (FlexAttention for the exactness oracle) -- no NeMo-RL.

For each tree SHAPE we compare, on identical work (the same leaf trajectories):
  * FUSED tree    -- flash_composed_forest_attention_fused over the DEDUPLICATED tree
                     (prompt/shared spine stored once): ``tree_tokens`` tokens.
  * BLOCK-DIAG    -- plain causal flash_attn_varlen over the SAME trajectories fully
                     expanded (no sharing; prompt re-encoded per leaf): ``base_tokens``.

Reported per shape (fwd+bwd, best-of-N):
  dup            = base_tokens / tree_tokens              (token reduction from sharing)
  per_tok_overhead = (fused_ms/tree_tokens) / (bd_ms/base_tokens)   (kernel penalty)
  net_speedup    = bd_ms / fused_ms  == dup / per_tok_overhead      (the bottom line)
  exact          = fused matches FlexAttention tree mask (correctness gate)

Throughput model: shared-prefix wins on attention iff ``dup > per_tok_overhead``.
The fused kernel does FEWER attention FLOPs than the expanded baseline, so the
overhead is merge/gather/launch, not real compute -- that's the headroom to close.

Run on a GPU node (flash_attn is bf16-only):
  python examples/shared_prefix_attention/bench_tree_kernel.py
  python examples/shared_prefix_attention/bench_tree_kernel.py --iters 10 --shapes branched_mc
"""

from __future__ import annotations

import argparse

import torch

# GQA dims (Qwen3-4B-like): 32 query heads over 8 kv heads, head_dim 128.
NP, NG, HN = 32, 8, 128


# --------------------------------------------------------------------------------------
# Tree shapes -> (node_start, node_len, node_parent, total_tokens, baseline_row_lens)
# baseline_row_lens = per trained-trajectory root->leaf path length (the no-sharing bin).
# Node arrays MUST be DFS-preorder (the fused kernel asserts it).
# --------------------------------------------------------------------------------------
def _finish(parent, length, leaf_rows):
    start, acc = [], 0
    for ln in length:
        start.append(acc)
        acc += ln
    return start, length, parent, acc, leaf_rows


def balanced_tree(P=1024, L=512, B=2, D=4):
    """Balanced tree: root len P, every other node len L, B children, depth D (the existing shape)."""
    parent, length = [], []
    path_len = {}  # node -> root..node token length

    def dfs(p, d, plen):
        i = len(parent)
        parent.append(p)
        ln = P if d == 0 else L
        length.append(ln)
        path_len[i] = plen + ln
        if d < D:
            for _ in range(B):
                dfs(i, d + 1, path_len[i])

    dfs(-1, 0, 0)
    is_parent = set(parent)
    leaf_rows = [path_len[i] for i in range(len(parent)) if i not in is_parent]
    return _finish(parent, length, leaf_rows)


def branched_mc_tree(P=512, seg=400, n_boundaries=6, sib_len=2900, n_sib=4,
                     branch_len=2500, branch_boundaries=(1, 2, 3, 4)):
    """Branched-MC shape: prompt root -> a seed spine of ``n_boundaries`` segments (a chain),
    with ``n_sib`` siblings forking at the root and one branch forking off the seed segment at
    each boundary in ``branch_boundaries``. Deep (spine depth ~n_boundaries), which is what makes
    the fused kernel's per-depth cross passes expensive -- the regime our live runs hit.

    Trained trajectories (the block-diag baseline rows): the seed (prompt+full spine), each sibling
    (prompt+sib_len), and each branch (prompt + spine-up-to-its-boundary + branch_len).
    """
    parent, length = [], []

    def add(p, ln):
        i = len(parent)
        parent.append(p)
        length.append(ln)
        return i

    branch_set = set(branch_boundaries)
    root = add(-1, P)               # prompt

    # CHAIN-FIRST DFS preorder (still a valid preorder -- children order is free): each node's
    # continuation child is emitted immediately after it, so the root+spine is one adjacent run
    # (a pure causal sequence for the chain-first plan) and every branch/sibling's ancestor set
    # is a contiguous prefix [0, boundary_end). Branches unwind after the spine, siblings last.
    def build_spine(parent_node, k):
        s = add(parent_node, seg)
        if k < n_boundaries:
            build_spine(s, k + 1)
        if k in branch_set:
            add(s, branch_len)      # branch continuation off this boundary

    build_spine(root, 1)
    for _ in range(n_sib):          # siblings fork at the root (emitted after the spine subtree)
        add(root, sib_len)

    # baseline row lengths (root->leaf path per trained trajectory)
    seed_path = P + n_boundaries * seg
    rows = [seed_path]                             # the seed trajectory
    rows += [P + sib_len] * n_sib                  # siblings
    rows += [P + b * seg + branch_len for b in branch_boundaries]  # branches
    return _finish(parent, length, rows)


SHAPES = {
    "balanced_d4": lambda: balanced_tree(),
    "branched_mc": lambda: branched_mc_tree(),
}


def replicate(shape, K):
    """Replicate a shape K times into a K-tree forest (more trajectories, SAME per-tree shape).

    Grows total tokens by adding trajectories -- NOT by lengthening sequences -- which is the
    packing regime (attention time is linear in token count at fixed per-sequence length). Used by
    --sweep to verify per-token cost is flat, so the overhead ratio is bin-size independent.
    """
    node_start, node_len, node_parent, _total, base_rows = shape
    n = len(node_len)
    nl, par = [], []
    for r in range(K):
        for i in range(n):
            nl.append(node_len[i])
            par.append(-1 if node_parent[i] == -1 else node_parent[i] + r * n)
    return _finish([p for p in par], nl, base_rows * K)


# --------------------------------------------------------------------------------------
# Timing + measurement
# --------------------------------------------------------------------------------------
def _best_fwd_bwd_ms(fn, q, k, v, iters, warmup):
    for _ in range(warmup):
        q.grad = k.grad = v.grad = None
        fn(q, k, v).float().sum().backward()
    best = float("inf")
    for _ in range(iters):
        q.grad = k.grad = v.grad = None
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn(q, k, v).float().sum().backward()
        e.record()
        torch.cuda.synchronize()
        best = min(best, s.elapsed_time(e))
    return best


def _best_fwd_ms(fn, q, k, v, iters, warmup):
    """Forward-only best-of-N ms (no backward). This is the logprob-phase cost: logprob is a single
    forward, so its SP penalty tracks the FORWARD overhead, whereas training (fwd+bwd) dilutes it
    with the backward's non-attention weight-gradient GEMMs."""
    with torch.no_grad():
        for _ in range(warmup):
            fn(q, k, v)
        best = float("inf")
        for _ in range(iters):
            torch.cuda.synchronize()
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            fn(q, k, v)
            e.record()
            torch.cuda.synchronize()
            best = min(best, s.elapsed_time(e))
    return best


def _attn_pairs_tree(node_len, node_parent):
    """Unmasked (query, key) attention pairs for the tree mask: each node attends its own tokens
    causally + every ancestor node's tokens fully. This is the ACTUAL attention math the mask
    implies -- fewer than the expanded baseline, since a shared prefix is attended once."""
    n = len(node_len)
    anc_sum = [0] * n  # total length of a node's strict ancestors
    for i in range(n):
        s, p = 0, node_parent[i]
        while p != -1:
            s += node_len[p]
            p = node_parent[p]
        anc_sum[i] = s
    return sum(node_len[i] * (node_len[i] + 1) // 2 + node_len[i] * anc_sum[i] for i in range(n))


def _attn_pairs_blockdiag(rows):
    """Unmasked (q,k) pairs for independent causal sequences (the no-sharing baseline)."""
    return sum(L * (L + 1) // 2 for L in rows)


def _tflops(pairs, ms, *, backward):
    """Approx achieved attention TFLOP/s. Per (q,k) pair per query-head: QK^T (2d) + AV (2d) = 4d
    fwd; backward ~2.5x fwd. Constant cancels in fused-vs-flash ratios, so absolute value is
    approximate but the RATIO (kernel efficiency) is meaningful."""
    fwd = 4.0 * HN * pairs * NP
    flop = (3.5 * fwd) if backward else fwd   # fwd+bwd = fwd + 2.5*fwd
    return flop / (ms / 1e3) / 1e12


def _flex_exact(sp, q, k, v, node_start, node_len, node_parent, total, dev):
    from megatron.core.models.hybrid import shared_prefix_fused as spf
    """max|fused - FlexAttention(tree mask)| over all rows (correctness gate)."""
    import torch.nn.attention.flex_attention as flex

    seg = []
    for i, ln in enumerate(node_len):
        seg += [i] * ln
    n = len(node_parent)
    anc = torch.zeros(n, n, dtype=torch.bool, device=dev)
    for a in range(n):
        x = a
        while x != -1:
            anc[a, x] = True
            x = node_parent[x]
    seg_t = torch.tensor(seg, device=dev)

    def mask_mod(b, h, qi, ki):
        return anc[seg_t[qi], seg_t[ki]] & (ki <= qi)

    bm = flex.create_block_mask(mask_mod, B=1, H=None, Q_LEN=total, KV_LEN=total, device=dev)
    o_fused = spf.flash_composed_forest_attention_fused(q, k, v, node_start, node_len, node_parent).squeeze(1)
    o_flex = sp.flex_tree_attention(q, k, v, bm).squeeze(1)
    return (o_fused.float() - o_flex.float()).abs().max().item()


def measure(sp, name, builder, iters, warmup, dev, dt, check_exact=True):
    from flash_attn import flash_attn_varlen_func

    node_start, node_len, node_parent, total, base_rows = builder()
    depth_max = 0
    for i in range(len(node_parent)):
        d, p = 0, node_parent[i]
        while p != -1:
            d += 1
            p = node_parent[p]
        depth_max = max(depth_max, d)

    torch.manual_seed(0)
    # exactness on non-grad tensors (the FlexAttention oracle is O(total^2) memory, so skip it for
    # very large replicated shapes -- exactness is shape-topology-dependent, not size-dependent).
    maxdiff = float("nan")
    if check_exact:
        q0 = torch.randn(total, 1, NP, HN, device=dev, dtype=dt)
        k0 = torch.randn(total, 1, NG, HN, device=dev, dtype=dt)
        v0 = torch.randn(total, 1, NG, HN, device=dev, dtype=dt)
        maxdiff = _flex_exact(sp, q0, k0, v0, node_start, node_len, node_parent, total, dev)

    # FUSED tree over the deduplicated tree -- fwd+bwd (training) AND fwd-only (logprob).
    def mkg(h):
        return torch.randn(total, 1, h, HN, device=dev, dtype=dt, requires_grad=True)

    from megatron.core.models.hybrid import shared_prefix_fused as spf
    fused_fn = lambda a, b, c: spf.flash_composed_forest_attention_fused(  # noqa: E731
        a, b, c, node_start, node_len, node_parent)
    qf, kf, vf = mkg(NP), mkg(NG), mkg(NG)
    t_fused = _best_fwd_bwd_ms(fused_fn, qf, kf, vf, iters, warmup)
    t_fused_fwd = _best_fwd_ms(fused_fn, qf, kf, vf, iters, warmup)

    # BLOCK-DIAG baseline: causal flash_varlen over the same leaves fully expanded (no sharing)
    base_tokens = sum(base_rows)
    cu = torch.tensor([0] + list(torch.cumsum(torch.tensor(base_rows), 0).tolist()),
                      dtype=torch.int32, device=dev)
    maxlen = max(base_rows)
    qb = torch.randn(base_tokens, NP, HN, device=dev, dtype=dt, requires_grad=True)
    kb = torch.randn(base_tokens, NG, HN, device=dev, dtype=dt, requires_grad=True)
    vb = torch.randn(base_tokens, NG, HN, device=dev, dtype=dt, requires_grad=True)
    bd_fn = lambda a, b, c: flash_attn_varlen_func(a, b, c, cu, cu, maxlen, maxlen, causal=True)  # noqa: E731
    t_bd = _best_fwd_bwd_ms(bd_fn, qb, kb, vb, iters, warmup)
    t_bd_fwd = _best_fwd_ms(bd_fn, qb, kb, vb, iters, warmup)

    dup = base_tokens / total
    # per-token overhead: fwd+bwd tracks TRAINING; fwd-only tracks LOGPROB (single forward, so its
    # SP penalty is higher -- the backward's non-attention weight-grad GEMMs dilute it in training).
    overhead_fb = (t_fused / total) / (t_bd / base_tokens)
    overhead_fwd = (t_fused_fwd / total) / (t_bd_fwd / base_tokens)
    net = t_bd / t_fused
    net_fwd = t_bd_fwd / t_fused_fwd

    # FLOP lens: the tree mask implies FEWER (q,k) pairs than the expanded baseline (shared prefix
    # attended once). flop_ceiling = the net speedup you'd get IF the tree kernel ran at flash's
    # achieved TFLOP/s -- i.e. the fastest we could go. achieved TFLOP/s (fused vs bd) is the kernel
    # efficiency gap; net == flop_ceiling * (fused_tflops / bd_tflops).
    pairs_tree = _attn_pairs_tree(node_len, node_parent)
    pairs_bd = _attn_pairs_blockdiag(base_rows)
    flop_ceiling = pairs_bd / pairs_tree
    return dict(name=name, nodes=len(node_len), rows=len(base_rows), depth=depth_max,
                tree_tokens=total, base_tokens=base_tokens, dup=dup,
                fused_ms=t_fused, bd_ms=t_bd, overhead=overhead_fb, overhead_fwd=overhead_fwd,
                net=net, net_fwd=net_fwd, maxdiff=maxdiff,
                pairs_tree=pairs_tree, pairs_bd=pairs_bd, flop_ceiling=flop_ceiling,
                fused_tflops=_tflops(pairs_tree, t_fused, backward=True),
                bd_tflops=_tflops(pairs_bd, t_bd, backward=True),
                fused_tflops_fwd=_tflops(pairs_tree, t_fused_fwd, backward=False),
                bd_tflops_fwd=_tflops(pairs_bd, t_bd_fwd, backward=False))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shapes", default="balanced_d4,branched_mc",
                    help="comma list of: " + ",".join(SHAPES))
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--sweep", default="",
                    help="comma list of replication factors K (e.g. 1,2,4,8): replicate each shape "
                         "into a K-tree forest and report per-token cost, to verify it is flat "
                         "(=> the overhead ratio is bin-size independent; no iso-bin mode needed)")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: run on a GPU node.")
        return 1
    try:
        import flash_attn  # noqa: F401
    except Exception:
        print("ERROR: flash_attn required.")
        return 1
    from megatron.core.models.hybrid import shared_prefix as sp

    if not sp.HAVE_FLEX_ATTENTION:
        print("ERROR: FlexAttention unavailable (need torch >= 2.5).")
        return 1

    dev = torch.device("cuda")
    dt = torch.bfloat16
    shapes = [s.strip() for s in args.shapes.split(",") if s.strip()]
    for nm in shapes:
        if nm not in SHAPES:
            print(f"  (unknown shape {nm}; known: {list(SHAPES)})")
            return 1
    print(f"# device={torch.cuda.get_device_name()} dtype=bf16 heads={NP} kv_heads={NG} head_dim={HN} "
          f"iters={args.iters}")

    if args.sweep:
        # Scaling check: replicate each shape into a K-tree forest (more trajectories, fixed per-tree
        # shape) and report per-token cost. Flat per-token across K => attention time is linear in
        # total tokens for BOTH paths => the overhead ratio is bin-size independent.
        Ks = [int(x) for x in args.sweep.split(",") if x.strip()]
        print("# SCALING SWEEP: replicate KxK (fixed per-tree shape). Flat per-token => linear regime.\n")
        hdr = ["shape", "K", "tree_tok", "base_tok", "fused_ns/tok", "bd_ns/tok", "per_tok_ovh"]
        print("  ".join(f"{c:>13}" for c in hdr))
        for nm in shapes:
            base = SHAPES[nm]()
            for K in Ks:
                r = measure(sp, nm, (lambda b=base, k=K: replicate(b, k)), args.iters, args.warmup,
                            dev, dt, check_exact=(K == 1))
                fpt = r["fused_ms"] / r["tree_tokens"] * 1e6
                bpt = r["bd_ms"] / r["base_tokens"] * 1e6
                vals = [nm, K, r["tree_tokens"], r["base_tokens"],
                        f"{fpt:.1f}", f"{bpt:.1f}", f"{r['overhead']:.2f}x"]
                print("  ".join(f"{str(v):>13}" for v in vals))
        print("\n# per-token flat across K => single-shape overhead is representative (bin-size independent).")
        return 0

    for nm in shapes:
        r = measure(sp, nm, SHAPES[nm], args.iters, args.warmup, dev, dt)
        eff_fb = 100 * r["fused_tflops"] / r["bd_tflops"]
        eff_fwd = 100 * r["fused_tflops_fwd"] / r["bd_tflops_fwd"]
        print(f"\n=== {r['name']}  nodes={r['nodes']} rows={r['rows']} depth={r['depth']} "
              f"tree_tok={r['tree_tokens']} base_tok={r['base_tokens']} dup={r['dup']:.2f} "
              f"exact(maxdiff_flex)={r['maxdiff']:.1e} ===")
        print(f"  attention pairs: tree={r['pairs_tree']:,} blockdiag={r['pairs_bd']:,}"
              f"  -> FLOP ceiling (max net if kernel hit flash TFLOP/s) = {r['flop_ceiling']:.2f}x")
        print(f"  fwd+bwd (TRAINING): fused={r['fused_ms']:.2f}ms bd={r['bd_ms']:.2f}ms  net={r['net']:.2f}x"
              f"  per_tok_ovh={r['overhead']:.2f}x  TFLOP/s fused={r['fused_tflops']:.0f}"
              f" bd={r['bd_tflops']:.0f} (eff {eff_fb:.0f}%)")
        print(f"  fwd-only (LOGPROB): net={r['net_fwd']:.2f}x"
              f"  per_tok_ovh={r['overhead_fwd']:.2f}x  TFLOP/s fused={r['fused_tflops_fwd']:.0f}"
              f" bd={r['bd_tflops_fwd']:.0f} (eff {eff_fwd:.0f}%)")
    print("\n# net = FLOP_ceiling * (fused_TFLOP/s / bd_TFLOP/s).")
    print("#   FLOP_ceiling = speedup if the tree kernel ran at flash's achieved TFLOP/s (best case,")
    print("#   from the sparser tree mask); eff% = how close it is -> the kernel work left to do.")
    print("# per_tok_ovh is the per-token view (fwd-only=logprob, fwd+bwd=training).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
