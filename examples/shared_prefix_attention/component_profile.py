"""Component-level time attribution for the fused tree attention (forward path).

Times each piece with CUDA events on the default stream (streams disabled for
attribution — overlap would blur the per-component costs): per-pass gathers,
per-pass flash calls, the Triton merge. Then times the flash calls of the
block-diag baseline for the efficiency reference. Answers: where do the
remaining milliseconds live?
"""
import os
import sys

os.environ["NRL_SP_STREAMS"] = "0"
os.environ["NRL_SP_COMBINE"] = "0"

import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from examples.shared_prefix_attention.bench_tree_kernel import SHAPES, NP, NG, HN  # noqa: E402
import megatron.core.models.hybrid.shared_prefix_fused as sp  # noqa: E402
from flash_attn import flash_attn_varlen_func  # noqa: E402

dev, dt = torch.device("cuda"), torch.bfloat16
ITERS = 20


def timed(fn, *a, **k):
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(ITERS):
        out = fn(*a, **k)
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / ITERS, out


for name, builder in SHAPES.items():
    node_start, node_len, node_parent, total, base_rows = builder()
    torch.manual_seed(0)
    q = torch.randn(total, NP, HN, device=dev, dtype=dt)
    k = torch.randn(total, NG, HN, device=dev, dtype=dt)
    v = torch.randn(total, NG, HN, device=dev, dtype=dt)

    total_c, passes, inv_maps, qidx64, slot_pass = sp._forest_attention_plan_cached(
        node_start, node_len, node_parent, dev
    )
    scale = HN ** -0.5

    print(f"\n=== {name} passes={len(passes)} total={total} ===")
    t_gather_all = 0.0
    t_flash_all = 0.0
    outs, lses = [], []
    for i, (q_idx, k_idx, cu_q, cu_k, mxq, mxk, causal) in enumerate(passes):
        if q_idx is None:
            qx, kx, vx = q, k, v
            t_g = 0.0
        else:
            def gather():
                a = sp._sel_rows(q, q_idx)
                b, c = sp._gather_kv(k, v, k_idx)
                return a, b, c
            t_g, (qx, kx, vx) = timed(gather)
        t_f, res = timed(
            flash_attn_varlen_func, qx, kx, vx, cu_q, cu_k, mxq, mxk,
            softmax_scale=scale, causal=causal, return_attn_probs=True,
        )
        o, lse, _ = res
        outs.append(o)
        lses.append(lse)
        rows = qx.shape[0]
        print(f"  pass{i} rows={rows:>7} gather={t_g:6.3f}ms flash={t_f:6.3f}ms")
        t_gather_all += t_g
        t_flash_all += t_f

    t_merge, _ = timed(
        sp._merge_passes_triton, slot_pass, inv_maps, outs, lses, total, NP, HN, dt, dev
    )
    base_tokens = sum(base_rows)
    cu = torch.tensor([0] + list(torch.cumsum(torch.tensor(base_rows), 0).tolist()),
                      dtype=torch.int32, device=dev)
    qb = torch.randn(base_tokens, NP, HN, device=dev, dtype=dt)
    kb = torch.randn(base_tokens, NG, HN, device=dev, dtype=dt)
    vb = torch.randn(base_tokens, NG, HN, device=dev, dtype=dt)
    t_bd, _ = timed(
        flash_attn_varlen_func, qb, kb, vb, cu, cu, max(base_rows), max(base_rows), causal=True
    )
    t_total = t_gather_all + t_flash_all + t_merge
    print(f"  SUM: gathers={t_gather_all:.3f}ms flash={t_flash_all:.3f}ms merge={t_merge:.3f}ms"
          f" -> fused_fwd={t_total:.3f}ms | bd_fwd={t_bd:.3f}ms (ratio {t_bd / t_total:.2f}x)")
print("\nPROFILE_DONE")
