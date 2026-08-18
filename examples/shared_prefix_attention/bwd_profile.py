"""Backward component attribution for the fused tree attention.

Times, per pass: the dox scale-gather kernel, the q/k/v/o gathers, the flash
backward, and the scatter-accumulates + the init-from-self float() conversions.
Streams disabled for attribution. Compares against the block-diag flash
backward for the reference.
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
from flash_attn.flash_attn_interface import _flash_attn_varlen_backward  # noqa: E402

dev, dt = torch.device("cuda"), torch.bfloat16
ITERS = 20


def timed(fn):
    fn()  # warmup (includes any Triton JIT)
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(ITERS):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / ITERS


for name, builder in SHAPES.items():
    node_start, node_len, node_parent, total, base_rows = builder()
    torch.manual_seed(0)
    q = torch.randn(total, NP, HN, device=dev, dtype=dt)
    k = torch.randn(total, NG, HN, device=dev, dtype=dt)
    v = torch.randn(total, NG, HN, device=dev, dtype=dt)
    do = torch.randn(total, NP, HN, device=dev, dtype=dt)
    scale = HN ** -0.5

    total_c, passes, inv_maps, qidx64, slot_pass = sp._forest_attention_plan_cached(
        node_start, node_len, node_parent, dev
    )
    # forward once to get outs/lses/o_merged/lse_final
    outs, lses = [], []
    for (q_idx, k_idx, cu_q, cu_k, mxq, mxk, causal) in passes:
        qx = sp._sel_rows(q, q_idx)
        kx = k if k_idx is None else k.index_select(0, k_idx)
        vx = v if k_idx is None else v.index_select(0, k_idx)
        o, lse, _ = flash_attn_varlen_func(qx, kx, vx, cu_q, cu_k, mxq, mxk,
                                           softmax_scale=scale, causal=causal,
                                           return_attn_probs=True)
        outs.append(o)
        lses.append(lse)
    o_merged, lse_final = sp._merge_passes_triton(
        slot_pass, inv_maps, outs, lses, total, NP, HN, dt, dev)

    print(f"\n=== {name} passes={len(passes)} total={total} ===")
    t_sg = t_gath = t_fb = t_scat = 0.0
    for i, ((q_idx, k_idx, cu_q, cu_k, mxq, mxk, causal), lse) in enumerate(zip(passes, lses)):
        qidx = qidx64[i]
        rows = qidx.numel()
        lse_c = lse.contiguous()

        dox = torch.empty(rows, NP, HN, dtype=dt, device=dev)
        t1 = timed(lambda: sp._sp_scale_gather_kernel[((rows + 15) // 16, NP)](
            do, lse_c, lse_final, qidx, dox, rows, total, np_=NP, HN=HN))

        def gathers():
            qx = sp._sel_rows(q, q_idx)
            kx, vx = (k, v) if k_idx is None else sp._gather_kv(k, v, k_idx)
            ox = sp._sel_rows(o_merged, q_idx)
            return qx, kx, vx, ox
        t2 = timed(gathers)
        qx, kx, vx, ox = gathers()

        dqx, dkx, dvx = torch.empty_like(qx), torch.empty_like(kx), torch.empty_like(vx)
        t3 = timed(lambda: _flash_attn_varlen_backward(
            dox, qx, kx, vx, ox, lse, dqx, dkx, dvx, cu_q, cu_k, mxq, mxk,
            0.0, scale, causal, -1, -1, 0.0, None, False, None, False))

        if i == 0:
            t4 = timed(lambda: (dqx.float(), dkx.float(), dvx.float()))
        else:
            dq = torch.zeros(total, NP, HN, dtype=torch.float32, device=dev)
            dk = torch.zeros(total, NG, HN, dtype=torch.float32, device=dev)
            dv = torch.zeros(total, NG, HN, dtype=torch.float32, device=dev)
            kro = k_idx.numel()
            t4 = timed(lambda: (
                sp._sp_scatter_accum_kernel[((rows + 15) // 16, NP)](dq, dqx, qidx, rows, n_=NP, HN=HN),
                sp._sp_scatter_accum_kernel[((kro + 15) // 16, NG)](dk, dkx, k_idx, kro, n_=NG, HN=HN),
                sp._sp_scatter_accum_kernel[((kro + 15) // 16, NG)](dv, dvx, k_idx, kro, n_=NG, HN=HN),
            ))
        print(f"  pass{i} rows={rows:>7} scale_gather={t1:6.3f} gathers={t2:6.3f}"
              f" flash_bwd={t3:6.3f} accum={t4:6.3f} ms")
        t_sg += t1; t_gath += t2; t_fb += t3; t_scat += t4

    # block-diag backward reference
    base_tokens = sum(base_rows)
    cu = torch.tensor([0] + list(torch.cumsum(torch.tensor(base_rows), 0).tolist()),
                      dtype=torch.int32, device=dev)
    maxlen = max(base_rows)
    qb = torch.randn(base_tokens, NP, HN, device=dev, dtype=dt)
    kb = torch.randn(base_tokens, NG, HN, device=dev, dtype=dt)
    vb = torch.randn(base_tokens, NG, HN, device=dev, dtype=dt)
    ob, lseb, _ = flash_attn_varlen_func(qb, kb, vb, cu, cu, maxlen, maxlen,
                                         softmax_scale=scale, causal=True, return_attn_probs=True)
    dob = torch.randn_like(ob)
    dqb, dkb, dvb = torch.empty_like(qb), torch.empty_like(kb), torch.empty_like(vb)
    t_bd = timed(lambda: _flash_attn_varlen_backward(
        dob, qb, kb, vb, ob, lseb, dqb, dkb, dvb, cu, cu, maxlen, maxlen,
        0.0, scale, True, -1, -1, 0.0, None, False, None, False))
    t_total = t_sg + t_gath + t_fb + t_scat
    print(f"  SUM: scale_gather={t_sg:.3f} gathers={t_gath:.3f} flash_bwd={t_fb:.3f}"
          f" accum={t_scat:.3f} -> fused_bwd={t_total:.3f}ms | bd_bwd={t_bd:.3f}ms"
          f" (ratio {t_bd / t_total:.2f}x)")
print("\nBWD_PROFILE_DONE")
