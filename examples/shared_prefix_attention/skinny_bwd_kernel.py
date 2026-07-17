"""Custom Triton backward for skinny-K non-causal cross attention (standalone).

Target: the tree kernel's cross passes — Q huge (thousands of rows/sequence),
K tiny (ancestor spans, 128..1024 rows), non-causal, GQA (32 q-heads over 8 kv
heads), head_dim 128. flash's generic backward underperforms on this shape.

Math per sequence, per q-head (dout here is the pass-scaled dox; out is the
merged output per the exact-backward trick — both supplied by the caller):
  S = q k^T * scale;  P = exp(S - lse[q])
  D = rowsum(dout * out)
  dv += P^T dout ;  dS = P * (dout v^T - D) * scale
  dk += dS^T q  ;   dq  = dS k

Harness: python skinny_bwd_kernel.py  — checks exactness vs
_flash_attn_varlen_backward and benchmarks both on cross-pass shapes.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _skinny_dq(
    q_ptr, k_ptr, v_ptr, do_ptr, o_ptr, lse_ptr, dq_ptr,
    cu_q_ptr, cu_k_ptr, scale, total_q,
    np_: tl.constexpr, ng: tl.constexpr, HN: tl.constexpr,
    BQ: tl.constexpr, BK: tl.constexpr,
):
    pid_q = tl.program_id(0)
    h = tl.program_id(1)
    seq = tl.program_id(2)
    q0 = tl.load(cu_q_ptr + seq)
    q1 = tl.load(cu_q_ptr + seq + 1)
    k0 = tl.load(cu_k_ptr + seq)
    k1 = tl.load(cu_k_ptr + seq + 1)
    if pid_q * BQ >= q1 - q0:
        return
    hk = h // (np_ // ng)
    rq = q0 + pid_q * BQ + tl.arange(0, BQ)
    qmask = rq < q1
    offs = tl.arange(0, HN)

    qv = tl.load(q_ptr + (rq[:, None] * np_ + h) * HN + offs[None, :],
                 mask=qmask[:, None], other=0.0)
    dov = tl.load(do_ptr + (rq[:, None] * np_ + h) * HN + offs[None, :],
                  mask=qmask[:, None], other=0.0)
    ov = tl.load(o_ptr + (rq[:, None] * np_ + h) * HN + offs[None, :],
                 mask=qmask[:, None], other=0.0)
    lse = tl.load(lse_ptr + h * total_q + rq, mask=qmask, other=float("inf"))
    D = tl.sum(dov.to(tl.float32) * ov.to(tl.float32), axis=1)

    acc = tl.zeros([BQ, HN], dtype=tl.float32)
    nkb = tl.cdiv(k1 - k0, BK)
    for kb in range(0, nkb):
        rk = k0 + kb * BK + tl.arange(0, BK)
        kmask = rk < k1
        kv = tl.load(k_ptr + (rk[:, None] * ng + hk) * HN + offs[None, :],
                     mask=kmask[:, None], other=0.0)
        vv = tl.load(v_ptr + (rk[:, None] * ng + hk) * HN + offs[None, :],
                     mask=kmask[:, None], other=0.0)
        S = tl.dot(qv, tl.trans(kv)).to(tl.float32) * scale
        P = tl.exp(S - lse[:, None])
        P = tl.where(kmask[None, :] & qmask[:, None], P, 0.0)
        dP = tl.dot(dov, tl.trans(vv)).to(tl.float32)
        dS = (P * (dP - D[:, None]) * scale).to(kv.dtype)
        acc += tl.dot(dS, kv).to(tl.float32)
    tl.store(dq_ptr + (rq[:, None] * np_ + h) * HN + offs[None, :],
             acc.to(dq_ptr.dtype.element_ty), mask=qmask[:, None])


@triton.jit
def _skinny_dkv(
    q_ptr, k_ptr, v_ptr, do_ptr, o_ptr, lse_ptr, dk_ptr, dv_ptr,
    cu_q_ptr, cu_k_ptr, scale, total_q,
    np_: tl.constexpr, ng: tl.constexpr, HN: tl.constexpr,
    BQ: tl.constexpr, BK: tl.constexpr,
):
    pid_k = tl.program_id(0)
    hk = tl.program_id(1)          # kv head; loops its q-head group internally
    seq = tl.program_id(2)
    q0 = tl.load(cu_q_ptr + seq)
    q1 = tl.load(cu_q_ptr + seq + 1)
    k0 = tl.load(cu_k_ptr + seq)
    k1 = tl.load(cu_k_ptr + seq + 1)
    if pid_k * BK >= k1 - k0:
        return
    rk = k0 + pid_k * BK + tl.arange(0, BK)
    kmask = rk < k1
    offs = tl.arange(0, HN)
    kv = tl.load(k_ptr + (rk[:, None] * ng + hk) * HN + offs[None, :],
                 mask=kmask[:, None], other=0.0)
    vv = tl.load(v_ptr + (rk[:, None] * ng + hk) * HN + offs[None, :],
                 mask=kmask[:, None], other=0.0)

    dk_acc = tl.zeros([BK, HN], dtype=tl.float32)
    dv_acc = tl.zeros([BK, HN], dtype=tl.float32)
    GROUP: tl.constexpr = np_ // ng
    for g in range(0, GROUP):
        h = hk * GROUP + g
        nqb = tl.cdiv(q1 - q0, BQ)
        for qb in range(0, nqb):
            rq = q0 + qb * BQ + tl.arange(0, BQ)
            qmask = rq < q1
            qv = tl.load(q_ptr + (rq[:, None] * np_ + h) * HN + offs[None, :],
                         mask=qmask[:, None], other=0.0)
            dov = tl.load(do_ptr + (rq[:, None] * np_ + h) * HN + offs[None, :],
                          mask=qmask[:, None], other=0.0)
            ov = tl.load(o_ptr + (rq[:, None] * np_ + h) * HN + offs[None, :],
                         mask=qmask[:, None], other=0.0)
            lse = tl.load(lse_ptr + h * total_q + rq, mask=qmask, other=float("inf"))
            D = tl.sum(dov.to(tl.float32) * ov.to(tl.float32), axis=1)
            S = tl.dot(qv, tl.trans(kv)).to(tl.float32) * scale
            P = tl.exp(S - lse[:, None])
            P = tl.where(qmask[:, None] & kmask[None, :], P, 0.0)
            dv_acc += tl.dot(tl.trans(P.to(dov.dtype)), dov).to(tl.float32)
            dP = tl.dot(dov, tl.trans(vv)).to(tl.float32)
            dS = (P * (dP - D[:, None]) * scale).to(qv.dtype)
            dk_acc += tl.dot(tl.trans(dS), qv).to(tl.float32)
    tl.store(dk_ptr + (rk[:, None] * ng + hk) * HN + offs[None, :],
             dk_acc.to(dk_ptr.dtype.element_ty), mask=kmask[:, None])
    tl.store(dv_ptr + (rk[:, None] * ng + hk) * HN + offs[None, :],
             dv_acc.to(dv_ptr.dtype.element_ty), mask=kmask[:, None])


@triton.jit
def _skinny_fused_bwd(
    q_ptr, k_ptr, v_ptr, do_ptr, o_ptr, lse_ptr,
    dq_ptr,            # fp32 [total_q, np, HN], pre-zeroed; atomic_add partials
    dk_ptr, dv_ptr,    # same dtype as k/v [total_k, ng, HN]; direct store per k-tile
    cu_q_ptr, cu_k_ptr, scale, total_q,
    np_: tl.constexpr, ng: tl.constexpr, HN: tl.constexpr,
    BQ: tl.constexpr, BK: tl.constexpr,
):
    # One pass over the data (P computed once per q/k block pair), like flash's
    # backward: dq via atomics across k-tiles, dk/dv in registers per k-tile.
    pid_k = tl.program_id(0)
    hk = tl.program_id(1)
    seq = tl.program_id(2)
    q0 = tl.load(cu_q_ptr + seq)
    q1 = tl.load(cu_q_ptr + seq + 1)
    k0 = tl.load(cu_k_ptr + seq)
    k1 = tl.load(cu_k_ptr + seq + 1)
    if pid_k * BK >= k1 - k0:
        return
    rk = k0 + pid_k * BK + tl.arange(0, BK)
    kmask = rk < k1
    offs = tl.arange(0, HN)
    kv = tl.load(k_ptr + (rk[:, None] * ng + hk) * HN + offs[None, :],
                 mask=kmask[:, None], other=0.0)
    vv = tl.load(v_ptr + (rk[:, None] * ng + hk) * HN + offs[None, :],
                 mask=kmask[:, None], other=0.0)
    dk_acc = tl.zeros([BK, HN], dtype=tl.float32)
    dv_acc = tl.zeros([BK, HN], dtype=tl.float32)
    GROUP: tl.constexpr = np_ // ng
    for g in range(0, GROUP):
        h = hk * GROUP + g
        nqb = tl.cdiv(q1 - q0, BQ)
        for qb in range(0, nqb):
            rq = q0 + qb * BQ + tl.arange(0, BQ)
            qmask = rq < q1
            qv = tl.load(q_ptr + (rq[:, None] * np_ + h) * HN + offs[None, :],
                         mask=qmask[:, None], other=0.0)
            dov = tl.load(do_ptr + (rq[:, None] * np_ + h) * HN + offs[None, :],
                          mask=qmask[:, None], other=0.0)
            ov = tl.load(o_ptr + (rq[:, None] * np_ + h) * HN + offs[None, :],
                         mask=qmask[:, None], other=0.0)
            lse = tl.load(lse_ptr + h * total_q + rq, mask=qmask, other=float("inf"))
            D = tl.sum(dov.to(tl.float32) * ov.to(tl.float32), axis=1)
            S = tl.dot(qv, tl.trans(kv)).to(tl.float32) * scale
            P = tl.exp(S - lse[:, None])
            P = tl.where(qmask[:, None] & kmask[None, :], P, 0.0)
            dv_acc += tl.dot(tl.trans(P.to(dov.dtype)), dov).to(tl.float32)
            dP = tl.dot(dov, tl.trans(vv)).to(tl.float32)
            dS = (P * (dP - D[:, None]) * scale).to(qv.dtype)
            dk_acc += tl.dot(tl.trans(dS), qv).to(tl.float32)
            dq_part = tl.dot(dS, kv).to(tl.float32)
            tl.atomic_add(dq_ptr + (rq[:, None] * np_ + h) * HN + offs[None, :],
                          dq_part, mask=qmask[:, None])
    tl.store(dk_ptr + (rk[:, None] * ng + hk) * HN + offs[None, :],
             dk_acc.to(dk_ptr.dtype.element_ty), mask=kmask[:, None])
    tl.store(dv_ptr + (rk[:, None] * ng + hk) * HN + offs[None, :],
             dv_acc.to(dv_ptr.dtype.element_ty), mask=kmask[:, None])


@triton.jit
def _skinny_qmajor_bwd(
    q_ptr, k_ptr, v_ptr, do_ptr, o_ptr, lse_ptr,
    dq_ptr,            # same dtype as q; direct store (each q row visited once/head)
    dk_ptr, dv_ptr,    # fp32 [total_k, ng, HN], pre-zeroed; atomic_add
    cu_q_ptr, cu_k_ptr, scale, total_q,
    np_: tl.constexpr, ng: tl.constexpr, HN: tl.constexpr,
    BQ: tl.constexpr, BK: tl.constexpr,
):
    # Q-block-major: the huge Q/dO/O side is read exactly once; the tiny K/V
    # side is re-read per q-block but stays L2-resident (a few hundred KB per
    # sequence). dk/dv contributions are flushed with atomics (~96MB total for
    # the target shapes -- negligible). P is computed once per (q,k) pair.
    pid_q = tl.program_id(0)
    h = tl.program_id(1)
    seq = tl.program_id(2)
    q0 = tl.load(cu_q_ptr + seq)
    q1 = tl.load(cu_q_ptr + seq + 1)
    k0 = tl.load(cu_k_ptr + seq)
    k1 = tl.load(cu_k_ptr + seq + 1)
    if pid_q * BQ >= q1 - q0:
        return
    hk = h // (np_ // ng)
    rq = q0 + pid_q * BQ + tl.arange(0, BQ)
    qmask = rq < q1
    offs = tl.arange(0, HN)
    qv = tl.load(q_ptr + (rq[:, None] * np_ + h) * HN + offs[None, :],
                 mask=qmask[:, None], other=0.0)
    dov = tl.load(do_ptr + (rq[:, None] * np_ + h) * HN + offs[None, :],
                  mask=qmask[:, None], other=0.0)
    ov = tl.load(o_ptr + (rq[:, None] * np_ + h) * HN + offs[None, :],
                 mask=qmask[:, None], other=0.0)
    lse = tl.load(lse_ptr + h * total_q + rq, mask=qmask, other=float("inf"))
    D = tl.sum(dov.to(tl.float32) * ov.to(tl.float32), axis=1)

    dq_acc = tl.zeros([BQ, HN], dtype=tl.float32)
    nkb = tl.cdiv(k1 - k0, BK)
    for kb in range(0, nkb):
        rk = k0 + kb * BK + tl.arange(0, BK)
        kmask = rk < k1
        kv = tl.load(k_ptr + (rk[:, None] * ng + hk) * HN + offs[None, :],
                     mask=kmask[:, None], other=0.0)
        vv = tl.load(v_ptr + (rk[:, None] * ng + hk) * HN + offs[None, :],
                     mask=kmask[:, None], other=0.0)
        S = tl.dot(qv, tl.trans(kv)).to(tl.float32) * scale
        P = tl.exp(S - lse[:, None])
        P = tl.where(qmask[:, None] & kmask[None, :], P, 0.0)
        Pc = P.to(dov.dtype)
        dv_part = tl.dot(tl.trans(Pc), dov).to(tl.float32)
        tl.atomic_add(dv_ptr + (rk[:, None] * ng + hk) * HN + offs[None, :],
                      dv_part, mask=kmask[:, None])
        dP = tl.dot(dov, tl.trans(vv)).to(tl.float32)
        dS = (P * (dP - D[:, None]) * scale).to(qv.dtype)
        dk_part = tl.dot(tl.trans(dS), qv).to(tl.float32)
        tl.atomic_add(dk_ptr + (rk[:, None] * ng + hk) * HN + offs[None, :],
                      dk_part, mask=kmask[:, None])
        dq_acc += tl.dot(dS, kv).to(tl.float32)
    tl.store(dq_ptr + (rq[:, None] * np_ + h) * HN + offs[None, :],
             dq_acc.to(dq_ptr.dtype.element_ty), mask=qmask[:, None])


def skinny_cross_backward_v3(dox, q, k, v, o, lse, cu_q, cu_k, max_q, max_k, scale,
                             BQ=128, BK=64, num_warps=8, num_stages=2):
    """Q-major single-pass backward. Returns (dq, dk fp32, dv fp32)."""
    total_q, np_, hn = q.shape
    ng = k.shape[1]
    nseq = cu_q.numel() - 1
    dq = torch.empty_like(q)
    dk = torch.zeros(k.shape, dtype=torch.float32, device=k.device)
    dv = torch.zeros(v.shape, dtype=torch.float32, device=v.device)
    _skinny_qmajor_bwd[(triton.cdiv(max_q, BQ), np_, nseq)](
        q, k, v, dox, o, lse.contiguous(), dq, dk, dv, cu_q, cu_k, scale, total_q,
        np_=np_, ng=ng, HN=hn, BQ=BQ, BK=BK,
        num_warps=num_warps, num_stages=num_stages,
    )
    return dq, dk, dv


def skinny_cross_backward_v2(dox, q, k, v, o, lse, cu_q, cu_k, max_q, max_k, scale,
                             BQ=64, BK=64, num_warps=8, num_stages=2):
    """Single-pass fused backward. Returns (dq fp32, dk, dv)."""
    total_q, np_, hn = q.shape
    ng = k.shape[1]
    nseq = cu_q.numel() - 1
    dq = torch.zeros(total_q, np_, hn, dtype=torch.float32, device=q.device)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    _skinny_fused_bwd[(triton.cdiv(max_k, BK), ng, nseq)](
        q, k, v, dox, o, lse.contiguous(), dq, dk, dv, cu_q, cu_k, scale, total_q,
        np_=np_, ng=ng, HN=hn, BQ=BQ, BK=BK,
        num_warps=num_warps, num_stages=num_stages,
    )
    return dq, dk, dv


def skinny_cross_backward(dox, q, k, v, o, lse, cu_q, cu_k, max_q, max_k, scale):
    """Drop-in for _flash_attn_varlen_backward on non-causal skinny-K passes.
    Returns (dq, dk, dv) in the dtypes of q/k/v."""
    total_q, np_, hn = q.shape
    ng = k.shape[1]
    nseq = cu_q.numel() - 1
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    BQ, BK = 64, 64
    lse_c = lse.contiguous()
    _skinny_dq[(triton.cdiv(max_q, BQ), np_, nseq)](
        q, k, v, dox, o, lse_c, dq, cu_q, cu_k, scale, total_q,
        np_=np_, ng=ng, HN=hn, BQ=BQ, BK=BK,
    )
    _skinny_dkv[(triton.cdiv(max_k, BK), ng, nseq)](
        q, k, v, dox, o, lse_c, dk, dv, cu_q, cu_k, scale, total_q,
        np_=np_, ng=ng, HN=hn, BQ=BQ, BK=BK,
    )
    return dq, dk, dv


if __name__ == "__main__":
    from flash_attn import flash_attn_varlen_func
    from flash_attn.flash_attn_interface import _flash_attn_varlen_backward

    dev, dt = torch.device("cuda"), torch.bfloat16
    NP, NG, HN = 32, 8, 128
    torch.manual_seed(0)

    # branched_mc cross-pass-like shape: 15 sequences, Q 400..24000, K 400..512
    q_lens = [2900] * 4 + [2500] * 4 + [2400, 2000, 1600, 1200, 800, 400, 400]
    k_lens = [512] * 4 + [400] * 4 + [400] * 7
    cu_q = torch.tensor([0] + list(torch.cumsum(torch.tensor(q_lens), 0)), dtype=torch.int32, device=dev)
    cu_k = torch.tensor([0] + list(torch.cumsum(torch.tensor(k_lens), 0)), dtype=torch.int32, device=dev)
    TQ, TK = int(cu_q[-1]), int(cu_k[-1])
    scale = HN ** -0.5

    q = torch.randn(TQ, NP, HN, device=dev, dtype=dt)
    k = torch.randn(TK, NG, HN, device=dev, dtype=dt)
    v = torch.randn(TK, NG, HN, device=dev, dtype=dt)
    o, lse, _ = flash_attn_varlen_func(q, k, v, cu_q, cu_k, max(q_lens), max(k_lens),
                                       softmax_scale=scale, causal=False, return_attn_probs=True)
    do = torch.randn_like(o)

    # reference
    dq_r = torch.empty_like(q); dk_r = torch.empty_like(k); dv_r = torch.empty_like(v)
    _flash_attn_varlen_backward(do, q, k, v, o, lse, dq_r, dk_r, dv_r, cu_q, cu_k,
                                max(q_lens), max(k_lens), 0.0, scale, False, -1, -1,
                                0.0, None, False, None, False)
    # custom
    dq_c, dk_c, dv_c = skinny_cross_backward(do, q, k, v, o, lse, cu_q, cu_k,
                                             max(q_lens), max(k_lens), scale)
    for nmm, a, b in (("dq", dq_c, dq_r), ("dk", dk_c, dk_r), ("dv", dv_c, dv_r)):
        rel = (a.float() - b.float()).abs().max().item() / max(b.float().abs().max().item(), 1e-6)
        print(f"{nmm} rel={rel:.3e}")

    def bench(fn, iters=30):
        fn(); torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters):
            fn()
        e.record(); torch.cuda.synchronize()
        return s.elapsed_time(e) / iters

    # v2 exactness
    dq2, dk2, dv2 = skinny_cross_backward_v2(do, q, k, v, o, lse, cu_q, cu_k,
                                             max(q_lens), max(k_lens), scale)
    for nmm, a, b in (("v2 dq", dq2, dq_r), ("v2 dk", dk2, dk_r), ("v2 dv", dv2, dv_r)):
        rel = (a.float() - b.float()).abs().max().item() / max(b.float().abs().max().item(), 1e-6)
        print(f"{nmm} rel={rel:.3e}")

    t_ref = bench(lambda: _flash_attn_varlen_backward(
        do, q, k, v, o, lse, dq_r, dk_r, dv_r, cu_q, cu_k, max(q_lens), max(k_lens),
        0.0, scale, False, -1, -1, 0.0, None, False, None, False))
    t_cus = bench(lambda: skinny_cross_backward(
        do, q, k, v, o, lse, cu_q, cu_k, max(q_lens), max(k_lens), scale))
    best = (None, float("inf"))
    for cfg in ((64, 64, 8, 2), (128, 64, 8, 1), (64, 64, 4, 3), (128, 64, 4, 2), (256, 64, 8, 1)):
        bq, bk, w, s = cfg
        try:
            t = bench(lambda: skinny_cross_backward_v2(
                do, q, k, v, o, lse, cu_q, cu_k, max(q_lens), max(k_lens), scale,
                BQ=bq, BK=bk, num_warps=w, num_stages=s))
            print(f"v2 cfg BQ={bq} BK={bk} warps={w} stages={s}: {t:.3f}ms ({t_ref / t:.2f}x)")
            if t < best[1]:
                best = (cfg, t)
        except Exception as ex:
            print(f"v2 cfg {cfg}: FAILED {type(ex).__name__}")
    print(f"flash_bwd={t_ref:.3f}ms v1={t_cus:.3f}ms v2_best={best[1]:.3f}ms cfg={best[0]}"
          f" v2_speedup={t_ref / best[1]:.2f}x")

    # v3: q-major
    dq3, dk3, dv3 = skinny_cross_backward_v3(do, q, k, v, o, lse, cu_q, cu_k,
                                             max(q_lens), max(k_lens), scale)
    for nmm, a, b in (("v3 dq", dq3, dq_r), ("v3 dk", dk3, dk_r), ("v3 dv", dv3, dv_r)):
        rel = (a.float() - b.float()).abs().max().item() / max(b.float().abs().max().item(), 1e-6)
        print(f"{nmm} rel={rel:.3e}")
    best3 = (None, float("inf"))
    for cfg in ((128, 64, 8, 2), (64, 64, 8, 2), (128, 128, 8, 1), (64, 128, 8, 2), (128, 64, 4, 2)):
        bq, bk, w, s = cfg
        try:
            t = bench(lambda: skinny_cross_backward_v3(
                do, q, k, v, o, lse, cu_q, cu_k, max(q_lens), max(k_lens), scale,
                BQ=bq, BK=bk, num_warps=w, num_stages=s))
            print(f"v3 cfg BQ={bq} BK={bk} warps={w} stages={s}: {t:.3f}ms ({t_ref / t:.2f}x)")
            if t < best3[1]:
                best3 = (cfg, t)
        except Exception as ex:
            print(f"v3 cfg {cfg}: FAILED {type(ex).__name__}")
    print(f"V3_BEST={best3[1]:.3f}ms cfg={best3[0]} speedup={t_ref / best3[1]:.2f}x")
    print("SKINNY_DONE")
