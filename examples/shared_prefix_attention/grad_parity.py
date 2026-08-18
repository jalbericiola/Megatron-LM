"""Gradient parity: Triton backward glue vs the proven eager backward, both shapes.

The forward maxdiff gate does not exercise the backward kernels, so compare
dq/dk/dv between NRL_SP_FUSED_MERGE paths directly (same inputs, same seed).
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from examples.shared_prefix_attention.bench_tree_kernel import SHAPES, NP, NG, HN  # noqa: E402
import megatron.core.models.hybrid.shared_prefix_fused as sp  # noqa: E402

dev, dt = torch.device("cuda"), torch.bfloat16
worst = 0.0
for name, builder in SHAPES.items():
    node_start, node_len, node_parent, total, _ = builder()
    torch.manual_seed(7)
    q = torch.randn(total, 1, NP, HN, device=dev, dtype=dt, requires_grad=True)
    k = torch.randn(total, 1, NG, HN, device=dev, dtype=dt, requires_grad=True)
    v = torch.randn(total, 1, NG, HN, device=dev, dtype=dt, requires_grad=True)
    do = torch.randn(total, 1, NP * HN, device=dev, dtype=dt)

    grads = {}
    for flag in (True, False):
        sp._SP_FUSED_MERGE = flag
        q.grad = k.grad = v.grad = None
        out = sp.flash_composed_forest_attention_fused(q, k, v, node_start, node_len, node_parent)
        out.backward(do)
        grads[flag] = (q.grad.float().clone(), k.grad.float().clone(), v.grad.float().clone())

    for label, a, b in zip("qkv", grads[True], grads[False]):
        diff = (a - b).abs().max().item()
        ref = b.abs().max().item()
        rel = diff / max(ref, 1e-6)
        worst = max(worst, rel)
        print(f"{name} d{label}: maxabs={diff:.3e} rel={rel:.3e}")

print(f"WORST_REL={worst:.3e}")
print("GRAD_PARITY_PASS" if worst < 5e-3 else "GRAD_PARITY_FAIL")
