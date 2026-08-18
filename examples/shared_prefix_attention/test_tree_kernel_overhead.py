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

"""Regression + optimization target for the fused shared-prefix tree-attention kernel.

Pins the per-token overhead of ``flash_composed_forest_attention_fused`` vs plain block-diagonal
flash so kernel edits can't silently make it worse, and marks the optimization goal (xfail until
met). Shares the exact shapes/measurement with ``bench_tree_kernel.py`` in this directory.

Per-token overhead = fused-tree fwd+bwd per token / block-diagonal-flash fwd+bwd per token, on the
SAME leaf trajectories (best-of-N). ``net_speedup = dup / overhead``; shared-prefix wins on attention
only when ``dup > overhead``. The overhead is the fp32 online-softmax merge + per-depth cross passes
+ index gathers -- NOT real FLOPs (the tree does fewer) -- so there is headroom; the target is ~1x.

Measured GB300: balanced_d4 ~4.5x, branched_mc ~3.1x (deeper but cheaper -- long segments amortize
the per-depth launches). CEILINGs are slack over those (best-of timing is stable).

GPU + flash_attn + FlexAttention required (skipped otherwise); bf16. Run:
  python -m pytest examples/shared_prefix_attention/test_tree_kernel_overhead.py -s
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))
from bench_tree_kernel import SHAPES, measure  # noqa: E402  (sibling module)

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

# Per-shape overhead guards (regression ceiling) and the shared optimization target.
OVERHEAD_TARGET = 2.0
OVERHEAD_CEILING = {"balanced_d4": 6.0, "branched_mc": 4.5}
SHARDED_SHAPES = list(OVERHEAD_CEILING)


def _measure(shape):
    pytest.importorskip("flash_attn")
    sp = pytest.importorskip("megatron.core.models.hybrid.shared_prefix")
    if not sp.HAVE_FLEX_ATTENTION:
        pytest.skip("FlexAttention unavailable in this torch build")
    return measure(sp, shape, SHAPES[shape], iters=6, warmup=8, dev=torch.device("cuda"), dt=torch.bfloat16)


@requires_cuda
@pytest.mark.parametrize("shape", SHARDED_SHAPES)
def test_tree_kernel_overhead_does_not_regress(shape):
    """Hard guard: the kernel stays exact and per-token overhead stays under the shape's CEILING."""
    r = _measure(shape)
    assert r["maxdiff"] < 2e-2, f"{shape}: fused diverged from FlexAttention (max|Δ|={r['maxdiff']:.2e}) -- exactness broke"
    assert r["overhead"] <= OVERHEAD_CEILING[shape], (
        f"{shape}: per-token overhead {r['overhead']:.2f}x exceeds ceiling {OVERHEAD_CEILING[shape]}x -- regressed"
    )


@requires_cuda
@pytest.mark.parametrize("shape", SHARDED_SHAPES)
@pytest.mark.xfail(strict=False, reason="tree-kernel per-token overhead optimization not yet done")
def test_tree_kernel_overhead_meets_target(shape):
    """Optimization goal (xfail until met): drive per-token overhead to TARGET. XPASSES when closed."""
    r = _measure(shape)
    assert r["maxdiff"] < 2e-2, f"{shape}: optimization must preserve exactness"
    assert r["overhead"] <= OVERHEAD_TARGET, (
        f"{shape}: per-token overhead {r['overhead']:.2f}x has not reached target {OVERHEAD_TARGET}x"
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-s", "-v"]))
