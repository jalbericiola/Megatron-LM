# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Functional tests for context-parallel logprob computation.

Each test spawns 2 GPU (or CPU) processes via torch.multiprocessing.spawn,
sets up a minimal process group with CP size = 2, and verifies that the
get_logprobs CP path returns the same tensor as a single-rank reference
computation.

Run with:
    pytest tests/unit_tests/rl/test_context_parallel_functional.py -v
or via torchrun for GPU:
    torchrun --nproc_per_node=2 -m pytest tests/unit_tests/rl/test_context_parallel_functional.py
"""

import os
import tempfile
from unittest.mock import MagicMock, patch, PropertyMock

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# ---------------------------------------------------------------------------
# Worker helpers
# ---------------------------------------------------------------------------

def _init_pg(rank: int, world_size: int, store_path: str) -> dist.ProcessGroup:
    """Create an in-process distributed group backed by a file store."""
    store = dist.FileStore(store_path, world_size)
    dist.init_process_group(
        backend="gloo",
        store=store,
        rank=rank,
        world_size=world_size,
    )
    return dist.new_group(ranks=list(range(world_size)), backend="gloo")


def _worker_get_logprobs_cp(
    rank: int,
    world_size: int,
    store_path: str,
    seq_len: int,
    batch: int,
    vocab: int,
    result_queue: mp.Queue,
) -> None:
    """Worker function: set up CP group, call get_logprobs, put result in queue."""
    try:
        cp_group = _init_pg(rank, world_size, store_path)

        from megatron.rl.rl_utils import _scatter_for_context_parallel, _gather_logprobs_context_parallel, selective_log_softmax
        from megatron.core.packed_seq_params import PackedSeqParams

        torch.manual_seed(0)
        tokens      = torch.randint(0, vocab, (batch, seq_len))
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch, -1)

        # Simulate full logits known to all ranks (deterministic, same on both)
        torch.manual_seed(0)
        logits_full = torch.randn(batch, seq_len, vocab)

        cu = torch.tensor([0, seq_len], dtype=torch.int32)
        psp = PackedSeqParams(
            qkv_format='thd',
            cu_seqlens_q=cu, cu_seqlens_kv=cu,
            max_seqlen_q=seq_len, max_seqlen_kv=seq_len,
            total_tokens=seq_len,
        )

        # Patch mpu to return our synthetic CP group.
        with patch('megatron.rl.rl_utils.mpu') as mock_mpu:
            mock_mpu.get_context_parallel_world_size.return_value = world_size
            mock_mpu.get_context_parallel_rank.return_value       = rank
            mock_mpu.get_context_parallel_group.return_value      = cp_group

            local_tokens, local_pos, cp_psp, local_labels = _scatter_for_context_parallel(
                tokens, position_ids, psp, world_size
            )
            # Use the matching local slice of the full logits as the model output.
            local_size  = seq_len // world_size
            start       = rank * local_size
            local_logits = logits_full[:, start : start + local_size, :]

            local_lp = selective_log_softmax(local_logits, local_labels)
            full_lp  = _gather_logprobs_context_parallel(local_lp, no_grad=True)

        result_queue.put(("ok", full_lp.cpu()))

    except Exception as exc:  # pragma: no cover
        import traceback
        result_queue.put(("err", traceback.format_exc()))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Helper to run a 2-process test
# ---------------------------------------------------------------------------

def _run_2rank_test(worker_fn, **kwargs) -> list:
    """Spawn world_size=2 workers and collect their results."""
    world_size = 2
    ctx        = mp.get_context("spawn")
    queue      = ctx.Queue()
    with tempfile.NamedTemporaryFile(delete=True) as f:
        store_path = f.name + ".store"
    procs = []
    for rank in range(world_size):
        p = ctx.Process(
            target=worker_fn,
            args=(rank, world_size, store_path),
            kwargs={**kwargs, "result_queue": queue},
        )
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
    results = [queue.get_nowait() for _ in range(world_size)]
    return results


# ---------------------------------------------------------------------------
# Functional tests
# ---------------------------------------------------------------------------

class TestCPLogprobsFunctional:

    def _check_results(self, results, expected_logprobs):
        """Assert all workers returned successfully and matching logprobs."""
        for status, payload in results:
            assert status == "ok", f"Worker failed:\n{payload}"
            torch.testing.assert_close(payload, expected_logprobs, atol=1e-5, rtol=1e-4)

    def _reference_logprobs(self, seq_len, batch, vocab):
        torch.manual_seed(0)
        tokens = torch.randint(0, vocab, (batch, seq_len))
        torch.manual_seed(0)
        logits = torch.randn(batch, seq_len, vocab)
        from megatron.rl.rl_utils import selective_log_softmax
        return selective_log_softmax(logits[:, :-1, :], tokens[:, 1:])

    def test_cp2_small_sequence(self):
        """CP=2 logprobs must match the reference single-rank computation."""
        seq_len, batch, vocab = 8, 1, 16
        expected = self._reference_logprobs(seq_len, batch, vocab)
        results  = _run_2rank_test(
            _worker_get_logprobs_cp,
            seq_len=seq_len, batch=batch, vocab=vocab,
        )
        self._check_results(results, expected)

    def test_cp2_larger_sequence(self):
        """CP=2 with a larger sequence (16 tokens)."""
        seq_len, batch, vocab = 16, 2, 32
        expected = self._reference_logprobs(seq_len, batch, vocab)
        results  = _run_2rank_test(
            _worker_get_logprobs_cp,
            seq_len=seq_len, batch=batch, vocab=vocab,
        )
        self._check_results(results, expected)

    def test_cp2_all_ranks_agree(self):
        """Both CP ranks must return the identical full-sequence logprob tensor."""
        seq_len, batch, vocab = 8, 1, 16
        results = _run_2rank_test(
            _worker_get_logprobs_cp,
            seq_len=seq_len, batch=batch, vocab=vocab,
        )
        statuses = [r[0] for r in results]
        assert all(s == "ok" for s in statuses), str(results)
        lp0, lp1 = results[0][1], results[1][1]
        torch.testing.assert_close(lp0, lp1)
