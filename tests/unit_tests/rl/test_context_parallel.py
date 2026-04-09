# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Unit tests for context-parallel helpers in megatron/rl/rl_utils.py.

These tests exercise _scatter_for_context_parallel and
_gather_logprobs_context_parallel without requiring a real distributed
environment by patching megatron.core.mpu.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

# ---------------------------------------------------------------------------
# Helpers to build a fake MagicMock process group that satisfies isinstance checks.
# ---------------------------------------------------------------------------

def _make_fake_pg():
    pg = MagicMock()
    pg.__class__ = torch.distributed.ProcessGroup
    return pg


# ---------------------------------------------------------------------------
# Tests for _scatter_for_context_parallel
# ---------------------------------------------------------------------------

class TestScatterForContextParallel:
    """Test _scatter_for_context_parallel in isolation."""

    def _run(self, cp_size, cp_rank, batch=1, seq_len=8, vocab=4):
        """Run scatter for one (cp_size, cp_rank) pair and return outputs."""
        from megatron.rl.rl_utils import _scatter_for_context_parallel
        from megatron.core.packed_seq_params import PackedSeqParams

        tokens      = torch.arange(batch * seq_len).reshape(batch, seq_len)
        position_ids = torch.arange(batch * seq_len).reshape(batch, seq_len)
        cu = torch.tensor([0, seq_len], dtype=torch.int32)
        packed_seq_params = PackedSeqParams(
            qkv_format='thd',
            cu_seqlens_q=cu,
            cu_seqlens_kv=cu,
            max_seqlen_q=seq_len,
            max_seqlen_kv=seq_len,
            total_tokens=seq_len,
        )
        fake_pg = _make_fake_pg()

        with patch('megatron.rl.rl_utils.mpu') as mock_mpu:
            mock_mpu.get_context_parallel_rank.return_value  = cp_rank
            mock_mpu.get_context_parallel_group.return_value = fake_pg
            result = _scatter_for_context_parallel(tokens, position_ids, packed_seq_params, cp_size)

        return result, tokens, packed_seq_params

    # --- shape tests ----------------------------------------------------------

    def test_local_tokens_shape(self):
        (lt, lp, _, _), tokens, _ = self._run(cp_size=2, cp_rank=0, seq_len=8)
        assert lt.shape == (1, 4), f"expected (1,4) got {lt.shape}"

    def test_local_position_ids_shape(self):
        (_, lp, _, _), tokens, _ = self._run(cp_size=2, cp_rank=0, seq_len=8)
        assert lp.shape == (1, 4)

    def test_local_labels_shape(self):
        (_, _, _, ll), tokens, _ = self._run(cp_size=2, cp_rank=0, seq_len=8)
        assert ll.shape == (1, 4)

    def test_cp4_shape(self):
        (lt, _, _, ll), _, _ = self._run(cp_size=4, cp_rank=2, seq_len=16)
        assert lt.shape == (1, 4)
        assert ll.shape == (1, 4)

    # --- value tests ----------------------------------------------------------

    def test_rank0_tokens_are_first_half(self):
        (lt, _, _, _), tokens, _ = self._run(cp_size=2, cp_rank=0, seq_len=8)
        torch.testing.assert_close(lt, tokens[:, :4])

    def test_rank1_tokens_are_second_half(self):
        (lt, _, _, _), tokens, _ = self._run(cp_size=2, cp_rank=1, seq_len=8)
        torch.testing.assert_close(lt, tokens[:, 4:])

    def test_rank0_labels_are_shifted_tokens(self):
        """Rank 0 labels = tokens[1:5] (the next-token for each local position)."""
        (_, _, _, ll), tokens, _ = self._run(cp_size=2, cp_rank=0, seq_len=8)
        torch.testing.assert_close(ll, tokens[:, 1:5])

    def test_rank1_labels_boundary(self):
        """Last label on the last rank wraps to tokens[-1] (padding placeholder)."""
        (_, _, _, ll), tokens, _ = self._run(cp_size=2, cp_rank=1, seq_len=8)
        # Labels for rank 1: tokens[5], tokens[6], tokens[7], tokens[7] (wrap)
        expected = torch.cat([tokens[:, 5:8], tokens[:, 7:8]], dim=1)
        torch.testing.assert_close(ll, expected)

    def test_labels_are_contiguous(self):
        (_, _, _, ll), _, _ = self._run(cp_size=2, cp_rank=0, seq_len=8)
        assert ll.is_contiguous()

    # --- PackedSeqParams mutation test ----------------------------------------

    def test_cp_fields_set_on_copy(self):
        """cp_group and local_cp_size must be set; original must be unchanged."""
        from megatron.core.packed_seq_params import PackedSeqParams
        (_, _, cp_params, _), _, orig = self._run(cp_size=2, cp_rank=0, seq_len=8)
        assert cp_params.local_cp_size == 2
        assert cp_params.cp_group is not None
        # Original must not have been mutated.
        assert orig.local_cp_size is None
        assert orig.cp_group is None

    def test_assertion_on_indivisible_seq_len(self):
        from megatron.rl.rl_utils import _scatter_for_context_parallel
        from megatron.core.packed_seq_params import PackedSeqParams
        tokens = torch.zeros(1, 9, dtype=torch.long)
        pos    = torch.zeros(1, 9, dtype=torch.long)
        cu     = torch.tensor([0, 9], dtype=torch.int32)
        psp    = PackedSeqParams(qkv_format='thd', cu_seqlens_q=cu, cu_seqlens_kv=cu,
                                 max_seqlen_q=9, max_seqlen_kv=9, total_tokens=9)
        fake_pg = _make_fake_pg()
        with patch('megatron.rl.rl_utils.mpu') as mock_mpu:
            mock_mpu.get_context_parallel_rank.return_value  = 0
            mock_mpu.get_context_parallel_group.return_value = fake_pg
            with pytest.raises(AssertionError, match="divisible"):
                _scatter_for_context_parallel(tokens, pos, psp, cp_size=2)


# ---------------------------------------------------------------------------
# Tests for _gather_logprobs_context_parallel
# ---------------------------------------------------------------------------

class TestGatherLogprobsContextParallel:
    """Test _gather_logprobs_context_parallel without a real dist backend."""

    def _gather_no_grad(self, cp_size, local_logprobs_per_rank):
        """Simulate the no_grad gather by patching all_gather."""
        from megatron.rl.rl_utils import _gather_logprobs_context_parallel
        fake_pg = _make_fake_pg()

        def fake_all_gather(out_list, tensor, group):
            for i, t in enumerate(local_logprobs_per_rank):
                out_list[i].copy_(t)

        with (
            patch('megatron.rl.rl_utils.mpu') as mock_mpu,
            patch('torch.distributed.all_gather', side_effect=fake_all_gather),
        ):
            mock_mpu.get_context_parallel_group.return_value   = fake_pg
            mock_mpu.get_context_parallel_world_size.return_value = cp_size
            result = _gather_logprobs_context_parallel(local_logprobs_per_rank[0], no_grad=True)
        return result

    def test_shape_after_gather(self):
        """Output shape must be [batch, cp_size*local_size - 1]."""
        cp_size    = 2
        local_size = 5
        ranks = [torch.arange(local_size, dtype=torch.float32).unsqueeze(0) for _ in range(cp_size)]
        out = self._gather_no_grad(cp_size, ranks)
        assert out.shape == (1, cp_size * local_size - 1)

    def test_values_correct_rank0_then_rank1(self):
        """Output must be rank0‖rank1 with last element dropped."""
        rank0 = torch.tensor([[1.0, 2.0, 3.0]])   # local_size=3
        rank1 = torch.tensor([[4.0, 5.0, 6.0]])
        out = self._gather_no_grad(cp_size=2, local_logprobs_per_rank=[rank0, rank1])
        expected = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])  # drop last (6.0)
        torch.testing.assert_close(out, expected)

    def test_cp4_shape(self):
        cp_size    = 4
        local_size = 4
        ranks = [torch.zeros(1, local_size) for _ in range(cp_size)]
        out = self._gather_no_grad(cp_size, ranks)
        assert out.shape == (1, cp_size * local_size - 1)


# ---------------------------------------------------------------------------
# Tests that verify _scatter + manual computation == reference logprobs
# ---------------------------------------------------------------------------

class TestScatterGatherEquivalence:
    """Verify that scattering tokens and computing logprobs per rank, then
    gathering, gives the same result as the single-rank reference computation."""

    @staticmethod
    def _reference_logprobs(logits, tokens):
        """selective_log_softmax(logits[:, :-1, :], tokens[:, 1:])."""
        from megatron.rl.rl_utils import selective_log_softmax
        return selective_log_softmax(logits[:, :-1, :], tokens[:, 1:])

    @staticmethod
    def _cp_logprobs(logits, tokens, cp_size):
        """Simulate the CP path: scatter, compute, gather (no actual dist)."""
        from megatron.rl.rl_utils import selective_log_softmax
        from megatron.core.packed_seq_params import PackedSeqParams
        seq_len    = tokens.shape[1]
        local_size = seq_len // cp_size
        tokens_extended = torch.cat([tokens, tokens[:, -1:]], dim=1)
        all_local_lp = []
        for cp_rank in range(cp_size):
            start = cp_rank * local_size
            end   = start + local_size
            local_logits = logits[:, start:end, :]
            local_labels = tokens_extended[:, start + 1 : end + 1]
            lp = selective_log_softmax(local_logits, local_labels)
            all_local_lp.append(lp)
        return torch.cat(all_local_lp, dim=1)[:, :-1]

    def test_cp2_matches_reference(self):
        torch.manual_seed(0)
        batch, seq_len, vocab = 1, 8, 16
        logits = torch.randn(batch, seq_len, vocab)
        tokens = torch.randint(0, vocab, (batch, seq_len))
        ref = self._reference_logprobs(logits, tokens)
        cp  = self._cp_logprobs(logits, tokens, cp_size=2)
        torch.testing.assert_close(ref, cp)

    def test_cp4_matches_reference(self):
        torch.manual_seed(42)
        batch, seq_len, vocab = 2, 16, 32
        logits = torch.randn(batch, seq_len, vocab)
        tokens = torch.randint(0, vocab, (batch, seq_len))
        ref = self._reference_logprobs(logits, tokens)
        cp  = self._cp_logprobs(logits, tokens, cp_size=4)
        torch.testing.assert_close(ref, cp)

    def test_cp2_with_boundary_spanning_sequence(self):
        """Sequence tokens that cross the CP boundary must still match reference."""
        torch.manual_seed(7)
        batch, seq_len, vocab = 1, 12, 8
        logits = torch.randn(batch, seq_len, vocab)
        tokens = torch.randint(0, vocab, (batch, seq_len))
        ref = self._reference_logprobs(logits, tokens)
        cp  = self._cp_logprobs(logits, tokens, cp_size=2)  # boundary at position 6
        torch.testing.assert_close(ref, cp)
