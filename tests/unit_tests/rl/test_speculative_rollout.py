# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Unit tests for speculative rollout generation.

Covers:
  - select_rollouts() with all three strategies
  - SpeculativeMixin.group_rollout() with mocked inference
  - EarlyExitGPTModel construction and parameter-sharing invariant
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch
import torch.nn as nn

from megatron.rl.agent.speculative_mixin import (
    SelectionStrategy,
    SpeculativeGroupedRolloutRequest,
    SpeculativeMixin,
    select_rollouts,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_rollout(reward: float):
    """Build a minimal TokenRollout-like object with a scalar reward."""
    r = MagicMock()
    r.reward = reward
    return r


def _rewards(rollouts) -> list[float]:
    return [float(r.reward) for r in rollouts]


# ---------------------------------------------------------------------------
# Tests for select_rollouts
# ---------------------------------------------------------------------------

class TestSelectRollouts:

    def _pool(self, rewards):
        return [_make_rollout(r) for r in rewards]

    # ---- passthrough when pool ≤ k ----------------------------------------

    def test_passthrough_when_less_than_k(self):
        pool = self._pool([1.0, 2.0])
        result = select_rollouts(pool, k=4, strategy=SelectionStrategy.TOP_K)
        assert result is pool  # same object

    def test_passthrough_when_equal_k(self):
        pool = self._pool([1.0, 2.0, 3.0])
        result = select_rollouts(pool, k=3, strategy=SelectionStrategy.TOP_K)
        assert result is pool

    # ---- TOP_K ---------------------------------------------------------------

    def test_top_k_basic(self):
        pool = self._pool([1.0, 5.0, 3.0, 4.0, 2.0])
        result = select_rollouts(pool, k=2, strategy=SelectionStrategy.TOP_K)
        assert sorted(_rewards(result), reverse=True) == [5.0, 4.0]

    def test_top_k_returns_exactly_k(self):
        pool = self._pool([1.0, 2.0, 3.0, 4.0, 5.0])
        result = select_rollouts(pool, k=3, strategy=SelectionStrategy.TOP_K)
        assert len(result) == 3

    def test_top_k_all_same_reward(self):
        pool = self._pool([1.0] * 5)
        result = select_rollouts(pool, k=2, strategy=SelectionStrategy.TOP_K)
        assert len(result) == 2

    # ---- VARIANCE_MAXIMIZING ------------------------------------------------

    def test_variance_maximizing_returns_k(self):
        pool = self._pool([1.0, 2.0, 3.0, 4.0, 5.0])
        result = select_rollouts(pool, k=3, strategy=SelectionStrategy.VARIANCE_MAXIMIZING)
        assert len(result) == 3

    def test_variance_maximizing_includes_extremes(self):
        """The variance-maximising selection should prefer diverse rewards."""
        pool = self._pool([1.0, 2.0, 3.0, 4.0, 10.0])
        result = select_rollouts(pool, k=2, strategy=SelectionStrategy.VARIANCE_MAXIMIZING)
        rewards = _rewards(result)
        # Min (1.0) and max (10.0) are the extremes — must both be in the result.
        assert 1.0 in rewards
        assert 10.0 in rewards

    def test_variance_maximizing_k1(self):
        pool = self._pool([3.0, 1.0, 4.0, 1.0, 5.0])
        result = select_rollouts(pool, k=1, strategy=SelectionStrategy.VARIANCE_MAXIMIZING)
        assert len(result) == 1
        assert _rewards(result)[0] == 5.0

    def test_variance_maximizing_all_equal(self):
        pool = self._pool([2.0] * 5)
        result = select_rollouts(pool, k=3, strategy=SelectionStrategy.VARIANCE_MAXIMIZING)
        assert len(result) == 3

    # ---- REWARD_DISTANCE_FROM_MEAN ------------------------------------------

    def test_distance_from_mean_returns_k(self):
        pool = self._pool([1.0, 2.0, 3.0, 4.0, 5.0])
        result = select_rollouts(pool, k=2, strategy=SelectionStrategy.REWARD_DISTANCE_FROM_MEAN)
        assert len(result) == 2

    def test_distance_from_mean_picks_extremes(self):
        # Mean = 3.0; extremes are 1.0 and 5.0 (distance 2.0 each).
        pool = self._pool([1.0, 2.0, 3.0, 4.0, 5.0])
        result = select_rollouts(pool, k=2, strategy=SelectionStrategy.REWARD_DISTANCE_FROM_MEAN)
        rewards = _rewards(result)
        assert set(rewards) == {1.0, 5.0}

    # ---- unknown strategy ---------------------------------------------------

    def test_unknown_strategy_raises(self):
        pool = self._pool([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="Unknown SelectionStrategy"):
            select_rollouts(pool, k=2, strategy="bad_strategy")  # type: ignore


# ---------------------------------------------------------------------------
# Tests for SpeculativeMixin.group_rollout
# ---------------------------------------------------------------------------

class _FakeInferenceInterface:
    pass


def _make_base_request(rollouts_per_group=4):
    req = MagicMock()
    req.rollouts_per_group = rollouts_per_group
    # Make model_copy return a modified copy.
    def _model_copy(update=None):
        copy = MagicMock()
        copy.rollouts_per_group = update.get("rollouts_per_group", rollouts_per_group)
        copy.inference_interface = update.get("inference_interface")
        return copy
    req.model_copy.side_effect = _model_copy
    return req


class _ConcreteAgent(SpeculativeMixin):
    """Minimal concrete agent for testing SpeculativeMixin."""

    async def group_rollout(self, request):  # type: ignore[override]
        # This is the *super()* side — called by SpeculativeMixin after building
        # the draft request.  Returns n fake rollouts.
        return [_make_rollout(float(i)) for i in range(request.rollouts_per_group)]


class TestSpeculativeMixin:

    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    def test_passthrough_for_plain_request(self):
        """Non-speculative requests must be forwarded to super().group_rollout()."""

        class PlainAgent(SpeculativeMixin):
            async def group_rollout(self, request):
                return ["plain_result"]

        agent = PlainAgent()
        plain_req = MagicMock()  # not a SpeculativeGroupedRolloutRequest
        result = self._run(agent.group_rollout(plain_req))
        assert result == ["plain_result"]

    def test_speculative_oversamples(self):
        """Mixin must call parent group_rollout with rollouts_per_group * oversample_factor."""
        agent = _ConcreteAgent()
        base = _make_base_request(rollouts_per_group=4)
        spec_req = SpeculativeGroupedRolloutRequest(
            base_request=base,
            draft_inference_interface=_FakeInferenceInterface(),
            oversample_factor=3,
            selection_strategy=SelectionStrategy.TOP_K,
        )
        result = self._run(agent.group_rollout(spec_req))
        # 4 * 3 = 12 candidates generated; 4 selected
        assert len(result) == 4

    def test_speculative_selects_top_k(self):
        """With TOP_K, the returned rollouts must have the highest rewards."""
        agent = _ConcreteAgent()
        base = _make_base_request(rollouts_per_group=2)
        spec_req = SpeculativeGroupedRolloutRequest(
            base_request=base,
            draft_inference_interface=_FakeInferenceInterface(),
            oversample_factor=4,
            selection_strategy=SelectionStrategy.TOP_K,
        )
        result = self._run(agent.group_rollout(spec_req))
        # _ConcreteAgent generates rewards 0..7; top-2 are 6.0 and 7.0.
        rewards = sorted(_rewards(result), reverse=True)
        assert rewards == [7.0, 6.0]

    def test_speculative_variance_maximizing(self):
        """VARIANCE_MAXIMIZING must include the reward extremes."""
        agent = _ConcreteAgent()
        base = _make_base_request(rollouts_per_group=2)
        spec_req = SpeculativeGroupedRolloutRequest(
            base_request=base,
            draft_inference_interface=_FakeInferenceInterface(),
            oversample_factor=4,
            selection_strategy=SelectionStrategy.VARIANCE_MAXIMIZING,
        )
        result = self._run(agent.group_rollout(spec_req))
        rewards = _rewards(result)
        # Pool rewards 0..7; extremes are 0.0 and 7.0.
        assert 0.0 in rewards
        assert 7.0 in rewards

    def test_speculative_uses_draft_inference_interface(self):
        """The draft request forwarded to super must use draft_inference_interface."""
        received = {}

        class TrackingAgent(SpeculativeMixin):
            async def group_rollout(self, request):
                received['interface'] = request.inference_interface
                received['n'] = request.rollouts_per_group
                return [_make_rollout(float(i)) for i in range(request.rollouts_per_group)]

        draft_ii = _FakeInferenceInterface()
        agent = TrackingAgent()
        base = _make_base_request(rollouts_per_group=2)
        spec_req = SpeculativeGroupedRolloutRequest(
            base_request=base,
            draft_inference_interface=draft_ii,
            oversample_factor=3,
            selection_strategy=SelectionStrategy.TOP_K,
        )
        self._run(agent.group_rollout(spec_req))
        assert received['interface'] is draft_ii
        assert received['n'] == 6  # 2 * 3


# ---------------------------------------------------------------------------
# Tests for EarlyExitGPTModel
# ---------------------------------------------------------------------------

class _FakeLayer(nn.Module):
    """Lightweight stand-in for a transformer layer."""

    def __init__(self, idx: int):
        super().__init__()
        self.idx = idx
        self.linear = nn.Linear(8, 8, bias=False)

    def forward(self, hidden_states, **kwargs):
        return self.linear(hidden_states), None  # (hidden, context)


class _FakeDecoder(nn.Module):
    def __init__(self, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([_FakeLayer(i) for i in range(num_layers)])
        self.final_layernorm = None


class _FakeGPTModel(nn.Module):
    """Minimal GPTModel stand-in with 6 transformer layers."""

    def __init__(self, num_layers: int = 6):
        super().__init__()
        self.config = MagicMock()
        self.config.hidden_size = 8
        self.pre_process = True
        self.post_process = True
        self.position_embedding_type = 'rope'
        self.parallel_output = False
        self.decoder = _FakeDecoder(num_layers)

    def _preprocess(self, input_ids, position_ids, **kwargs):
        b, s = input_ids.shape
        hidden = torch.zeros(b, s, 8)
        return hidden, None, None, None, None, None

    def output_layer(self, hidden):
        return hidden, None  # (logits, bias)


class TestEarlyExitGPTModel:
    """Tests for EarlyExitGPTModel without GPU / real TransformerBlock."""

    def _make(self, num_layers=6, exit_layer=3):
        from megatron.rl.inference.draft_model import EarlyExitGPTModel
        return EarlyExitGPTModel(_FakeGPTModel(num_layers), exit_layer)

    # ---- construction --------------------------------------------------------

    def test_valid_exit_layer(self):
        model = self._make(num_layers=6, exit_layer=3)
        assert model.exit_layer == 3

    def test_invalid_exit_layer_zero_raises(self):
        from megatron.rl.inference.draft_model import EarlyExitGPTModel
        with pytest.raises(ValueError, match="exit_layer"):
            EarlyExitGPTModel(_FakeGPTModel(6), exit_layer=0)

    def test_invalid_exit_layer_equal_total_raises(self):
        from megatron.rl.inference.draft_model import EarlyExitGPTModel
        with pytest.raises(ValueError, match="exit_layer"):
            EarlyExitGPTModel(_FakeGPTModel(6), exit_layer=6)

    def test_config_exposed(self):
        fm = _FakeGPTModel(6)
        from megatron.rl.inference.draft_model import EarlyExitGPTModel
        em = EarlyExitGPTModel(fm, exit_layer=2)
        assert em.config is fm.config

    # ---- parameter sharing ---------------------------------------------------

    def test_full_model_params_not_registered(self):
        """The full model's parameters must not appear in draft.parameters()."""
        from megatron.rl.inference.draft_model import EarlyExitGPTModel
        fm = _FakeGPTModel(6)
        em = EarlyExitGPTModel(fm, exit_layer=3)
        # EarlyExitGPTModel has no registered parameters itself.
        assert list(em.parameters()) == []

    def test_layers_are_shared_by_reference(self):
        """Layer objects in the draft must be the same Python objects as in the full model."""
        from megatron.rl.inference.draft_model import EarlyExitGPTModel
        fm = _FakeGPTModel(6)
        em = EarlyExitGPTModel(fm, exit_layer=3)
        for i in range(3):
            assert em._full_model.decoder.layers[i] is fm.decoder.layers[i]

    def test_weight_modification_visible_in_draft(self):
        """Writing to a full-model layer weight must be immediately visible in the draft."""
        from megatron.rl.inference.draft_model import EarlyExitGPTModel
        fm = _FakeGPTModel(6)
        em = EarlyExitGPTModel(fm, exit_layer=3)

        # Modify a weight in layer 0 of the full model.
        with torch.no_grad():
            fm.decoder.layers[0].linear.weight.fill_(42.0)

        # The draft sees the same tensor object → same values.
        assert em._full_model.decoder.layers[0].linear.weight[0, 0].item() == 42.0

    # ---- sync_draft_weights --------------------------------------------------

    def test_sync_is_noop_for_early_exit(self):
        from megatron.rl.inference.draft_model import EarlyExitGPTModel, sync_draft_weights
        fm = _FakeGPTModel(6)
        em = EarlyExitGPTModel(fm, exit_layer=3)
        # Should complete without error.
        sync_draft_weights(em, fm, num_layers=3)

    # ---- DraftDistillationConfig --------------------------------------------

    def test_distillation_config_disabled_by_default(self):
        from megatron.rl.inference.draft_model import DraftDistillationConfig
        cfg = DraftDistillationConfig()
        assert not cfg.enabled
        assert cfg.num_warmup_steps == 0

    def test_distillation_config_enabled(self):
        from megatron.rl.inference.draft_model import DraftDistillationConfig
        cfg = DraftDistillationConfig(num_warmup_steps=100, temperature=2.0)
        assert cfg.enabled
        assert cfg.temperature == 2.0
