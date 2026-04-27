# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Unit tests for speculative rollout generation.

Covers:
  - select_rollouts() with all three strategies
  - SpeculativeMixin.group_rollout() with mocked inference
  - EarlyExitGPTModel construction, parameter sharing, attribute forwarding,
    and the forward path (tied embeddings + [b, s, h] layout).
"""

import asyncio
from unittest.mock import MagicMock

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

    def test_passthrough_when_less_than_k_returns_copy(self):
        """Passthrough must return a *copy* so callers can mutate safely."""
        pool = self._pool([1.0, 2.0])
        result = select_rollouts(pool, k=4, strategy=SelectionStrategy.TOP_K)
        assert result is not pool
        assert _rewards(result) == _rewards(pool)

    def test_passthrough_when_equal_k_returns_copy(self):
        pool = self._pool([1.0, 2.0, 3.0])
        result = select_rollouts(pool, k=3, strategy=SelectionStrategy.TOP_K)
        assert result is not pool
        assert _rewards(result) == _rewards(pool)

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

    def test_variance_maximizing_returns_k(self):
        pool = self._pool([1.0, 2.0, 3.0, 4.0, 5.0])
        result = select_rollouts(pool, k=3, strategy=SelectionStrategy.VARIANCE_MAXIMIZING)
        assert len(result) == 3

    def test_variance_maximizing_includes_extremes(self):
        pool = self._pool([1.0, 2.0, 3.0, 4.0, 10.0])
        result = select_rollouts(pool, k=2, strategy=SelectionStrategy.VARIANCE_MAXIMIZING)
        rewards = _rewards(result)
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

    def test_distance_from_mean_returns_k(self):
        pool = self._pool([1.0, 2.0, 3.0, 4.0, 5.0])
        result = select_rollouts(pool, k=2, strategy=SelectionStrategy.REWARD_DISTANCE_FROM_MEAN)
        assert len(result) == 2

    def test_distance_from_mean_picks_extremes(self):
        pool = self._pool([1.0, 2.0, 3.0, 4.0, 5.0])
        result = select_rollouts(pool, k=2, strategy=SelectionStrategy.REWARD_DISTANCE_FROM_MEAN)
        rewards = _rewards(result)
        assert set(rewards) == {1.0, 5.0}

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

    def _model_copy(update=None):
        copy = MagicMock()
        copy.rollouts_per_group = update.get("rollouts_per_group", rollouts_per_group)
        copy.inference_interface = update.get("inference_interface")
        return copy

    req.model_copy.side_effect = _model_copy
    return req


class _ConcreteAgent(SpeculativeMixin):
    async def group_rollout(self, request):  # type: ignore[override]
        return [_make_rollout(float(i)) for i in range(request.rollouts_per_group)]


class TestSpeculativeMixin:

    def test_passthrough_for_plain_request(self):
        class PlainAgent(SpeculativeMixin):
            async def group_rollout(self, request):
                return ["plain_result"]

        agent = PlainAgent()
        plain_req = MagicMock()
        result = asyncio.run(agent.group_rollout(plain_req))
        assert result == ["plain_result"]

    def test_speculative_oversamples(self):
        agent = _ConcreteAgent()
        base = _make_base_request(rollouts_per_group=4)
        spec_req = SpeculativeGroupedRolloutRequest(
            base_request=base,
            draft_inference_interface=_FakeInferenceInterface(),
            oversample_factor=3,
            selection_strategy=SelectionStrategy.TOP_K,
        )
        result = asyncio.run(agent.group_rollout(spec_req))
        assert len(result) == 4

    def test_speculative_selects_top_k(self):
        agent = _ConcreteAgent()
        base = _make_base_request(rollouts_per_group=2)
        spec_req = SpeculativeGroupedRolloutRequest(
            base_request=base,
            draft_inference_interface=_FakeInferenceInterface(),
            oversample_factor=4,
            selection_strategy=SelectionStrategy.TOP_K,
        )
        result = asyncio.run(agent.group_rollout(spec_req))
        rewards = sorted(_rewards(result), reverse=True)
        assert rewards == [7.0, 6.0]

    def test_speculative_variance_maximizing(self):
        agent = _ConcreteAgent()
        base = _make_base_request(rollouts_per_group=2)
        spec_req = SpeculativeGroupedRolloutRequest(
            base_request=base,
            draft_inference_interface=_FakeInferenceInterface(),
            oversample_factor=4,
            selection_strategy=SelectionStrategy.VARIANCE_MAXIMIZING,
        )
        result = asyncio.run(agent.group_rollout(spec_req))
        rewards = _rewards(result)
        assert 0.0 in rewards
        assert 7.0 in rewards

    def test_speculative_uses_draft_inference_interface(self):
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
        asyncio.run(agent.group_rollout(spec_req))
        assert received['interface'] is draft_ii
        assert received['n'] == 6


# ---------------------------------------------------------------------------
# Tests for EarlyExitGPTModel
# ---------------------------------------------------------------------------

class _FakeLayer(nn.Module):
    def __init__(self, idx: int):
        super().__init__()
        self.idx = idx
        self.linear = nn.Linear(8, 8, bias=False)

    def forward(self, hidden_states, **kwargs):
        # Return [s, b, h] unchanged plus None context — matches the real
        # TransformerLayer return signature.
        return self.linear(hidden_states), None


class _FakeDecoder(nn.Module):
    def __init__(self, num_layers: int, with_final_ln: bool = False):
        super().__init__()
        self.layers = nn.ModuleList([_FakeLayer(i) for i in range(num_layers)])
        self.final_layernorm = nn.LayerNorm(8) if with_final_ln else None


class _FakeOutputLayer(nn.Module):
    """Mimics ColumnParallelLinear: accepts (hidden, weight=, runtime_gather_output=)."""

    def __init__(self, vocab_size: int = 16, hidden: int = 8):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(vocab_size, hidden))
        self.last_call_weight = None  # records `weight` kwarg for assertions
        self.last_call_gather = None

    def forward(self, hidden_states, weight=None, runtime_gather_output=None):
        self.last_call_weight = weight
        self.last_call_gather = runtime_gather_output
        w = weight if weight is not None else self.weight
        return hidden_states @ w.t(), None


class _FakeGPTModel(nn.Module):
    """Minimal GPTModel stand-in.  Provides enough surface for the draft's forward."""

    def __init__(
        self,
        num_layers: int = 6,
        with_final_ln: bool = True,
        share_embeddings_and_output_weights: bool = False,
    ):
        super().__init__()
        self.config = MagicMock()
        self.config.hidden_size = 8
        self.pre_process = True
        self.post_process = True
        self.position_embedding_type = 'rope'
        self.parallel_output = False
        self.decoder = _FakeDecoder(num_layers, with_final_ln=with_final_ln)
        self.output_layer = _FakeOutputLayer()
        self.max_sequence_length = 2048
        self.model_type = "GPT"
        self.xattn_needed = False
        self.pg_collection = MagicMock()
        self.share_embeddings_and_output_weights = share_embeddings_and_output_weights
        self._shared_weight = nn.Parameter(torch.full((16, 8), 7.0))

    def _preprocess(self, input_ids, position_ids, decoder_input=None,
                    inference_context=None, packed_seq_params=None):
        s, b = input_ids.shape[1], input_ids.shape[0]
        # Megatron convention: [s, b, h]
        hidden = torch.zeros(s, b, 8)
        return hidden, None, None, None, None, None

    def shared_embedding_or_output_weight(self):
        return self._shared_weight

    def _scale_logits(self, logits):
        return logits  # no-op


class TestEarlyExitGPTModel:

    def _make(self, num_layers=6, exit_layer=3, **fm_kwargs):
        from megatron.rl.inference.draft_model import EarlyExitGPTModel
        fm = _FakeGPTModel(num_layers, **fm_kwargs)
        return fm, EarlyExitGPTModel(fm, exit_layer)

    # ---- construction --------------------------------------------------------

    def test_valid_exit_layer(self):
        _, em = self._make(num_layers=6, exit_layer=3)
        assert em.exit_layer == 3

    def test_invalid_exit_layer_zero_raises(self):
        from megatron.rl.inference.draft_model import EarlyExitGPTModel
        with pytest.raises(ValueError, match="exit_layer"):
            EarlyExitGPTModel(_FakeGPTModel(6), exit_layer=0)

    def test_invalid_exit_layer_equal_total_raises(self):
        from megatron.rl.inference.draft_model import EarlyExitGPTModel
        with pytest.raises(ValueError, match="exit_layer"):
            EarlyExitGPTModel(_FakeGPTModel(6), exit_layer=6)

    def test_config_exposed(self):
        fm, em = self._make(exit_layer=2)
        assert em.config is fm.config

    # ---- forwarded attributes for the inference engine ----------------------

    def test_forwards_max_sequence_length(self):
        fm, em = self._make()
        assert em.max_sequence_length == fm.max_sequence_length

    def test_forwards_pg_collection(self):
        fm, em = self._make()
        assert em.pg_collection is fm.pg_collection

    def test_forwards_model_type_and_xattn(self):
        fm, em = self._make()
        assert em.model_type == fm.model_type
        assert em.xattn_needed == fm.xattn_needed

    def test_decoder_property_does_not_register_children(self):
        """Exposing decoder as a property must NOT register its parameters."""
        _, em = self._make()
        # decoder is the full transformer block, but draft's parameters() must be empty.
        assert em.decoder is not None
        assert list(em.parameters()) == []

    # ---- parameter sharing --------------------------------------------------

    def test_full_model_params_not_registered(self):
        _, em = self._make()
        assert list(em.parameters()) == []

    def test_layers_are_shared_by_reference(self):
        fm, em = self._make()
        for i in range(3):
            assert em._full_model.decoder.layers[i] is fm.decoder.layers[i]

    def test_weight_modification_visible_in_draft(self):
        fm, em = self._make()
        with torch.no_grad():
            fm.decoder.layers[0].linear.weight.fill_(42.0)
        assert em._full_model.decoder.layers[0].linear.weight[0, 0].item() == 42.0

    # ---- forward path -------------------------------------------------------

    def test_forward_returns_bsh_layout(self):
        """Logits must be returned in [b, s, h] layout (transposed from [s, b, h])."""
        _, em = self._make(num_layers=6, exit_layer=3)
        b, s = 2, 4
        input_ids = torch.zeros(b, s, dtype=torch.long)
        position_ids = torch.zeros(b, s, dtype=torch.long)
        attention_mask = torch.ones(b, 1, s, s)
        logits = em(input_ids, position_ids, attention_mask)
        # vocab=16 in _FakeOutputLayer, hidden=8.  After transpose: [b, s, vocab].
        assert logits.shape == (b, s, 16)

    def test_forward_passes_tied_weight(self):
        """When share_embeddings_and_output_weights=True, output_layer must
        receive the shared weight via the `weight` kwarg."""
        fm, em = self._make(share_embeddings_and_output_weights=True)
        b, s = 1, 2
        em(torch.zeros(b, s, dtype=torch.long),
           torch.zeros(b, s, dtype=torch.long),
           torch.ones(b, 1, s, s))
        # The fake output layer records the kwarg; it must be the shared weight tensor.
        assert fm.output_layer.last_call_weight is fm._shared_weight

    def test_forward_skips_tied_weight_when_untied(self):
        fm, em = self._make(share_embeddings_and_output_weights=False)
        b, s = 1, 2
        em(torch.zeros(b, s, dtype=torch.long),
           torch.zeros(b, s, dtype=torch.long),
           torch.ones(b, 1, s, s))
        assert fm.output_layer.last_call_weight is None

    def test_forward_only_runs_first_exit_layers(self):
        """The forward must call exactly `exit_layer` layers, not more."""
        from unittest.mock import patch
        fm, em = self._make(num_layers=6, exit_layer=2)
        call_count = [0]
        original_forwards = [layer.forward for layer in fm.decoder.layers]

        def make_counting_forward(orig, idx):
            def fwd(*a, **kw):
                call_count[0] = max(call_count[0], idx + 1)
                return orig(*a, **kw)
            return fwd

        for i, layer in enumerate(fm.decoder.layers):
            layer.forward = make_counting_forward(original_forwards[i], i)

        b, s = 1, 2
        em(torch.zeros(b, s, dtype=torch.long),
           torch.zeros(b, s, dtype=torch.long),
           torch.ones(b, 1, s, s))
        assert call_count[0] == 2  # only layers 0 and 1 ran

    # ---- sync_draft_weights -------------------------------------------------

    def test_sync_is_noop_for_early_exit(self):
        from megatron.rl.inference.draft_model import sync_draft_weights
        fm, em = self._make()
        sync_draft_weights(em, fm, num_layers=3)  # must not raise
