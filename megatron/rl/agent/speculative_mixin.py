# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Speculative grouped rollout mixin for RewardOnlyAgent.

SpeculativeMixin overrides ``group_rollout()`` to:

1. Generate ``rollouts_per_group * oversample_factor`` candidate rollouts via a
   fast draft inference interface (typically backed by EarlyExitGPTModel).
2. Score each candidate with the agent's reward function (already embedded in
   ``_run_episode()`` via ``get_trajectory_reward()``).
3. Select ``rollouts_per_group`` candidates according to ``selection_strategy``.

The training pipeline recomputes exact log-probs with the full model anyway
(``get_logprobs()`` in ``train_rl.py``), so draft log-probs are discarded and
no importance-sampling correction is required.

Usage::

    class MyAgent(SpeculativeMixin, RewardOnlyAgent):
        ...

    request = SpeculativeGroupedRolloutRequest(
        base_request=GroupedRolloutRequest(
            num_groups=8,
            rollouts_per_group=4,
            inference_interface=full_inference_interface,
        ),
        draft_inference_interface=draft_inference_interface,
        oversample_factor=4,
        selection_strategy=SelectionStrategy.VARIANCE_MAXIMIZING,
    )
    groups = await agent.group_rollout(request)
"""
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel

from .api import GroupedRolloutRequest, Rollout, TokenRollout


# ---------------------------------------------------------------------------
# SelectionStrategy
# ---------------------------------------------------------------------------


class SelectionStrategy(str, Enum):
    """Strategy used to select ``k`` rollouts from oversampled candidates.

    Attributes:
        TOP_K: Keep the rollouts with the highest reward.
        VARIANCE_MAXIMIZING: Keep the subset that maximises reward variance —
            ideal for GRPO, which benefits from diverse reward signals.
        REWARD_DISTANCE_FROM_MEAN: Keep rollouts whose reward deviates most
            from the group mean, ensuring coverage of both extremes.
    """

    TOP_K = "top_k"
    VARIANCE_MAXIMIZING = "variance_maximizing"
    REWARD_DISTANCE_FROM_MEAN = "reward_distance_from_mean"


# ---------------------------------------------------------------------------
# SpeculativeGroupedRolloutRequest
# ---------------------------------------------------------------------------


class SpeculativeGroupedRolloutRequest(BaseModel):
    """Extended rollout request for speculative grouped generation.

    Wraps a standard :class:`GroupedRolloutRequest` and adds the fields needed
    to drive the draft-based oversampling.

    Args:
        base_request: The underlying ``GroupedRolloutRequest`` that defines
            reward computation and the *full* inference interface used for any
            non-draft evaluation.
        draft_inference_interface: Faster inference interface (e.g. backed by
            :class:`~megatron.rl.inference.draft_model.EarlyExitGPTModel`)
            used to generate the oversampled candidate pool.
        oversample_factor: Multiplier on ``rollouts_per_group``.  For example,
            ``oversample_factor=4`` with ``rollouts_per_group=4`` generates 16
            candidates from which the best 4 are selected.
        selection_strategy: How to select the final rollout set from the
            candidates.
    """

    model_config = {"arbitrary_types_allowed": True}

    base_request: GroupedRolloutRequest
    draft_inference_interface: Any  # InferenceInterface
    oversample_factor: int = 4
    selection_strategy: SelectionStrategy = SelectionStrategy.VARIANCE_MAXIMIZING


# ---------------------------------------------------------------------------
# Selection logic (pure function, easily unit-testable)
# ---------------------------------------------------------------------------


def _scalar_reward(rollout: Rollout | TokenRollout) -> float:
    """Extract a scalar reward from a rollout (reward may be a list)."""
    rw = rollout.reward
    if isinstance(rw, list):
        return float(rw[-1]) if rw else 0.0
    return float(rw)


def select_rollouts(
    rollouts: list[Rollout | TokenRollout],
    k: int,
    strategy: SelectionStrategy,
) -> list[Rollout | TokenRollout]:
    """Select ``k`` rollouts from ``rollouts`` according to ``strategy``.

    Args:
        rollouts: Pool of candidate rollouts.
        k: Number of rollouts to return.
        strategy: Selection criterion.

    Returns:
        List of ``k`` selected rollouts (order is not guaranteed).
    """
    if len(rollouts) <= k:
        return rollouts

    rewards = [_scalar_reward(r) for r in rollouts]

    if strategy == SelectionStrategy.TOP_K:
        indices = sorted(range(len(rollouts)), key=lambda i: rewards[i], reverse=True)[:k]

    elif strategy == SelectionStrategy.VARIANCE_MAXIMIZING:
        # Greedy subset selection that maximises reward variance.
        # Initialise with the two extremes (min + max) to maximise the initial
        # spread, then greedily add the candidate that most increases variance.
        if k == 1:
            indices = [max(range(len(rewards)), key=lambda i: rewards[i])]
        else:
            min_i = min(range(len(rewards)), key=lambda i: rewards[i])
            max_i = max(range(len(rewards)), key=lambda i: rewards[i])
            if min_i == max_i:
                # All rewards equal — fall back to arbitrary selection.
                indices = list(range(k))
            else:
                chosen = [min_i, max_i]
                remaining = [i for i in range(len(rollouts)) if i not in chosen]
                while len(chosen) < k and remaining:
                    # Variance of chosen ∪ {candidate}: use the online formula
                    #   Var = E[x²] − (E[x])²
                    n = len(chosen) + 1
                    cur_sum = sum(rewards[i] for i in chosen)
                    cur_sum_sq = sum(rewards[i] ** 2 for i in chosen)
                    best_i, best_var = None, -1.0
                    for i in remaining:
                        new_sum = cur_sum + rewards[i]
                        new_sum_sq = cur_sum_sq + rewards[i] ** 2
                        var = (new_sum_sq / n) - (new_sum / n) ** 2
                        if var > best_var:
                            best_var, best_i = var, i
                    chosen.append(best_i)
                    remaining.remove(best_i)
                indices = chosen

    elif strategy == SelectionStrategy.REWARD_DISTANCE_FROM_MEAN:
        mean = sum(rewards) / len(rewards)
        indices = sorted(
            range(len(rollouts)), key=lambda i: abs(rewards[i] - mean), reverse=True
        )[:k]

    else:
        raise ValueError(f"Unknown SelectionStrategy: {strategy!r}")

    return [rollouts[i] for i in indices]


# ---------------------------------------------------------------------------
# SpeculativeMixin
# ---------------------------------------------------------------------------


class SpeculativeMixin:
    """Mixin that overrides ``group_rollout()`` with speculative oversampling.

    Place this mixin *before* ``RewardOnlyAgent`` in the MRO::

        class MyAgent(SpeculativeMixin, RewardOnlyAgent):
            ...

    When ``group_rollout()`` receives a :class:`SpeculativeGroupedRolloutRequest`,
    the mixin:

    1. Builds a draft request with ``rollouts_per_group * oversample_factor``
       candidates backed by ``draft_inference_interface``.
    2. Awaits the parent ``group_rollout()`` — which uses the agent's own
       ``_run_episode()`` / ``get_trajectory_reward()`` to score every rollout.
    3. Selects ``rollouts_per_group`` candidates with ``select_rollouts()``.

    All other request types are forwarded unchanged to ``super().group_rollout()``.
    """

    async def group_rollout(self, request) -> list[Rollout | TokenRollout]:
        if not isinstance(request, SpeculativeGroupedRolloutRequest):
            return await super().group_rollout(request)

        base = request.base_request
        k = base.rollouts_per_group
        n_candidates = k * request.oversample_factor

        # Build a request that uses the draft interface and asks for more rollouts.
        draft_request = base.model_copy(
            update={
                "rollouts_per_group": n_candidates,
                "inference_interface": request.draft_inference_interface,
            }
        )

        # Generate oversampled candidates (each is scored by the agent's reward).
        candidates = await super().group_rollout(draft_request)

        # Select the best k from the candidate pool.
        return select_rollouts(candidates, k, request.selection_strategy)
