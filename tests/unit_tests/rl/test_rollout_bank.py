# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Durable rollout-bank persistence, recovery, and compaction.

Format v3 stores one ledger record per rollout, so a group exists on disk only
through its members. Completeness is inferred from the member count rather than a
seal record, and a group that lost members to a kill reads back as *incomplete*,
carrying the problem state needed to regenerate the rest.
"""

import json
from contextlib import aclosing

import numpy as np
import pytest

from megatron.rl import rl_utils
from megatron.rl.agent.api import GroupedRolloutRequest, Rollout, RolloutGroup, TokenRollout
from megatron.rl.agent.weighted_multi_task import AgentConfig, WeightedMultiTask
from megatron.rl.rollout_bank import (
    _CONSUMED,
    _FORMAT_VERSION,
    _GENERATIONS,
    _LEDGER,
    _MANIFEST,
    _TOKENS_BIN,
    RolloutBank,
    _segment_name,
)
from megatron.rl.types import Rollout as SharedRollout
from megatron.rl.types import RolloutGroup as SharedRolloutGroup
from megatron.rl.types import TokenRollout as SharedTokenRollout
from tests.unit_tests.rl.test_grouped_rollouts import MockGenerator, MockInferenceInterface


def _token_group(batch_id=0, *, empty=False):
    members = (
        [([], [], [])]
        if empty
        else [
            (
                [[1, 2, 3], [4, 5]],
                [[-0.1, -0.2, -0.3], [-0.4, -0.5]],
                [[False, True, True], [True, True]],
            ),
            ([[7, 8]], [[-1.5, -2.5]], [[True, True]]),
        ]
    )
    return RolloutGroup(
        rollouts=[
            TokenRollout(
                trajectory=tokens,
                reward=1.0,
                logprobs=logprobs,
                generation_mask=mask,
                env_id="test",
                problem_id="p",
                policy_epoch=[[(0, 0)]],
                kv_cache_epoch=[[(0, 0)]],
                num_evictions=[0],
                completion_ids=[f"completion-{index}" for index in range(len(tokens))],
            )
            for tokens, logprobs, mask in members
        ],
        batch_id=batch_id,
    )


def _text_group():
    return RolloutGroup(
        rollouts=[
            Rollout(
                trajectory=["hello world"],
                reward=0.5,
                env_id="text",
                policy_epoch=[[(0, 0)]],
                kv_cache_epoch=[[(0, 0)]],
                num_evictions=[0],
            )
        ]
    )

PROBLEM = {"prompt": "Natalia sold clips to 48 friends", "golden": {"answer": "72"}}
ITER = 3


def _token(reward=1.0, tokens=((1, 2, 3), (4, 5))):
    trajectory = [list(turn) for turn in tokens]
    return TokenRollout(
        trajectory=trajectory,
        reward=reward,
        logprobs=[[-0.1 * (i + 1) for i in range(len(turn))] for turn in trajectory],
        generation_mask=[[True] * len(turn) for turn in trajectory],
        env_id="gsm8k",
        problem_id="p0",
        policy_epoch=[[(0, 0)] for _ in trajectory],
        kv_cache_epoch=[[(0, 0)] for _ in trajectory],
        num_evictions=[0 for _ in trajectory],
    )


def _text(reward=0.5):
    return Rollout(
        trajectory=["hello world"],
        reward=reward,
        env_id="text",
        policy_epoch=[[(0, 0)]],
        kv_cache_epoch=[[(0, 0)]],
        num_evictions=[0],
    )


def _bank(path, group_size=2, **kwargs):
    return RolloutBank(str(path), rollouts_per_group=group_size, **kwargs)


def _generation(bank_dir):
    manifest = json.loads((bank_dir / _MANIFEST).read_text())
    return bank_dir / _GENERATIONS / manifest["active_generation"]


def _segment(bank_dir, iteration=ITER):
    return _generation(bank_dir) / _segment_name(iteration)


def _records(bank_dir, iteration=ITER):
    ledger = _segment(bank_dir, iteration) / _LEDGER
    return [json.loads(line) for line in ledger.read_text().splitlines() if line.strip()]


def _write(bank, members, uid=None, problem_state=None, indices=None):
    """Persist one group's members, optionally at explicit slots."""
    uid = uid or bank.reserve_group_uid()
    if problem_state is not None:
        bank.append_problem(uid, problem_state)
    for slot, member in zip(indices or range(len(members)), members, strict=True):
        bank.append_rollout(uid, slot, member)
    return uid


def test_shared_types_are_the_same_objects_as_the_agent_api_re_exports():
    """api.py re-exports the megatron.rl.types models; drift would split the schema."""
    assert (SharedRollout, SharedRolloutGroup, SharedTokenRollout) == (
        Rollout,
        RolloutGroup,
        TokenRollout,
    )
    assert (
        Rollout(
            trajectory=["prompt"],
            reward=None,
            policy_epoch=[[(0, 0)]],
            kv_cache_epoch=[[(0, 0)]],
            num_evictions=[0],
        ).reward
        is None
    )


@pytest.mark.parametrize(
    "members, group_size",
    [
        pytest.param([_token(), _token(reward=0.0, tokens=((7, 8),))], 2, id="ragged-token"),
        pytest.param([_text()], 1, id="inline-text"),
        pytest.param([_token(tokens=())], 1, id="empty-trajectory"),
    ],
)
def test_payload_shapes_survive_a_restart(tmp_path, members, group_size):
    """Every supported member shape decodes back to what was written."""
    bank = _bank(tmp_path, group_size)
    bank.set_collection(ITER)
    uid = _write(bank, members)
    bank.close()

    records = _records(tmp_path)
    assert len(records) == group_size, "one record per member; no group record"
    assert {record["format_version"] for record in records} == {_FORMAT_VERSION}
    assert (_generation(tmp_path) / _CONSUMED).exists()

    restored = _bank(tmp_path, group_size).restore(trained_through=0)
    assert [group.uid for group in restored] == [uid]
    for actual, expected in zip(restored[0].rollouts, members, strict=True):
        assert type(actual) is type(expected)
        assert actual.trajectory == expected.trajectory
        assert actual.reward == expected.reward
        if isinstance(expected, TokenRollout):
            assert actual.generation_mask == expected.generation_mask
            for got, want in zip(actual.logprobs, expected.logprobs, strict=True):
                assert np.allclose(got, want, atol=1e-3)


@pytest.mark.parametrize(
    "members, indices, problem_state, drop_zero_variance, expect",
    [
        pytest.param([_token(0.0), _token(1.0)], None, PROBLEM, False, "complete", id="complete"),
        pytest.param([_token()], [0], PROBLEM, False, "incomplete", id="incomplete-restores"),
        pytest.param([_token()], [0], None, False, "dropped", id="incomplete-without-problem"),
        pytest.param([_token(1.0), _token(1.0)], None, None, True, "dropped", id="zero-variance"),
        pytest.param(
            [_token(1.0), _token(1.0)], None, None, False, "complete", id="zero-variance-allowed"
        ),
    ],
)
def test_restore_fold(tmp_path, members, indices, problem_state, drop_zero_variance, expect):
    """Completeness comes from the member count; there is no seal and no tombstone."""
    bank = _bank(tmp_path, 2, drop_zero_variance=drop_zero_variance)
    try:
        bank.set_collection(ITER)
        uid = _write(bank, members, problem_state=problem_state, indices=indices)
        restored = bank.restore(0)
    finally:
        bank.close()

    if expect == "dropped":
        assert restored == []
        return
    assert [group.uid for group in restored] == [uid]
    group = restored[0]
    assert group.is_complete(2) is (expect == "complete")
    if expect == "incomplete":
        assert group.problem_state == PROBLEM
        assert group.missing_indices(2) == [1]


def test_members_restore_in_slot_order_with_gaps_preserved(tmp_path):
    """Records interleave on disk, so restore must order by slot, not arrival."""
    bank = _bank(tmp_path, 4)
    try:
        bank.set_collection(ITER)
        uid = bank.reserve_group_uid()
        bank.append_problem(uid, PROBLEM)
        bank.append_rollout(uid, 2, _token(reward=1.0))
        bank.append_rollout(uid, 0, _token(reward=0.0))
        restored = bank.restore(0)
    finally:
        bank.close()

    assert restored[0].member_indices == [0, 2]
    assert [rollout.reward for rollout in restored[0].rollouts] == [0.0, 1.0]
    assert restored[0].missing_indices(4) == [1, 3]


def test_ledger_holds_only_problem_and_rollout_records(tmp_path):
    """Format v3 has exactly two record kinds."""
    bank = _bank(tmp_path, 2)
    try:
        bank.set_collection(ITER)
        uid = _write(bank, [_token(), _token()], problem_state=PROBLEM)
    finally:
        bank.close()

    records = _records(tmp_path)
    assert [record["kind"] for record in records] == ["problem", "rollout", "rollout"]
    assert records[1]["uid"] == f"{uid}#0"


def _tear_ledger(bank_dir):
    ledger = _segment(bank_dir) / _LEDGER
    lines = ledger.read_bytes().splitlines(keepends=True)
    ledger.write_bytes(b"".join(lines[:3]) + lines[3][: len(lines[3]) // 2])


def _corrupt_checksum(bank_dir):
    ledger = _segment(bank_dir) / _LEDGER
    records = [json.loads(line) for line in ledger.read_text().splitlines()]
    records[-1]["checksum"] = "0" * 32
    ledger.write_text("".join(json.dumps(record) + "\n" for record in records))


def _truncate_sidecar(bank_dir):
    tokens = _segment(bank_dir) / _TOKENS_BIN
    tokens.write_bytes(tokens.read_bytes()[:-1])


@pytest.mark.parametrize(
    "damage",
    [
        pytest.param(_tear_ledger, id="torn-final-record"),
        pytest.param(_corrupt_checksum, id="bad-checksum"),
        pytest.param(_truncate_sidecar, id="truncated-sidecar"),
    ],
)
def test_damage_drops_only_the_affected_group(tmp_path, damage):
    """A damaged record leaves its group short, which is then unrestorable.

    Losing a member can only make a group look *less* complete, never more, so
    corruption degrades into the incomplete-group case rather than corrupting.
    """
    bank = _bank(tmp_path, 2)
    bank.set_collection(ITER)
    durable = _write(bank, [_token(), _token()])
    _write(bank, [_token(), _token()])
    bank.close()

    damage(tmp_path)

    restarted = _bank(tmp_path, 2)
    restarted.set_collection(ITER)
    assert [group.uid for group in restarted.restore(trained_through=0)] == [durable]


def test_empty_group_persists_nothing(tmp_path):
    """A group exists on disk only through its members, so a memberless one cannot."""
    bank = _bank(tmp_path, 1)
    try:
        bank.set_collection(ITER)
        bank.append(RolloutGroup(rollouts=[]))
        assert (_segment(tmp_path) / _LEDGER).exists() is False
        assert bank.restore(trained_through=0) == []
    finally:
        bank.close()


def test_append_group_writes_each_member_and_keeps_its_uid(tmp_path):
    """The whole-group convenience path is the per-member path underneath."""
    bank = _bank(tmp_path, 2)
    try:
        bank.set_collection(ITER)
        group = RolloutGroup(rollouts=[_token(0.0), _token(1.0)])
        uid = bank.append(group, problem_state=PROBLEM)
        restored = bank.restore(0)
    finally:
        bank.close()

    assert [group.uid for group in restored] == [uid]
    assert restored[0].member_indices == [0, 1]
    assert restored[0].problem_state == PROBLEM


def test_uids_do_not_collide_across_bank_instances(tmp_path):
    """Uids carry a per-run nonce, so a restart cannot reuse one."""
    first, second = _bank(tmp_path / "a"), _bank(tmp_path / "b")
    try:
        uids = {first.reserve_group_uid() for _ in range(5)}
        uids |= {second.reserve_group_uid() for _ in range(5)}
        assert len(uids) == 10
    finally:
        first.close()
        second.close()


def test_rollouts_per_group_change_across_resume_is_rejected(tmp_path):
    """Completeness is inferred from the count, so the count must not shift."""
    bank = _bank(tmp_path, 4)
    try:
        bank.set_collection(ITER)
        _write(bank, [_token()], indices=[0])
    finally:
        bank.close()

    with pytest.raises(ValueError, match="rollouts_per_group"):
        _bank(tmp_path, 8)


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        pytest.param({}, [0, 1], id="defaults-to-positional"),
        pytest.param({"member_indices": [0, 2]}, [0, 2], id="explicit-slots-preserved"),
    ],
)
def test_member_indices_are_always_populated(kwargs, expected):
    """One source of truth: readers never fall back to positional slots."""
    assert RolloutGroup(rollouts=[_token(), _token()], **kwargs).member_indices == expected


def test_member_indices_must_match_the_member_count():
    """A slot list that disagrees with the members is a bug, not a fallback."""
    with pytest.raises(ValueError, match="member_indices"):
        RolloutGroup(rollouts=[_token()], member_indices=[0, 1])


@pytest.mark.parametrize(
    "consumed_offset, should_restore",
    [
        pytest.param(-1, False, id="consumed-before-checkpoint"),
        pytest.param(0, False, id="consumed-at-checkpoint"),
        pytest.param(1, True, id="consumed-after-checkpoint"),
    ],
)
def test_checkpoint_compacts_at_the_consumption_boundary(tmp_path, consumed_offset, should_restore):
    """Compaction prunes trained groups and carries forward future markers."""
    checkpoint = 10
    bank = _bank(tmp_path, 2)
    bank.set_collection(5)
    consumed = _write(bank, [_token(0.0), _token(1.0)])
    kept = _write(bank, [_token(0.0), _token(1.0)])
    bank.mark_consumed(consumed, checkpoint + consumed_offset)
    old_generation = _generation(tmp_path)

    before = {group.uid for group in bank.restore(checkpoint)}
    assert (consumed in before) is should_restore
    assert kept in before

    bank.checkpoint(checkpoint)
    manifest = json.loads((tmp_path / _MANIFEST).read_text())
    assert manifest["trained_through"] == checkpoint
    assert manifest["segments"] == [_segment_name(checkpoint)]
    assert not old_generation.exists()

    restarted = _bank(tmp_path, 2)
    restored = {group.uid for group in restarted.restore(checkpoint)}
    assert restored == ({consumed, kept} if should_restore else {kept})

    markers = [
        json.loads(line) for line in (_generation(tmp_path) / _CONSUMED).read_text().splitlines()
    ]
    expected_markers = (
        [{"uid": consumed, "iter": checkpoint + 1}] if should_restore else []
    )
    assert markers == expected_markers

    if should_restore:
        restarted.checkpoint(checkpoint + 1)
        assert {group.uid for group in restarted.restore(checkpoint + 1)} == {kept}


def test_checkpoint_preserves_the_active_collection_for_late_writes(tmp_path):
    """Async checkpoint finalization must not leave the background writer closed."""
    bank = _bank(tmp_path, 2)
    bank.set_collection(8)
    before = _write(bank, [_token(0.0), _token(1.0)])

    bank.checkpoint_preserving_collection(7)

    assert bank.collection_iteration == 8
    after = _write(bank, [_token(0.0), _token(1.0)])
    assert {group.uid for group in bank.restore(7)} == {before, after}


@pytest.mark.parametrize(
    "checkpoint_iterations",
    [pytest.param([7], id="one-boundary"), pytest.param([7, 8], id="two-boundaries")],
)
def test_checkpoint_preserves_problem_for_members_finishing_after_its_fifo_boundary(
    tmp_path, checkpoint_iterations
):
    """A problem before compaction must remain paired with members written after it."""
    bank = _bank(tmp_path, 2)
    bank.set_collection(9)
    uid = bank.reserve_group_uid()
    bank.append_problem(uid, PROBLEM)

    for iteration in checkpoint_iterations:
        bank.checkpoint_preserving_collection(iteration)

    bank.append_rollout(uid, 1, _token())
    bank.close()

    restored = _bank(tmp_path, 2).restore(checkpoint_iterations[-1])
    assert [group.uid for group in restored] == [uid]
    assert restored[0].problem_state == PROBLEM
    assert restored[0].member_indices == [1]
    assert restored[0].missing_indices(2) == [0]


def test_recovery_discards_problem_only_compaction_carriers(tmp_path):
    """After a restart no producer remains to finish a problem-only group."""
    bank = _bank(tmp_path, 2)
    bank.set_collection(8)
    uid = bank.reserve_group_uid()
    bank.append_problem(uid, PROBLEM)
    bank.checkpoint_preserving_collection(7)
    bank.close()

    restarted = _bank(tmp_path, 2)
    assert restarted.recover(7) == []
    assert restarted.restore(7) == []


def _env_group(env_id, problem_id):
    return RolloutGroup(
        rollouts=[
            Rollout(
                trajectory=["cached"],
                reward=1.0,
                env_id=env_id,
                problem_id=problem_id,
                policy_epoch=[[(0, 0)]],
                kv_cache_epoch=[[(0, 0)]],
                num_evictions=[0],
            )
        ]
    )


def _weighted_agent(env_weights):
    return WeightedMultiTask(
        [
            AgentConfig(agent_type=MockGenerator, agent_args={"env_id": env_id}, weight=weight)
            for env_id, weight in env_weights
        ]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "env_weights, restored_env, restored_count, request_groups, take_count, "
    "expected_envs, expected_fresh_calls, write_through",
    [
        pytest.param(
            [("a", 1.0), ("b", 1.0)],
            "a",
            2,
            4,
            4,
            {"a": 2, "b": 2},
            [0, 2],
            False,
            id="restored-groups-replace-fresh",
        ),
        pytest.param(
            [("a", 1.0), ("b", 1.0)],
            "a",
            4,
            4,
            12,
            {"a": 6, "b": 6},
            [2, 6],
            False,
            id="streaming-backlog-drains-before-fresh",
        ),
        pytest.param(
            [("", 1.0)], "", 1, 2, 2, {"": 2}, [1], False, id="single-environment-without-env-id"
        ),
        pytest.param(
            [("a", 1.0)], "a", 0, 4, 4, {"a": 4}, [4], True, id="fresh-groups-write-through"
        ),
    ],
)
async def test_pipeline_uses_restored_groups_before_fresh_generation(
    tmp_path,
    env_weights,
    restored_env,
    restored_count,
    request_groups,
    take_count,
    expected_envs,
    expected_fresh_calls,
    write_through,
):
    """Restored groups retain weighted routing; fresh groups remain durable."""
    agent = _weighted_agent(env_weights)
    restored = [
        _env_group(restored_env, problem_id=f"cached-{index}") for index in range(restored_count)
    ]
    assert agent.set_restored_groups(restored) == restored_count

    if len(env_weights) > 1:
        with pytest.raises(ValueError, match="not in the current"):
            agent.set_restored_groups([_env_group("unknown", "drift")])
        assert agent.set_restored_groups(restored) == restored_count

    bank = RolloutBank(str(tmp_path)) if write_through else None
    if bank is not None:
        bank.set_collection(0)
    agent._rollout_bank = bank
    request = GroupedRolloutRequest(
        num_groups=request_groups,
        rollouts_per_group=1,
        inference_interface=MockInferenceInterface(),
        streaming=take_count > request_groups,
    )

    async with aclosing(agent.get_grouped_rollouts(request)) as groups:
        produced = [await asyncio.wait_for(anext(groups), timeout=10) for _ in range(take_count)]

    assert len(produced) == take_count
    assert Counter(group[0].env_id for group in produced) == expected_envs
    assert [generator.prepare_group_rollout_calls for generator in agent.agents] == (
        expected_fresh_calls
    )
    cached_problem_ids = {f"cached-{index}" for index in range(restored_count)}
    assert cached_problem_ids <= {group[0].problem_id for group in produced}
    assert not agent._restored_groups.get(restored_env)

    if bank is not None:
        bank.close()
        persisted = RolloutBank(str(tmp_path)).restore(trained_through=0)
        assert len(persisted) == take_count
        assert {group.uid for group in persisted} == {group.uid for group in produced}


@pytest.mark.asyncio
async def test_restored_group_metrics_are_reported_per_step_and_env(monkeypatch):
    agent = _weighted_agent([("a", 1.0), ("b", 1.0)])
    agent.set_restored_groups(
        [
            *[_env_group("a", problem_id=f"a{index}") for index in range(3)],
            _env_group("b", problem_id="b0"),
        ]
    )
    request = GroupedRolloutRequest(
        num_groups=4,
        rollouts_per_group=1,
        inference_interface=MockInferenceInterface(num_slow_calls=100),
        streaming=True,
        submission_granularity="B",
        consumption_granularity="B",
    )
    monkeypatch.setattr(rl_utils, "_ROLLOUT_AGENT", agent)

    async with aclosing(agent.get_grouped_rollouts(request)) as groups:
        [await anext(groups) for _ in range(4)]
        first_step_metrics = rl_utils._collect_rollout_pipeline_metrics()

        assert first_step_metrics["rollout_pipeline_restored_count"] == 3
        assert first_step_metrics["rollout_pipeline_yielded_count"] == 4
        assert first_step_metrics["a_restored_groups"] == 2
        assert first_step_metrics["b_restored_groups"] == 1
        assert first_step_metrics["a_restored_groups_percentage"] == 100.0
        assert first_step_metrics["a_fresh_groups_percentage"] == 0.0
        assert first_step_metrics["b_restored_groups_percentage"] == 50.0
        assert first_step_metrics["b_fresh_groups_percentage"] == 50.0

        [await anext(groups) for _ in range(4)]
        second_step_metrics = rl_utils._collect_rollout_pipeline_metrics()

        assert second_step_metrics["rollout_pipeline_restored_count"] == 1
        assert second_step_metrics["rollout_pipeline_yielded_count"] == 4
        assert second_step_metrics["a_restored_groups"] == 1
        assert second_step_metrics["b_restored_groups"] == 0
        assert second_step_metrics["a_restored_groups_percentage"] == 50.0
        assert second_step_metrics["a_fresh_groups_percentage"] == 50.0
        assert second_step_metrics["b_restored_groups_percentage"] == 0.0
        assert second_step_metrics["b_fresh_groups_percentage"] == 100.0


def test_multiple_envs_require_env_ids_for_restore_routing():
    agent = _weighted_agent([("", 1.0), ("b", 1.0)])

    with pytest.raises(ValueError, match="configuring multiple active agents"):
        agent.set_restored_groups([])
