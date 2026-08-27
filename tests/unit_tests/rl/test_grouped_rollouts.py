# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import asyncio
import time
from collections import deque
from unittest.mock import MagicMock

import numpy as np
import pytest
from pydantic import ValidationError

from megatron.rl.agent.api import (
    GroupedRolloutGenerator,
    GroupedRolloutRequest,
    GroupRolloutParams,
    Rollout,
    RolloutGenerator,
    RolloutGroup,
    RolloutRequest,
    _BankRecordItem,
    _RolloutPipeline,
    _SubmissionGate,
)
from megatron.rl.agent.reward_only_agent import RewardOnlyAgent
from megatron.rl.agent.weighted_multi_task import AgentConfig, WeightedMultiTask
from megatron.rl.inference import InferenceResponse, LLMChatMessage, ReturnsRaw


class MockInferenceInterface(ReturnsRaw):
    """Mock raw-text inference interface with configurable per-prompt delays."""

    num_slow_calls: int = 0
    active_requests: int = 0
    max_active_requests: int = 0

    async def base_generate(self, request):
        prompt = request.prompt[0].content
        idx = int(prompt.removeprefix("t"))
        self.active_requests += 1
        self.max_active_requests = max(self.max_active_requests, self.active_requests)
        try:
            if idx < self.num_slow_calls:
                await asyncio.sleep(0.03)
            else:
                await asyncio.sleep(0)
            return InferenceResponse(
                response=LLMChatMessage(role="assistant", content=prompt),
                raw_text=prompt,
                finish_reason="stop",
                policy_epoch=[(0, 0)],
                kv_cache_epoch=[(0, 0)],
                num_evictions=0,
            )
        finally:
            self.active_requests -= 1


class MockGenerator(RolloutGenerator, GroupedRolloutGenerator):
    """Mock generator with configurable per-call delays."""

    def __init__(self, env_id="test", **kwargs):
        super().__init__(**kwargs)
        self.env_id = env_id
        self._call_count = 0
        self.prepare_group_rollout_calls = 0

    async def get_reward_rollouts(self, request):
        raise NotImplementedError

    async def get_rollout_response(self, request, inference_request):
        return await request.inference_interface.agenerate(inference_request)

    async def prepare_group_rollout(self, request, *, problem_state=None):
        idx = problem_state["idx"] if problem_state is not None else self._call_count
        self._call_count += 1
        self.prepare_group_rollout_calls += 1
        inference_request = request.inference_interface.prepare_request(
            f"t{idx}", request.generation_args
        )

        async def build_rollout(response):
            response_idx = int(response.response.content.removeprefix("t"))
            return Rollout(
                trajectory=[response.raw_text],
                reward=float(response_idx),
                env_id=self.env_id,
                policy_epoch=[response.policy_epoch],
                kv_cache_epoch=[response.kv_cache_epoch],
                num_evictions=[response.num_evictions],
            )

        return GroupRolloutParams(
            inference_request=inference_request,
            build_rollout=build_rollout,
            problem_state={"idx": idx},
        )


def test_grouped_rollout_generator_initializes_rollout_bank():
    generator = MockGenerator()
    assert generator._rollout_bank is None
    assert generator._restored_groups is None


def test_rollout_group_defines_optional_uid():
    assert RolloutGroup.model_fields["uid"].default is None


def _banked_pipeline(bank):
    request = GroupedRolloutRequest(
        num_groups=1,
        rollouts_per_group=1,
        inference_interface=MockInferenceInterface(),
        submission_granularity="R",
        consumption_granularity="G",
    )
    return _RolloutPipeline(MockGenerator(), request, 1, bank=bank)


def test_bank_writer_progresses_without_pumping_asyncio():
    """The writer keeps draining while the trainer owns the loop thread."""
    written = []
    bank = MagicMock()

    def slow_write(records):
        time.sleep(0.05)
        written.extend(records)

    bank.write_records.side_effect = slow_write
    pipeline = _banked_pipeline(bank)
    pipeline.bank_queue.put(_BankRecordItem("record"))
    deadline = time.monotonic() + 2
    while not written and time.monotonic() < deadline:
        time.sleep(0.01)
    assert written == ["record"]


def test_consumed_markers_follow_queued_records():
    ops = []
    bank = MagicMock()
    bank.write_records.side_effect = lambda records: ops.append(("write", list(records)))
    bank.mark_consumed_many.side_effect = lambda uids, iteration: ops.append(
        ("mark", list(uids), iteration)
    )
    pipeline = _banked_pipeline(bank)
    pipeline.bank_queue.put(_BankRecordItem("record"))
    pipeline.enqueue_consumed_markers(["group-1"], iteration=3)
    pipeline.drain_bank()
    assert ops == [("write", ["record"]), ("mark", ["group-1"], 3)]


def test_bank_drain_is_a_synchronous_barrier():
    bank = MagicMock()
    pipeline = _banked_pipeline(bank)
    pipeline.bank_queue.put(_BankRecordItem("record"))
    assert pipeline.drain_bank() is None
    bank.write_records.assert_called_once_with(["record"])


def test_weighted_environment_pipelines_share_one_bank_writer():
    ops = []
    bank = MagicMock()
    bank.write_records.side_effect = lambda records: ops.append(("write", list(records)))
    bank.mark_consumed_many.side_effect = lambda uids, iteration: ops.append(
        ("mark", list(uids), iteration)
    )
    first = _banked_pipeline(bank)
    second = _banked_pipeline(bank)
    assert first.bank_queue is second.bank_queue
    assert first._bank_writer is second._bank_writer
    first.bank_queue.put(_BankRecordItem("env-a"))
    second.bank_queue.put(_BankRecordItem("env-b"))
    second.enqueue_consumed_markers(["group-a", "group-b"], iteration=4)
    second.drain_bank()
    marker_index = next(i for i, op in enumerate(ops) if op[0] == "mark")
    written_before_marker = [record for op in ops[:marker_index] for record in op[1]]
    assert written_before_marker == ["env-a", "env-b"]


def test_collection_transition_is_ordered_between_record_batches():
    """Lifecycle items flush earlier records before switching the bank segment."""
    ops = []
    bank = MagicMock()
    bank.write_records.side_effect = lambda records: ops.append(("write", list(records)))
    bank.set_collection.side_effect = lambda iteration: ops.append(
        ("set_collection", iteration)
    )
    pipeline = _banked_pipeline(bank)
    pipeline.bank_queue.put(_BankRecordItem("old-record"))
    pipeline.set_bank_collection(8)
    pipeline.bank_queue.put(_BankRecordItem("new-record"))
    pipeline.drain_bank()
    assert ops == [
        ("write", ["old-record"]),
        ("set_collection", 8),
        ("write", ["new-record"]),
    ]


def test_checkpoint_transition_is_ordered_between_record_batches():
    """Checkpoint compaction executes in the writer FIFO, not beside it."""
    ops = []
    bank = MagicMock()
    bank.write_records.side_effect = lambda records: ops.append(("write", list(records)))
    bank.checkpoint_preserving_collection.side_effect = lambda iteration: ops.append(
        ("checkpoint", iteration)
    )
    pipeline = _banked_pipeline(bank)
    pipeline.bank_queue.put(_BankRecordItem("before-checkpoint"))
    pipeline.compact_bank_at_checkpoint(7)
    pipeline.bank_queue.put(_BankRecordItem("after-checkpoint"))
    pipeline.drain_bank()
    assert ops == [
        ("write", ["before-checkpoint"]),
        ("checkpoint", 7),
        ("write", ["after-checkpoint"]),
    ]


class CountingRewardAgent(RewardOnlyAgent):
    """Minimal RewardOnlyAgent: prompts t0, t1, ... and reward = echoed index."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.env_id = "reward-test"
        self._prompt_count = 0

    async def get_prompt(self, validation):
        idx = self._prompt_count
        self._prompt_count += 1
        return f"t{idx}", {"idx": idx}

    async def get_reward(self, response, golden, finish_reason):
        return float(int(response.removeprefix("t")) == golden["idx"])


async def _flush(rounds: int = 50):
    """Let pipeline stage tasks settle (mock inference is zero-delay)."""
    for _ in range(rounds):
        await asyncio.sleep(0)


class TestSubmissionGate:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("submission", ["R", "G", "B"])
    async def test_release_requires_matching_granularity(self, submission):
        gate = _SubmissionGate(capacity=1, submission=submission)
        await gate.acquire_for(submission)
        assert gate.held == 1
        for granularity in ("R", "G", "B"):
            if granularity == submission:
                continue
            gate.release_for(granularity)
        assert gate.held == 1
        assert gate.release_calls == 0
        gate.release_for(submission)
        assert gate.held == 0
        assert gate.release_calls == 1


class TestConsumptionRelease:
    """G-submission gate slots must recycle on trainer consumption, not assembly."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "consumption_granularity, num_groups",
        [
            pytest.param("G", 1, id="group_consumption"),
            pytest.param("B", 2, id="batch_consumption"),
        ],
    )
    async def test_group_submission_stalls_until_consumption(
        self, consumption_granularity, num_groups
    ):
        capacity = 4
        gen = MockGenerator(parallel_generation_tasks=capacity)
        request = GroupedRolloutRequest(
            num_groups=num_groups,
            rollouts_per_group=1,
            inference_interface=MockInferenceInterface(),
            streaming=True,
            submission_granularity="G",
            consumption_granularity=consumption_granularity,
        )
        it = gen.get_grouped_rollouts(request)
        try:
            for pulled in range(1, capacity + 3):
                # wait_for turns the deadlock failure mode (a slot never freed)
                # into a test failure instead of a hang.
                await asyncio.wait_for(anext(it), timeout=10)
                await _flush()
                # Each yield frees exactly one group slot on the consumer's next
                # resume, so submission tracks consumption with a one-slot skew
                # (the release for the latest pull hasn't fired yet). On
                # assembly-release semantics this runs away unbounded; if no
                # consume-site release existed, the loop would deadlock at
                # `pulled == capacity + 1`.
                assert gen.prepare_group_rollout_calls == capacity + pulled - 1
        finally:
            await it.aclose()

    @pytest.mark.asyncio
    async def test_batch_submission_releases_once_per_batch(self):
        gen = MockGenerator(parallel_generation_tasks=1)
        request = GroupedRolloutRequest(
            num_groups=2,
            rollouts_per_group=1,
            inference_interface=MockInferenceInterface(),
            streaming=True,
            submission_granularity="B",
            consumption_granularity="B",
        )
        it = gen.get_grouped_rollouts(request)
        try:
            await asyncio.wait_for(anext(it), timeout=10)
            await asyncio.wait_for(anext(it), timeout=10)
            await _flush()
            gate = gen._active_pipeline.gate
            # Batch 0 fully yielded but the consumer hasn't come back yet: its
            # single batch slot is still held (a per-group release here would
            # show release_calls == 2 and prepared == 4).
            assert gate.release_calls == 0
            assert gen.prepare_group_rollout_calls == 2
            await asyncio.wait_for(anext(it), timeout=10)
            await _flush()
            assert gate.release_calls == 1
            assert gen.prepare_group_rollout_calls == 4
        finally:
            await it.aclose()


class TestRewardRollouts:
    @pytest.mark.asyncio
    async def test_get_reward_rollouts_matches_per_rollout_composition(self):
        agent = CountingRewardAgent()
        request = RolloutRequest(num_rollouts=4, inference_interface=MockInferenceInterface())
        rollouts = await agent.get_reward_rollouts(request)
        assert len(rollouts) == 4
        assert sorted(r.trajectory[0] for r in rollouts) == ["t0", "t1", "t2", "t3"]
        assert all(r.reward == 1.0 for r in rollouts)
        assert all(r.env_id == "reward-test" for r in rollouts)


class TestGroupedRollouts:
    @pytest.mark.parametrize("field", ["submission_granularity", "consumption_granularity"])
    def test_grouped_rollout_request_rejects_unknown_granularity(self, field):
        request_kwargs = {
            "num_groups": 1,
            "rollouts_per_group": 1,
            "inference_interface": MagicMock(spec=ReturnsRaw),
            field: "X",
        }
        with pytest.raises(ValidationError) as exc_info:
            GroupedRolloutRequest(**request_kwargs)
        assert any(error["loc"] == (field,) for error in exc_info.value.errors())

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "num_groups, submission_granularity, consumption_granularity",
        [
            pytest.param(1, "B", "B", id="num_groups_1_batch"),
            pytest.param(4, "G", "G", id="num_groups_gt_1_group"),
            pytest.param(4, "R", "B", id="num_groups_gt_1_rollout"),
        ],
    )
    async def test_filter_groups_with_same_reward_rejected(
        self, num_groups, submission_granularity, consumption_granularity
    ):
        gen = MockGenerator(parallel_generation_tasks=8)
        request = GroupedRolloutRequest(
            num_groups=num_groups,
            rollouts_per_group=2,
            inference_interface=MockInferenceInterface(),
            filter_groups_with_same_reward=True,
            submission_granularity=submission_granularity,
            consumption_granularity=consumption_granularity,
        )
        with pytest.raises(AssertionError, match="filter_groups_with_same_reward"):
            async for _ in gen.get_grouped_rollouts(request):
                pass

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        (
            "num_slow_calls, streaming, num_groups, submission_granularity, "
            "consumption_granularity, expected_count, expected_batch_ids, "
            "expected_trajectories"
        ),
        [
            pytest.param(0, False, 8, "B", "B", 8, None, None, id="non_batched"),
            pytest.param(
                0, False, 4, "B", "B", 4, None, None, id="non_streaming_fewer_than_parallel"
            ),
            pytest.param(
                4,
                True,
                2,
                "B",
                "B",
                8,
                [0, 0, 1, 1, 2, 2, 3, 3],
                None,
                id="batched_submission_order",
            ),
            pytest.param(0, True, 1, "G", "B", 10, None, None, id="streaming"),
            pytest.param(
                4,
                True,
                1,
                "G",
                "G",
                8,
                None,
                [f"t{i}" for i in range(4, 8)],
                id="group_consume_completion_order",
            ),
            pytest.param(
                4,
                True,
                1,
                "G",
                "B",
                8,
                list(range(8)),
                [f"t{i}" for i in range(8)],
                id="batch_consume_submission_order",
            ),
        ],
    )
    async def test_get_grouped_rollouts(
        self,
        num_slow_calls,
        streaming,
        num_groups,
        submission_granularity,
        consumption_granularity,
        expected_count,
        expected_batch_ids,
        expected_trajectories,
    ):
        gen = MockGenerator(parallel_generation_tasks=8)
        request = GroupedRolloutRequest(
            num_groups=num_groups,
            rollouts_per_group=1,
            inference_interface=MockInferenceInterface(num_slow_calls=num_slow_calls),
            streaming=streaming,
            submission_granularity=submission_granularity,
            consumption_granularity=consumption_granularity,
        )

        groups = []
        async for group in gen.get_grouped_rollouts(request):
            groups.append(group)
            if request.streaming and len(groups) >= expected_count:
                break

        assert len(groups) == expected_count
        if expected_batch_ids is not None:
            assert [g.batch_id for g in groups] == expected_batch_ids
        if expected_trajectories is not None:
            trajectories = [group[0].trajectory[0] for group in groups]
            assert trajectories[: len(expected_trajectories)] == expected_trajectories

    @pytest.mark.asyncio
    async def test_batch_order_starts_at_initial_batch_id(self):
        gen = MockGenerator(parallel_generation_tasks=1)
        request = GroupedRolloutRequest(
            num_groups=2,
            rollouts_per_group=1,
            inference_interface=MockInferenceInterface(),
            streaming=True,
            submission_granularity="B",
            consumption_granularity="B",
            initial_batch_id=10,
        )

        groups = []
        async for group in gen.get_grouped_rollouts(request):
            groups.append(group)
            if len(groups) == 4:
                break

        assert [group.batch_id for group in groups] == [10, 10, 11, 11]
        assert [group.index_in_batch for group in groups] == [0, 1, 0, 1]

    @pytest.mark.asyncio
    async def test_restored_group_replaces_fresh_work_with_current_batch_metadata(self):
        gen = MockGenerator(parallel_generation_tasks=1)
        restored = RolloutGroup(
            rollouts=[
                Rollout(
                    trajectory=["cached"],
                    reward=1.0,
                    env_id="test",
                    policy_epoch=[[(0, 0)]],
                    kv_cache_epoch=[[(0, 0)]],
                    num_evictions=[0],
                )
            ],
            batch_id=99,
            index_in_batch=99,
        )
        gen._restored_groups = deque([restored])
        request = GroupedRolloutRequest(
            num_groups=2,
            rollouts_per_group=1,
            inference_interface=MockInferenceInterface(),
            submission_granularity="B",
            consumption_granularity="B",
            initial_batch_id=10,
        )

        groups = [group async for group in gen.get_grouped_rollouts(request)]

        assert [group.batch_id for group in groups] == [10, 10]
        assert [group.index_in_batch for group in groups] == [0, 1]
        assert groups[0][0].trajectory == ["cached"]
        assert gen.prepare_group_rollout_calls == 1
        assert gen._active_pipeline is None

    @pytest.mark.asyncio
    async def test_rollout_submission_granularity_limits_inference_concurrency(self):
        gen = MockGenerator(parallel_generation_tasks=2)
        inference_interface = MockInferenceInterface(num_slow_calls=100)
        request = GroupedRolloutRequest(
            num_groups=1,
            rollouts_per_group=4,
            inference_interface=inference_interface,
            streaming=True,
            submission_granularity="R",
            consumption_granularity="B",
        )

        groups = []
        async for group in gen.get_grouped_rollouts(request):
            groups.append(group)
            break

        assert len(groups) == 1
        assert len(groups[0]) == 4
        assert inference_interface.max_active_requests <= gen.parallel_generation_tasks

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "submission_granularity, consumption_granularity, expected_parallel_generation_tasks",
        [
            pytest.param("B", "B", [4, 4], id="batch_submission"),
            pytest.param("G", "G", [3, 1], id="group_submission"),
        ],
    )
    async def test_weighted_multi_task(
        self, submission_granularity, consumption_granularity, expected_parallel_generation_tasks
    ):
        configs = [
            AgentConfig(agent_type=MockGenerator, agent_args={"env_id": "a"}, weight=3.0),
            AgentConfig(agent_type=MockGenerator, agent_args={"env_id": "b"}, weight=1.0),
        ]
        mt = WeightedMultiTask(configs)
        mt.parallel_generation_tasks = 4

        captured = []
        for agent in mt.agents:
            original = agent.get_grouped_rollouts

            async def spy(req, orig=original):
                captured.append(req)
                async for group in orig(req):
                    yield group

            agent.get_grouped_rollouts = spy

        request = GroupedRolloutRequest(
            num_groups=4,
            rollouts_per_group=1,
            inference_interface=MockInferenceInterface(),
            streaming=False,
            submission_granularity=submission_granularity,
            consumption_granularity=consumption_granularity,
        )
        groups = []
        async for group in mt.get_grouped_rollouts(request):
            groups.append(group)

        assert len(groups) == 4
        # Weights 3:1 → agent "a" produces 3 groups, agent "b" produces 1.
        env_ids = [g[0].env_id for g in groups]
        assert sorted(env_ids) == ["a", "a", "a", "b"]
        for sub_req in captured:
            assert sub_req.num_groups in (1, 3)  # distributed proportionally by weight
            assert sub_req.streaming == request.streaming
            assert sub_req.submission_granularity == request.submission_granularity
            assert sub_req.consumption_granularity == request.consumption_granularity
            assert sub_req.initial_batch_id == request.initial_batch_id
        assert [agent.parallel_generation_tasks for agent in mt.agents] == (
            expected_parallel_generation_tasks
        )

    @pytest.mark.parametrize(
        "num_groups, all_envs_active",
        [
            pytest.param(1, False, id="num_groups_1_starves_an_env"),
            pytest.param(8, True, id="trainer_batch_size_keeps_all_envs_active"),
        ],
    )
    def test_multi_env_distribution_requires_num_groups_above_one(
        self, num_groups, all_envs_active
    ):
        """Regression for the removed ``num_groups=1`` streaming override.

        With multiple weighted environments, ``num_groups=1`` hands the single
        group to one environment and leaves the other with zero groups. It also
        collapses ``agent_slots`` (computed without remainder distribution) to all
        zeros, so ``np.gcd.reduce`` is 0 and the per-agent slot counts become
        ``nan`` -- which stalls ``get_grouped_rollouts``. Keeping ``num_groups`` at
        the trainer batch size (> 1) keeps every environment active with a valid,
        non-zero slot count.
        """
        configs = [
            AgentConfig(agent_type=MockGenerator, agent_args={"env_id": "a"}, weight=3.0),
            AgentConfig(agent_type=MockGenerator, agent_args={"env_id": "b"}, weight=1.0),
        ]
        mt = WeightedMultiTask(configs)

        agent_groups = mt._distribute_counts(num_groups)
        agent_slots = mt._distribute_counts(num_groups, distribute_remainder=False)

        assert all(groups > 0 for groups in agent_groups) is all_envs_active
        if all_envs_active:
            assert min(agent_slots) > 0
            assert np.gcd.reduce(agent_slots) > 0
        else:
            assert min(agent_groups) == 0
            assert all(slots == 0 for slots in agent_slots)
            assert np.gcd.reduce(agent_slots) == 0
