# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import asyncio
import logging
import queue as thread_queue
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncIterator, Awaitable, Callable, Generic, Iterable, NamedTuple, TypeAlias, TypeVar

import numpy as np

from megatron.core.inference.utils import asyncio_Queue, asyncio_QueueShutDown
from megatron.core.utils import trace_async_exceptions

from ..__init__ import Request, TypeLookupable
from ..inference import (
    InferenceInterface,
    InferenceRequest,
    InferenceResponse,
    LLMChatMessage,
    ReturnsRaw,
)
from ..inflight_tracker import add_inflight, remove_inflight
from ..rollout_bank import RolloutBank, _PendingProblem, _PendingRollout
from ..rollout_granularity import ConsumptionGranularity, SubmissionGranularity
from ..types import (
    AgentBaseModel,
    EnvId,
    GroupedRollouts,
    GroupQueuesPerEnv,
    Rollout,
    RolloutGroup,
    Rollouts,
    TokenRollout,
)


BANK_WRITE_MAX_RECORDS = 64


class _BankQueueItem(ABC):
    """An item that can be processed by the rollout-bank writer."""

    @abstractmethod
    def process(self, writer: "_RolloutPipeline", pending_records: list) -> None:
        """Apply this item in FIFO order."""


@dataclass(frozen=True)
class _BankRecordItem(_BankQueueItem):
    """A problem or rollout record waiting to be coalesced and persisted."""

    record: _PendingProblem | _PendingRollout

    def process(self, writer: "_RolloutPipeline", pending_records: list) -> None:
        pending_records.append(self.record)


@dataclass(frozen=True)
class _ConsumedBankGroups(_BankQueueItem):
    """Consumption markers queued behind the rollout records they name."""

    uids: tuple[str, ...]
    iteration: int

    def process(self, writer: "_RolloutPipeline", pending_records: list) -> None:
        writer._flush_bank_records(pending_records)
        writer._mark_consumed_latching(self.uids, self.iteration)


@dataclass(kw_only=True)
class _BankLifecycleItem(_BankQueueItem, ABC):
    """A synchronous bank lifecycle boundary carried by the writer FIFO."""

    done: threading.Event = field(default_factory=threading.Event)
    error: BaseException | None = None

    def process(self, writer: "_RolloutPipeline", pending_records: list) -> None:
        writer._flush_bank_records(pending_records)
        try:
            writer._raise_latched_bank_error()
            self.execute(writer.bank)
        except BaseException as exc:
            self.error = exc
        finally:
            # The submitting thread must never hang, even when lifecycle work fails.
            self.done.set()

    @abstractmethod
    def execute(self, bank: RolloutBank) -> None:
        """Apply this lifecycle transition on the bank-writer thread."""


@dataclass(kw_only=True)
class _SetRolloutBankCollection(_BankLifecycleItem):
    """Switch subsequent records to an iteration's collection segment."""

    iteration: int

    def execute(self, bank: RolloutBank) -> None:
        bank.set_collection(self.iteration)


@dataclass(kw_only=True)
class _CompactRolloutBankCheckpoint(_BankLifecycleItem):
    """Compact a checkpoint without leaving the live collection closed."""

    iteration: int

    def execute(self, bank: RolloutBank) -> None:
        bank.checkpoint_preserving_collection(self.iteration)


# TODO: Move these models to ``megatron.rl.types`` after moving ``Request``,
# ``InferenceInterface``, and their dependencies there to avoid circular imports.
class RolloutRequest(Request):
    """Request to agent to generate Rollouts."""

    num_rollouts: int
    inference_interface: InferenceInterface
    validation: bool = False


class GroupedRolloutRequest(Request):
    """Request to agent to generate grouped Rollouts."""

    num_groups: int
    rollouts_per_group: int
    inference_interface: InferenceInterface
    validation: bool = False
    filter_groups_with_same_reward: bool = False
    streaming: bool = False
    submission_granularity: SubmissionGranularity = "B"
    consumption_granularity: ConsumptionGranularity = "B"
    initial_batch_id: int = 0


class GroupRolloutParams(NamedTuple):
    """Returned by agent.prepare_group_rollout.

    One instance is created per group call and reused for all rollouts in that group.
    ``problem_state`` is optional agent-owned state used to regenerate missing
    members of a restored group against the same problem.
    """

    inference_request: InferenceRequest
    build_rollout: Callable[[InferenceResponse], Awaitable[Rollout]]
    problem_state: dict | None = None


class ContrastiveRollout(AgentBaseModel):
    """Contrastive/Preference data for language-based Rollout."""

    chosen_trajectory: list[str]
    rejected_trajectory: list[str]


class Head2HeadRolloutRequest(Request):
    num_rollouts: int
    inference_interface: list[InferenceInterface]
    validation: bool = False


class EvaluationRequest(Request):
    """Request to evaluate N prompts, optionally distributed across ranks."""

    inference_interface: InferenceInterface
    num_prompts: int
    rank_info: tuple[int, int] | None = (
        None  # (rank, total_ranks) if distributed, None for full evaluation
    )
    validation: bool = True


class EvaluationResult(AgentBaseModel):
    prompt: str | list[LLMChatMessage]
    response: str | LLMChatMessage


class RewardEvaluationResult(EvaluationResult):
    reward: float
    problem_id: str | None = None


T = TypeVar('T', bound=EvaluationResult)


class EvaluationResponse(AgentBaseModel, TypeLookupable, Generic[T]):
    env_id: str
    results: list[T]

    def metrics(self):
        raise NotImplementedError(f"{type(self)} did not provide metric aggregation.")


class Agent(ABC, AgentBaseModel):

    @abstractmethod
    async def get_rollout_response(
        self,
        request: "RolloutRequest | GroupedRolloutRequest | EvaluationRequest",
        inference_request: InferenceRequest,
    ) -> InferenceResponse:
        """Obtain the model response for a single rollout. Subclasses implement how."""
        ...


class RolloutGenerator(Agent, ABC):
    """An agent that produces Rollout objects containing rollout string and associated reward."""

    @abstractmethod
    async def get_reward_rollouts(self, request: RolloutRequest) -> list[Rollout]: ...


class ContrastiveRolloutGenerator(Agent, ABC):
    """An agent that produces ContrastiveRollout objects containing two rollout strings, one chosen and one rejected."""

    @abstractmethod
    async def get_contrastive_rollouts(
        self, request: RolloutRequest
    ) -> list[ContrastiveRollout]: ...


class TokenizedRolloutGenerator(Agent, ABC):
    """An agent that produces TokenRollout objects containing rollout token ids and associated rewards.

    Optionally can also provide generation masks to indicate which tokens were generated and token masks to indicate which
    tokens were possible at any given step.
    """

    @abstractmethod
    async def get_reward_rollouts(self, request: RolloutRequest) -> list[TokenRollout]: ...


class _GranularityConfig(NamedTuple):
    submission: SubmissionGranularity
    consumption: ConsumptionGranularity
    num_groups_per_batch: int

    @classmethod
    def from_request(cls, request: GroupedRolloutRequest) -> "_GranularityConfig":
        cls._validate(request)
        return cls(
            submission=request.submission_granularity,
            consumption=request.consumption_granularity,
            num_groups_per_batch=request.num_groups,
        )

    @property
    def prevent_dataset_reorder(self) -> bool:
        return self.consumption == "B"

    @staticmethod
    def _validate(request: GroupedRolloutRequest) -> None:
        assert not (
            request.submission_granularity == "B" and request.consumption_granularity == "G"
        ), "Batch submission with group consumption is not supported."
        assert not request.filter_groups_with_same_reward, (
            "filter_groups_with_same_reward is not currently supported: dropped groups "
            "are not regenerated, so non-streaming callers receive fewer groups than "
            "requested and batch-order consumers stall on incomplete batches."
        )


class _SubmissionGate:
    """Gate capacity is measured in units of the configured submission granularity.

    Each granularity has a single release point: R slots free when inference
    completes, so the gate bounds engine concurrency in rollouts. G and B
    slots free when the trainer consumes the group/batch, so the gate
    enforces the --rl-generation-lag run-ahead cap in groups/batches
    respectively.
    """

    def __init__(
        self,
        *,
        capacity: int,
        submission: SubmissionGranularity,
    ) -> None:
        self._sem = asyncio.Semaphore(capacity)
        self._submission = submission
        self.capacity = capacity
        # Observability counters, updated only on the configured submission
        # granularity (the only path that touches the semaphore). `held`
        # counts slots currently held; `prepare_blocked_seconds` accumulates
        # time stage_prepare spent waiting on the semaphore.
        self.held = 0
        self.prepare_blocked_seconds = 0.0
        self.acquire_calls = 0
        self.release_calls = 0

    async def acquire_for(self, granularity: SubmissionGranularity) -> None:
        if self._submission == granularity:
            start = time.monotonic()
            await self._sem.acquire()
            self.prepare_blocked_seconds += time.monotonic() - start
            self.held += 1
            self.acquire_calls += 1

    def release_for(self, granularity: SubmissionGranularity) -> None:
        if self._submission == granularity:
            self._sem.release()
            self.held -= 1
            self.release_calls += 1


class _InferWorkItem(NamedTuple):
    """One rollout's worth of work flowing from prepare to infer.

    Timestamps are wall-clock monotonic seconds: `prepared_at` is stamped at
    construction and `infer_dequeued_at` is filled in via `_replace` when an
    infer worker dequeues the item. Zero means "not yet reached".
    """

    group_id: int
    rollout_idx: int
    batch_id: int
    index_in_batch: int
    params: GroupRolloutParams
    bank_uid: str | None = None
    prepared_at: float = 0.0
    infer_dequeued_at: float = 0.0


class _InferredItem(NamedTuple):
    """One rollout post-inference, flowing from infer to assemble."""

    item: _InferWorkItem
    response: InferenceResponse
    inferred_at: float = 0.0


class _RolloutPipeline:
    """Per-call orchestrator for grouped rollout generation."""

    def __init__(
        self,
        agent: "GroupedRolloutGenerator",
        request: GroupedRolloutRequest,
        parallel_generation_tasks: int,
        bank: RolloutBank | None = None,
    ) -> None:
        self.agent = agent
        self.request = request
        # Optional durable rollout bank. Injected (never read from a global) so the
        # pipeline stays testable; when None, all bank calls are skipped.
        self.bank = bank
        self.initial_batch_id = request.initial_batch_id
        self.gran_policy = _GranularityConfig.from_request(request)
        self.gate = _SubmissionGate(
            capacity=parallel_generation_tasks,
            submission=self.gran_policy.submission,
        )
        rollouts_per_submission_unit = {
            "R": 1,
            "G": request.rollouts_per_group,
            "B": self.gran_policy.num_groups_per_batch * request.rollouts_per_group,
        }[self.gran_policy.submission]
        self.num_infer_workers = parallel_generation_tasks * rollouts_per_submission_unit
        if not request.streaming:
            self.num_infer_workers = min(
                self.num_infer_workers, request.num_groups * request.rollouts_per_group
            )
        self.infer_queue = asyncio_Queue()
        self.assemble_queue = asyncio_Queue()
        # Unbounded: flow control is owned entirely by the submission gate.
        # Bounding this queue would add a second backpressure that silently
        # clamps the run-ahead configured via --rl-generation-lag.
        self.output_queue = asyncio_Queue()
        # The trainer does not pump asyncio while it trains. Keep durability
        # writes moving independently so the next collection does not inherit
        # the entire previous iteration's fsync backlog.
        self.bank_queue: thread_queue.Queue = thread_queue.Queue()
        self._bank_error: Exception | None = None
        self._bank_writer: threading.Thread | None = None
        if bank is not None:
            # This branch's weighted agent owns one pipeline per environment,
            # unlike the newer single-pipeline architecture of cb555e8c8. Share
            # one FIFO and writer across all of them so RolloutBank remains a
            # single-writer store and markers order behind every producer.
            owner = getattr(bank, "__dict__", {}).get("_pipeline_writer_owner")
            if owner is None:
                owner = self
                bank._pipeline_writer_owner = owner
                self._bank_writer = threading.Thread(
                    target=self._bank_writer_loop,
                    name="rollout-bank-writer",
                    daemon=True,
                )
                self._bank_writer.start()
            else:
                self.bank_queue = owner.bank_queue
                self._bank_writer = owner._bank_writer
            self._bank_writer_owner = owner
        else:
            self._bank_writer_owner = self
        # Buffer of pending groups (incomplete groups being filled by
        # stage_assemble). Held here so metric collection can report its size.
        self._assemble_pending: dict[int, dict[int, Rollout]] = {}
        # Pending groups waiting for their batch to fill in stage_consume
        # (only populated when prevent_dataset_reorder is True).
        self._consume_pending: dict[int, list[RolloutGroup]] = {}
        # Per-group "output entry" times, keyed by (batch_id, index_in_batch),
        # so stage_consume can compute output_queue_dwell when yielding.
        self._output_enqueued_at: dict[tuple[int, int], float] = {}
        self._restored_output_keys: set[tuple[int, int]] = set()
        # Observability accumulators. Measured here; snapshot/reset and
        # wandb formatting happen in rl_utils during metric logging.
        self.infer_queue_dwell: list[float] = []
        self.engine_dwell: list[float] = []
        self.assemble_queue_dwell: list[float] = []
        self.output_queue_dwell: list[float] = []
        self.prepared_count = 0
        self.inferred_count = 0
        self.assembled_count = 0
        self.restored_count = 0
        self.yielded_count = 0

    async def _submit_group(
        self,
        *,
        group_id: int,
        batch_id: int,
        index_in_batch: int,
        restored: RolloutGroup | None = None,
    ) -> None:
        """Prepare a fresh or incomplete group and enqueue only missing members."""
        params = await self.agent.prepare_group_rollout(
            self.request, problem_state=restored.problem_state if restored else None
        )
        if restored is not None:
            bank_uid = restored.uid
            indices = restored.missing_indices(self.request.rollouts_per_group)
            self._assemble_pending[group_id] = dict(
                zip(restored.member_indices, restored.rollouts, strict=True)
            )
        else:
            bank_uid = self.bank.reserve_group_uid() if self.bank is not None else None
            indices = list(range(self.request.rollouts_per_group))
            if self.bank is not None and params.problem_state is not None:
                self.bank_queue.put_nowait(
                    _BankRecordItem(_PendingProblem(bank_uid, params.problem_state))
                )

        add_inflight(self.request.rollouts_per_group)
        for rollout_idx in indices:
            await self.gate.acquire_for("R")
            await self.infer_queue.put(
                _InferWorkItem(
                    group_id=group_id,
                    rollout_idx=rollout_idx,
                    batch_id=batch_id,
                    index_in_batch=index_in_batch,
                    params=params,
                    bank_uid=bank_uid,
                    prepared_at=time.monotonic(),
                )
            )
            self.prepared_count += 1

    async def stage_prepare(self) -> None:
        """Generate gated inference work items."""
        assert (
            self.request.streaming
            or self.request.num_groups % self.gran_policy.num_groups_per_batch == 0
        ), "non-streaming requires num_groups to be a multiple of num_groups_per_batch"
        group_id = 0
        try:
            while self.request.streaming or group_id < self.request.num_groups:
                await self.gate.acquire_for("B")
                batch_id = self.initial_batch_id + group_id // self.gran_policy.num_groups_per_batch

                for index_in_batch in range(self.gran_policy.num_groups_per_batch):
                    await self.gate.acquire_for("G")
                    restored = self.agent.take_restored_group()
                    if restored is not None:
                        expected_env_id = getattr(self.agent, "env_id", "")
                        assert all(rollout.env_id == expected_env_id for rollout in restored), (
                            f"Restored rollout group routed to env {expected_env_id!r} contains "
                            f"members for {[rollout.env_id for rollout in restored]}"
                        )
                        output_key = (batch_id, index_in_batch)
                        self._restored_output_keys.add(output_key)
                        if restored.is_complete(self.request.rollouts_per_group):
                            restored.batch_id = batch_id
                            restored.index_in_batch = index_in_batch
                            self._output_enqueued_at[output_key] = time.monotonic()
                            add_inflight(len(restored))
                            await self.output_queue.put(restored)
                            group_id += 1
                            continue
                    await self._submit_group(
                        group_id=group_id,
                        batch_id=batch_id,
                        index_in_batch=index_in_batch,
                        restored=restored,
                    )
                    group_id += 1
        finally:
            self.infer_queue.shutdown()

    def _bank_writer_loop(self) -> None:
        """Write bank items on a dedicated thread for the pipeline lifetime."""
        while True:
            items = [self.bank_queue.get()]
            while len(items) < BANK_WRITE_MAX_RECORDS:
                try:
                    items.append(self.bank_queue.get_nowait())
                except thread_queue.Empty:
                    break
            try:
                self._process_bank_items(items)
            finally:
                for _ in items:
                    self.bank_queue.task_done()

    def _process_bank_items(self, items: list[_BankQueueItem]) -> None:
        """Process bank items polymorphically in FIFO order."""
        pending_records = []
        for item in items:
            item.process(self, pending_records)
        self._flush_bank_records(pending_records)

    def _flush_bank_records(self, pending_records: list) -> None:
        """Persist and clear one coalesced batch of problem/rollout records."""
        if not pending_records:
            return
        records = list(pending_records)
        pending_records.clear()
        self._write_records_latching(records)

    def _write_records_latching(self, records: list) -> None:
        if not records:
            return
        try:
            self.bank.write_records(records)
        except Exception as exc:
            self._bank_error = exc
            logging.getLogger(__name__).exception(
                "Rollout bank write failed; %d record(s) lost", len(records)
            )

    def _mark_consumed_latching(self, uids: tuple[str, ...], iteration: int) -> None:
        try:
            self.bank.mark_consumed_many(uids, iteration)
        except Exception as exc:
            self._bank_error = exc
            logging.getLogger(__name__).exception(
                "Rollout bank consumption-marker write failed; %d marker(s) lost",
                len(uids),
            )

    def _raise_latched_bank_error(self) -> None:
        if self._bank_error is not None:
            error, self._bank_error = self._bank_error, None
            raise error

    def enqueue_consumed_markers(self, uids: Iterable[str | None], iteration: int) -> None:
        """Queue markers after this producer's records without blocking training."""
        if self.bank is None:
            return
        owner = self._bank_writer_owner
        owner._raise_latched_bank_error()
        markers = _ConsumedBankGroups(
            tuple(uid for uid in uids if uid), iteration
        )
        if not markers.uids:
            return
        owner.bank_queue.put(markers)
        if owner._bank_writer is None or not owner._bank_writer.is_alive():
            owner.drain_bank()

    def _submit_bank_lifecycle_item(self, item: _BankLifecycleItem) -> None:
        """Submit a FIFO lifecycle boundary and wait for its acknowledgement."""
        if self.bank is None:
            return
        owner = self._bank_writer_owner
        owner._raise_latched_bank_error()
        owner.bank_queue.put(item)
        if owner._bank_writer is not None and owner._bank_writer.is_alive():
            item.done.wait()
        else:
            owner.drain_bank()
        if item.error is not None:
            raise item.error
        owner._raise_latched_bank_error()

    def set_bank_collection(self, iteration: int) -> None:
        """Switch collection after all earlier FIFO records are durable."""
        self._submit_bank_lifecycle_item(
            _SetRolloutBankCollection(iteration=iteration)
        )

    def compact_bank_at_checkpoint(self, iteration: int) -> None:
        """Compact after all earlier FIFO records and markers are durable."""
        self._submit_bank_lifecycle_item(
            _CompactRolloutBankCheckpoint(iteration=iteration)
        )

    def drain_bank(self) -> None:
        """Block until all queued rollout-bank records and markers are durable."""
        if self.bank is None:
            return
        owner = self._bank_writer_owner
        if owner is not self:
            return owner.drain_bank()
        if self._bank_writer is not None and self._bank_writer.is_alive():
            self.bank_queue.join()
        else:
            pending = []
            while True:
                try:
                    pending.append(self.bank_queue.get_nowait())
                except thread_queue.Empty:
                    break
            try:
                self._process_bank_items(pending)
            finally:
                for _ in pending:
                    self.bank_queue.task_done()
        self._raise_latched_bank_error()

    async def stage_infer(self) -> None:
        """Run a persistent pool of inference workers, spawned once per pipeline."""
        workers = [
            asyncio.create_task(self._infer_worker()) for _ in range(self.num_infer_workers)
        ]
        try:
            await asyncio.gather(*workers, return_exceptions=True)
        finally:
            for worker in workers:
                worker.cancel()
            self.assemble_queue.shutdown()

    async def _infer_worker(self) -> None:
        while True:
            try:
                item = await self.infer_queue.get()
            except asyncio_QueueShutDown:
                return
            item = item._replace(infer_dequeued_at=time.monotonic())
            if item.prepared_at:
                self.infer_queue_dwell.append(item.infer_dequeued_at - item.prepared_at)
            await self._infer_one(item)

    @trace_async_exceptions(verbose=True)
    async def _infer_one(self, item: _InferWorkItem) -> None:
        response = await self.agent.get_rollout_response(
            self.request, item.params.inference_request
        )
        inferred_at = time.monotonic()
        self.gate.release_for("R")
        if item.infer_dequeued_at:
            self.engine_dwell.append(inferred_at - item.infer_dequeued_at)
        self.inferred_count += 1
        await self.assemble_queue.put(
            _InferredItem(item=item, response=response, inferred_at=inferred_at)
        )

    async def stage_assemble(self) -> None:
        """Build complete rollout groups from inferred items."""
        pending = self._assemble_pending
        try:
            while True:
                try:
                    inferred = await self.assemble_queue.get()
                except asyncio_QueueShutDown:
                    break
                dequeued_at = time.monotonic()
                if inferred.inferred_at:
                    self.assemble_queue_dwell.append(dequeued_at - inferred.inferred_at)
                item = inferred.item
                rollout = await item.params.build_rollout(inferred.response)
                if self.bank is not None and item.bank_uid is not None:
                    self.bank_queue.put_nowait(
                        _BankRecordItem(
                            _PendingRollout(item.bank_uid, item.rollout_idx, rollout)
                        )
                    )
                bucket = pending.setdefault(item.group_id, {})
                bucket[item.rollout_idx] = rollout
                if len(bucket) < self.request.rollouts_per_group:
                    continue
                pending.pop(item.group_id)
                indices = sorted(bucket)
                rollouts = [bucket[index] for index in indices]
                self.assembled_count += 1
                # NOTE: this filter is currently non-functional dead code:
                # _GranularityConfig._validate rejects filter_groups_with_same_reward
                # at pipeline construction, so `keep` is always True. Kept for a
                # future PR that regenerates dropped groups instead of
                # under-delivering to the caller. That PR must also release the
                # gate slot on the drop path: G/B slots free on consumption, and
                # a dropped group never reaches stage_consume, so its slot (and
                # eventually its batch's) would leak permanently.
                keep = (
                    not self.request.filter_groups_with_same_reward
                    or np.std([rollout.reward for rollout in rollouts]) > 1e-6
                )
                if keep:
                    output_enqueued_at = time.monotonic()
                    self._output_enqueued_at[
                        (item.batch_id, item.index_in_batch)
                    ] = output_enqueued_at
                    group = RolloutGroup(
                        rollouts=rollouts,
                        batch_id=item.batch_id,
                        index_in_batch=item.index_in_batch,
                        uid=item.bank_uid,
                        member_indices=indices,
                    )
                    await self.output_queue.put(group)
                else:
                    # Filtered out (all-equal reward): these rollouts are dropped
                    # and will never be consumed, so they leave the in-flight set here.
                    remove_inflight(len(rollouts))

        finally:
            self.output_queue.shutdown()

    def _record_output_dwell(self, group: RolloutGroup) -> None:
        """Record how long a group sat in output_queue before being dequeued."""
        key = (group.batch_id, group.index_in_batch)
        enqueued_at = self._output_enqueued_at.pop(key, 0.0)
        if enqueued_at:
            self.output_queue_dwell.append(time.monotonic() - enqueued_at)

    def _record_yield(self, group: RolloutGroup) -> None:
        """Record a group handed to the trainer, including whether it was restored."""
        key = (group.batch_id, group.index_in_batch)
        if key in self._restored_output_keys:
            self._restored_output_keys.remove(key)
            self.restored_count += 1
        self.yielded_count += 1

    async def stage_consume(self) -> AsyncIterator[RolloutGroup]:
        if not self.gran_policy.prevent_dataset_reorder:
            while True:
                try:
                    group = await self.output_queue.get()
                except asyncio_QueueShutDown:
                    return
                self._record_output_dwell(group)
                self._record_yield(group)
                yield group
                self.gate.release_for("G")

        next_batch_id = self.initial_batch_id
        pending = self._consume_pending
        while True:
            try:
                group = await self.output_queue.get()
            except asyncio_QueueShutDown:
                return
            self._record_output_dwell(group)
            pending.setdefault(group.batch_id, []).append(group)
            while (
                len(pending.get(next_batch_id, []))
                >= self.gran_policy.num_groups_per_batch
            ):
                batch = pending.pop(next_batch_id)
                batch.sort(key=lambda group: group.index_in_batch)
                next_batch_id += 1
                for group in batch:
                    self._record_yield(group)
                    yield group
                    self.gate.release_for("G")
                self.gate.release_for("B")


class GroupedRolloutGenerator(Agent, ABC):
    """An interface to return grouped Rollout objects to support algorithms like GRPO."""

    parallel_generation_tasks: int = 512

    def __init__(self, *, parallel_generation_tasks: int | None = None, **kwargs):
        super().__init__(**kwargs)
        self._rollout_bank = None
        self._restored_groups = None
        if parallel_generation_tasks is not None:
            self.parallel_generation_tasks = parallel_generation_tasks

    def take_restored_group(self) -> RolloutGroup | None:
        """Return the next recovered group assigned to this producer, if available."""
        return self._restored_groups.popleft() if self._restored_groups else None

    @abstractmethod
    async def prepare_group_rollout(
        self, request: GroupedRolloutRequest, *, problem_state: dict | None = None
    ) -> GroupRolloutParams:
        """Return the params for one group's rollouts.

        Args:
            request: The grouped rollout request being served.
            problem_state: When None, draw a fresh problem as usual. When given, it
                is a state this agent previously returned on ``GroupRolloutParams``;
                prepare for that same problem instead of drawing a new one, so a
                restored group's missing members are regenerated against the prompt
                its existing members already answered.
        """
        ...

    async def get_grouped_rollouts(
        self, request: GroupedRolloutRequest
    ) -> AsyncIterator[RolloutGroup]:
        assert isinstance(
            request.inference_interface, ReturnsRaw
        ), "InferenceInterface must support raw_text return to provide rollouts."
        pipeline = _RolloutPipeline(
            agent=self,
            request=request,
            parallel_generation_tasks=self.parallel_generation_tasks,
            bank=self._rollout_bank,
        )
        # Expose the live pipeline for observability; rl_utils reads its
        # queue sizes, gate state, and timing accumulators during logging.
        self._active_pipeline = pipeline
        stage_prepare_task = asyncio.create_task(pipeline.stage_prepare())
        infer_task = asyncio.create_task(pipeline.stage_infer())
        assemble_task = asyncio.create_task(pipeline.stage_assemble())
        tasks = [stage_prepare_task, infer_task, assemble_task]
        try:
            async for group in pipeline.stage_consume():
                yield group
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            self._last_pipeline = pipeline
            self._active_pipeline = None


class EvaluationAgent(Agent, ABC):
    """An agent that can take an inference interface and return a benchmark score."""

    @abstractmethod
    async def run_evaluation(self, request: EvaluationRequest) -> EvaluationResponse: ...
