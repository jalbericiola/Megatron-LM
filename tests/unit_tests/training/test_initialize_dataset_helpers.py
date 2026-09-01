# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from types import ModuleType

import pytest
import torch

from megatron.core.datasets import utils as dataset_utils


def make_helpers_module():
    module = ModuleType(dataset_utils._HELPERS_MODULE_NAME)
    for export in dataset_utils._HELPERS_REQUIRED_EXPORTS:
        setattr(module, export, lambda: None)
    return module


def configure_distributed_mocks(monkeypatch, *, rank=1):
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: rank)
    monkeypatch.setattr(torch.distributed, "get_backend", lambda: torch.distributed.Backend.GLOO)
    monkeypatch.setattr(
        torch.distributed,
        "all_gather_object",
        lambda *args, **kwargs: pytest.fail("bootstrap must not use an object collective"),
    )


def test_compile_helpers_distributed_runs_on_every_rank_and_all_reduces_success(monkeypatch):
    compile_calls = []
    all_reduce_calls = []
    module = make_helpers_module()

    def fake_compile():
        compile_calls.append(True)
        return module

    monkeypatch.setattr(dataset_utils, "compile_helpers", fake_compile)
    configure_distributed_mocks(monkeypatch)

    def fake_all_reduce(flag, op):
        all_reduce_calls.append((flag.item(), flag.device.type, op))

    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)

    assert dataset_utils.compile_helpers_distributed() is module

    assert compile_calls == [True]
    assert all_reduce_calls == [(0, "cpu", torch.distributed.ReduceOp.MAX)]


def test_compile_helpers_distributed_uses_explicit_cuda_device_for_nccl(monkeypatch):
    module = make_helpers_module()
    observed = {}

    class FakeFlag:
        @staticmethod
        def item():
            return 0

    monkeypatch.setattr(dataset_utils, "compile_helpers", lambda: module)
    configure_distributed_mocks(monkeypatch)
    monkeypatch.setattr(torch.distributed, "get_backend", lambda: torch.distributed.Backend.NCCL)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 3)

    def fake_tensor(value, *, dtype, device):
        observed.update(value=value, dtype=dtype, device=device)
        return FakeFlag()

    monkeypatch.setattr(torch, "tensor", fake_tensor)
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda flag, op: None)

    assert dataset_utils.compile_helpers_distributed() is module
    assert observed == {"value": 0, "dtype": torch.int32, "device": torch.device("cuda", 3)}


def test_compile_helpers_distributed_reports_remote_failure(monkeypatch):
    monkeypatch.setattr(dataset_utils, "compile_helpers", lambda: None)
    configure_distributed_mocks(monkeypatch)

    def fake_all_reduce(flag, op):
        assert flag.item() == 0
        assert op == torch.distributed.ReduceOp.MAX
        flag.fill_(1)

    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)

    with pytest.raises(RuntimeError, match="on another rank"):
        dataset_utils.compile_helpers_distributed()


def test_compile_helpers_distributed_converges_after_local_exception(monkeypatch):
    def fail_compile():
        raise RuntimeError("compiler failed")

    monkeypatch.setattr(dataset_utils, "compile_helpers", fail_compile)
    configure_distributed_mocks(monkeypatch, rank=3)

    def fake_all_reduce(flag, op):
        assert flag.item() == 1
        assert op == torch.distributed.ReduceOp.MAX

    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)

    with pytest.raises(RuntimeError, match="rank 3: RuntimeError: compiler failed"):
        dataset_utils.compile_helpers_distributed()


def test_compile_helpers_distributed_without_process_group_uses_direct_path(monkeypatch):
    module = make_helpers_module()
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    monkeypatch.setattr(dataset_utils, "compile_helpers", lambda: module)
    monkeypatch.setattr(
        torch.distributed,
        "all_reduce",
        lambda *args, **kwargs: pytest.fail("non-distributed build must not use a collective"),
    )

    assert dataset_utils.compile_helpers_distributed() is module
