# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Focused CPU tests for opt-in semantic MoE route diagnostics."""

import json

import pytest
import torch

from megatron.core.transformer.moe.router_trace import RouterTracer
from megatron.core.transformer.moe.router_trace_compare import (
    GREEN_MARKER,
    RED_MARKER,
    SCHEMA,
    compare_router_traces,
    dense_semantic_token_ids,
    main,
    shared_prefix_semantic_token_ids,
)


def _record(*, layer, semantic_ids, routes, rank=0, step=0, block="decoder"):
    return {
        "step": step,
        "stage": "pre_dispatch",
        "block": block,
        "layer": layer,
        "rank": rank,
        "num_tokens": len(routes),
        "topk": len(routes[0]),
        "semantic_token_ids": semantic_ids,
        "top_indices": routes,
    }


def test_dense_and_shared_layouts_align_semantically():
    dense = dense_semantic_token_ids(2, [2, 1])
    shared = shared_prefix_semantic_token_ids(2, [2, 1])

    # Router flattening is sequence-major, so prompt positions repeat over the
    # dense batch while the packed star contains one physical prompt copy.
    assert dense == [
        "0:prefix:0",
        "0:prefix:0",
        "0:prefix:1",
        "0:prefix:1",
        "0:completion:0:0",
        "0:completion:1:0",
        "0:completion:0:1",
        None,
    ]
    assert shared == [
        "0:prefix:0",
        "0:prefix:1",
        "0:completion:0:0",
        "0:completion:0:1",
        "0:completion:1:0",
    ]


def test_semantic_layouts_follow_cp_zigzag_then_tp_sequence_parallel_ownership():
    shared = shared_prefix_semantic_token_ids(
        4,
        [4, 4],
        physical_len=16,
        cp_size=2,
        cp_rank=1,
        tp_size=2,
        tp_rank=1,
        sequence_parallel=True,
    )
    assert shared == [
        "0:completion:1:0",
        "0:completion:1:1",
        "0:completion:1:2",
        "0:completion:1:3",
    ]

    dense = dense_semantic_token_ids(
        4, [4, 4], cp_size=2, cp_rank=0, tp_size=2, tp_rank=1, sequence_parallel=True
    )
    assert dense == ["0:completion:0:2", "0:completion:1:2", "0:completion:0:3", "0:completion:1:3"]


def test_router_tracer_semantic_scope_is_opt_in_and_restored(tmp_path):
    tracer = RouterTracer(str(tmp_path), max_steps=10, rank=0)
    indices = torch.tensor([[[1, 2]], [[3, 4]]], dtype=torch.int32)

    tracer.record_indices(indices, step=0)
    with tracer.semantic_token_scope(["prompt:0", None]):
        tracer.record_indices(indices, step=1)
    tracer.record_indices(indices, step=2)
    tracer.flush()

    records = [json.loads(line) for line in open(tracer.output_path) if line.strip()]
    assert "semantic_token_ids" not in records[0]
    assert records[1]["semantic_token_ids"] == ["prompt:0", None]
    assert "semantic_token_ids" not in records[2]


def test_router_tracer_fails_closed_on_mapping_length_mismatch(tmp_path):
    tracer = RouterTracer(str(tmp_path), max_steps=10, rank=0)
    tracer.set_semantic_token_ids(["only-one"])
    indices = torch.zeros(2, 1, 2, dtype=torch.int32)
    with pytest.raises(RuntimeError, match="mapping length does not match"):
        tracer.record_indices(indices)


def test_semantic_compare_accepts_dense_prefix_duplicates_and_rank_replicas():
    off = [
        _record(
            layer=3,
            semantic_ids=["prefix:0", "prefix:0", "completion:0:0"],
            routes=[[7, 2], [7, 2], [5, 1]],
            rank=0,
        ),
        _record(
            layer=3, semantic_ids=["prefix:0", "completion:0:0"], routes=[[7, 2], [5, 1]], rank=1
        ),
    ]
    on = [
        # MTP labeling is a separate scope; an intentionally unlabeled record
        # must not prevent a decoder-only semantic comparison.
        {
            "step": 0,
            "block": "mtp",
            "mtp_idx": 0,
            "layer": 0,
            "rank": 0,
            "num_tokens": 1,
            "topk": 2,
            "top_indices": [[3, 4]],
        },
        _record(
            layer=3,
            semantic_ids=["prefix:0", "completion:0:0", None],
            routes=[[7, 2], [5, 1], [99, 98]],
        ),
    ]

    with pytest.raises(ValueError, match="missing semantic_token_ids"):
        compare_router_traces(off, on)

    report = compare_router_traces(off, on, blocks=["decoder"])
    assert report["schema"] == SCHEMA
    assert report["status"] == "parity"
    assert report["compared_semantic_routes"] == 2
    assert report["off_semantic_rows"] == 5
    assert report["on_semantic_rows"] == 2
    assert report["included_blocks"] == ["decoder"]


def test_semantic_compare_reports_first_layer_and_token_bifurcation():
    off = [
        _record(layer=1, semantic_ids=["token:b", "token:a"], routes=[[4, 5], [1, 2]]),
        _record(layer=2, semantic_ids=["token:a"], routes=[[3, 4]]),
    ]
    on = [
        _record(layer=1, semantic_ids=["token:b", "token:a"], routes=[[9, 5], [1, 2]]),
        _record(layer=2, semantic_ids=["token:a"], routes=[[8, 4]]),
    ]

    report = compare_router_traces(off, on)
    assert report["status"] == "bifurcation"
    mismatch = report["first_mismatch"]
    assert mismatch["reason"] == "off_on_routing_mismatch"
    assert mismatch["identity"]["layer"] == 1
    assert mismatch["semantic_token_id"] == "token:b"
    assert mismatch["off_top_indices"] == [[4, 5]]
    assert mismatch["on_top_indices"] == [[9, 5]]


def test_first_bifurcation_uses_causal_generated_token_order():
    semantic_ids = ["0:completion:0:0", "0:prefix:10", "0:prefix:2"]
    off = [_record(layer=1, semantic_ids=semantic_ids, routes=[[1, 2], [3, 4], [5, 6]])]
    on = [_record(layer=1, semantic_ids=semantic_ids, routes=[[9, 2], [8, 4], [7, 6]])]

    report = compare_router_traces(off, on)
    # Prefix precedes completion, and numeric 2 precedes lexicographic-looking 10.
    assert report["first_mismatch"]["semantic_token_id"] == "0:prefix:2"


def test_semantic_compare_reports_dense_duplicate_disagreement_before_on_compare():
    off = [_record(layer=0, semantic_ids=["prefix:0", "prefix:0"], routes=[[1, 2], [1, 3]])]
    on = [_record(layer=0, semantic_ids=["prefix:0"], routes=[[1, 2]])]

    report = compare_router_traces(off, on)
    assert report["first_mismatch"]["reason"] == "off_internal_routing_mismatch"
    assert report["first_mismatch"]["off_top_indices"] == [[1, 2], [1, 3]]


def test_semantic_compare_rejects_padding_only_trace():
    padding_only = [_record(layer=0, semantic_ids=[None], routes=[[1, 2]])]
    with pytest.raises(ValueError, match="no labeled semantic router rows"):
        compare_router_traces(padding_only, padding_only)


def test_cli_writes_report_and_machine_marker(tmp_path, capsys):
    off_dir = tmp_path / "off"
    on_dir = tmp_path / "on"
    off_dir.mkdir()
    on_dir.mkdir()
    record = _record(layer=0, semantic_ids=["token:0"], routes=[[1, 2]])
    (off_dir / "router_trace_rank0.jsonl").write_text(json.dumps(record) + "\n")
    (on_dir / "router_trace_rank0.jsonl").write_text(json.dumps(record) + "\n")
    output = tmp_path / "report.json"

    assert main(["--off", str(off_dir), "--on", str(on_dir), "--output", str(output)]) == 0
    assert json.loads(output.read_text())["status"] == "parity"
    assert GREEN_MARKER in capsys.readouterr().out

    bad_record = dict(record, top_indices=[[2, 1]])
    (on_dir / "router_trace_rank0.jsonl").write_text(json.dumps(bad_record) + "\n")
    assert main(["--off", str(off_dir), "--on", str(on_dir)]) == 2
    assert RED_MARKER in capsys.readouterr().out
