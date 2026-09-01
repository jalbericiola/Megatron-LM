# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Semantic OFF/ON comparison for MoE router traces.

Physical token rows are not directly comparable when shared-prefix execution
deduplicates a prompt.  This module compares the optional ``semantic_token_ids``
written by :class:`~megatron.core.transformer.moe.router_trace.RouterTracer` and
reports the earliest layer/token whose ordered top-K expert ids differ.

Run as::

    python -m megatron.core.transformer.moe.router_trace_compare \
        --off off_trace_dir --on on_trace_dir --block decoder \
        --output parity_report.json

Directories are expanded to ``router_trace_rank*.jsonl``.  Repeated semantic
ids are intentional: dense OFF has one copy of each prompt token per completion,
and EP replicas can capture the same decision on multiple ranks.  All copies in
one arm must agree before the two arms are compared.

The comparator is strict by default: every input record must be labeled.  Use
the repeatable ``--block`` allowlist to intentionally restrict a comparison;
records cannot be silently omitted because a scope was accidentally absent.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

SCHEMA = "mcore-moe-router-semantic-parity-v1"
GREEN_MARKER = "MCORE_MOE_ROUTER_SEMANTIC_PARITY_GREEN"
RED_MARKER = "MCORE_MOE_ROUTER_SEMANTIC_PARITY_RED"

_SemanticKey = tuple[int, str, int, int, str, str | int]


def shared_prefix_semantic_token_ids(
    prefix_len: int,
    completion_lens: Sequence[int],
    *,
    physical_len: int | None = None,
    sample_id: str | int = 0,
    cp_size: int = 1,
    cp_rank: int = 0,
    tp_size: int = 1,
    tp_rank: int = 0,
    sequence_parallel: bool = False,
) -> list[str | None]:
    """Labels a packed ``[prefix, completion_0, ...]`` star.

    Topology padding up to ``physical_len`` receives ``None`` and is therefore
    excluded from semantic parity.  The returned list follows this rank's
    MCore CP-zigzag and optional TP sequence-parallel ownership.
    """
    prefix_len, completion_lens = _validate_layout(prefix_len, completion_lens)
    labels = [f"{sample_id}:prefix:{position}" for position in range(prefix_len)]
    for branch, length in enumerate(completion_lens):
        labels.extend(f"{sample_id}:completion:{branch}:{position}" for position in range(length))
    if physical_len is None:
        physical_len = len(labels)
    if physical_len < len(labels):
        raise ValueError(
            f"physical_len {physical_len} is shorter than packed star length {len(labels)}"
        )
    labels.extend([None] * (physical_len - len(labels)))
    local_indices = _local_sequence_indices(
        physical_len,
        cp_size=cp_size,
        cp_rank=cp_rank,
        tp_size=tp_size,
        tp_rank=tp_rank,
        sequence_parallel=sequence_parallel,
    )
    return [labels[index] for index in local_indices]


def dense_semantic_token_ids(
    prefix_len: int,
    completion_lens: Sequence[int],
    *,
    sequence_length: int | None = None,
    sample_id: str | int = 0,
    cp_size: int = 1,
    cp_rank: int = 0,
    tp_size: int = 1,
    tp_rank: int = 0,
    sequence_parallel: bool = False,
) -> list[str | None]:
    """Labels the router's flattened ``[sequence, batch]`` dense layout.

    Prefix labels deliberately repeat across completion branches.  Rectangular
    batch padding after a shorter completion receives ``None``.  The returned
    flattened rows follow this rank's MCore CP-zigzag and optional TP/SP
    ownership before the batch dimension is flattened.
    """
    prefix_len, completion_lens = _validate_layout(prefix_len, completion_lens)
    minimum_sequence_length = prefix_len + max(completion_lens)
    if sequence_length is None:
        sequence_length = minimum_sequence_length
    if sequence_length < minimum_sequence_length:
        raise ValueError(
            f"sequence_length {sequence_length} is shorter than the longest branch "
            f"{minimum_sequence_length}"
        )

    rows: list[list[str | None]] = []
    for sequence_position in range(sequence_length):
        row = []
        for branch, completion_len in enumerate(completion_lens):
            if sequence_position < prefix_len:
                row.append(f"{sample_id}:prefix:{sequence_position}")
            elif sequence_position - prefix_len < completion_len:
                row.append(f"{sample_id}:completion:{branch}:{sequence_position - prefix_len}")
            else:
                row.append(None)
        rows.append(row)
    local_indices = _local_sequence_indices(
        sequence_length,
        cp_size=cp_size,
        cp_rank=cp_rank,
        tp_size=tp_size,
        tp_rank=tp_rank,
        sequence_parallel=sequence_parallel,
    )
    return [label for index in local_indices for label in rows[index]]


def _validate_layout(
    prefix_len: int, completion_lens: Sequence[int]
) -> tuple[int, tuple[int, ...]]:
    prefix_len = int(prefix_len)
    completion_lens = tuple(int(length) for length in completion_lens)
    if prefix_len < 1:
        raise ValueError("prefix_len must be positive")
    if not completion_lens or any(length < 1 for length in completion_lens):
        raise ValueError("completion_lens must contain only positive lengths")
    return prefix_len, completion_lens


def _local_sequence_indices(
    sequence_length: int,
    *,
    cp_size: int,
    cp_rank: int,
    tp_size: int,
    tp_rank: int,
    sequence_parallel: bool,
) -> list[int]:
    """Return MCore's CP-zigzag then optional TP/SP sequence ownership."""
    sequence_length = int(sequence_length)
    cp_size, cp_rank = int(cp_size), int(cp_rank)
    tp_size, tp_rank = int(tp_size), int(tp_rank)
    if cp_size < 1 or not 0 <= cp_rank < cp_size:
        raise ValueError(f"invalid CP geometry: {cp_size=}, {cp_rank=}")
    if tp_size < 1 or not 0 <= tp_rank < tp_size:
        raise ValueError(f"invalid TP geometry: {tp_size=}, {tp_rank=}")

    if cp_size == 1:
        indices = list(range(sequence_length))
    else:
        if sequence_length % (2 * cp_size):
            raise ValueError(
                f"sequence length {sequence_length} must be divisible by 2 * CP size {cp_size}"
            )
        chunk = sequence_length // (2 * cp_size)
        front_start = cp_rank * chunk
        back_start = (2 * cp_size - cp_rank - 1) * chunk
        indices = list(range(front_start, front_start + chunk))
        indices.extend(range(back_start, back_start + chunk))

    if sequence_parallel:
        if len(indices) % tp_size:
            raise ValueError(
                f"CP-local sequence length {len(indices)} must be divisible by TP size {tp_size}"
            )
        tp_chunk = len(indices) // tp_size
        indices = indices[tp_rank * tp_chunk : (tp_rank + 1) * tp_chunk]
    return indices


def _trace_paths(path: str | os.PathLike[str]) -> list[Path]:
    path = Path(path)
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise ValueError(f"trace path does not exist: {path}")
    paths = [Path(item) for item in glob.glob(str(path / "router_trace_rank*.jsonl"))]
    if not paths:
        raise ValueError(f"trace directory has no router_trace_rank*.jsonl files: {path}")
    return sorted(paths)


def load_trace_records(path: str | os.PathLike[str]) -> list[dict]:
    """Load one trace file or every per-rank trace in a directory."""
    records = []
    for trace_path in _trace_paths(path):
        with trace_path.open() as trace_file:
            for line_number, line in enumerate(trace_file, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"invalid JSON in {trace_path}:{line_number}: {error.msg}"
                    ) from error
                record["_source"] = f"{trace_path}:{line_number}"
                records.append(record)
    return records


def _record_identity(record: dict) -> tuple[int, str, int, int, str]:
    try:
        step = int(record["step"])
        block = str(record["block"])
        mtp_idx = int(record.get("mtp_idx", -1))
        layer = int(record["layer"])
        stage = str(record.get("stage", "pre_dispatch"))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"invalid router record identity at {record.get('_source')}: {error}"
        ) from error
    return step, block, mtp_idx, layer, stage


def _route_index(records: Iterable[dict], arm: str):
    routes: dict[_SemanticKey, set[tuple[int, ...]]] = defaultdict(set)
    sources: dict[_SemanticKey, list[str]] = defaultdict(list)
    record_count = 0
    compared_rows = 0
    for record in records:
        record_count += 1
        source = record.get("_source", f"{arm}:record:{record_count}")
        semantic_ids = record.get("semantic_token_ids")
        top_indices = record.get("top_indices")
        if semantic_ids is None:
            raise ValueError(f"record is missing semantic_token_ids at {source}")
        if not isinstance(semantic_ids, list):
            raise ValueError(f"semantic_token_ids is not a list at {source}")
        if not isinstance(top_indices, list):
            raise ValueError(f"record is missing top_indices at {source}")
        if len(semantic_ids) != len(top_indices):
            raise ValueError(
                f"semantic/top-index row count differs at {source}: "
                f"{len(semantic_ids)} != {len(top_indices)}"
            )
        num_tokens = record.get("num_tokens")
        if num_tokens is not None and int(num_tokens) != len(top_indices):
            raise ValueError(
                f"num_tokens differs from serialized rows at {source}: "
                f"{num_tokens} != {len(top_indices)}"
            )
        identity = _record_identity(record)
        topk = record.get("topk")
        for semantic_id, expert_ids in zip(semantic_ids, top_indices, strict=True):
            if semantic_id is None:
                continue
            if isinstance(semantic_id, bool) or not isinstance(semantic_id, (str, int)):
                raise ValueError(
                    f"semantic token id must be a string, integer, or null at {source}"
                )
            if not isinstance(expert_ids, list) or any(
                isinstance(expert_id, bool) or not isinstance(expert_id, int)
                for expert_id in expert_ids
            ):
                raise ValueError(f"invalid top_indices row at {source}")
            if topk is not None and int(topk) != len(expert_ids):
                raise ValueError(
                    f"topk differs from serialized expert ids at {source}: "
                    f"{topk} != {len(expert_ids)}"
                )
            key: _SemanticKey = (*identity, semantic_id)
            routes[key].add(tuple(expert_ids))
            sources[key].append(source)
            compared_rows += 1
    return routes, sources, record_count, compared_rows


def _sort_key(key: _SemanticKey):
    step, block, mtp_idx, layer, stage, semantic_id = key
    block_order = 0 if block == "decoder" else 1
    semantic_order = _semantic_sort_key(semantic_id)
    return step, block_order, block, mtp_idx, layer, stage, semantic_order


def _natural_string_key(value: str):
    return tuple(
        (0, int(piece)) if piece.isdigit() else (1, piece)
        for piece in re.split(r"(\d+)", value)
        if piece
    )


def _semantic_sort_key(semantic_id: str | int):
    """Sort generated labels in causal order and arbitrary labels naturally."""
    if isinstance(semantic_id, int):
        return 0, semantic_id
    parts = semantic_id.rsplit(":", 3)
    if len(parts) >= 3 and parts[-2] == "prefix" and parts[-1].isdigit():
        sample = ":".join(parts[:-2])
        return 1, _natural_string_key(sample), 0, 0, int(parts[-1])
    if (
        len(parts) == 4
        and parts[-3] == "completion"
        and parts[-2].isdigit()
        and parts[-1].isdigit()
    ):
        return (1, _natural_string_key(parts[0]), 1, int(parts[-2]), int(parts[-1]))
    return 2, _natural_string_key(semantic_id)


def _key_payload(key: _SemanticKey) -> dict:
    step, block, mtp_idx, layer, stage, semantic_id = key
    identity = {"step": step, "block": block, "layer": layer, "stage": stage}
    if mtp_idx >= 0:
        identity["mtp_idx"] = mtp_idx
    return {"identity": identity, "semantic_token_id": semantic_id}


def _filter_blocks(records: Iterable[dict], blocks: Sequence[str] | None) -> list[dict]:
    records = list(records)
    if blocks is None:
        return records
    block_set = set(blocks)
    if not block_set or any(not isinstance(block, str) or not block for block in block_set):
        raise ValueError("blocks must contain one or more non-empty block names")
    return [record for record in records if _record_identity(record)[1] in block_set]


def compare_router_traces(
    off_records: Iterable[dict], on_records: Iterable[dict], *, blocks: Sequence[str] | None = None
) -> dict:
    """Compare ordered top-K expert ids and return a machine-readable report."""
    off_routes, off_sources, off_record_count, off_rows = _route_index(
        _filter_blocks(off_records, blocks), "off"
    )
    on_routes, on_sources, on_record_count, on_rows = _route_index(
        _filter_blocks(on_records, blocks), "on"
    )
    if not off_routes:
        raise ValueError("OFF trace has no labeled semantic router rows")
    if not on_routes:
        raise ValueError("ON trace has no labeled semantic router rows")
    all_keys = sorted(set(off_routes) | set(on_routes), key=_sort_key)

    first_mismatch = None
    for key in all_keys:
        off_values = off_routes.get(key, set())
        on_values = on_routes.get(key, set())
        if len(off_values) > 1:
            reason = "off_internal_routing_mismatch"
        elif len(on_values) > 1:
            reason = "on_internal_routing_mismatch"
        elif not off_values:
            reason = "missing_from_off"
        elif not on_values:
            reason = "missing_from_on"
        elif off_values != on_values:
            reason = "off_on_routing_mismatch"
        else:
            continue
        first_mismatch = {
            "reason": reason,
            **_key_payload(key),
            "off_top_indices": [list(route) for route in sorted(off_values)],
            "on_top_indices": [list(route) for route in sorted(on_values)],
            "off_sources": off_sources.get(key, []),
            "on_sources": on_sources.get(key, []),
        }
        break

    return {
        "schema": SCHEMA,
        "status": "parity" if first_mismatch is None else "bifurcation",
        "included_blocks": "all" if blocks is None else sorted(set(blocks)),
        "off_record_count": off_record_count,
        "on_record_count": on_record_count,
        "off_semantic_rows": off_rows,
        "on_semantic_rows": on_rows,
        "compared_semantic_routes": len(set(off_routes) & set(on_routes)),
        "first_mismatch": first_mismatch,
    }


def compare_trace_paths(
    off_path: str | os.PathLike[str],
    on_path: str | os.PathLike[str],
    *,
    blocks: Sequence[str] | None = None,
) -> dict:
    """Load and compare trace paths."""
    return compare_router_traces(
        load_trace_records(off_path), load_trace_records(on_path), blocks=blocks
    )


def _write_report(report: dict, output: str | os.PathLike[str] | None) -> None:
    serialized = json.dumps(report, indent=2, sort_keys=True)
    if output is None:
        print(serialized)
    else:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--off", required=True, help="OFF JSONL trace file or trace directory")
    parser.add_argument("--on", required=True, help="ON JSONL trace file or trace directory")
    parser.add_argument("--output", help="optional JSON report path")
    parser.add_argument(
        "--block",
        action="append",
        dest="blocks",
        help=(
            "explicit block allowlist (repeatable, for example --block decoder); "
            "without it every record must be semantically labeled"
        ),
    )
    args = parser.parse_args(argv)

    try:
        report = compare_trace_paths(args.off, args.on, blocks=args.blocks)
    except ValueError as error:
        report = {"schema": SCHEMA, "status": "invalid", "error": str(error)}
    _write_report(report, args.output)

    marker = GREEN_MARKER if report["status"] == "parity" else RED_MARKER
    summary = f"status={report['status']} schema={SCHEMA}"
    if report["status"] != "invalid":
        summary += f" compared={report['compared_semantic_routes']}"
    print(f"{marker} {summary}")
    return 0 if report["status"] == "parity" else 2


if __name__ == "__main__":
    raise SystemExit(main())
