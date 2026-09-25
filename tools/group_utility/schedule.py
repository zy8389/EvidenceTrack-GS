"""Pre-registered 39-call schedules for group utility continuations."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from evidence_track.diffusion.pseudo_schedule import (
    expected_pseudo_iterations,
    pseudo_call_trace_sha256,
    pseudo_camera_pool_sha256,
)

from .manifest import canonical_sha256

SCHEDULE_SCHEMA = "group_utility_schedule_v1"


def build_group_call_trace(
    camera_keys: Sequence[str],
    *,
    start: int = 10000,
    end: int = 12000,
    interval: int = 50,
) -> list[dict[str, Any]]:
    keys = [str(key) for key in camera_keys]
    if len(keys) != 8:
        raise ValueError(f"Group Utility Protocol v1 requires exactly 8 camera keys, got {len(keys)}")
    if len(set(keys)) != len(keys) or any(not key for key in keys):
        raise ValueError("group camera keys must be unique and non-empty")
    iterations = expected_pseudo_iterations(start, end, interval)
    if len(iterations) != 39:
        raise ValueError(f"protocol requires exactly 39 pseudo iterations, got {len(iterations)}")
    return [
        {
            "call_index": index,
            "iteration": iteration,
            "camera_index": index % len(keys),
            "camera_key": keys[index % len(keys)],
        }
        for index, iteration in enumerate(iterations)
    ]


def build_group_schedule(
    group_id: str,
    camera_keys: Sequence[str],
    *,
    parent_manifest_sha256: str,
    partition_sha256: str,
    start: int = 10000,
    end: int = 12000,
    interval: int = 50,
) -> dict[str, Any]:
    keys = [str(key) for key in camera_keys]
    trace = build_group_call_trace(keys, start=start, end=end, interval=interval)
    exposure_counts = dict(sorted(Counter(item["camera_key"] for item in trace).items()))
    schedule = {
        "schema": SCHEDULE_SCHEMA,
        "group_id": str(group_id),
        "parent_manifest_sha256": str(parent_manifest_sha256),
        "partition_sha256": str(partition_sha256),
        "start": int(start),
        "end": int(end),
        "interval": int(interval),
        "calls": len(trace),
        "unique_views": len(keys),
        "camera_keys": keys,
        "camera_pool_sha256": pseudo_camera_pool_sha256(keys),
        "trace": trace,
        "trace_sha256": pseudo_call_trace_sha256(trace),
        "exposure_counts": exposure_counts,
    }
    schedule["schedule_sha256"] = canonical_sha256({k: v for k, v in schedule.items() if k != "schedule_sha256"})
    validate_group_schedule(schedule)
    return schedule


def validate_group_schedule(
    schedule: Mapping[str, Any] | str | Path,
    *,
    expected_group_id: str | None = None,
    expected_parent_manifest_sha256: str | None = None,
    expected_partition_sha256: str | None = None,
) -> None:
    if isinstance(schedule, (str, Path)):
        with Path(schedule).open("r", encoding="utf-8") as handle:
            schedule = json.load(handle)
    value = dict(schedule)
    required = (
        "schema", "group_id", "parent_manifest_sha256", "partition_sha256", "start", "end",
        "interval", "calls", "unique_views", "camera_keys", "camera_pool_sha256", "trace",
        "trace_sha256", "exposure_counts", "schedule_sha256",
    )
    missing = [field for field in required if field not in value]
    if missing:
        raise ValueError(f"schedule missing fields: {missing}")
    if value["schema"] != SCHEDULE_SCHEMA:
        raise ValueError("wrong group utility schedule schema")
    if expected_group_id is not None and value["group_id"] != expected_group_id:
        raise ValueError("schedule group ID mismatch")
    if expected_parent_manifest_sha256 is not None and value["parent_manifest_sha256"] != expected_parent_manifest_sha256:
        raise ValueError("schedule parent manifest hash mismatch")
    if expected_partition_sha256 is not None and value["partition_sha256"] != expected_partition_sha256:
        raise ValueError("schedule partition hash mismatch")
    keys = [str(key) for key in value["camera_keys"]]
    if len(keys) != 8 or len(set(keys)) != 8:
        raise ValueError("schedule must contain 8 unique camera keys")
    if int(value["calls"]) != 39 or int(value["unique_views"]) != 8:
        raise ValueError("schedule dimensions are not 39 calls / 8 views")
    if value["camera_pool_sha256"] != pseudo_camera_pool_sha256(keys):
        raise ValueError("schedule camera pool hash mismatch")
    trace = list(value["trace"])
    expected = build_group_call_trace(keys, start=int(value["start"]), end=int(value["end"]), interval=int(value["interval"]))
    if trace != expected:
        raise ValueError("schedule trace is not the pre-registered round-robin trace")
    if value["trace_sha256"] != pseudo_call_trace_sha256(trace):
        raise ValueError("schedule trace hash mismatch")
    counts = dict(sorted(Counter(item["camera_key"] for item in trace).items()))
    if value["exposure_counts"] != counts or sum(counts.values()) != 39 or set(counts.values()) - {4, 5} or min(counts.values()) != 4 or max(counts.values()) != 5:
        raise ValueError("schedule exposure counts are not the expected 4/5 distribution")
    body = {key: value[key] for key in value if key != "schedule_sha256"}
    if value["schedule_sha256"] != canonical_sha256(body):
        raise ValueError("schedule_sha256 does not match schedule contents")


def write_group_schedule(path: str | Path, schedule: Mapping[str, Any]) -> str:
    validate_group_schedule(schedule)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(dict(schedule), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return str(schedule["schedule_sha256"])
