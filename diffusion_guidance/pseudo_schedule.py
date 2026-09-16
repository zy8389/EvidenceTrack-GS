"""Deterministic pseudo-view schedule records used by controlled continuations."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Sequence


def canonical_sha256(value: Any) -> str:
    serialized = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def pseudo_camera_pool_sha256(camera_keys: Sequence[str]) -> str:
    keys = [str(key) for key in camera_keys]
    if not keys or len(keys) != len(set(keys)):
        raise ValueError("Pseudo-camera keys must be non-empty and unique")
    return canonical_sha256(keys)


def pseudo_call_trace_sha256(trace: Sequence[dict[str, Any]]) -> str:
    return canonical_sha256(list(trace))


def expected_pseudo_iterations(start: int, end: int, interval: int) -> list[int]:
    start, end, interval = int(start), int(end), int(interval)
    if interval <= 0 or end <= start:
        raise ValueError("Invalid pseudo supervision interval or iteration range")
    return [
        iteration
        for iteration in range(start + 1, end)
        if (iteration - start) % interval == 0
    ]


def expected_pseudo_call_trace(
    camera_keys: Sequence[str], start: int, end: int, interval: int
) -> list[dict[str, Any]]:
    keys = [str(key) for key in camera_keys]
    pseudo_camera_pool_sha256(keys)
    return [
        {
            "call_index": call_index,
            "iteration": iteration,
            "camera_index": call_index % len(keys),
            "camera_key": keys[call_index % len(keys)],
        }
        for call_index, iteration in enumerate(
            expected_pseudo_iterations(start, end, interval)
        )
    ]


def validate_pseudo_usage(
    usage: dict[str, Any],
    *,
    expected_calls: int,
    expected_unique_views: int,
) -> list[str]:
    """Return fail-closed validation failures for one realized pseudo schedule."""
    failures: list[str] = []
    required = (
        "pseudo_supervision_calls",
        "unique_pseudo_views_used",
        "pseudo_view_pool_size",
        "pseudo_view_indices_used",
        "pseudo_camera_keys",
        "pseudo_camera_pool_sha256",
        "pseudo_call_trace",
        "pseudo_call_trace_sha256",
        "pseudo_rgb_interval",
        "pseudo_rgb_start",
        "pseudo_rgb_end",
    )
    missing = [key for key in required if key not in usage]
    if missing:
        return [f"pseudo_usage_missing:{key}" for key in missing]

    try:
        calls = int(usage["pseudo_supervision_calls"])
        unique = int(usage["unique_pseudo_views_used"])
        pool_size = int(usage["pseudo_view_pool_size"])
        indices = [int(value) for value in usage["pseudo_view_indices_used"]]
        keys = [str(value) for value in usage["pseudo_camera_keys"]]
        trace = list(usage["pseudo_call_trace"])
        start = int(usage["pseudo_rgb_start"])
        end = int(usage["pseudo_rgb_end"])
        interval = int(usage["pseudo_rgb_interval"])
    except (TypeError, ValueError) as exc:
        return [f"pseudo_usage_invalid_type:{exc}"]

    if calls != expected_calls:
        failures.append(f"unexpected_pseudo_update_count:{calls}")
    if unique != expected_unique_views:
        failures.append(f"unexpected_unique_pseudo_view_count:{unique}")
    if pool_size != expected_unique_views or len(keys) != pool_size:
        failures.append(
            f"unexpected_pseudo_pool_size:declared={pool_size},keys={len(keys)}"
        )
    if len(keys) != len(set(keys)):
        failures.append("duplicate_pseudo_camera_keys")
    if indices != list(range(expected_unique_views)):
        failures.append("pseudo_view_indices_do_not_cover_precommitted_pool")
    if len(trace) != calls:
        failures.append(f"pseudo_call_trace_length_mismatch:{len(trace)}!={calls}")

    try:
        pool_hash = pseudo_camera_pool_sha256(keys)
        if usage["pseudo_camera_pool_sha256"] != pool_hash:
            failures.append("pseudo_camera_pool_hash_mismatch")
        trace_hash = pseudo_call_trace_sha256(trace)
        if usage["pseudo_call_trace_sha256"] != trace_hash:
            failures.append("pseudo_call_trace_hash_mismatch")
        expected_trace = expected_pseudo_call_trace(keys, start, end, interval)
        if trace != expected_trace:
            failures.append("pseudo_call_trace_not_precommitted_round_robin_schedule")
    except (TypeError, ValueError) as exc:
        failures.append(f"pseudo_schedule_invalid:{exc}")
    return failures
