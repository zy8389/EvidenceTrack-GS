"""Deterministic 32-view -> 4x8 partitioning for Group Utility Protocol v1."""

from __future__ import annotations

import hashlib
import json
import copy
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .manifest import (
    GROUP_MANIFEST_SCHEMA,
    PARENT_MANIFEST_SCHEMA,
    ParentManifest,
    canonical_sha256,
    record_camera_fingerprint,
    record_camera_key,
    write_jsonl,
)

PARTITION_SCHEMA = "group_utility_partition_v1"
DEFAULT_BALANCE_FIELDS = (
    "trajectory",
    "view_direction",
    "source_support",
    "interpolation_distance",
)


def _stable_digest(seed: int, key: str) -> str:
    return hashlib.sha256(f"{int(seed)}:{key}".encode("utf-8")).hexdigest()


def _feature_value(record: Mapping[str, Any], field: str) -> Any:
    value = record.get(field)
    if value is None and isinstance(record.get("camera"), dict):
        value = record["camera"].get(field)
    if value is None and isinstance(record.get("features"), dict):
        value = record["features"].get(field)
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return value


def _sort_token(value: Any) -> tuple[int, str]:
    if value is None:
        return (0, "")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return (1, f"{float(value):.12g}")
    return (2, str(value))


@dataclass(frozen=True)
class GroupPartition:
    schema: str
    parent_manifest_sha256: str
    group_count: int
    group_size: int
    seed: int
    algorithm: str
    balance_fields: tuple[str, ...]
    groups: tuple[dict[str, Any], ...]
    partition_sha256: str

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema": self.schema,
            "parent_manifest_sha256": self.parent_manifest_sha256,
            "group_count": self.group_count,
            "group_size": self.group_size,
            "seed": self.seed,
            "algorithm": self.algorithm,
            "balance_fields": list(self.balance_fields),
            "groups": list(self.groups),
        }
        if include_hash:
            value["partition_sha256"] = self.partition_sha256
        return value


def _partition_digest(payload: Mapping[str, Any]) -> str:
    return canonical_sha256(payload)


def build_balanced_partition(
    records: Sequence[Mapping[str, Any]] | ParentManifest,
    *,
    group_count: int = 4,
    group_size: int = 8,
    seed: int = 0,
    balance_fields: Sequence[str] = DEFAULT_BALANCE_FIELDS,
) -> GroupPartition:
    """Build a deterministic, balanced, non-overlapping partition.

    Records are first ordered by the pre-registered balance fields (when
    present), with a seed-keyed stable digest as the tie-breaker.  A snake
    round-robin assignment avoids placing adjacent sorted records in one
    group while remaining entirely reproducible.  The original parent index is
    retained and group records are emitted in parent order for schedule
    stability.
    """

    parent = records if isinstance(records, ParentManifest) else None
    values = list(parent.records if parent else records)
    parent_hash = parent.sha256 if parent else ""
    if not values:
        raise ValueError("cannot partition an empty manifest")
    group_count, group_size = int(group_count), int(group_size)
    if group_count <= 0 or group_size <= 0:
        raise ValueError("group_count and group_size must be positive")
    if len(values) != group_count * group_size:
        raise ValueError(f"record count {len(values)} != group_count*group_size {group_count * group_size}")
    if not parent_hash:
        raise ValueError("partition requires a ParentManifest with a byte-level SHA256")

    indexed = list(enumerate(values))
    indexed.sort(
        key=lambda item: (
            tuple(_sort_token(_feature_value(item[1], field)) for field in balance_fields),
            _stable_digest(seed, record_camera_key(item[1])),
        )
    )
    assignments: list[list[tuple[int, Mapping[str, Any]]]] = [[] for _ in range(group_count)]
    for position, item in enumerate(indexed):
        cycle, offset = divmod(position, group_count)
        group_index = offset if cycle % 2 == 0 else group_count - 1 - offset
        assignments[group_index].append(item)

    groups: list[dict[str, Any]] = []
    for group_index, members in enumerate(assignments):
        members.sort(key=lambda item: item[0])
        entries = [
            {
                "parent_index": int(index),
                "camera_key": record_camera_key(record),
                "camera_fingerprint": record_camera_fingerprint(record),
            }
            for index, record in members
        ]
        groups.append(
            {
                "group_id": f"group_{group_index:02d}",
                "group_index": group_index,
                "count": len(entries),
                "camera_keys": [entry["camera_key"] for entry in entries],
                "camera_fingerprints": [entry["camera_fingerprint"] for entry in entries],
                "members": entries,
            }
        )

    payload = {
        "schema": PARTITION_SCHEMA,
        "parent_manifest_sha256": parent_hash,
        "group_count": group_count,
        "group_size": group_size,
        "seed": int(seed),
        "algorithm": "stratified_snake_round_robin_v1",
        "balance_fields": list(balance_fields),
        "groups": groups,
    }
    digest = _partition_digest(payload)
    return GroupPartition(partition_sha256=digest, **payload)


def validate_partition(
    partition: GroupPartition | Mapping[str, Any] | str | Path,
    *,
    parent_manifest_sha256: str | None = None,
    parent_records: Sequence[Mapping[str, Any]] | ParentManifest | None = None,
    expected_group_count: int = 4,
    expected_group_size: int = 8,
) -> None:
    """Fail-closed validation of a partition and optional parent membership."""

    if isinstance(partition, (str, Path)):
        with Path(partition).open("r", encoding="utf-8") as handle:
            partition = json.load(handle)
    value = partition.to_dict() if isinstance(partition, GroupPartition) else dict(partition)
    required = ("schema", "parent_manifest_sha256", "group_count", "group_size", "groups", "partition_sha256")
    missing = [field for field in required if field not in value]
    if missing:
        raise ValueError(f"partition missing fields: {missing}")
    if value["schema"] != PARTITION_SCHEMA:
        raise ValueError(f"wrong partition schema: {value['schema']}")
    if parent_manifest_sha256 is not None and value["parent_manifest_sha256"] != parent_manifest_sha256:
        raise ValueError("partition parent manifest hash mismatch")
    group_count, group_size = int(value["group_count"]), int(value["group_size"])
    if group_count != int(expected_group_count) or group_size != int(expected_group_size):
        raise ValueError("partition dimensions do not match the pre-registered protocol")
    groups = value["groups"]
    if not isinstance(groups, list) or len(groups) != group_count:
        raise ValueError("partition group list is malformed")
    payload = dict(value)
    declared_hash = str(payload.pop("partition_sha256"))
    if _partition_digest(payload) != declared_hash:
        raise ValueError("partition_sha256 does not match partition contents")

    all_keys: list[str] = []
    all_fingerprints: list[str] = []
    group_ids: list[str] = []
    all_parent_indices: list[int] = []
    for expected_index, group in enumerate(groups):
        if not isinstance(group, dict):
            raise ValueError("partition group must be an object")
        if group.get("group_id") != f"group_{expected_index:02d}":
            raise ValueError("partition group IDs are not canonical")
        if int(group.get("group_index", -1)) != expected_index:
            raise ValueError("partition group indices are not canonical")
        members = group.get("members")
        if not isinstance(members, list) or len(members) != group_size:
            raise ValueError(f"group {group.get('group_id')} has wrong member count")
        if int(group.get("count", -1)) != group_size:
            raise ValueError("group count field mismatch")
        keys = [str(item.get("camera_key", "")) for item in members]
        fingerprints = [str(item.get("camera_fingerprint", "")) for item in members]
        try:
            parent_indices = [int(item["parent_index"]) for item in members]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("partition member lacks a valid parent_index") from exc
        if any(not key for key in keys) or any(not fp for fp in fingerprints):
            raise ValueError("partition member lacks key or fingerprint")
        if group.get("camera_keys") != keys or group.get("camera_fingerprints") != fingerprints:
            raise ValueError("partition group summary does not match members")
        all_keys.extend(keys)
        all_fingerprints.extend(fingerprints)
        all_parent_indices.extend(parent_indices)
        group_ids.append(str(group["group_id"]))

    if len(set(group_ids)) != group_count:
        raise ValueError("duplicate group ID")
    if len(set(all_parent_indices)) != len(all_parent_indices):
        raise ValueError("partition parent indices overlap")
    if len(set(all_keys)) != len(all_keys) or len(set(all_fingerprints)) != len(all_fingerprints):
        raise ValueError("partition groups overlap")
    if parent_records is not None:
        parent = parent_records if isinstance(parent_records, ParentManifest) else None
        records = list(parent.records if parent else parent_records)
        if sorted(all_parent_indices) != list(range(len(records))):
            raise ValueError("partition parent indices do not exactly cover parent records")
        parent_keys = [record_camera_key(record) for record in records]
        parent_fps = [record_camera_fingerprint(record) for record in records]
        for group in groups:
            for member in group["members"]:
                record = records[int(member["parent_index"])]
                if record_camera_key(record) != member["camera_key"] or record_camera_fingerprint(record) != member["camera_fingerprint"]:
                    raise ValueError("partition member identity does not match parent record")
        if sorted(all_keys) != sorted(parent_keys) or sorted(all_fingerprints) != sorted(parent_fps):
            raise ValueError("partition does not exactly cover parent manifest")


def write_partition(path: str | Path, partition: GroupPartition) -> str:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    validate_partition(partition)
    destination.write_text(json.dumps(partition.to_dict(), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return partition.partition_sha256


def write_group_manifests(
    parent: ParentManifest,
    partition: GroupPartition,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write new group JSONL files and metadata without touching the parent."""

    validate_partition(
        partition,
        parent_manifest_sha256=parent.sha256,
        parent_records=parent,
        expected_group_count=partition.group_count,
        expected_group_size=partition.group_size,
    )
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    result: dict[str, Path] = {}
    for group in partition.groups:
        group_id = str(group["group_id"])
        group_dir = root / group_id
        group_dir.mkdir(parents=True, exist_ok=True)
        records_by_index = {index: record for index, record in enumerate(parent.records)}
        records = []
        for member in group["members"]:
            record = copy.deepcopy(records_by_index[int(member["parent_index"])])
            # Parent manifests commonly use paths relative to the parent
            # manifest.  Group manifests live in a new directory, so rewrite
            # only those path strings while preserving every immutable hash
            # and camera field byte-for-byte in meaning.
            for field in ("input", "input_path", "gs_render", "target", "target_path", "pseudo_target", "reference_image", "reference", "ref", "mask"):
                value = record.get(field)
                if value and not Path(str(value)).expanduser().is_absolute():
                    source = (parent.path.parent / str(value)).resolve()
                    record[field] = os.path.relpath(source, group_dir)
            records.append(record)
        manifest_path = group_dir / "manifest.jsonl"
        digest = write_jsonl(manifest_path, records)
        metadata = {
            "schema": GROUP_MANIFEST_SCHEMA,
            "group_id": group_id,
            "group_index": int(group["group_index"]),
            "group_size": int(group["count"]),
            "parent_manifest_sha256": parent.sha256,
            "partition_sha256": partition.partition_sha256,
            "group_manifest_sha256": digest,
            "camera_keys": list(group["camera_keys"]),
            "camera_fingerprints": list(group["camera_fingerprints"]),
            "source_manifest": parent.path.name,
        }
        (group_dir / "manifest_metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        result[group_id] = manifest_path
    return result


def write_group_manifests_reusing_partition(
    parent: ParentManifest,
    partition: GroupPartition | Mapping[str, Any] | str | Path,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Materialize a second target-kind parent using an existing camera partition.

    A utility pilot commonly has two byte-different 32-view parents (for
    example ``difix`` and ``self_render_a0``) that must share exactly the same
    camera assignment.  The partition itself remains bound to the discovery
    parent that created it, while this function validates only the immutable
    camera identities and parent indices against the second parent.  The
    emitted metadata records both hashes so the distinction is auditable.
    """

    if isinstance(partition, (str, Path)):
        with Path(partition).open("r", encoding="utf-8") as handle:
            partition = json.load(handle)
    value = partition.to_dict() if isinstance(partition, GroupPartition) else dict(partition)
    validate_partition(
        value,
        parent_records=parent,
        expected_group_count=int(value.get("group_count", 4)),
        expected_group_size=int(value.get("group_size", 8)),
    )
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    result: dict[str, Path] = {}
    records_by_index = {index: record for index, record in enumerate(parent.records)}
    for group in value["groups"]:
        group_id = str(group["group_id"])
        group_dir = root / group_id
        group_dir.mkdir(parents=True, exist_ok=True)
        records = []
        for member in group["members"]:
            record = copy.deepcopy(records_by_index[int(member["parent_index"])])
            for field in (
                "input", "input_path", "gs_render", "target", "target_path",
                "pseudo_target", "reference_image", "reference", "ref", "mask",
            ):
                item = record.get(field)
                if item and not Path(str(item)).expanduser().is_absolute():
                    source = (parent.path.parent / str(item)).resolve()
                    record[field] = os.path.relpath(source, group_dir)
            records.append(record)
        manifest_path = group_dir / "manifest.jsonl"
        digest = write_jsonl(manifest_path, records)
        metadata = {
            "schema": GROUP_MANIFEST_SCHEMA,
            "group_id": group_id,
            "group_index": int(group["group_index"]),
            "group_size": int(group["count"]),
            "parent_manifest_sha256": parent.sha256,
            "partition_parent_manifest_sha256": value["parent_manifest_sha256"],
            "partition_sha256": value["partition_sha256"],
            "group_manifest_sha256": digest,
            "camera_keys": list(group["camera_keys"]),
            "camera_fingerprints": list(group["camera_fingerprints"]),
            "source_manifest": parent.path.name,
            "partition_reuse": "camera_identity_partition_v1",
        }
        (group_dir / "manifest_metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        result[group_id] = manifest_path
    return result
