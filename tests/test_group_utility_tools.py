from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

# Some environments preload an unrelated top-level ``tools`` package.  Make
# this repository's package win deterministically during test collection.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.modules.pop("tools", None)

from tools.group_utility.manifest import load_parent_manifest, sha256_file, write_jsonl
from tools.group_utility.partition import (
    build_balanced_partition,
    validate_partition,
    write_group_manifests,
    write_group_manifests_reusing_partition,
    write_partition,
)
from tools.group_utility.schedule import (
    build_group_call_trace,
    build_group_schedule,
    validate_group_schedule,
)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _records(tmp_path: Path, count: int = 32) -> list[dict]:
    records = []
    for index in range(count):
        input_path = tmp_path / f"input_{index:02d}.bin"
        target_path = tmp_path / f"target_{index:02d}.bin"
        reference_path = tmp_path / f"reference_{index:02d}.bin"
        input_path.write_bytes(f"input-{index}".encode())
        target_path.write_bytes(f"target-{index}".encode())
        reference_path.write_bytes(f"reference-{index}".encode())
        records.append(
            {
                "manifest_schema": "synthetic_parent_v1",
                "key": f"camera-{index:02d}",
                "camera_fingerprint": f"fingerprint-{index:02d}",
                "camera": {"width": 16, "height": 16, "R": [1, 0, 0], "T": [index, 0, 0]},
                "is_heldout": False,
                "supervision_target_kind": "difix",
                "input": input_path.name,
                "target": target_path.name,
                "reference_image": reference_path.name,
                "input_sha256": sha256_file(input_path),
                "target_sha256": sha256_file(target_path),
                "reference_image_sha256": sha256_file(reference_path),
                "source_support": index % 4,
                "trajectory": index // 8,
            }
        )
    return records


def _parent(tmp_path: Path) -> tuple[Path, object]:
    manifest_path = tmp_path / "parent.jsonl"
    write_jsonl(manifest_path, _records(tmp_path))
    return manifest_path, load_parent_manifest(manifest_path, require_files=True)


def test_parent_manifest_and_partition_are_byte_stable(tmp_path: Path):
    manifest_path, parent = _parent(tmp_path)
    first = build_balanced_partition(parent, seed=17)
    second = build_balanced_partition(parent, seed=17)
    assert first.partition_sha256 == second.partition_sha256
    assert first.to_dict() == second.to_dict()
    validate_partition(first, parent_manifest_sha256=parent.sha256, parent_records=parent)
    keys = [key for group in first.groups for key in group["camera_keys"]]
    assert len(keys) == 32
    assert len(set(keys)) == 32
    assert sorted(keys) == sorted(record["key"] for record in parent)
    assert manifest_path.read_bytes()  # parent remains readable and untouched


def test_group_manifests_and_partition_hashes(tmp_path: Path):
    _, parent = _parent(tmp_path)
    partition = build_balanced_partition(parent, seed=0)
    output = tmp_path / "group_artifacts"
    partition_path = output / "partition.json"
    write_partition(partition_path, partition)
    paths = write_group_manifests(parent, partition, output / "groups")
    assert set(paths) == {"group_00", "group_01", "group_02", "group_03"}
    for group_id, path in paths.items():
        group_parent = load_parent_manifest(path, expected_count=8, require_files=True)
        metadata = json.loads((path.parent / "manifest_metadata.json").read_text(encoding="utf-8"))
        assert group_parent.sha256 == metadata["group_manifest_sha256"]
        assert metadata["group_id"] == group_id
        assert metadata["partition_sha256"] == partition.partition_sha256


def test_secondary_target_parent_reuses_camera_partition(tmp_path: Path):
    difix_root = tmp_path / "difix"
    difix_root.mkdir()
    _, difix_parent = _parent(difix_root)
    partition = build_balanced_partition(difix_parent, seed=0)

    self_root = tmp_path / "self"
    self_root.mkdir()
    self_records = _records(self_root)
    for record in self_records:
        record["supervision_target_kind"] = "self_render_a0"
    self_manifest_path = self_root / "parent.jsonl"
    write_jsonl(self_manifest_path, self_records)
    self_parent = load_parent_manifest(self_manifest_path, require_files=True)

    output = tmp_path / "self_group_artifacts"
    paths = write_group_manifests_reusing_partition(self_parent, partition, output)
    assert set(paths) == {"group_00", "group_01", "group_02", "group_03"}
    for group_id, path in paths.items():
        group_parent = load_parent_manifest(path, expected_count=8, require_files=True)
        metadata = json.loads((path.parent / "manifest_metadata.json").read_text(encoding="utf-8"))
        assert group_parent.sha256 == metadata["group_manifest_sha256"]
        assert metadata["parent_manifest_sha256"] == self_parent.sha256
        assert metadata["partition_parent_manifest_sha256"] == difix_parent.sha256
        assert metadata["partition_sha256"] == partition.partition_sha256
        assert {record["supervision_target_kind"] for record in group_parent.records} == {"self_render_a0"}


def test_group_schedule_is_39_call_round_robin_with_4_5_exposures():
    keys = [f"camera-{index}" for index in range(8)]
    trace = build_group_call_trace(keys)
    assert len(trace) == 39
    assert trace[0] == {"call_index": 0, "iteration": 10050, "camera_index": 0, "camera_key": "camera-0"}
    assert trace[-1]["iteration"] == 11950
    schedule = build_group_schedule(
        "group_00",
        keys,
        parent_manifest_sha256="a" * 64,
        partition_sha256="b" * 64,
    )
    validate_group_schedule(schedule)
    assert sorted(schedule["exposure_counts"].values()) == [4, 5, 5, 5, 5, 5, 5, 5]


def test_duplicate_key_and_heldout_fail_closed(tmp_path: Path):
    records = _records(tmp_path)
    records[1]["key"] = records[0]["key"]
    path = tmp_path / "duplicate.jsonl"
    write_jsonl(path, records)
    with pytest.raises(ValueError, match="duplicate camera key"):
        load_parent_manifest(path)

    records = _records(tmp_path)
    records[3]["is_heldout"] = True
    path = tmp_path / "heldout.jsonl"
    write_jsonl(path, records)
    with pytest.raises(ValueError, match="held-out camera"):
        load_parent_manifest(path)

    records = _records(tmp_path)
    records[4]["uses_heldout_pose_information"] = True
    path = tmp_path / "heldout_pose_flag.jsonl"
    write_jsonl(path, records)
    with pytest.raises(ValueError, match="held-out camera"):
        load_parent_manifest(path)


def test_wrong_parent_hash_and_changed_asset_fail_closed(tmp_path: Path):
    manifest_path, parent = _parent(tmp_path)
    with pytest.raises(ValueError, match="parent manifest hash mismatch"):
        load_parent_manifest(manifest_path, expected_sha256="0" * 64, require_files=True)
    record = _records(tmp_path)[0]
    (tmp_path / record["target"]).write_bytes(b"tampered")
    tampered_path = tmp_path / "tampered.jsonl"
    write_jsonl(tampered_path, _records(tmp_path))
    # Rebuild the record with an intentionally stale declared target hash.
    tampered = _records(tmp_path)
    tampered[0]["target_sha256"] = "0" * 64
    write_jsonl(tampered_path, tampered)
    with pytest.raises(ValueError, match="target hash mismatch"):
        load_parent_manifest(tampered_path, require_files=True)


def test_partition_tampering_and_schedule_tampering_fail_closed(tmp_path: Path):
    _, parent = _parent(tmp_path)
    partition = build_balanced_partition(parent)
    value = partition.to_dict()
    value["groups"][1]["members"][0]["camera_key"] = value["groups"][0]["members"][0]["camera_key"]
    with pytest.raises(ValueError, match="partition_sha256"):
        validate_partition(value)

    schedule = build_group_schedule(
        "group_00",
        [f"camera-{i}" for i in range(8)],
        parent_manifest_sha256="a" * 64,
        partition_sha256="b" * 64,
    )
    schedule["trace"][0]["camera_key"] = schedule["camera_keys"][1]
    with pytest.raises(ValueError, match="schedule trace"):
        validate_group_schedule(schedule)


def test_wrong_group_dimensions_fail_closed(tmp_path: Path):
    _, parent = _parent(tmp_path)
    with pytest.raises(ValueError, match="record count"):
        build_balanced_partition(parent, group_count=3, group_size=8)
    with pytest.raises(ValueError, match="exactly 8"):
        build_group_call_trace(["one"])
