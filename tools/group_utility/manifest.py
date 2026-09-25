"""Strict loading and writing of Group Utility Protocol v1 manifests.

The existing 32-view controlled manifests are evidence for an experiment, not
group-utility manifests themselves.  This module therefore validates the
immutable parent records and emits new, explicitly versioned artifacts.  It
does not modify the parent file and never silently repairs malformed input.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

PARENT_MANIFEST_SCHEMA = "group_utility_parent_manifest_v1"
GROUP_MANIFEST_SCHEMA = "group_utility_group_manifest_v1"
_ALLOWED_TARGET_KINDS = {"difix", "self_render_a0"}
_HELDOUT_WORDS = {"heldout", "held-out", "held_out", "test", "validation", "val"}


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _required_string(record: Mapping[str, Any], fields: Sequence[str], label: str) -> str:
    for field in fields:
        value = record.get(field)
        if value is not None and str(value).strip():
            return str(value)
    raise ValueError(f"manifest record lacks {label}: expected one of {list(fields)}")


def record_camera_key(record: Mapping[str, Any]) -> str:
    return _required_string(record, ("camera_key", "key", "camera_fingerprint"), "camera key")


def record_camera_fingerprint(record: Mapping[str, Any]) -> str:
    return _required_string(
        record,
        ("camera_fingerprint", "fingerprint", "camera_key", "key"),
        "camera fingerprint",
    )


def record_target_kind(record: Mapping[str, Any]) -> str | None:
    for field in ("supervision_target_kind", "pseudo_target_kind", "target_kind"):
        if field in record and record[field] not in (None, ""):
            return str(record[field]).strip().lower()
    return None


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "heldout", "held-out"}


def record_is_heldout(record: Mapping[str, Any]) -> bool:
    for field in ("is_heldout", "heldout", "held_out", "is_held_out"):
        if field in record and _as_bool(record[field]):
            return True
    for field in ("split", "pose_split", "camera_role", "role"):
        value = record.get(field)
        if value is not None and str(value).strip().lower().replace(" ", "-") in _HELDOUT_WORDS:
            return True
    return False


def _resolve_record_path(manifest: Path, value: Any) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (manifest.parent / path).resolve()


def validate_record_integrity(
    record: Mapping[str, Any],
    *,
    manifest_path: str | Path,
    require_files: bool = False,
) -> None:
    """Validate declared input/target/reference files when a record declares them.

    ``require_files=False`` is useful for a portable synthetic manifest.  On a
    real pilot, callers should use ``True`` so missing or changed assets fail
    before any group is generated.
    """

    manifest = Path(manifest_path).resolve()
    path_hash_pairs = (
        (("input", "input_path", "gs_render"), "input_sha256", "input"),
        (("target", "target_path", "pseudo_target"), "target_sha256", "target"),
        (("reference_image", "reference", "ref"), "reference_image_sha256", "reference"),
    )
    for path_fields, hash_field, label in path_hash_pairs:
        path_value = next((record.get(field) for field in path_fields if record.get(field)), None)
        declared_hash = record.get(hash_field)
        if path_value is None:
            if require_files:
                raise ValueError(f"manifest record lacks {label} path")
            continue
        path = _resolve_record_path(manifest, path_value)
        if not path.is_file():
            if require_files:
                raise FileNotFoundError(f"missing {label} file: {path}")
            continue
        if declared_hash:
            actual = sha256_file(path)
            if str(declared_hash).lower() != actual:
                raise ValueError(f"{label} hash mismatch: declared={declared_hash} actual={actual}")


@dataclass(frozen=True)
class ParentManifest:
    """Immutable parent manifest plus its byte-level hash."""

    path: Path
    records: tuple[dict[str, Any], ...]
    sha256: str
    source_schema: str | None = None

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return iter(self.records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.records[index]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"blank JSONL line is not allowed: {path}:{line_number}")
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"manifest record must be an object: {path}:{line_number}")
            records.append(value)
    if not records:
        raise ValueError(f"manifest is empty: {path}")
    return records


def load_parent_manifest(
    path: str | Path,
    *,
    expected_count: int = 32,
    expected_sha256: str | None = None,
    require_source_only: bool = True,
    require_files: bool = False,
) -> ParentManifest:
    """Load and fail-closed validate a frozen 32-view parent JSONL manifest."""

    manifest = Path(path).expanduser().resolve()
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    digest = sha256_file(manifest)
    if expected_sha256 is not None and digest != str(expected_sha256).lower():
        raise ValueError(f"parent manifest hash mismatch: expected={expected_sha256} actual={digest}")
    records = _read_jsonl(manifest)
    if expected_count is not None and len(records) != int(expected_count):
        raise ValueError(f"parent manifest count mismatch: {len(records)} != {expected_count}")

    keys: set[str] = set()
    fingerprints: set[str] = set()
    target_kinds: set[str] = set()
    for index, record in enumerate(records):
        key = record_camera_key(record)
        fingerprint = record_camera_fingerprint(record)
        if key in keys:
            raise ValueError(f"duplicate camera key at index {index}: {key}")
        if fingerprint in fingerprints:
            raise ValueError(f"duplicate camera fingerprint at index {index}: {fingerprint}")
        keys.add(key)
        fingerprints.add(fingerprint)
        if "camera" not in record or not isinstance(record["camera"], dict):
            raise ValueError(f"record {index} lacks a camera object")
        if require_source_only and record_is_heldout(record):
            raise ValueError(f"held-out camera is not allowed in parent manifest: {key}")
        target_kind = record_target_kind(record)
        if target_kind is not None:
            if target_kind not in _ALLOWED_TARGET_KINDS:
                raise ValueError(f"unsupported target kind at index {index}: {target_kind}")
            target_kinds.add(target_kind)
        validate_record_integrity(record, manifest_path=manifest, require_files=require_files)

    if len(target_kinds) > 1:
        raise ValueError(f"parent manifest mixes target kinds: {sorted(target_kinds)}")
    source_schema = str(records[0].get("manifest_schema")) if records[0].get("manifest_schema") else None
    return ParentManifest(path=manifest, records=tuple(records), sha256=digest, source_schema=source_schema)


def write_jsonl(path: str | Path, records: Iterable[Mapping[str, Any]]) -> str:
    """Write canonical JSONL and return the resulting SHA256."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(canonical_json(dict(record)) + "\n")
    return sha256_file(destination)

