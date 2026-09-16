"""Fail-closed provenance checks for a frozen Difix cache."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping

from PIL import Image


A0_PSEUDO_MANIFEST_SCHEMA = "a0_pseudo_supervision_v2"
HELDOUT_DIAGNOSTIC_MANIFEST_SCHEMA = "heldout_identity_diagnostic_v2"
SUPPORTED_MANIFEST_SCHEMAS = frozenset(
    {A0_PSEUDO_MANIFEST_SCHEMA, HELDOUT_DIAGNOSTIC_MANIFEST_SCHEMA}
)

CACHE_IDENTITY_FIELDS = (
    "manifest_schema",
    "manifest",
    "manifest_sha256",
    "record_count",
    "seed",
    "model_id",
    "model_revision",
    "difix_code_commit",
    "difix_code_clean",
    "coordinate_policy",
    "dtype",
    "timesteps",
    "guidance_scale",
    "prompt",
    "reproducibility_check_sha256",
)

SIDECAR_TARGET_FIELDS = (
    "input_image",
    "reference_image",
    "target_image",
    "input_sha256",
    "reference_sha256",
    "output_sha256",
    "camera_fingerprint",
    "resolution",
    "source_manifest_record",
    "source_manifest_record_sha256",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def difix_run_metadata_path(manifest: Path) -> Path:
    return manifest.with_suffix(manifest.suffix + ".difix_metadata.json")


def _missing_fields(payload: Mapping[str, Any], fields: tuple[str, ...]) -> list[str]:
    return [field for field in fields if field not in payload or payload[field] is None]


def _require_immutable_commit(value: Any, label: str) -> None:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-fA-F]{40,64}", value) is None:
        raise ValueError(f"{label} must be a full 40- or 64-character hexadecimal commit")


def cache_run_fingerprint(payload: Mapping[str, Any]) -> str:
    missing = _missing_fields(payload, CACHE_IDENTITY_FIELDS)
    if missing:
        raise ValueError(f"Incomplete Difix cache identity: missing {missing}")
    identity = {field: payload[field] for field in CACHE_IDENTITY_FIELDS}
    canonical = json.dumps(
        identity, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(canonical).hexdigest()


def reproducibility_check_sha256(payload: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        dict(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(canonical).hexdigest()


def manifest_record_sha256(payload: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        dict(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(canonical).hexdigest()


def require_manifest_schema(value: Any) -> str:
    schema = str(value)
    if schema not in SUPPORTED_MANIFEST_SCHEMAS:
        raise ValueError(
            f"Unsupported Difix manifest schema {schema!r}; expected one of "
            f"{sorted(SUPPORTED_MANIFEST_SCHEMAS)}"
        )
    return schema


def _resolve_manifest_asset(manifest: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Difix manifest asset path must be a non-empty string")
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (manifest.parent / path).resolve()


def _validate_reproducibility_record(
    manifest: Path, payload: Mapping[str, Any]
) -> None:
    """Bind the one repeated inference to an actual cache-manifest record."""
    record = payload.get("reproducibility_check")
    if not isinstance(record, Mapping) or record.get("passed") is not True:
        raise ValueError(
            "Difix run metadata requires reproducibility_check.passed=true"
        )
    required = (
        "camera_fingerprint",
        "input_sha256",
        "reference_sha256",
        "output_sha256",
        "repeated_output_sha256",
        "mode",
    )
    missing = _missing_fields(record, required)
    if missing:
        raise ValueError(
            f"Difix reproducibility record is incomplete: missing {missing}"
        )
    for field in (
        "input_sha256",
        "reference_sha256",
        "output_sha256",
        "repeated_output_sha256",
    ):
        if re.fullmatch(r"[0-9a-fA-F]{64}", str(record[field])) is None:
            raise ValueError(f"Difix reproducibility {field} is not a SHA256 digest")
    if str(record["mode"]) not in {"byte_exact", "pixel_exact", "metric_tolerance"}:
        raise ValueError("Difix reproducibility record has an invalid mode")

    matching = []
    with manifest.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                candidate = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid Difix manifest JSON at line {line_number}") from exc
            if str(candidate.get("key", "")) == str(record["camera_fingerprint"]):
                matching.append(candidate)
    if len(matching) != 1:
        raise ValueError(
            "Difix reproducibility record does not identify exactly one manifest camera"
        )
    candidate = matching[0]
    input_path = _resolve_manifest_asset(manifest, candidate.get("input"))
    reference_path = _resolve_manifest_asset(
        manifest, candidate.get("reference_image", candidate.get("ref"))
    )
    target_path = _resolve_manifest_asset(manifest, candidate.get("target"))
    for label, path in (
        ("input", input_path),
        ("reference", reference_path),
        ("target", target_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(
                f"Difix reproducibility {label} asset is missing: {path}"
            )
    expected = {
        "input_sha256": sha256_file(input_path),
        "reference_sha256": sha256_file(reference_path),
        "output_sha256": sha256_file(target_path),
    }
    for field, value in expected.items():
        if str(record[field]).lower() != value:
            raise ValueError(
                f"Difix reproducibility {field} is not bound to the manifest asset"
            )


def load_difix_run_metadata(
    manifest: Path,
    *,
    expected_record_count: int | None = None,
    require_reproducibility: bool = False,
) -> dict[str, Any]:
    manifest = manifest.expanduser().resolve()
    metadata_path = difix_run_metadata_path(manifest)
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing Difix run metadata: {metadata_path}")
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Difix run metadata must be a JSON object: {metadata_path}")
    required = CACHE_IDENTITY_FIELDS + ("cache_run_fingerprint",)
    missing = _missing_fields(payload, required)
    if missing:
        raise ValueError(f"Incomplete Difix run metadata {metadata_path}: missing {missing}")
    if payload["manifest_sha256"] != sha256_file(manifest):
        raise ValueError(f"Difix run metadata manifest hash mismatch: {metadata_path}")
    if Path(str(payload["manifest"])).expanduser().resolve() != manifest:
        raise ValueError(f"Difix run metadata points to a different manifest: {metadata_path}")
    if int(payload["record_count"]) <= 0:
        raise ValueError("Difix run metadata record_count must be positive")
    require_manifest_schema(payload["manifest_schema"])
    if expected_record_count is not None and int(payload["record_count"]) != int(
        expected_record_count
    ):
        raise ValueError(
            "Difix run metadata record count mismatch: "
            f"metadata={payload['record_count']}, manifest={expected_record_count}"
        )
    _require_immutable_commit(payload["model_revision"], "Difix model revision")
    _require_immutable_commit(payload["difix_code_commit"], "Difix code commit")
    if payload["difix_code_clean"] is not True:
        raise ValueError("Difix run metadata must record a clean code checkout")
    expected_fingerprint = cache_run_fingerprint(payload)
    if payload["cache_run_fingerprint"] != expected_fingerprint:
        raise ValueError(f"Difix run fingerprint mismatch: {metadata_path}")
    if require_reproducibility:
        reproducibility = payload.get("reproducibility_check")
        if not isinstance(reproducibility, Mapping) or payload.get(
            "reproducibility_check_sha256"
        ) != reproducibility_check_sha256(reproducibility):
            raise ValueError(
                "Difix reproducibility record hash is missing or inconsistent"
            )
        _validate_reproducibility_record(manifest, payload)
    return payload


def validate_difix_target(
    *,
    target_path: Path,
    input_path: Path,
    reference_path: Path,
    camera_fingerprint: str,
    camera_resolution: tuple[int, int],
    run_metadata: Mapping[str, Any],
    require_reproducibility: bool = False,
) -> dict[str, Any]:
    """Validate one target and bind its sidecar to the cache-wide run metadata."""
    reproducibility = run_metadata.get("reproducibility_check")
    if require_reproducibility and (
        not isinstance(reproducibility, Mapping)
        or reproducibility.get("passed") is not True
    ):
        raise ValueError("Difix target belongs to a cache without a passed reproducibility gate")
    for label, path in (
        ("GS input", input_path),
        ("reference", reference_path),
        ("Difix target", target_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label}: {path}")

    sidecar = target_path.with_suffix(target_path.suffix + ".metadata.json")
    if not sidecar.is_file():
        raise FileNotFoundError(f"Missing Difix target sidecar: {sidecar}")
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid Difix sidecar object: {sidecar}")
    required_sidecar = SIDECAR_TARGET_FIELDS
    if not require_reproducibility:
        required_sidecar = tuple(
            field
            for field in SIDECAR_TARGET_FIELDS
            if field not in {"source_manifest_record", "source_manifest_record_sha256"}
        )
    required = CACHE_IDENTITY_FIELDS + ("cache_run_fingerprint",) + required_sidecar
    missing = _missing_fields(payload, required)
    if missing:
        raise ValueError(f"Incomplete Difix provenance sidecar {sidecar}: missing {missing}")

    for field in CACHE_IDENTITY_FIELDS + ("cache_run_fingerprint",):
        if payload[field] != run_metadata.get(field):
            raise ValueError(
                f"Difix sidecar/run metadata mismatch for {field}: {sidecar}"
            )
    if payload["cache_run_fingerprint"] != cache_run_fingerprint(payload):
        raise ValueError(f"Difix sidecar cache fingerprint is internally inconsistent: {sidecar}")

    expected_paths = {
        "input_image": input_path,
        "reference_image": reference_path,
        "target_image": target_path,
    }
    for field, expected in expected_paths.items():
        declared = Path(str(payload[field])).expanduser().resolve()
        if declared != expected.expanduser().resolve():
            raise ValueError(f"Difix provenance path mismatch for {field}: {sidecar}")

    expected_values = {
        "input_sha256": sha256_file(input_path),
        "reference_sha256": sha256_file(reference_path),
        "output_sha256": sha256_file(target_path),
        "camera_fingerprint": camera_fingerprint,
    }
    for field, expected in expected_values.items():
        if payload[field] != expected:
            raise ValueError(f"Difix provenance mismatch for {field}: {sidecar}")

    if require_reproducibility:
        source_record = payload["source_manifest_record"]
        if not isinstance(source_record, Mapping):
            raise ValueError(f"Difix sidecar source manifest record is invalid: {sidecar}")
        source_record = dict(source_record)
        source_digest = manifest_record_sha256(source_record)
        if payload["source_manifest_record_sha256"] != source_digest:
            raise ValueError(f"Difix sidecar source manifest record hash mismatch: {sidecar}")
        manifest = Path(str(run_metadata["manifest"])).expanduser().resolve()
        candidates = []
        with manifest.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    candidate = json.loads(line)
                    if str(candidate.get("key", "")) == str(camera_fingerprint):
                        candidates.append(candidate)
        if len(candidates) != 1 or candidates[0] != source_record:
            raise ValueError(
                f"Difix sidecar does not reproduce its exact source manifest row: {sidecar}"
            )

    with Image.open(target_path) as image:
        actual_resolution = tuple(map(int, image.size))
    if list(payload["resolution"]) != list(actual_resolution):
        raise ValueError(
            f"Difix target resolution mismatch: sidecar={payload['resolution']}, "
            f"actual={list(actual_resolution)}"
        )
    if actual_resolution != tuple(map(int, camera_resolution)):
        raise ValueError(
            "Difix target resolution differs from manifest camera domain: "
            f"target={actual_resolution}, camera={camera_resolution}"
        )
    return payload


def validate_self_render_target(
    *,
    input_path: Path,
    target_path: Path,
    camera_resolution: tuple[int, int],
    input_sha256: str,
    target_sha256: str,
) -> None:
    """Require the SelfRender target to remain the exact exported A0 render."""
    if input_path != target_path:
        raise ValueError("SelfRender target must be exactly the immutable A0 input render")
    if not target_path.is_file():
        raise FileNotFoundError(f"Missing SelfRender A0 render: {target_path}")
    actual_sha256 = sha256_file(target_path)
    if input_sha256 != actual_sha256 or target_sha256 != actual_sha256:
        raise ValueError("SelfRender A0 render hash differs from its frozen manifest value")
    with Image.open(target_path) as image:
        actual_resolution = tuple(map(int, image.size))
    if actual_resolution != tuple(map(int, camera_resolution)):
        raise ValueError(
            "SelfRender target resolution differs from manifest camera domain: "
            f"target={actual_resolution}, camera={camera_resolution}"
        )
