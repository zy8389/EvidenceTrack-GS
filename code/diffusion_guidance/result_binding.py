"""Fail-closed binding between measured result rows and controlled-pair audits.

The audit path is evidence for one concrete ``dataset/scene/seed`` group.  It
is deliberately not a global switch that can authorize a different result row.
"""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any, Mapping

from .control_identity import (
    canonical_controlled_checkpoint_provenance,
    canonical_source_image_inventory,
    canonical_training_camera_inventory,
    controlled_checkpoint_provenance_from_metadata,
    controlled_checkpoint_provenance_sha256,
    controlled_pair_id,
    controlled_training_protocol_sha256,
    normalized_camera_names,
    source_camera_set_sha256,
    source_image_inventory_sha256,
    training_camera_inventory_sha256,
    validate_a0_checkpoint_provenance,
    validate_controlled_checkpoint_provenance_files,
    validate_preregistered_controlled_training_protocol,
)
from .checkpoint_state import load_checkpoint_summary
from .experiment_registry import require_scene_role

RESULT_BINDING_SCHEMA = "controlled_result_binding_v5"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _audited_input_fingerprint(records: list[dict[str, str]]) -> str:
    payload = json.dumps(
        records, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _require_sha256(value: Any, label: str, audit: Path) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"Pair audit {label} is not a lowercase SHA256: {audit}")
    return value


def _canonical_path(value: Any, label: str, audit: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Pair audit {label} is invalid: {audit}")
    path = Path(value).expanduser().resolve()
    if value != str(path):
        raise ValueError(f"Pair audit {label} is not a canonical absolute path: {audit}")
    return path


def resolve_pair_audit_path(
    path: str | Path, *, base_dir: Path | None = None
) -> Path:
    """Resolve a CSV audit reference without depending on the process CWD."""
    candidate = Path(path).expanduser()
    if not candidate.is_absolute() and base_dir is not None:
        candidate = base_dir / candidate
    return candidate.resolve()


def read_pair_audit(
    path: str | Path, *, base_dir: Path | None = None
) -> tuple[Path, dict[str, Any]]:
    resolved = resolve_pair_audit_path(path, base_dir=base_dir)
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Pair audit is unreadable: {resolved}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Pair audit must contain one JSON object: {resolved}")
    if payload.get("passed") is not True:
        raise ValueError(f"Controlled-pair audit failed or is incomplete: {resolved}")
    if payload.get("audit_schema") != RESULT_BINDING_SCHEMA:
        raise ValueError(
            "Pair audit has no supported result-binding schema "
            f"({RESULT_BINDING_SCHEMA!r} required): {resolved}"
        )
    required = (
        "pair_id",
        "controlled_pair_id",
        "dataset",
        "scene",
        "role",
        "seed",
        "scene_source_path",
        "start_checkpoint_path",
        "start_checkpoint_sha256",
        "checkpoint_iteration",
        "final_iteration",
        "track_h5_path",
        "track_h5_sha256",
        "source_camera_names",
        "source_camera_set_sha256",
        "source_images_dir",
        "source_image_inventory",
        "source_image_inventory_sha256",
        "source_training_camera_inventory",
        "source_training_camera_inventory_sha256",
        "controlled_training_protocol",
        "controlled_training_protocol_sha256",
        "start_checkpoint_controlled_provenance",
        "start_checkpoint_controlled_provenance_sha256",
        "strict_geometry_protocol",
        "audited_methods",
        "audited_contrasts",
        "pseudo_validated_methods",
        "final_checkpoints",
        "audited_input_files",
        "audited_input_file_count",
        "audited_input_fingerprint",
    )
    missing = [key for key in required if payload.get(key) in (None, "")]
    if missing:
        raise ValueError(f"Pair audit lacks result-binding fields {missing}: {resolved}")
    if payload["pair_id"] != payload["controlled_pair_id"]:
        raise ValueError(f"Pair audit pair_id does not match controlled run identity: {resolved}")
    if not isinstance(payload["pair_id"], str) or not payload["pair_id"].strip():
        raise ValueError(f"Pair audit pair_id is invalid: {resolved}")
    for key in (
        "dataset",
        "scene",
        "role",
    ):
        if not isinstance(payload[key], str) or not payload[key].strip():
            raise ValueError(f"Pair audit {key} is invalid: {resolved}")
    if isinstance(payload["seed"], bool):
        raise ValueError(f"Pair audit seed is invalid: {resolved}")
    try:
        int(payload["seed"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Pair audit seed is invalid: {resolved}") from exc
    for key in ("checkpoint_iteration", "final_iteration"):
        value = payload[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"Pair audit {key} is invalid: {resolved}")
    if payload["final_iteration"] <= payload["checkpoint_iteration"]:
        raise ValueError(f"Pair audit continuation span is invalid: {resolved}")
    require_scene_role(payload["dataset"], payload["scene"], payload["role"])

    scene_path = _canonical_path(payload["scene_source_path"], "scene_source_path", resolved)
    checkpoint_path = _canonical_path(
        payload["start_checkpoint_path"], "start_checkpoint_path", resolved
    )
    track_path = _canonical_path(payload["track_h5_path"], "track_h5_path", resolved)
    images_dir = _canonical_path(payload["source_images_dir"], "source_images_dir", resolved)
    if not scene_path.is_dir():
        raise ValueError(f"Pair audit scene source is missing: {resolved}")
    if not images_dir.is_dir():
        raise ValueError(f"Pair audit source-image directory is missing: {resolved}")
    try:
        images_dir.relative_to(scene_path)
    except ValueError as exc:
        raise ValueError(f"Pair audit source-image directory is outside its scene: {resolved}") from exc
    checkpoint_digest = _require_sha256(
        payload["start_checkpoint_sha256"], "start_checkpoint_sha256", resolved
    )
    track_digest = _require_sha256(
        payload["track_h5_sha256"], "track_h5_sha256", resolved
    )
    if not checkpoint_path.is_file() or _sha256_file(checkpoint_path) != checkpoint_digest:
        raise ValueError(f"Pair audit A0 checkpoint is missing or changed: {resolved}")
    if not track_path.is_file() or _sha256_file(track_path) != track_digest:
        raise ValueError(f"Pair audit Track H5 is missing or changed: {resolved}")
    source_names = payload["source_camera_names"]
    if not isinstance(source_names, list) or source_names != normalized_camera_names(source_names):
        raise ValueError(f"Pair audit source-camera list is not canonical: {resolved}")
    source_set_digest = _require_sha256(
        payload["source_camera_set_sha256"], "source_camera_set_sha256", resolved
    )
    if source_camera_set_sha256(source_names) != source_set_digest:
        raise ValueError(f"Pair audit source-camera set is invalid: {resolved}")

    source_inventory = canonical_source_image_inventory(
        payload["source_image_inventory"]
    )
    if payload["source_image_inventory"] != source_inventory:
        raise ValueError(f"Pair audit source-image inventory is not canonical: {resolved}")
    if [item["camera_name"] for item in source_inventory] != source_names:
        raise ValueError(f"Pair audit source-image inventory has another camera set: {resolved}")
    source_inventory_digest = _require_sha256(
        payload["source_image_inventory_sha256"],
        "source_image_inventory_sha256",
        resolved,
    )
    if source_image_inventory_sha256(source_inventory) != source_inventory_digest:
        raise ValueError(f"Pair audit source-image inventory hash is invalid: {resolved}")
    for item in source_inventory:
        image_path = Path(item["path"])
        try:
            image_path.relative_to(images_dir)
        except ValueError as exc:
            raise ValueError(f"Pair audit source image is outside its image root: {image_path}") from exc
        if not image_path.is_file() or _sha256_file(image_path) != item["sha256"]:
            raise ValueError(f"Pair audit source image is missing or changed: {image_path}")

    camera_inventory = canonical_training_camera_inventory(
        payload["source_training_camera_inventory"]
    )
    if payload["source_training_camera_inventory"] != camera_inventory:
        raise ValueError(f"Pair audit training-camera inventory is not canonical: {resolved}")
    if [item["camera_name"] for item in camera_inventory] != source_names:
        raise ValueError(f"Pair audit training-camera inventory has another camera set: {resolved}")
    camera_inventory_digest = _require_sha256(
        payload["source_training_camera_inventory_sha256"],
        "source_training_camera_inventory_sha256",
        resolved,
    )
    if training_camera_inventory_sha256(camera_inventory) != camera_inventory_digest:
        raise ValueError(f"Pair audit training-camera inventory hash is invalid: {resolved}")

    training_protocol = validate_preregistered_controlled_training_protocol(
        payload["controlled_training_protocol"], role="A1"
    )
    if payload["controlled_training_protocol"] != training_protocol:
        raise ValueError(f"Pair audit training protocol is not canonical: {resolved}")
    training_protocol_digest = _require_sha256(
        payload["controlled_training_protocol_sha256"],
        "controlled_training_protocol_sha256",
        resolved,
    )
    if controlled_training_protocol_sha256(training_protocol) != training_protocol_digest:
        raise ValueError(f"Pair audit training protocol hash is invalid: {resolved}")
    strict_geometry = payload["strict_geometry_protocol"]
    if not isinstance(strict_geometry, dict) or not strict_geometry:
        raise ValueError(f"Pair audit strict geometry protocol is invalid: {resolved}")
    strict_expected = {
        "strict_tracks": True,
        "strict_source_only_geometry": True,
        "strict_track_weight": 0.1,
        "densify_until_iter": 10000,
        "mixed_precision": False,
        "disable_legacy_pseudo_depth": True,
        "disable_depth_loss": True,
        "geometry_reg_enabled": False,
        "use_gt_dca": False,
    }
    strict_mismatch = sorted(
        key for key, expected in strict_expected.items()
        if strict_geometry.get(key) != expected
    )
    if strict_mismatch:
        raise ValueError(
            f"Pair audit strict geometry protocol is off-protocol: {strict_mismatch}: "
            f"{resolved}"
        )

    a0_provenance = canonical_controlled_checkpoint_provenance(
        payload["start_checkpoint_controlled_provenance"]
    )
    if payload["start_checkpoint_controlled_provenance"] != a0_provenance:
        raise ValueError(f"Pair audit A0 provenance is not canonical: {resolved}")
    a0_provenance_digest = _require_sha256(
        payload["start_checkpoint_controlled_provenance_sha256"],
        "start_checkpoint_controlled_provenance_sha256",
        resolved,
    )
    if (
        controlled_checkpoint_provenance_sha256(a0_provenance)
        != a0_provenance_digest
    ):
        raise ValueError(f"Pair audit A0 provenance hash is invalid: {resolved}")
    continuation_provenance = controlled_checkpoint_provenance_from_metadata(
        {
            **payload,
            "role": "A1",
            "start_checkpoint": payload["start_checkpoint_path"],
            "pseudo_manifest_path": None,
            "pseudo_manifest_sha256": None,
            "pseudo_camera_pool_sha256": None,
            "pseudo_target_kind": None,
            "pseudo_rgb_strict_cache": False,
        }
    )
    validate_a0_checkpoint_provenance(a0_provenance, continuation_provenance)
    checkpoint_summary = load_checkpoint_summary(
        checkpoint_path,
        expected_iteration=payload["checkpoint_iteration"],
        require_cuda_rng=True,
        require_controlled_provenance=True,
    )
    if (
        checkpoint_summary["controlled_provenance"] != a0_provenance
        or checkpoint_summary["controlled_provenance_sha256"]
        != a0_provenance_digest
    ):
        raise ValueError(
            f"Pair audit A0 provenance is not embedded in its checkpoint: {resolved}"
        )
    if controlled_pair_id(payload) != payload["controlled_pair_id"]:
        raise ValueError(f"Pair audit controlled pair identity is inconsistent: {resolved}")
    methods = payload["audited_methods"]
    contrasts = payload["audited_contrasts"]
    if (
        not isinstance(methods, list)
        or not methods
        or any(not isinstance(method, str) or not method for method in methods)
        or len(methods) != len(set(methods))
    ):
        raise ValueError(f"Pair audit audited_methods is invalid: {resolved}")
    if not isinstance(contrasts, list) or not contrasts:
        raise ValueError(f"Pair audit audited_contrasts is invalid: {resolved}")
    normalized_contrasts: list[tuple[str, str]] = []
    for contrast in contrasts:
        if (
            not isinstance(contrast, list)
            or len(contrast) != 2
            or any(not isinstance(method, str) or not method for method in contrast)
            or contrast[0] not in methods
            or contrast[1] not in methods
            or contrast[0] == contrast[1]
        ):
            raise ValueError(f"Pair audit contains an invalid contrast {contrast}: {resolved}")
        normalized_contrasts.append((contrast[0], contrast[1]))
    if len(normalized_contrasts) != len(set(normalized_contrasts)):
        raise ValueError(f"Pair audit repeats an audited contrast: {resolved}")

    pseudo_validated_methods = payload["pseudo_validated_methods"]
    expected_pseudo_methods = sorted(
        method for method in methods if method in {"SelfRender", "B"}
    )
    if pseudo_validated_methods != expected_pseudo_methods:
        raise ValueError(
            "Pair audit does not prove strict pseudo usage/manifest validation for "
            f"all pseudo-supervised methods: {resolved}"
        )

    final_checkpoints = payload["final_checkpoints"]
    role_by_method = {"A1": "A1", "SelfRender": "SELFRENDER", "B": "B"}
    expected_final_methods = {
        method for method in methods if method in role_by_method
    }
    if (
        not isinstance(final_checkpoints, dict)
        or set(final_checkpoints) != expected_final_methods
    ):
        raise ValueError(
            f"Pair audit final-checkpoint methods are invalid: {resolved}"
        )
    final_checkpoint_records: set[tuple[str, str]] = set()
    final_record_fields = {
        "path",
        "sha256",
        "iteration",
        "state_format",
        "gaussian_count",
        "render_state_schema",
        "render_state_sha256",
        "controlled_provenance",
        "controlled_provenance_sha256",
    }
    for method in sorted(expected_final_methods):
        record = final_checkpoints[method]
        if not isinstance(record, dict) or set(record) != final_record_fields:
            raise ValueError(
                f"Pair audit final-checkpoint record is invalid for {method}: {resolved}"
            )
        final_path = _canonical_path(
            record["path"], f"final_checkpoints.{method}.path", resolved
        )
        final_digest = _require_sha256(
            record["sha256"], f"final_checkpoints.{method}.sha256", resolved
        )
        if not final_path.is_file() or _sha256_file(final_path) != final_digest:
            raise ValueError(
                f"Pair audit final checkpoint is missing or changed for {method}: {resolved}"
            )
        final_summary = load_checkpoint_summary(
            final_path,
            expected_iteration=payload["final_iteration"],
            require_cuda_rng=True,
            require_controlled_provenance=True,
        )
        final_provenance = validate_controlled_checkpoint_provenance_files(
            record["controlled_provenance"]
        )
        if final_provenance["role"] != role_by_method[method]:
            raise ValueError(
                f"Pair audit final checkpoint has the wrong role for {method}: {resolved}"
            )
        if (
            final_provenance["start_checkpoint"] != str(checkpoint_path)
            or final_provenance["start_checkpoint_sha256"] != checkpoint_digest
            or final_provenance[
                "start_checkpoint_controlled_provenance_sha256"
            ]
            != a0_provenance_digest
        ):
            raise ValueError(
                f"Pair audit final checkpoint has the wrong exact A0 lineage for "
                f"{method}: {resolved}"
            )
        validate_a0_checkpoint_provenance(a0_provenance, final_provenance)
        final_provenance_digest = _require_sha256(
            record["controlled_provenance_sha256"],
            f"final_checkpoints.{method}.controlled_provenance_sha256",
            resolved,
        )
        if (
            controlled_checkpoint_provenance_sha256(final_provenance)
            != final_provenance_digest
            or final_summary["controlled_provenance"] != final_provenance
            or final_summary["controlled_provenance_sha256"]
            != final_provenance_digest
            or record["iteration"] != final_summary["iteration"]
            or record["state_format"] != final_summary["format"]
            or record["gaussian_count"] != final_summary["gaussian_count"]
            or record["render_state_schema"]
            != final_summary["render_state_schema"]
            or record["render_state_sha256"]
            != final_summary["render_state_sha256"]
        ):
            raise ValueError(
                f"Pair audit final-checkpoint evidence is inconsistent for {method}: {resolved}"
            )
        _require_sha256(
            record["render_state_sha256"],
            f"final_checkpoints.{method}.render_state_sha256",
            resolved,
        )
        final_checkpoint_records.add((str(final_path), final_digest))
    input_files = payload["audited_input_files"]
    input_count = payload["audited_input_file_count"]
    if (
        isinstance(input_count, bool)
        or not isinstance(input_count, int)
        or input_count <= 0
        or not isinstance(input_files, list)
        or len(input_files) != input_count
    ):
        raise ValueError(f"Pair audit input-file inventory is invalid: {resolved}")
    normalized_inputs: list[dict[str, str]] = []
    seen_paths: set[Path] = set()
    for item in input_files:
        if not isinstance(item, dict) or set(item) != {"path", "sha256"}:
            raise ValueError(f"Pair audit input-file record is invalid: {resolved}")
        path_value = item["path"]
        digest = item["sha256"]
        if not isinstance(path_value, str) or not path_value.strip():
            raise ValueError(f"Pair audit input-file path is invalid: {resolved}")
        path = Path(path_value).expanduser().resolve()
        if path_value != str(path):
            raise ValueError(f"Pair audit input-file path is not canonical: {path_value}")
        if path in seen_paths:
            raise ValueError(f"Pair audit repeats an input-file path: {path}")
        seen_paths.add(path)
        _require_sha256(digest, f"input-file hash for {path}", resolved)
        if not path.is_file() or _sha256_file(path) != digest:
            raise ValueError(f"Pair audit input file is missing or changed: {path}")
        normalized_inputs.append({"path": str(path), "sha256": digest})
    if normalized_inputs != sorted(normalized_inputs, key=lambda item: item["path"]):
        raise ValueError(f"Pair audit input-file inventory is not canonical: {resolved}")
    input_fingerprint = _require_sha256(
        payload["audited_input_fingerprint"], "audited_input_fingerprint", resolved
    )
    if input_fingerprint != _audited_input_fingerprint(normalized_inputs):
        raise ValueError(f"Pair audit input-file fingerprint is inconsistent: {resolved}")
    audited_records = {(item["path"], item["sha256"]) for item in normalized_inputs}
    required_records = {
        (str(checkpoint_path), checkpoint_digest),
        (str(track_path), track_digest),
        *((item["path"], item["sha256"]) for item in source_inventory),
        *final_checkpoint_records,
    }
    missing_records = sorted(required_records - audited_records)
    if missing_records:
        raise ValueError(
            f"Pair audit input-file inventory omits controlled inputs: {missing_records}"
        )
    return resolved, payload


def require_audit_binding(
    payload: dict[str, Any],
    *,
    dataset: str,
    scene: str,
    seed: int,
    method: str,
    pair_id: str | None = None,
    role: str | None = None,
) -> None:
    if isinstance(seed, bool):
        raise ValueError("Result seed must be an integer, not a boolean")
    expected = {
        "dataset": str(dataset),
        "scene": str(scene),
        "seed": int(seed),
    }
    for key, value in expected.items():
        observed = int(payload[key]) if key == "seed" else str(payload[key])
        if observed != value:
            raise ValueError(
                f"Pair audit {key} mismatch: row={value!r}, audit={observed!r}"
            )
    expected_role = require_scene_role(dataset, scene, role or payload["role"])
    if payload["role"] != expected_role:
        raise ValueError(
            f"Pair audit role mismatch: row={expected_role!r}, audit={payload['role']!r}"
        )
    if pair_id is not None and str(pair_id) != str(payload["pair_id"]):
        raise ValueError(
            f"Result pair_id does not match audit: row={pair_id!r}, "
            f"audit={payload['pair_id']!r}"
        )
    if method not in payload["audited_methods"]:
        raise ValueError(
            f"Method {method!r} is not covered by pair audit methods "
            f"{payload['audited_methods']!r}"
        )


def audit_supports_contrast(
    payload: dict[str, Any], baseline: str, treatment: str
) -> bool:
    return [str(baseline), str(treatment)] in payload["audited_contrasts"]


def require_final_checkpoint_binding(
    checkpoint_summary: Mapping[str, Any],
    audit_payload: Mapping[str, Any],
    *,
    method: str,
    checkpoint_path: str | Path,
) -> dict[str, Any]:
    """Bind a final 12k checkpoint to the run authorized by a pair audit."""
    normalized_method = str(method)
    role_by_method = {"A1": "A1", "SelfRender": "SELFRENDER", "B": "B"}
    if normalized_method not in role_by_method:
        raise ValueError(f"Unsupported controlled checkpoint method: {method!r}")
    if normalized_method not in audit_payload.get("audited_methods", []):
        raise ValueError(
            f"Pair audit does not authorize final checkpoint method {normalized_method!r}"
        )
    bindings = audit_payload.get("final_checkpoints")
    if not isinstance(bindings, Mapping) or normalized_method not in bindings:
        raise ValueError(
            f"Pair audit lacks final checkpoint evidence for {normalized_method!r}"
        )
    binding = bindings[normalized_method]
    if not isinstance(binding, Mapping):
        raise ValueError("Pair audit final checkpoint evidence is invalid")
    resolved_checkpoint = Path(checkpoint_path).expanduser().resolve()
    expected_path = Path(str(binding.get("path", ""))).expanduser().resolve()
    if resolved_checkpoint != expected_path:
        raise ValueError(
            f"Final {normalized_method} checkpoint path differs from the pair audit"
        )
    expected_file_digest = binding.get("sha256")
    if (
        not resolved_checkpoint.is_file()
        or not isinstance(expected_file_digest, str)
        or _sha256_file(resolved_checkpoint) != expected_file_digest
    ):
        raise ValueError(
            f"Final {normalized_method} checkpoint bytes differ from the pair audit"
        )
    expected_iteration = audit_payload.get("final_iteration")
    if (
        isinstance(expected_iteration, bool)
        or not isinstance(expected_iteration, int)
        or checkpoint_summary.get("iteration") != expected_iteration
    ):
        raise ValueError("Final checkpoint iteration differs from the pair audit")
    expected = canonical_controlled_checkpoint_provenance(
        binding.get("controlled_provenance")
    )
    if expected["role"] != role_by_method[normalized_method]:
        raise ValueError(
            f"Final {normalized_method} checkpoint role differs from the pair audit"
        )
    audit_a0_path = Path(
        str(audit_payload.get("start_checkpoint_path", ""))
    ).expanduser().resolve()
    audit_a0_digest = audit_payload.get("start_checkpoint_sha256")
    audit_a0_provenance_digest = audit_payload.get(
        "start_checkpoint_controlled_provenance_sha256"
    )
    if (
        expected["start_checkpoint"] != str(audit_a0_path)
        or expected["start_checkpoint_sha256"] != audit_a0_digest
        or expected["start_checkpoint_controlled_provenance_sha256"]
        != audit_a0_provenance_digest
    ):
        raise ValueError(
            f"Final {normalized_method} checkpoint has the wrong exact A0 lineage"
        )
    observed = canonical_controlled_checkpoint_provenance(
        checkpoint_summary.get("controlled_provenance")
    )
    if observed != expected:
        raise ValueError(
            f"Final {normalized_method} checkpoint provenance differs from its pair audit"
        )
    expected_digest = controlled_checkpoint_provenance_sha256(expected)
    if (
        binding.get("controlled_provenance_sha256") != expected_digest
        or checkpoint_summary.get("controlled_provenance_sha256") != expected_digest
        or binding.get("iteration") != checkpoint_summary.get("iteration")
        or binding.get("state_format") != checkpoint_summary.get("format")
        or binding.get("gaussian_count") != checkpoint_summary.get("gaussian_count")
        or binding.get("render_state_schema")
        != checkpoint_summary.get("render_state_schema")
        or binding.get("render_state_sha256")
        != checkpoint_summary.get("render_state_sha256")
    ):
        raise ValueError(
            f"Final {normalized_method} checkpoint evidence is inconsistent"
        )
    return observed
