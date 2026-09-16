"""Pure validation helpers for the preregistered identity protocol."""

from __future__ import annotations

import math
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from .checkpoint_state import FORMAT as CHECKPOINT_FORMAT, RENDER_STATE_SCHEMA
from .control_identity import (
    canonical_source_image_inventory,
    normalized_camera_names,
    source_camera_set_sha256,
    source_image_inventory_sha256,
)
from .experiment_registry import require_scene_role
from .result_binding import (
    audit_supports_contrast,
    read_pair_audit,
    require_audit_binding,
)


PAIRED_RECONSTRUCTION_FIELDS = (
    "a1_difix_protocol",
    "b_difix_protocol",
    "a1_checkpoint",
    "a1_checkpoint_sha256",
    "a1_checkpoint_iteration",
    "a1_checkpoint_state_format",
    "a1_checkpoint_gaussian_count",
    "a1_checkpoint_render_state_schema",
    "a1_checkpoint_render_state_sha256",
    "a1_checkpoint_controlled_provenance_sha256",
    "a1_scene_ply_gaussian_count",
    "a1_scene_ply_checkpoint_count_match",
    "a1_scene_ply_render_state_schema",
    "a1_scene_ply_render_state_sha256",
    "a1_scene_ply_checkpoint_render_state_match",
    "b_checkpoint",
    "b_checkpoint_sha256",
    "b_checkpoint_iteration",
    "b_checkpoint_state_format",
    "b_checkpoint_gaussian_count",
    "b_checkpoint_render_state_schema",
    "b_checkpoint_render_state_sha256",
    "b_checkpoint_controlled_provenance_sha256",
    "b_scene_ply_gaussian_count",
    "b_scene_ply_checkpoint_count_match",
    "b_scene_ply_render_state_schema",
    "b_scene_ply_render_state_sha256",
    "b_scene_ply_checkpoint_render_state_match",
)

PAIRED_METADATA_BINDING_FIELDS = (
    "dataset",
    "scene",
    "seed",
    "experiment_role",
    "scene_source_path",
    "controlled_pair_id",
    "pair_audit",
    "pair_audit_sha256",
    "track_h5",
    "track_h5_sha256",
    "source_camera_names",
    "source_camera_set_sha256",
    "source_images_dir",
    "source_image_inventory",
    "source_image_inventory_sha256",
    "a1_evidence_manifest",
    "a1_evidence_manifest_sha256",
    "b_evidence_manifest",
    "b_evidence_manifest_sha256",
    "a1_difix_manifest",
    "a1_difix_manifest_sha256",
    "b_difix_manifest",
    "b_difix_manifest_sha256",
    "a1_difix_run_metadata",
    "a1_difix_run_metadata_sha256",
    "b_difix_run_metadata",
    "b_difix_run_metadata_sha256",
    "a1_difix_cache_run_fingerprint",
    "b_difix_cache_run_fingerprint",
) + PAIRED_RECONSTRUCTION_FIELDS

_PAIRED_PATH_FIELDS = frozenset(
    {
        "pair_audit",
        "track_h5",
        "source_images_dir",
        "a1_evidence_manifest",
        "b_evidence_manifest",
        "a1_difix_manifest",
        "b_difix_manifest",
        "a1_difix_run_metadata",
        "b_difix_run_metadata",
        "a1_checkpoint",
        "b_checkpoint",
    }
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_bound_file(path_value: Any, digest_value: Any, label: str) -> Path:
    path = Path(str(path_value)).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    digest = str(digest_value)
    if re.fullmatch(r"[0-9a-f]{64}", digest) is None or _sha256_file(path) != digest:
        raise ValueError(f"{label} hash is stale: {path}")
    return path


def _same_json(left: Any, right: Any) -> bool:
    return json.dumps(
        left, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ) == json.dumps(right, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def validate_primary_radius(primary_radius: float, window_radii: Sequence[float]) -> float:
    primary = float(primary_radius)
    radii = [float(value) for value in window_radii]
    if not math.isfinite(primary) or primary <= 0.0:
        raise ValueError("--primary-radius must be a finite positive value")
    if any(not math.isfinite(value) or value <= 0.0 for value in radii):
        raise ValueError("--window-radii must contain only finite positive values")
    if len(set(radii)) != len(radii):
        raise ValueError("--window-radii must not contain duplicate values")
    if primary not in radii:
        raise ValueError(
            "--primary-radius must be one of --window-radii; it cannot be selected by list order"
        )
    return primary


def validate_hard_negative_radius(primary_radius: float, hard_negative_radius: float) -> float:
    hard_radius = float(hard_negative_radius)
    if not math.isfinite(hard_radius) or hard_radius <= 0.0:
        raise ValueError("--hard-negative-radius must be a finite positive value")
    if hard_radius != float(primary_radius):
        raise ValueError(
            "--hard-negative-radius must equal --primary-radius; each robustness "
            "window uses its own radius for local-negative selection"
        )
    return hard_radius


def dinov2_protocol_from_metadata(metadata: Mapping[str, Any]) -> dict:
    """Validate and canonicalize the immutable DINOv2 identity fields."""
    provenance = metadata.get("extractor_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("extractor_provenance must be an object")
    if str(metadata.get("feature_backend", "")) != "dinov2":
        raise ValueError("DINOv2 provenance requires feature_backend=dinov2")
    dino_fields = (
        "backend",
        "local_repository_path",
        "repository_commit",
        "pretrained_weight_path",
        "pretrained_weight_sha256",
        "weight_loading",
    )
    missing = [field for field in dino_fields if not provenance.get(field)]
    if missing:
        raise ValueError(f"Incomplete DINOv2 provenance: missing {missing}")
    model = metadata.get("dinov2_model")
    if not model or provenance["backend"] != f"dinov2:{model}":
        raise ValueError("DINOv2 model name and extractor backend identity disagree")
    if re.fullmatch(r"[0-9a-fA-F]{40,64}", str(provenance["repository_commit"])) is None:
        raise ValueError("DINOv2 repository_commit must be a full hexadecimal commit")
    if re.fullmatch(r"[0-9a-fA-F]{64}", str(provenance["pretrained_weight_sha256"])) is None:
        raise ValueError("DINOv2 pretrained_weight_sha256 must be a SHA256 digest")
    if provenance["weight_loading"] != "explicit_local_state_dict":
        raise ValueError("DINOv2 weights must come from an explicit local state dict")
    repository = Path(str(provenance["local_repository_path"])).expanduser().resolve()
    if not repository.is_dir() or not (repository / "hubconf.py").is_file():
        raise ValueError("DINOv2 local repository is missing or invalid")
    weights = _require_bound_file(
        provenance["pretrained_weight_path"],
        provenance["pretrained_weight_sha256"],
        "DINOv2 pretrained weights",
    )
    return {
        "feature_backend_identity": str(provenance["backend"]),
        "dinov2_model": str(model),
        "dinov2_local_repository_path": str(repository),
        "dinov2_repository_commit": str(provenance["repository_commit"]),
        "dinov2_pretrained_weight_path": str(weights),
        "dinov2_pretrained_weight_sha256": str(
            provenance["pretrained_weight_sha256"]
        ),
    }


def identity_protocol_from_metadata(metadata: Mapping[str, Any], radius: float) -> dict:
    """Extract the comparison-critical protocol and reject incomplete DINO metadata."""
    required = (
        "feature_backend",
        "window_radii",
        "primary_radius",
        "extractor_provenance",
        "hard_negative_radius",
        "temperature",
        "track_h5_sha256",
        "feature_maps",
        "target_manifest",
        "target_manifest_sha256",
        "paired_identity_arm",
        "dataset",
        "scene",
        "seed",
        "experiment_role",
        "scene_source_path",
        "controlled_pair_id",
        "pair_audit",
        "pair_audit_sha256",
        "track_h5",
        "source_camera_names",
        "source_camera_set_sha256",
        "source_images_dir",
        "source_image_inventory",
        "source_image_inventory_sha256",
        "paired_manifest_metadata",
        "paired_manifest_metadata_sha256",
        "paired_manifest_record_count",
        "a1_evidence_manifest",
        "a1_evidence_manifest_sha256",
        "b_evidence_manifest",
        "b_evidence_manifest_sha256",
        "a1_difix_manifest",
        "a1_difix_manifest_sha256",
        "b_difix_manifest",
        "b_difix_manifest_sha256",
        "a1_difix_run_metadata",
        "a1_difix_run_metadata_sha256",
        "b_difix_run_metadata",
        "b_difix_run_metadata_sha256",
        "a1_difix_cache_run_fingerprint",
        "b_difix_cache_run_fingerprint",
        "per_track_csv",
        "per_track_csv_sha256",
    ) + PAIRED_RECONSTRUCTION_FIELDS
    missing = [field for field in required if field not in metadata]
    if missing:
        raise ValueError(f"Incomplete identity metadata: missing {missing}")
    primary = validate_primary_radius(metadata["primary_radius"], metadata["window_radii"])
    if primary != float(radius):
        raise ValueError(
            f"Requested report radius {radius} is not metadata primary_radius {primary}"
        )
    hard_negative_radius = validate_hard_negative_radius(
        primary, metadata["hard_negative_radius"]
    )
    provenance = metadata["extractor_provenance"]
    if not isinstance(provenance, Mapping):
        raise ValueError("extractor_provenance must be an object")
    backend = str(metadata["feature_backend"])
    feature_maps = metadata["feature_maps"]
    if not isinstance(feature_maps, list) or not feature_maps:
        raise ValueError("feature_maps must be a non-empty list")
    geometry_counts: dict[str, tuple[dict, int]] = {}
    for index, feature_map in enumerate(feature_maps):
        if not isinstance(feature_map, Mapping):
            raise ValueError(f"feature_maps[{index}] must be an object")
        required_geometry = (
            "backend",
            "input_image_resolution",
            "model_input_resolution",
            "feature_resolution",
            "feature_stride",
            "effective_feature_stride",
            "interpolation_method",
        )
        missing = [field for field in required_geometry if field not in feature_map]
        if missing:
            raise ValueError(f"Incomplete feature_maps[{index}]: missing {missing}")
        geometry = {
            "role": str(feature_map.get("target_variant", "source")),
            "backend": str(feature_map["backend"]),
            "input_image_resolution": [
                int(value) for value in feature_map["input_image_resolution"]
            ],
            "model_input_resolution": [
                int(value) for value in feature_map["model_input_resolution"]
            ],
            "feature_resolution": [
                int(value) for value in feature_map["feature_resolution"]
            ],
            "feature_stride": [float(value) for value in feature_map["feature_stride"]],
            "effective_feature_stride": float(
                feature_map["effective_feature_stride"]
            ),
            "interpolation_method": str(feature_map["interpolation_method"]),
        }
        if any(
            len(geometry[field]) != 2 or any(value <= 0 for value in geometry[field])
            for field in (
                "input_image_resolution",
                "model_input_resolution",
                "feature_resolution",
                "feature_stride",
            )
        ) or not math.isfinite(geometry["effective_feature_stride"]) or geometry[
            "effective_feature_stride"
        ] <= 0:
            raise ValueError(f"Invalid resolution/stride in feature_maps[{index}]")
        key = json.dumps(geometry, sort_keys=True, separators=(",", ":"))
        if key in geometry_counts:
            previous, count = geometry_counts[key]
            geometry_counts[key] = (previous, count + 1)
        else:
            geometry_counts[key] = (geometry, 1)
    feature_map_geometry = []
    for key in sorted(geometry_counts):
        geometry, count = geometry_counts[key]
        feature_map_geometry.append({**geometry, "count": count})
    protocol = {
        "feature_backend": backend,
        "feature_backend_identity": provenance.get("backend"),
        "hard_negative_radius": hard_negative_radius,
        "temperature": float(metadata["temperature"]),
        "track_h5_sha256": str(metadata["track_h5_sha256"]),
        "target_manifest": str(metadata["target_manifest"]),
        "target_manifest_sha256": str(metadata["target_manifest_sha256"]),
        "paired_identity_arm": metadata["paired_identity_arm"],
        "dataset": str(metadata["dataset"]),
        "scene": str(metadata["scene"]),
        "seed": int(metadata["seed"]),
        "experiment_role": str(metadata["experiment_role"]),
        "scene_source_path": str(metadata["scene_source_path"]),
        "controlled_pair_id": str(metadata["controlled_pair_id"]),
        "pair_audit": str(metadata["pair_audit"]),
        "pair_audit_sha256": str(metadata["pair_audit_sha256"]),
        "source_camera_names": list(metadata["source_camera_names"]),
        "source_camera_set_sha256": str(metadata["source_camera_set_sha256"]),
        "source_images_dir": str(metadata["source_images_dir"]),
        "source_image_inventory": metadata["source_image_inventory"],
        "source_image_inventory_sha256": str(
            metadata["source_image_inventory_sha256"]
        ),
        "paired_manifest_metadata": str(metadata["paired_manifest_metadata"]),
        "paired_manifest_metadata_sha256": str(
            metadata["paired_manifest_metadata_sha256"]
        ),
        "a1_evidence_manifest_sha256": str(metadata["a1_evidence_manifest_sha256"]),
        "a1_evidence_manifest": str(metadata["a1_evidence_manifest"]),
        "b_evidence_manifest_sha256": str(metadata["b_evidence_manifest_sha256"]),
        "b_evidence_manifest": str(metadata["b_evidence_manifest"]),
        "a1_difix_manifest_sha256": str(metadata["a1_difix_manifest_sha256"]),
        "a1_difix_manifest": str(metadata["a1_difix_manifest"]),
        "b_difix_manifest_sha256": str(metadata["b_difix_manifest_sha256"]),
        "b_difix_manifest": str(metadata["b_difix_manifest"]),
        "a1_difix_run_metadata_sha256": str(
            metadata["a1_difix_run_metadata_sha256"]
        ),
        "a1_difix_run_metadata": str(metadata["a1_difix_run_metadata"]),
        "b_difix_run_metadata_sha256": str(
            metadata["b_difix_run_metadata_sha256"]
        ),
        "b_difix_run_metadata": str(metadata["b_difix_run_metadata"]),
        "a1_difix_cache_run_fingerprint": str(
            metadata["a1_difix_cache_run_fingerprint"]
        ),
        "b_difix_cache_run_fingerprint": str(
            metadata["b_difix_cache_run_fingerprint"]
        ),
        "paired_manifest_record_count": int(metadata["paired_manifest_record_count"]),
        "per_track_csv": str(metadata["per_track_csv"]),
        "per_track_csv_sha256": str(metadata["per_track_csv_sha256"]),
        "feature_map_geometry": feature_map_geometry,
        "dinov2_model": None,
        "dinov2_repository_commit": None,
        "dinov2_pretrained_weight_sha256": None,
    }
    protocol.update({field: metadata[field] for field in PAIRED_RECONSTRUCTION_FIELDS})
    if re.fullmatch(r"[0-9a-fA-F]{64}", protocol["track_h5_sha256"]) is None:
        raise ValueError("track_h5_sha256 must be a SHA256 digest")
    if not protocol["target_manifest"]:
        raise ValueError("target_manifest must be an immutable manifest path")
    if re.fullmatch(r"[0-9a-fA-F]{64}", protocol["target_manifest_sha256"]) is None:
        raise ValueError("target_manifest_sha256 must be a SHA256 digest")
    if protocol["paired_identity_arm"] not in {"A1", "B"}:
        raise ValueError("paired_identity_arm must be A1 or B for final A1/B comparison")
    role = require_scene_role(
        protocol["dataset"], protocol["scene"], protocol["experiment_role"]
    )
    protocol["experiment_role"] = role
    scene_source_path = Path(protocol["scene_source_path"]).expanduser().resolve()
    if not scene_source_path.is_dir() or scene_source_path.name != protocol["scene"]:
        raise ValueError("Identity metadata scene_source_path is missing or names another scene")
    protocol["scene_source_path"] = str(scene_source_path)
    pair_audit_path = _require_bound_file(
        metadata["pair_audit"], metadata["pair_audit_sha256"], "pair audit"
    )
    protocol["pair_audit"] = str(pair_audit_path)
    _, pair_audit = read_pair_audit(pair_audit_path)
    require_audit_binding(
        pair_audit,
        dataset=protocol["dataset"],
        scene=protocol["scene"],
        seed=protocol["seed"],
        method=protocol["paired_identity_arm"],
        pair_id=protocol["controlled_pair_id"],
        role=role,
    )
    if not audit_supports_contrast(pair_audit, "A1", "B"):
        raise ValueError("Identity metadata pair audit does not authorize A1/B")
    if Path(pair_audit["scene_source_path"]).expanduser().resolve() != scene_source_path:
        raise ValueError("Identity metadata scene path differs from its pair audit")
    track_path = _require_bound_file(
        metadata["track_h5"], metadata["track_h5_sha256"], "Track H5"
    )
    protocol["track_h5"] = str(track_path)
    source_names = metadata["source_camera_names"]
    if not isinstance(source_names, list) or source_camera_set_sha256(source_names) != protocol[
        "source_camera_set_sha256"
    ]:
        raise ValueError("Identity metadata source-camera binding is invalid")
    protocol["source_camera_names"] = normalized_camera_names(source_names)
    images_dir = Path(protocol["source_images_dir"]).expanduser().resolve()
    if images_dir != (scene_source_path / "images").resolve() or not images_dir.is_dir():
        raise ValueError("Identity metadata source_images_dir is not the audited scene image root")
    protocol["source_images_dir"] = str(images_dir)
    inventory = canonical_source_image_inventory(metadata["source_image_inventory"])
    if [item["camera_name"] for item in inventory] != protocol["source_camera_names"]:
        raise ValueError("Identity source-image inventory differs from its source-camera set")
    for item in inventory:
        image_path = Path(item["path"])
        try:
            image_path.relative_to(images_dir)
        except ValueError as exc:
            raise ValueError("Identity source image is outside the audited scene image root") from exc
        _require_bound_file(image_path, item["sha256"], f"source image {item['camera_name']}")
    inventory_digest = source_image_inventory_sha256(inventory)
    if inventory_digest != protocol["source_image_inventory_sha256"]:
        raise ValueError("Identity source-image inventory fingerprint is stale")
    protocol["source_image_inventory"] = inventory
    for path_field, digest_field in (
        ("target_manifest", "target_manifest_sha256"),
        ("paired_manifest_metadata", "paired_manifest_metadata_sha256"),
        ("a1_evidence_manifest", "a1_evidence_manifest_sha256"),
        ("b_evidence_manifest", "b_evidence_manifest_sha256"),
        ("a1_difix_manifest", "a1_difix_manifest_sha256"),
        ("b_difix_manifest", "b_difix_manifest_sha256"),
        ("a1_difix_run_metadata", "a1_difix_run_metadata_sha256"),
        ("b_difix_run_metadata", "b_difix_run_metadata_sha256"),
        ("a1_checkpoint", "a1_checkpoint_sha256"),
        ("b_checkpoint", "b_checkpoint_sha256"),
        ("per_track_csv", "per_track_csv_sha256"),
    ):
        protocol[path_field] = str(
            _require_bound_file(metadata[path_field], metadata[digest_field], path_field)
        )
    target_manifest = Path(protocol["target_manifest"])
    paired_metadata_path = Path(protocol["paired_manifest_metadata"])
    expected_metadata_path = target_manifest.with_suffix(
        target_manifest.suffix + ".metadata.json"
    )
    if paired_metadata_path != expected_metadata_path:
        raise ValueError("Identity evaluator metadata is bound to the wrong paired sidecar")
    paired_metadata = json.loads(paired_metadata_path.read_text(encoding="utf-8"))
    if (
        not isinstance(paired_metadata, Mapping)
        or paired_metadata.get("schema") != "paired_identity_manifest_metadata_v4"
        or paired_metadata.get("passed") is not True
        or Path(str(paired_metadata.get("paired_manifest", ""))).expanduser().resolve()
        != target_manifest
        or paired_metadata.get("paired_manifest_sha256")
        != protocol["target_manifest_sha256"]
        or int(paired_metadata.get("record_count", -1))
        != protocol["paired_manifest_record_count"]
        or protocol["paired_manifest_record_count"] <= 0
    ):
        raise ValueError("Identity paired-manifest sidecar is stale or inconsistent")
    for field in PAIRED_METADATA_BINDING_FIELDS:
        if field not in paired_metadata:
            raise ValueError(f"Identity paired-manifest sidecar lacks {field}")
        if field in _PAIRED_PATH_FIELDS:
            observed = Path(str(metadata[field])).expanduser().resolve()
            expected = Path(str(paired_metadata[field])).expanduser()
            if not expected.is_absolute():
                expected = paired_metadata_path.parent / expected
            matches = observed == expected.resolve()
        else:
            matches = _same_json(metadata[field], paired_metadata[field])
        if not matches:
            raise ValueError(
                f"Identity evaluator metadata differs from paired sidecar for {field}"
            )
    if not isinstance(protocol["a1_difix_protocol"], Mapping) or not isinstance(
        protocol["b_difix_protocol"], Mapping
    ):
        raise ValueError("Identity metadata Difix protocols must be objects")
    if not _same_json(protocol["a1_difix_protocol"], protocol["b_difix_protocol"]):
        raise ValueError("Identity metadata A1/B Difix protocols differ")
    for prefix in ("a1", "b"):
        checkpoint_count = int(protocol[f"{prefix}_checkpoint_gaussian_count"])
        scene_ply_count = int(protocol[f"{prefix}_scene_ply_gaussian_count"])
        state_digest = str(protocol[f"{prefix}_checkpoint_render_state_sha256"])
        provenance_digest = str(
            protocol[f"{prefix}_checkpoint_controlled_provenance_sha256"]
        )
        if (
            int(protocol[f"{prefix}_checkpoint_iteration"]) != 12000
            or protocol[f"{prefix}_checkpoint_state_format"] != CHECKPOINT_FORMAT
            or checkpoint_count <= 0
            or scene_ply_count != checkpoint_count
            or protocol[f"{prefix}_checkpoint_render_state_schema"]
            != RENDER_STATE_SCHEMA
            or protocol[f"{prefix}_scene_ply_render_state_schema"]
            != RENDER_STATE_SCHEMA
            or re.fullmatch(r"[0-9a-f]{64}", state_digest) is None
            or re.fullmatch(r"[0-9a-f]{64}", provenance_digest) is None
            or protocol[f"{prefix}_scene_ply_render_state_sha256"] != state_digest
            or protocol[f"{prefix}_scene_ply_checkpoint_count_match"] is not True
            or protocol[f"{prefix}_scene_ply_checkpoint_render_state_match"] is not True
        ):
            raise ValueError(
                f"Identity metadata {prefix.upper()} checkpoint/Scene PLY binding is invalid"
            )
    if backend == "dinov2":
        protocol.update(dinov2_protocol_from_metadata(metadata))
    elif backend == "rgb":
        if provenance.get("backend") != "rgb" or provenance.get("weight_loading") != "none":
            raise ValueError("RGB backend provenance is inconsistent")
    else:
        raise ValueError(f"Unsupported feature backend in metadata: {backend}")
    return protocol
