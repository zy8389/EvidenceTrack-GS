#!/usr/bin/env python3
"""Projection/GS/Difix/real Track evidence diagnostic with counterfactuals."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from diffusion_guidance.camera_utils import camera_fingerprint_from_payload
from diffusion_guidance.checkpoint_state import (
    RENDER_STATE_SCHEMA,
    load_checkpoint_summary,
)
from diffusion_guidance.control_identity import (
    normalized_camera_names,
    source_camera_set_sha256,
)
from diffusion_guidance.difix_provenance import (
    HELDOUT_DIAGNOSTIC_MANIFEST_SCHEMA,
    difix_run_metadata_path,
    load_difix_run_metadata,
    validate_difix_target,
)
from diffusion_guidance.evidence_protocol import (
    PAIRED_RECONSTRUCTION_FIELDS,
    canonical_source_image_inventory,
    source_image_inventory_sha256,
    validate_hard_negative_radius,
    validate_primary_radius,
)
from diffusion_guidance.experiment_registry import require_scene_role
from diffusion_guidance.result_binding import (
    audit_supports_contrast,
    read_pair_audit,
    require_audit_binding,
    require_final_checkpoint_binding,
)
from diffusion_guidance.evidence_features import (
    FeatureExtractor,
    sample_features,
)
from diffusion_guidance.evidence_matching import (
    error_metrics,
    local_feature_match,
    project_anchor_numpy,
    select_hard_negatives,
    shifted_centers,
    valid_shift_recovery_mask,
)
from geometric_constraints.strict_track_store import (
    StrictTrackStore,
    normalize_image_name,
)


VARIANTS = {
    "GS Render + feature matching": "gs_render",
    "Difix Output + feature matching": "difix_output",
    "Real Target + feature matching [reference diagnostic]": "real_target",
}
PAIRED_MANIFEST_SCHEMA = "paired_identity_manifest_v4"
PAIRED_MANIFEST_METADATA_SCHEMA = "paired_identity_manifest_metadata_v4"


def resolve_path(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def resolve_image(images_dir: Path, image_name: str) -> Path:
    direct = images_dir / image_name
    if direct.exists():
        return direct
    matches = list(images_dir.glob(normalize_image_name(image_name) + ".*"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Cannot resolve {image_name!r} in {images_dir}")
    return matches[0]


def sha256_file(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_evidence_difix(
    record: dict, cache_record: dict, run_metadata: dict
) -> dict:
    path_pairs = (
        ("input", "gs_render"),
        ("reference_image", "reference_image"),
        ("target", "difix_output"),
    )
    for cache_field, evidence_field in path_pairs:
        if Path(cache_record[cache_field]) != Path(record[evidence_field]):
            raise ValueError(
                f"Difix/evidence manifest path mismatch for {evidence_field}: "
                f"{record['image_name']}"
            )
    if cache_record["key"] != record["camera_fingerprint"]:
        raise ValueError(f"Difix/evidence camera mismatch for {record['image_name']}")
    context_fields = (
        "manifest_schema",
        "camera",
        "camera_source",
        "dataset",
        "scene",
        "seed",
        "experiment_role",
        "scene_source_path",
        "paired_identity_arm",
        "controlled_pair_id",
        "pair_audit",
        "pair_audit_sha256",
        "track_h5",
        "track_h5_sha256",
        "source_camera_names",
        "source_camera_set_sha256",
        "checkpoint",
        "checkpoint_sha256",
        "checkpoint_iteration",
        "checkpoint_state_format",
        "checkpoint_gaussian_count",
        "checkpoint_render_state_schema",
        "checkpoint_render_state_sha256",
        "checkpoint_controlled_provenance_sha256",
        "scene_ply_gaussian_count",
        "scene_ply_checkpoint_count_match",
        "scene_ply_render_state_schema",
        "scene_ply_render_state_sha256",
        "scene_ply_checkpoint_render_state_match",
    )
    for field in context_fields:
        evidence_value = record.get(field)
        cache_value = cache_record.get(field)
        if field in {"checkpoint", "pair_audit", "track_h5", "scene_source_path"}:
            evidence_value = str(Path(str(evidence_value)).expanduser().resolve())
            cache_value = str(Path(str(cache_value)).expanduser().resolve())
        if json.dumps(
            evidence_value, sort_keys=True, separators=(",", ":")
        ) != json.dumps(cache_value, sort_keys=True, separators=(",", ":")):
            raise ValueError(
                f"Difix/evidence scientific context mismatch for {field}: "
                f"{record['image_name']}"
            )
    hash_pairs = (
        ("input_sha256", "gs_render_sha256"),
        ("reference_image_sha256", "reference_image_sha256"),
    )
    for cache_field, evidence_field in hash_pairs:
        if cache_record.get(cache_field) != record.get(evidence_field):
            raise ValueError(
                f"Difix/evidence frozen-asset hash mismatch for {evidence_field}: "
                f"{record['image_name']}"
            )
    camera = record["camera"]
    return validate_difix_target(
        target_path=Path(record["difix_output"]),
        input_path=Path(record["gs_render"]),
        reference_path=Path(record["reference_image"]),
        camera_fingerprint=record["camera_fingerprint"],
        camera_resolution=(int(camera["width"]), int(camera["height"])),
        run_metadata=run_metadata,
        require_reproducibility=True,
    )


def _load_difix_manifest(path: Path) -> dict[str, dict]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing Difix cache manifest: {path}")
    records: dict[str, dict] = {}
    required = ("key", "camera", "input", "reference_image", "target")
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            missing = [key for key in required if key not in record]
            if missing:
                raise ValueError(f"Incomplete Difix manifest at {path}:{line_number}: {missing}")
            key = str(record["key"])
            if key in records:
                raise ValueError(f"Duplicate Difix camera fingerprint: {key}")
            expected = camera_fingerprint_from_payload(record["camera"], prefix="heldout")
            if key != expected:
                raise ValueError(f"Difix camera fingerprint mismatch at {path}:{line_number}")
            for field in ("input", "reference_image", "target"):
                record[field] = str(resolve_path(path.parent, record[field]))
            for field in ("checkpoint", "pair_audit", "track_h5", "scene_source_path"):
                if field in record:
                    record[field] = str(resolve_path(path.parent, record[field]))
            records[key] = record
    if not records:
        raise RuntimeError(f"Difix manifest is empty: {path}")
    return records


def load_target_manifest(path: Path) -> List[dict]:
    path = path.expanduser().resolve()
    records = []
    required = (
        "manifest_schema",
        "image_name",
        "camera_fingerprint",
        "camera",
        "camera_source",
        "gs_render",
        "difix_output",
        "real_target",
        "reference_image",
        "gs_render_sha256",
        "reference_image_sha256",
        "real_target_sha256",
        "dataset",
        "scene",
        "seed",
        "experiment_role",
        "scene_source_path",
        "paired_identity_arm",
        "controlled_pair_id",
        "pair_audit",
        "pair_audit_sha256",
        "track_h5",
        "track_h5_sha256",
        "source_camera_names",
        "source_camera_set_sha256",
        "checkpoint",
        "checkpoint_sha256",
        "checkpoint_iteration",
        "checkpoint_state_format",
        "checkpoint_gaussian_count",
        "checkpoint_render_state_schema",
        "checkpoint_render_state_sha256",
        "checkpoint_controlled_provenance_sha256",
        "scene_ply_gaussian_count",
        "scene_ply_checkpoint_count_match",
        "scene_ply_render_state_schema",
        "scene_ply_render_state_sha256",
        "scene_ply_checkpoint_render_state_match",
    )
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            missing = [key for key in required if key not in record]
            if missing:
                raise ValueError(f"Missing evidence provenance fields at {path}:{line_number}: {missing}")
            if record["manifest_schema"] != HELDOUT_DIAGNOSTIC_MANIFEST_SCHEMA:
                raise ValueError(f"Evidence manifest uses the wrong schema at {path}:{line_number}")
            if record["camera_source"] != "heldout_evaluation_camera":
                raise ValueError(f"Evidence manifest camera is not held-out at {path}:{line_number}")
            role = require_scene_role(
                record["dataset"], record["scene"], record["experiment_role"]
            )
            if str(record["paired_identity_arm"]).upper() not in {"A1", "B"}:
                raise ValueError(f"Evidence manifest arm is invalid at {path}:{line_number}")
            pair_audit_path = resolve_path(path.parent, record["pair_audit"])
            if record["pair_audit_sha256"] != sha256_file(pair_audit_path):
                raise ValueError(f"Evidence pair-audit hash mismatch at {path}:{line_number}")
            _, pair_audit = read_pair_audit(pair_audit_path)
            require_audit_binding(
                pair_audit,
                dataset=str(record["dataset"]),
                scene=str(record["scene"]),
                seed=int(record["seed"]),
                method=str(record["paired_identity_arm"]).upper(),
                pair_id=str(record["controlled_pair_id"]),
                role=role,
            )
            scene_source_path = resolve_path(path.parent, record["scene_source_path"])
            if (
                not scene_source_path.is_dir()
                or scene_source_path.name != str(record["scene"])
                or scene_source_path
                != Path(pair_audit["scene_source_path"]).expanduser().resolve()
            ):
                raise ValueError(
                    f"Evidence scene-source binding mismatch at {path}:{line_number}"
                )
            for field in ("gs_render", "difix_output", "real_target", "reference_image"):
                record[field] = str(resolve_path(path.parent, record[field]))
            record["checkpoint"] = str(resolve_path(path.parent, record["checkpoint"]))
            record["pair_audit"] = str(pair_audit_path)
            record["track_h5"] = str(resolve_path(path.parent, record["track_h5"]))
            record["scene_source_path"] = str(scene_source_path)
            expected = camera_fingerprint_from_payload(record["camera"], prefix="heldout")
            if record["camera_fingerprint"] != expected:
                raise ValueError(f"Evidence camera fingerprint mismatch at {path}:{line_number}")
            if record["gs_render_sha256"] != sha256_file(Path(record["gs_render"])):
                raise ValueError(f"Evidence GS input hash mismatch at {path}:{line_number}")
            if record["reference_image_sha256"] != sha256_file(Path(record["reference_image"])):
                raise ValueError(f"Evidence reference hash mismatch at {path}:{line_number}")
            if record["real_target_sha256"] != sha256_file(Path(record["real_target"])):
                raise ValueError(f"Evidence real target hash mismatch at {path}:{line_number}")
            if record["checkpoint_sha256"] != sha256_file(Path(record["checkpoint"])):
                raise ValueError(f"Evidence checkpoint hash mismatch at {path}:{line_number}")
            checkpoint_summary = load_checkpoint_summary(
                Path(record["checkpoint"]),
                expected_iteration=12000,
                require_cuda_rng=True,
                require_controlled_provenance=True,
            )
            require_final_checkpoint_binding(
                checkpoint_summary,
                pair_audit,
                method=str(record["paired_identity_arm"]).upper(),
                checkpoint_path=record["checkpoint"],
            )
            if (
                int(record["checkpoint_iteration"]) != 12000
                or record["checkpoint_state_format"] != checkpoint_summary["format"]
                or int(record["checkpoint_gaussian_count"])
                != checkpoint_summary["gaussian_count"]
                or record["checkpoint_render_state_schema"]
                != checkpoint_summary["render_state_schema"]
                or record["checkpoint_render_state_sha256"]
                != checkpoint_summary["render_state_sha256"]
                or record["checkpoint_controlled_provenance_sha256"]
                != checkpoint_summary["controlled_provenance_sha256"]
                or int(record["scene_ply_gaussian_count"])
                != checkpoint_summary["gaussian_count"]
                or record["scene_ply_checkpoint_count_match"] is not True
                or record["scene_ply_render_state_schema"]
                != checkpoint_summary["render_state_schema"]
                or record["scene_ply_render_state_sha256"]
                != checkpoint_summary["render_state_sha256"]
                or record["scene_ply_checkpoint_render_state_match"] is not True
            ):
                raise ValueError(
                    f"Evidence checkpoint render-state provenance mismatch at {path}:{line_number}"
                )
            records.append(record)
    if not records:
        raise RuntimeError("Target manifest is empty")
    context_fields = (
        "dataset",
        "scene",
        "seed",
        "experiment_role",
        "scene_source_path",
        "paired_identity_arm",
        "controlled_pair_id",
        "pair_audit",
        "pair_audit_sha256",
        "track_h5",
        "track_h5_sha256",
        "source_camera_names",
        "source_camera_set_sha256",
        "checkpoint",
        "checkpoint_sha256",
        "checkpoint_iteration",
        "checkpoint_state_format",
        "checkpoint_gaussian_count",
        "checkpoint_render_state_schema",
        "checkpoint_render_state_sha256",
        "checkpoint_controlled_provenance_sha256",
        "scene_ply_gaussian_count",
        "scene_ply_checkpoint_count_match",
        "scene_ply_render_state_schema",
        "scene_ply_render_state_sha256",
        "scene_ply_checkpoint_render_state_match",
    )
    first = records[0]
    for record in records[1:]:
        for field in context_fields:
            if record.get(field) != first.get(field):
                raise ValueError(
                    f"Evidence manifest changes immutable context field {field}"
                )
    if not isinstance(first["source_camera_names"], list) or source_camera_set_sha256(
        first["source_camera_names"]
    ) != first["source_camera_set_sha256"]:
        raise ValueError("Evidence source-camera set hash is inconsistent")
    if first["track_h5_sha256"] != sha256_file(
        resolve_path(path.parent, first["track_h5"])
    ):
        raise ValueError("Evidence Track H5 hash is stale")
    cache_manifest_path = path.parent / "difix_manifest.jsonl"
    cache_records = _load_difix_manifest(cache_manifest_path)
    evidence_keys = {record["camera_fingerprint"] for record in records}
    if evidence_keys != set(cache_records):
        raise ValueError("Evidence and Difix manifests contain different camera sets")
    if len(evidence_keys) != len(records):
        raise ValueError("Evidence manifest contains duplicate camera fingerprints")
    run_metadata = load_difix_run_metadata(
        cache_manifest_path,
        expected_record_count=len(records),
        require_reproducibility=True,
    )
    for record in records:
        sidecar = _validate_evidence_difix(
            record, cache_records[record["camera_fingerprint"]], run_metadata
        )
        record["difix_output_sha256"] = sidecar["output_sha256"]
    return records


def validate_paired_manifest_metadata(path: Path, arm: str) -> dict:
    path = path.expanduser().resolve()
    metadata_path = path.with_suffix(path.suffix + ".metadata.json")
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Paired identity manifest metadata is missing: {metadata_path}"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    required = (
        "schema",
        "passed",
        "paired_manifest",
        "paired_manifest_sha256",
        "record_count",
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
    if not isinstance(metadata, dict):
        raise ValueError("Paired identity manifest metadata must be an object")
    missing = [field for field in required if metadata.get(field) in (None, "")]
    if missing:
        raise ValueError(f"Paired identity metadata is incomplete: {missing}")
    if (
        metadata["schema"] != PAIRED_MANIFEST_METADATA_SCHEMA
        or metadata["passed"] is not True
        or Path(metadata["paired_manifest"]).expanduser().resolve() != path
        or metadata["paired_manifest_sha256"] != sha256_file(path)
        or int(metadata["record_count"]) <= 0
    ):
        raise ValueError("Paired identity manifest metadata is stale or invalid")
    role = require_scene_role(
        metadata["dataset"], metadata["scene"], metadata["experiment_role"]
    )
    pair_audit_path = resolve_path(path.parent, metadata["pair_audit"])
    if metadata["pair_audit_sha256"] != sha256_file(pair_audit_path):
        raise ValueError("Paired identity pair-audit hash is stale")
    _, pair_audit = read_pair_audit(pair_audit_path)
    require_audit_binding(
        pair_audit,
        dataset=metadata["dataset"],
        scene=metadata["scene"],
        seed=int(metadata["seed"]),
        method=arm.upper(),
        pair_id=metadata["controlled_pair_id"],
        role=role,
    )
    if not audit_supports_contrast(pair_audit, "A1", "B"):
        raise ValueError("Paired identity audit does not authorize A1/B")
    scene_source_path = resolve_path(path.parent, metadata["scene_source_path"])
    if (
        not scene_source_path.is_dir()
        or scene_source_path.name != str(metadata["scene"])
        or scene_source_path
        != Path(pair_audit["scene_source_path"]).expanduser().resolve()
    ):
        raise ValueError("Paired identity scene-source binding is invalid")
    track_path = resolve_path(path.parent, metadata["track_h5"])
    if metadata["track_h5_sha256"] != sha256_file(track_path):
        raise ValueError("Paired identity Track H5 hash is stale")
    names = metadata["source_camera_names"]
    if not isinstance(names, list) or source_camera_set_sha256(names) != metadata[
        "source_camera_set_sha256"
    ]:
        raise ValueError("Paired identity source-camera binding is invalid")
    source_images_dir = resolve_path(path.parent, metadata["source_images_dir"])
    if source_images_dir != (scene_source_path / "images").resolve() or not source_images_dir.is_dir():
        raise ValueError("Paired identity source image root is invalid")
    inventory = canonical_source_image_inventory(metadata["source_image_inventory"])
    if [item["camera_name"] for item in inventory] != normalized_camera_names(names):
        raise ValueError("Paired identity source-image inventory differs from source cameras")
    for item in inventory:
        image_path = Path(item["path"]).expanduser().resolve()
        try:
            image_path.relative_to(source_images_dir)
        except ValueError as exc:
            raise ValueError("Paired identity source image is outside the scene image root") from exc
        if str(item["path"]) != str(image_path):
            raise ValueError("Paired identity source image path is not canonical")
        if not image_path.is_file() or sha256_file(image_path) != item["sha256"]:
            raise ValueError(f"Paired identity source image hash is stale: {image_path}")
    if source_image_inventory_sha256(inventory) != metadata["source_image_inventory_sha256"]:
        raise ValueError("Paired identity source-image inventory hash is stale")
    metadata["source_images_dir"] = str(source_images_dir)
    metadata["source_image_inventory"] = inventory
    metadata["source_image_inventory_sha256"] = source_image_inventory_sha256(inventory)
    for prefix in ("a1", "b"):
        for kind in ("evidence_manifest", "difix_manifest", "difix_run_metadata"):
            asset = resolve_path(path.parent, metadata[f"{prefix}_{kind}"])
            if metadata[f"{prefix}_{kind}_sha256"] != sha256_file(asset):
                raise ValueError(
                    f"Paired identity {prefix} {kind} is missing or changed"
                )
            metadata[f"{prefix}_{kind}"] = str(asset)
        difix_manifest = resolve_path(
            path.parent, metadata[f"{prefix}_difix_manifest"]
        )
        run_metadata = load_difix_run_metadata(
            difix_manifest,
            expected_record_count=int(metadata["record_count"]),
            require_reproducibility=True,
        )
        declared_run_metadata = resolve_path(
            path.parent, metadata[f"{prefix}_difix_run_metadata"]
        )
        if declared_run_metadata != difix_run_metadata_path(difix_manifest).resolve():
            raise ValueError(
                f"Paired identity {prefix} Difix metadata path is not bound to its manifest"
            )
        if run_metadata["cache_run_fingerprint"] != metadata[
            f"{prefix}_difix_cache_run_fingerprint"
        ]:
            raise ValueError(
                f"Paired identity {prefix} Difix cache fingerprint is stale"
            )
        actual_protocol = {
            field: run_metadata[field]
            for field in (
                "model_id",
                "model_revision",
                "difix_code_commit",
                "coordinate_policy",
                "dtype",
                "timesteps",
                "guidance_scale",
                "prompt",
            )
        }
        if metadata[f"{prefix}_difix_protocol"] != actual_protocol:
            raise ValueError(
                f"Paired identity {prefix} Difix protocol is stale or fabricated"
            )
    if metadata["a1_difix_protocol"] != metadata["b_difix_protocol"]:
        raise ValueError("Paired identity arms use different Difix scientific configuration")
    for prefix in ("a1", "b"):
        checkpoint = resolve_path(path.parent, metadata[f"{prefix}_checkpoint"])
        if metadata[f"{prefix}_checkpoint_sha256"] != sha256_file(checkpoint):
            raise ValueError(f"Paired identity {prefix} checkpoint hash is stale")
        metadata[f"{prefix}_checkpoint"] = str(checkpoint)
        summary = load_checkpoint_summary(
            checkpoint,
            expected_iteration=12000,
            require_cuda_rng=True,
            require_controlled_provenance=True,
        )
        method = "A1" if prefix == "a1" else "B"
        require_final_checkpoint_binding(
            summary,
            pair_audit,
            method=method,
            checkpoint_path=checkpoint,
        )
        if (
            int(metadata[f"{prefix}_checkpoint_iteration"]) != 12000
            or metadata[f"{prefix}_checkpoint_state_format"] != summary["format"]
            or int(metadata[f"{prefix}_checkpoint_gaussian_count"])
            != summary["gaussian_count"]
            or metadata[f"{prefix}_checkpoint_render_state_schema"]
            != summary["render_state_schema"]
            or metadata[f"{prefix}_checkpoint_render_state_sha256"]
            != summary["render_state_sha256"]
            or metadata[f"{prefix}_checkpoint_controlled_provenance_sha256"]
            != summary["controlled_provenance_sha256"]
            or int(metadata[f"{prefix}_scene_ply_gaussian_count"])
            != summary["gaussian_count"]
            or metadata[f"{prefix}_scene_ply_checkpoint_count_match"] is not True
            or metadata[f"{prefix}_scene_ply_render_state_schema"]
            != summary["render_state_schema"]
            or metadata[f"{prefix}_scene_ply_checkpoint_render_state_match"] is not True
            or metadata[f"{prefix}_scene_ply_render_state_sha256"]
            != summary["render_state_sha256"]
        ):
            raise ValueError(f"Paired identity {prefix} checkpoint binding is invalid")
    metadata["paired_manifest"] = str(path)
    metadata["pair_audit"] = str(pair_audit_path)
    metadata["track_h5"] = str(track_path)
    metadata["scene_source_path"] = str(scene_source_path)
    metadata["metadata_path"] = str(metadata_path.resolve())
    metadata["metadata_sha256"] = sha256_file(metadata_path)
    metadata["experiment_role"] = role
    return metadata


def load_paired_target_manifest(
    path: Path, arm: str
) -> tuple[List[dict], dict]:
    """Materialize one A1/B arm from an immutable paired-support manifest.

    Both arms receive the same camera/reference bytes and only their own GS and
    Difix outputs.  Hashes are rechecked before the usual evidence/cache
    provenance validator runs, so a paired-manifest path cannot silently point
    at a later output file.
    """
    path = path.expanduser().resolve()
    arm = str(arm).lower()
    if arm not in {"a1", "b"}:
        raise ValueError("--paired-arm must be A1 or B")
    binding = validate_paired_manifest_metadata(path, arm)
    evidence_by_arm = {}
    for evidence_arm in ("a1", "b"):
        evidence_path = resolve_path(
            path.parent, binding[f"{evidence_arm}_evidence_manifest"]
        )
        evidence_by_arm[evidence_arm] = {
            row["image_name"]: row for row in load_target_manifest(evidence_path)
        }
    records: list[dict] = []
    paired_image_names: set[str] = set()
    required = (
        "image_name",
        "manifest_schema",
        "camera",
        "camera_fingerprint",
        "reference_image",
        "real_target",
        "reference_image_sha256",
        "real_target_sha256",
        "source_images_dir",
        "source_image_inventory",
        "source_image_inventory_sha256",
        f"{arm}_checkpoint",
        f"{arm}_checkpoint_sha256",
        f"{arm}_checkpoint_iteration",
        f"{arm}_checkpoint_state_format",
        f"{arm}_checkpoint_gaussian_count",
        f"{arm}_checkpoint_render_state_schema",
        f"{arm}_checkpoint_render_state_sha256",
        f"{arm}_checkpoint_controlled_provenance_sha256",
        f"{arm}_scene_ply_gaussian_count",
        f"{arm}_scene_ply_checkpoint_count_match",
        f"{arm}_scene_ply_render_state_schema",
        f"{arm}_scene_ply_render_state_sha256",
        f"{arm}_scene_ply_checkpoint_render_state_match",
        f"{arm}_render",
        f"{arm}_render_sha256",
        f"{arm}_difix_output",
        f"{arm}_difix_output_sha256",
        "a1_difix_protocol",
        "b_difix_protocol",
    )
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            missing = [field for field in required if field not in raw]
            if missing:
                raise ValueError(
                    f"Missing paired identity provenance at {path}:{line_number}: {missing}"
                )
            if raw["manifest_schema"] != PAIRED_MANIFEST_SCHEMA:
                raise ValueError(
                    f"Paired identity row uses the wrong schema at {path}:{line_number}"
                )
            image_name = str(raw["image_name"])
            if image_name in paired_image_names:
                raise ValueError(
                    f"Paired identity manifest contains duplicate image: {image_name}"
                )
            paired_image_names.add(image_name)
            for evidence_arm in ("a1", "b"):
                evidence = evidence_by_arm[evidence_arm].get(image_name)
                if evidence is None:
                    raise ValueError(
                        f"Paired identity row is absent from {evidence_arm} evidence: "
                        f"{image_name}"
                    )
                arm_fields = {
                    "camera": evidence["camera"],
                    "camera_fingerprint": evidence["camera_fingerprint"],
                    f"{evidence_arm}_render": evidence["gs_render"],
                    f"{evidence_arm}_render_sha256": evidence["gs_render_sha256"],
                    f"{evidence_arm}_difix_output": evidence["difix_output"],
                    f"{evidence_arm}_difix_output_sha256": evidence[
                        "difix_output_sha256"
                    ],
                    f"{evidence_arm}_checkpoint": evidence["checkpoint"],
                    f"{evidence_arm}_checkpoint_sha256": evidence[
                        "checkpoint_sha256"
                    ],
                    f"{evidence_arm}_checkpoint_iteration": evidence[
                        "checkpoint_iteration"
                    ],
                    f"{evidence_arm}_checkpoint_state_format": evidence[
                        "checkpoint_state_format"
                    ],
                    f"{evidence_arm}_checkpoint_gaussian_count": evidence[
                        "checkpoint_gaussian_count"
                    ],
                    f"{evidence_arm}_checkpoint_render_state_schema": evidence[
                        "checkpoint_render_state_schema"
                    ],
                    f"{evidence_arm}_checkpoint_render_state_sha256": evidence[
                        "checkpoint_render_state_sha256"
                    ],
                    f"{evidence_arm}_checkpoint_controlled_provenance_sha256": evidence[
                        "checkpoint_controlled_provenance_sha256"
                    ],
                    f"{evidence_arm}_scene_ply_gaussian_count": evidence[
                        "scene_ply_gaussian_count"
                    ],
                    f"{evidence_arm}_scene_ply_checkpoint_count_match": evidence[
                        "scene_ply_checkpoint_count_match"
                    ],
                    f"{evidence_arm}_scene_ply_render_state_schema": evidence[
                        "scene_ply_render_state_schema"
                    ],
                    f"{evidence_arm}_scene_ply_render_state_sha256": evidence[
                        "scene_ply_render_state_sha256"
                    ],
                    f"{evidence_arm}_scene_ply_checkpoint_render_state_match": evidence[
                        "scene_ply_checkpoint_render_state_match"
                    ],
                }
                if evidence_arm == "a1":
                    arm_fields.update(
                        {
                            "reference_image": evidence["reference_image"],
                            "reference_image_sha256": evidence[
                                "reference_image_sha256"
                            ],
                            "real_target": evidence["real_target"],
                            "real_target_sha256": evidence["real_target_sha256"],
                        }
                    )
                for field, expected in arm_fields.items():
                    observed = raw.get(field)
                    if field.endswith(("_render", "_difix_output", "_checkpoint")):
                        if observed is not None:
                            observed = str(resolve_path(path.parent, observed))
                    if field in {"reference_image", "real_target"} and observed is not None:
                        observed = str(resolve_path(path.parent, observed))
                    if json.dumps(observed, sort_keys=True, separators=(",", ":")) != json.dumps(
                        expected, sort_keys=True, separators=(",", ":")
                    ):
                        raise ValueError(
                            f"Paired identity row/evidence mismatch for {field}: "
                            f"{image_name}"
                        )
            for field in (
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
                "a1_difix_protocol",
                "b_difix_protocol",
            ):
                if raw.get(field) != binding.get(field):
                    raise ValueError(
                        f"Paired identity row/metadata mismatch for {field}"
                    )
            record = {
                **evidence_by_arm[arm][image_name],
                "image_name": image_name,
                "camera": raw["camera"],
                "camera_fingerprint": raw["camera_fingerprint"],
                "gs_render": str(resolve_path(path.parent, raw[f"{arm}_render"])),
                "difix_output": str(
                    resolve_path(path.parent, raw[f"{arm}_difix_output"])
                ),
                "difix_output_sha256": raw[f"{arm}_difix_output_sha256"],
                "reference_image": str(
                    resolve_path(path.parent, raw["reference_image"])
                ),
                "real_target": str(resolve_path(path.parent, raw["real_target"])),
                "gs_render_sha256": raw[f"{arm}_render_sha256"],
                "reference_image_sha256": raw["reference_image_sha256"],
                "real_target_sha256": raw["real_target_sha256"],
                "source_images_dir": str(
                    resolve_path(path.parent, raw["source_images_dir"])
                ),
                "source_image_inventory": raw["source_image_inventory"],
                "source_image_inventory_sha256": raw[
                    "source_image_inventory_sha256"
                ],
                "checkpoint": str(resolve_path(path.parent, raw[f"{arm}_checkpoint"])),
                "checkpoint_sha256": raw[f"{arm}_checkpoint_sha256"],
                "checkpoint_iteration": raw[f"{arm}_checkpoint_iteration"],
                "checkpoint_state_format": raw[f"{arm}_checkpoint_state_format"],
                "checkpoint_gaussian_count": raw[f"{arm}_checkpoint_gaussian_count"],
                "checkpoint_render_state_schema": raw[
                    f"{arm}_checkpoint_render_state_schema"
                ],
                "checkpoint_render_state_sha256": raw[
                    f"{arm}_checkpoint_render_state_sha256"
                ],
                "checkpoint_controlled_provenance_sha256": raw[
                    f"{arm}_checkpoint_controlled_provenance_sha256"
                ],
                "scene_ply_gaussian_count": raw[f"{arm}_scene_ply_gaussian_count"],
                "scene_ply_checkpoint_count_match": raw[
                    f"{arm}_scene_ply_checkpoint_count_match"
                ],
                "scene_ply_render_state_schema": raw[
                    f"{arm}_scene_ply_render_state_schema"
                ],
                "scene_ply_render_state_sha256": raw[
                    f"{arm}_scene_ply_render_state_sha256"
                ],
                "scene_ply_checkpoint_render_state_match": raw[
                    f"{arm}_scene_ply_checkpoint_render_state_match"
                ],
            }
            if raw["a1_difix_protocol"] != binding["a1_difix_protocol"] or raw[
                "b_difix_protocol"
            ] != binding["b_difix_protocol"]:
                raise ValueError(
                    f"Paired identity row/Difix protocol mismatch at {path}:{line_number}"
                )
            expected = camera_fingerprint_from_payload(record["camera"], prefix="heldout")
            if record["camera_fingerprint"] != expected:
                raise ValueError(
                    f"Paired identity camera fingerprint mismatch at {path}:{line_number}"
                )
            for path_field, hash_field in (
                ("gs_render", "gs_render_sha256"),
                ("reference_image", "reference_image_sha256"),
                ("real_target", "real_target_sha256"),
                ("checkpoint", "checkpoint_sha256"),
                ("difix_output", "difix_output_sha256"),
            ):
                asset = Path(record[path_field])
                if not asset.is_file():
                    raise FileNotFoundError(
                        f"Paired identity asset is missing at {path}:{line_number}: {asset}"
                    )
                if record[hash_field] != sha256_file(asset):
                    raise ValueError(
                        f"Paired identity asset hash mismatch at {path}:{line_number}: {path_field}"
                    )
            state_hash = str(record["checkpoint_render_state_sha256"])
            if (
                len(state_hash) != 64
                or any(character not in "0123456789abcdef" for character in state_hash.lower())
            ):
                raise ValueError(
                    f"Paired identity checkpoint render-state hash is invalid at {path}:{line_number}"
                )
            checkpoint_summary = load_checkpoint_summary(
                Path(record["checkpoint"]),
                expected_iteration=12000,
                require_cuda_rng=True,
                require_controlled_provenance=True,
            )
            if (
                record["checkpoint_state_format"] != checkpoint_summary["format"]
                or int(record["checkpoint_gaussian_count"])
                != checkpoint_summary["gaussian_count"]
                or record["checkpoint_render_state_schema"]
                != checkpoint_summary["render_state_schema"]
                or state_hash != checkpoint_summary["render_state_sha256"]
                or record["checkpoint_controlled_provenance_sha256"]
                != checkpoint_summary["controlled_provenance_sha256"]
                or int(record["scene_ply_gaussian_count"])
                != checkpoint_summary["gaussian_count"]
                or record["scene_ply_checkpoint_count_match"] is not True
                or record["scene_ply_render_state_schema"]
                != checkpoint_summary["render_state_schema"]
                or record["scene_ply_checkpoint_render_state_match"] is not True
                or record["scene_ply_render_state_sha256"] != state_hash
            ):
                raise ValueError(
                    f"Paired identity checkpoint/Scene PLY binding mismatch at {path}:{line_number}"
                )
            records.append(record)
    if not records:
        raise RuntimeError("Paired identity target manifest is empty")
    if len({record["camera_fingerprint"] for record in records}) != len(records):
        raise ValueError("Paired identity target manifest has duplicate camera fingerprints")
    for evidence_arm, evidence_records in evidence_by_arm.items():
        if paired_image_names != set(evidence_records):
            raise ValueError(
                f"Paired identity row set differs from {evidence_arm} evidence manifest"
            )

    cache_manifest_path = resolve_path(
        path.parent, binding[f"{arm}_difix_manifest"]
    )
    cache_root = cache_manifest_path.parent
    for record in records:
        try:
            Path(record["difix_output"]).resolve().relative_to(cache_root)
        except ValueError as exc:
            raise ValueError(
                "Paired identity Difix output is outside the declared cache root: "
                f"{record['image_name']}"
            ) from exc
    cache_records = _load_difix_manifest(cache_manifest_path)
    evidence_keys = {record["camera_fingerprint"] for record in records}
    if evidence_keys != set(cache_records):
        raise ValueError("Paired identity arm and Difix manifest have different camera sets")
    run_metadata = load_difix_run_metadata(
        cache_manifest_path,
        expected_record_count=len(records),
        require_reproducibility=True,
    )
    for record in records:
        _validate_evidence_difix(
            record, cache_records[record["camera_fingerprint"]], run_metadata
        )
    if len(records) != int(binding["record_count"]):
        raise ValueError("Paired identity row count differs from its metadata")
    return records, binding


def build_source_queries(
    store: StrictTrackStore,
    extractor: FeatureExtractor,
    images_dir: Path,
) -> tuple[Dict[int, torch.Tensor], List[dict]]:
    feature_cache = {}
    metadata = []
    queries: Dict[int, torch.Tensor] = {}
    for track_index in range(len(store)):
        observations = store.observations_for(track_index)
        samples = []
        for xy, image_name, use_source in zip(
            observations["xy"],
            observations["image_name"],
            observations["use_for_source_features"],
        ):
            if not bool(use_source):
                continue
            path = resolve_image(images_dir, str(image_name))
            key = str(path)
            if key not in feature_cache:
                feature_cache[key] = extractor.extract(path)
                metadata.append({"path": key, **feature_cache[key].metadata()})
            feature_map = feature_cache[key]
            coordinate = torch.as_tensor(
                np.asarray(xy)[None],
                device=feature_map.features.device,
                dtype=feature_map.features.dtype,
            )
            samples.append(sample_features(feature_map, coordinate)[0])
        if samples:
            queries[track_index] = F.normalize(torch.stack(samples).mean(dim=0), dim=0)
    return queries, metadata


def build_source_image_inventory(
    images_dir: Path, source_camera_names: List[str]
) -> List[dict]:
    inventory = []
    for camera_name in sorted(source_camera_names):
        image_path = resolve_image(images_dir, camera_name).resolve()
        inventory.append(
            {
                "camera_name": normalize_image_name(camera_name),
                "path": str(image_path),
                "sha256": sha256_file(image_path),
            }
        )
    # The digest helper also rejects normalized-name collisions.
    source_image_inventory_sha256(inventory)
    return inventory


def target_track_data(
    store: StrictTrackStore,
    image_name: str,
    source_queries: Dict[int, torch.Tensor],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, torch.Tensor]:
    key = normalize_image_name(image_name)
    track_indices, ground_truth, projections, queries = [], [], [], []
    calibration = store.camera_calibration(image_name)
    for track_index in range(len(store)):
        if track_index not in source_queries:
            continue
        observations = store.observations_for(track_index)
        matches = [
            offset
            for offset, name in enumerate(observations["image_name"])
            if normalize_image_name(name) == key
            and not bool(observations["use_for_anchor"][offset])
        ]
        if len(matches) != 1:
            continue
        projection = project_anchor_numpy(store.xyz[track_index], calibration)
        if not np.isfinite(projection).all():
            continue
        track_indices.append(track_index)
        ground_truth.append(observations["xy"][matches[0]])
        projections.append(projection)
        queries.append(source_queries[track_index])
    if not track_indices:
        raise RuntimeError(f"No held-out GT Track observations for {image_name}")
    return (
        np.asarray(track_indices, dtype=np.int64),
        np.asarray(ground_truth, dtype=np.float64),
        np.asarray(projections, dtype=np.float64),
        torch.stack(queries),
    )


def write_csv(path: Path, rows: List[dict]) -> None:
    fields = sorted(set().union(*(row.keys() for row in rows))) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def prefix_metrics(prefix: str, metrics: dict) -> dict:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--track-h5", required=True, type=Path)
    parser.add_argument("--images-dir", required=True, type=Path)
    parser.add_argument("--target-manifest", required=True, type=Path)
    parser.add_argument(
        "--paired-arm",
        choices=["A1", "B", "a1", "b"],
        help="Read one arm from an immutable --target-manifest paired identity selector",
    )
    parser.add_argument("--feature-backend", choices=["dinov2", "rgb"], default="dinov2")
    parser.add_argument("--dinov2-model", default="dinov2_vits14")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--window-radii", nargs="+", type=float, default=[32.0])
    parser.add_argument("--primary-radius", type=float, default=32.0)
    parser.add_argument("--shift-pixels", nargs="+", type=float, default=[4.0, 8.0, 16.0])
    parser.add_argument("--hard-negative-radius", type=float, default=32.0)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    store = StrictTrackStore.load(args.track_h5)
    audit = store.leakage_audit()
    audit.emit(prefix="[Evidence Leakage Audit]")
    audit.require_pass()
    images_dir = args.images_dir.expanduser().resolve()
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Source image directory is missing: {images_dir}")
    target_manifest = args.target_manifest.expanduser().resolve()
    paired_binding = None
    if args.paired_arm:
        records, paired_binding = load_paired_target_manifest(
            target_manifest, args.paired_arm
        )
        current_track = args.track_h5.expanduser().resolve()
        if (
            current_track != resolve_path(target_manifest.parent, paired_binding["track_h5"])
            or sha256_file(current_track) != paired_binding["track_h5_sha256"]
            or source_camera_set_sha256(store.source_images)
            != paired_binding["source_camera_set_sha256"]
        ):
            raise ValueError(
                "Evaluator Track H5/source-camera set differs from the paired manifest"
            )
        expected_images_dir = (
            Path(paired_binding["scene_source_path"]).expanduser().resolve() / "images"
        ).resolve()
        if images_dir != expected_images_dir:
            raise ValueError(
                "Evaluator --images-dir differs from the pair-audited scene image root"
            )
    else:
        records = load_target_manifest(target_manifest)

    primary_radius = validate_primary_radius(
        args.primary_radius, args.window_radii
    )
    hard_negative_radius = validate_hard_negative_radius(
        primary_radius, args.hard_negative_radius
    )

    # Validate every file-backed scientific input before loading DINOv2 or
    # computing features.  A stale paired/cache/checkpoint binding must fail
    # before the expensive diagnostic starts.
    extractor = FeatureExtractor(
        args.feature_backend,
        device=args.device,
        dinov2_model=args.dinov2_model,
    )
    source_queries, feature_metadata = build_source_queries(
        store, extractor, images_dir
    )
    extractor_provenance = extractor.reproducibility_metadata()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: List[dict] = []
    per_track_rows: List[dict] = []
    aggregate_projection_pred, aggregate_projection_gt = [], []
    aggregate_predictions = {label: [] for label in VARIANTS}
    aggregate_ground_truth = {label: [] for label in VARIANTS}
    aggregate_strides = {label: [] for label in VARIANTS}

    for record in records:
        if normalize_image_name(record["image_name"]) in set(store.source_images):
            raise RuntimeError("Evidence target must be held-out, not a training source")
        sizes = []
        for field in VARIANTS.values():
            with Image.open(record[field]) as image:
                sizes.append(image.size)
        if len(set(sizes)) != 1:
            raise RuntimeError("GS/Difix/real target sizes differ; paired errors would be incomparable")
        track_indices, ground_truth, projections, queries = target_track_data(
            store, record["image_name"], source_queries
        )
        target_calibration = store.camera_calibration(record["image_name"])
        # Training-free null controls. CPU generation fixes RNG across devices.
        generator = torch.Generator(device="cpu").manual_seed(20260906)
        random_queries = F.normalize(torch.randn(queries.shape, generator=generator), dim=-1).to(queries)
        null_queries = {
            "global_shuffle": torch.roll(queries, shifts=1, dims=0),
            "random_query": random_queries,
            "uniform_query": torch.zeros_like(queries),
        }


        for variant_label, manifest_field in VARIANTS.items():
            feature_map = extractor.extract(record[manifest_field])
            coordinate_scale = np.asarray(
                [
                    feature_map.input_resolution[1]
                    / float(target_calibration["width"]),
                    feature_map.input_resolution[0]
                    / float(target_calibration["height"]),
                ],
                dtype=np.float64,
            )
            variant_ground_truth = (ground_truth + 0.5) * coordinate_scale - 0.5
            variant_projections = (projections + 0.5) * coordinate_scale - 0.5
            if variant_label == "Real Target + feature matching [reference diagnostic]":
                aggregate_projection_pred.append(variant_projections)
                aggregate_projection_gt.append(variant_ground_truth)
            feature_metadata.append(
                {
                    "path": record[manifest_field],
                    "target_variant": variant_label,
                    **extractor_provenance,
                    **feature_map.metadata(),
                }
            )
            for radius in args.window_radii:
                hard_selection = select_hard_negatives(
                    queries,
                    variant_projections,
                    nearby_radius=radius,
                )
                hard_indices = hard_selection.indices
                shuffled_queries = queries.index_select(
                    0,
                    torch.as_tensor(
                        hard_indices, device=queries.device, dtype=torch.long
                    ),
                )
                prediction, diagnostics = local_feature_match(
                    queries,
                    feature_map,
                    variant_projections,
                    window_radius=radius,
                    temperature=args.temperature,
                )
                correct_metrics = error_metrics(
                    prediction,
                    variant_ground_truth,
                    feature_stride=feature_map.effective_stride,
                )
                shuffled_prediction, shuffled_diagnostics = local_feature_match(
                    shuffled_queries,
                    feature_map,
                    variant_projections,
                    window_radius=radius,
                    temperature=args.temperature,
                )
                missing_hard_negative = (
                    hard_selection.selection_mode == "missing"
                )
                # ``indices`` deliberately points to self for a missing
                # negative so batch indexing remains safe.  Erase that result
                # before any metric is computed: a self query is not a wrong
                # identity and must never be credited as one.
                shuffled_prediction[missing_hard_negative] = np.nan
                local_hard_negative = hard_selection.within_radius
                global_fallback = (
                    hard_selection.selection_mode == "global_fallback"
                )
                local_hard_metrics = error_metrics(
                    shuffled_prediction[local_hard_negative],
                    variant_ground_truth[local_hard_negative],
                    feature_stride=feature_map.effective_stride,
                )
                global_fallback_metrics = error_metrics(
                    shuffled_prediction[global_fallback],
                    variant_ground_truth[global_fallback],
                    feature_stride=feature_map.effective_stride,
                )
                null_predictions = {}
                null_metrics = {}
                for control_name, control_queries in null_queries.items():
                    control_prediction, _ = local_feature_match(
                        control_queries, feature_map, variant_projections,
                        window_radius=radius, temperature=args.temperature)
                    null_predictions[control_name] = control_prediction
                    null_metrics.update(prefix_metrics(control_name, error_metrics(
                        control_prediction, variant_ground_truth,
                        feature_stride=feature_map.effective_stride)))
                projection_metrics = error_metrics(
                    variant_projections,
                    variant_ground_truth,
                    feature_stride=feature_map.effective_stride,
                )
                row = {
                    "image_name": record["image_name"],
                    "target_variant": variant_label,
                    "window_radius": radius,
                    "feature_backend": args.feature_backend,
                    "feature_resolution": "x".join(map(str, feature_map.feature_resolution)),
                    "feature_stride": feature_map.effective_stride,
                    **prefix_metrics("projection", projection_metrics),
                    **prefix_metrics("correct", correct_metrics),
                    **prefix_metrics("local_hard_shuffled", local_hard_metrics),
                    **prefix_metrics("global_fallback_hard_shuffled", global_fallback_metrics),
                    **null_metrics,
                    "hard_negative_valid_count": int(
                        hard_selection.within_radius.sum()
                    ),
                    "hard_negative_radius": float(radius),
                    "hard_negative_global_fallback_count": int(
                        (hard_selection.selection_mode == "global_fallback").sum()
                    ),
                    "hard_negative_missing_count": int(missing_hard_negative.sum()),
                    "correct_vs_local_hard_shuffled_mean_error_reduction": (
                        (local_hard_metrics["mean_error"] - correct_metrics["mean_error"])
                        / max(local_hard_metrics["mean_error"], 1e-12)
                    ),
                }
                for shift in args.shift_pixels:
                    shifted = shifted_centers(variant_projections, shift)
                    valid = valid_shift_recovery_mask(
                        shifted, variant_ground_truth, radius
                    )
                    label = str(int(shift) if float(shift).is_integer() else shift)
                    row[f"shift_{label}_eligible_tracks"] = int(valid.sum())
                    if bool(valid.any()):
                        recovered, _ = local_feature_match(
                            queries[torch.as_tensor(valid, device=queries.device)],
                            feature_map,
                            shifted[valid],
                            window_radius=radius,
                            temperature=args.temperature,
                        )
                        recovery_metrics = error_metrics(
                            recovered,
                            variant_ground_truth[valid],
                            feature_stride=feature_map.effective_stride,
                        )
                        center_metrics = error_metrics(
                            shifted[valid],
                            variant_ground_truth[valid],
                            feature_stride=feature_map.effective_stride,
                        )
                        row.update(prefix_metrics(f"shift_{label}_recovery", recovery_metrics))
                        row.update(prefix_metrics(f"shift_{label}_center", center_metrics))
                summary_rows.append(row)

                correct_errors = np.linalg.norm(
                    prediction - variant_ground_truth, axis=1
                )
                shuffled_errors = np.linalg.norm(
                    shuffled_prediction - variant_ground_truth, axis=1
                )
                for offset, track_index in enumerate(track_indices):
                    per_track_rows.append(
                        {
                            "image_name": record["image_name"],
                            "track_id": int(store.track_ids[track_index]),
                            "track_index": int(track_index),
                            "target_variant": variant_label,
                            "window_radius": radius,
                            "feature_backend": args.feature_backend,
                            "image_width": feature_map.input_resolution[1],
                            "image_height": feature_map.input_resolution[0],
                            "feature_width": feature_map.feature_resolution[1],
                            "feature_height": feature_map.feature_resolution[0],
                            "feature_stride": feature_map.effective_stride,
                            "hard_negative_radius": float(radius),
                            "projection_x": float(variant_projections[offset, 0]),
                            "projection_y": float(variant_projections[offset, 1]),
                            "ground_truth_x": float(variant_ground_truth[offset, 0]),
                            "ground_truth_y": float(variant_ground_truth[offset, 1]),
                            "hard_negative_valid": bool(
                                hard_selection.within_radius[offset]
                            ),
                            "hard_negative_within_radius": bool(
                                hard_selection.within_radius[offset]
                            ),
                            "hard_negative_projection_distance_px": float(
                                hard_selection.projection_distance_px[offset]
                            ),
                            "hard_negative_selection_mode": str(
                                hard_selection.selection_mode[offset]
                            ),
                            "global_shuffle_valid": bool(len(queries) > 1),
                            **{name + "_error": float(np.linalg.norm(pred[offset] - variant_ground_truth[offset]))
                               for name, pred in null_predictions.items()},
                            "projection_error": float(
                                np.linalg.norm(
                                    variant_projections[offset]
                                    - variant_ground_truth[offset]
                                )
                            ),
                            "matching_error": float(correct_errors[offset]),
                            "hard_shuffled_error": float(shuffled_errors[offset]),
                            "entropy": float(diagnostics["entropy"][offset]),
                            "peak_probability": float(
                                diagnostics["peak_probability"][offset]
                            ),
                            "hard_negative_track_id": (
                                None if hard_selection.selection_mode[offset] == "missing"
                                else int(store.track_ids[track_indices[hard_indices[offset]]])
                            ),
                            "gt_inside_correct_window": bool(
                                valid_shift_recovery_mask(
                                    variant_projections[offset : offset + 1],
                                    variant_ground_truth[offset : offset + 1],
                                    radius,
                                )[0]
                            ),
                        }
                    )
                if radius == primary_radius:
                    aggregate_predictions[variant_label].append(prediction)
                    aggregate_ground_truth[variant_label].append(
                        variant_ground_truth
                    )
                    aggregate_strides[variant_label].extend(
                        [feature_map.effective_stride] * len(prediction)
                    )

    projection_pred = np.concatenate(aggregate_projection_pred)
    projection_gt = np.concatenate(aggregate_projection_gt)
    stride_values = aggregate_strides["Real Target + feature matching [reference diagnostic]"]
    projection_stride = float(np.mean(stride_values)) if stride_values else 1.0
    projection_metrics = error_metrics(
        projection_pred, projection_gt, feature_stride=projection_stride
    )
    exact = {
        "primary_radius": primary_radius,
        "projection_mean_error": projection_metrics["mean_error"],
        "projection_median_error": projection_metrics["median_error"],
        "projection_pck3": projection_metrics["pck3"],
        "projection_pck5": projection_metrics["pck5"],
        "projection_pck8": projection_metrics["pck8"],
        "projection_pck16": projection_metrics["pck16"],
        "projection_normalized_error_feature_stride": projection_metrics[
            "normalized_error_feature_stride"
        ],
    }
    variant_metrics = {}
    for label in VARIANTS:
        prediction = np.concatenate(aggregate_predictions[label])
        ground_truth = np.concatenate(aggregate_ground_truth[label])
        stride = float(np.mean(aggregate_strides[label]))
        variant_metrics[label] = error_metrics(
            prediction, ground_truth, feature_stride=stride
        )
    gs_error = variant_metrics["GS Render + feature matching"]["mean_error"]
    difix_error = variant_metrics["Difix Output + feature matching"]["mean_error"]
    real_error = variant_metrics[
        "Real Target + feature matching [reference diagnostic]"
    ]["mean_error"]
    exact.update(
        {
            "gs_mean_error": gs_error,
            "difix_mean_error": difix_error,
            "real_reference_mean_error": real_error,
            "difix_vs_projection_error_reduction": (
                (exact["projection_mean_error"] - difix_error)
                / max(exact["projection_mean_error"], 1e-12)
            ),
            "difix_vs_gs_error_reduction": (
                (gs_error - difix_error) / max(gs_error, 1e-12)
            ),
            "primary_window_radius": primary_radius,
            "variant_metrics": variant_metrics,
        }
    )

    summary_csv = args.output_dir / "summary.csv"
    per_track_csv = args.output_dir / "per_track.csv"
    write_csv(summary_csv, summary_rows)
    write_csv(per_track_csv, per_track_rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(exact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    metadata = {
        "feature_backend": args.feature_backend,
        "dinov2_model": args.dinov2_model if args.feature_backend == "dinov2" else None,
        "window_radii": args.window_radii,
        "primary_radius": args.primary_radius,
        "track_h5_sha256": sha256_file(args.track_h5.expanduser().resolve()),
        "extractor_provenance": extractor_provenance,
        "shift_pixels": args.shift_pixels,
        "hard_negative_policy": (
            "highest_appearance_similarity_within_radius_for_primary_margin; "
            "global_fallback_secondary_only; missing_is_not_a_negative"
        ),
        "hard_negative_radius": hard_negative_radius,
        "hard_negative_radius_policy": "equal_to_each_row_window_radius",
        "temperature": args.temperature,
        "interpolation_method": "recorded_per_feature_map",
        "feature_maps": feature_metadata,
        "source_images_dir": str(images_dir),
        "target_manifest": str(target_manifest),
        "target_manifest_sha256": sha256_file(target_manifest),
        "paired_identity_arm": args.paired_arm.upper() if args.paired_arm else None,
        "pck_primary_thresholds": [3, 5, 8, 16],
        "heldout_usage": "evaluation_labels_only",
        "primary_comparison": "paired_evidence_report.py; never compare conditional means alone",
        "shift_analysis": "secondary_oracle_stratified_by_gt_inside_window",
        "real_target_role": "privileged reference diagnostic, not a mathematical upper bound",
        "summary_csv": str(summary_csv.resolve()),
        "summary_csv_sha256": sha256_file(summary_csv),
        "per_track_csv": str(per_track_csv.resolve()),
        "per_track_csv_sha256": sha256_file(per_track_csv),
    }
    if paired_binding is not None:
        for field in (
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
            "metadata_path",
            "metadata_sha256",
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
        ):
            metadata[field] = paired_binding[field]
        for field in PAIRED_RECONSTRUCTION_FIELDS:
            metadata[field] = paired_binding[field]
        source_inventory = build_source_image_inventory(
            images_dir, paired_binding["source_camera_names"]
        )
        metadata["source_image_inventory"] = source_inventory
        metadata["source_image_inventory_sha256"] = (
            source_image_inventory_sha256(source_inventory)
        )
        metadata["paired_manifest_record_count"] = int(
            paired_binding["record_count"]
        )
        metadata["paired_manifest_metadata"] = paired_binding["metadata_path"]
        metadata["paired_manifest_metadata_sha256"] = paired_binding[
            "metadata_sha256"
        ]
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(exact, indent=2))


if __name__ == "__main__":
    main()
