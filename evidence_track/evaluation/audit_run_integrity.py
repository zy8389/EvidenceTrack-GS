#!/usr/bin/env python3
"""Revalidate and summarize one controlled dataset/scene/seed run."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

from diffusion_guidance.checkpoint_state import load_checkpoint_summary
from diffusion_guidance.control_identity import (
    normalized_camera_names,
    source_camera_set_sha256,
)
from diffusion_guidance.evidence_protocol import (
    canonical_source_image_inventory,
    source_image_inventory_sha256,
)
from diffusion_guidance.experiment_registry import expected_scene_role
from diffusion_guidance.result_binding import (
    audit_supports_contrast,
    read_pair_audit,
    require_audit_binding,
)
from evidence_track.evaluation.aggregate_results import _require_row_bound_pair_audits
from evidence_track.evaluation.audit_pseudo_camera_manifest import audit as audit_pseudo_manifest
from evidence_track.evaluation.compare_identity_reports import compare_reports, read as read_identity_report
from evidence_track.evaluation.evaluate_track_evidence import (
    load_paired_target_manifest,
    validate_paired_manifest_metadata,
)
from evidence_track.evaluation.metrics_from_log import extract_metric_row


SCHEMA = "single_scene_seed_integrity_v1"
PINNED_UPSTREAM_COMMIT = "81ada6a32c918591ae7c7a0279dc6ca7a8018e2f"
RECOVERY_SCHEMA = "real_gaussian_geometry_recovery_v1"
RECOVERY_GATE = "P0_real_gaussian_geometry_recovery"
RECOVERY_RATIOS = (0.005, 0.01, 0.02)
RECOVERY_STEPS = 100
RECOVERY_MINIMUM_REDUCTION = 0.10
RECOVERY_TRACK_COUNT = 32
PROJECTION_CODE_INPUTS = (
    "train.py",
    "arguments/__init__.py",
    "evidence_track/evaluation/check_camera_projection_equivalence.py",
    "evidence_track/geometry/repaired_geometry.py",
    "evidence_track/geometry/strict_track_store.py",
    "evidence_track/diffusion/calibration_guard.py",
    "scene/__init__.py",
    "scene/cameras.py",
    "scene/dataset_readers.py",
    "utils/camera_utils.py",
    "utils/graphics_utils.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_passed_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("passed") is not True:
        raise ValueError(f"{label} lacks exact passed=true: {path}")
    return payload


def _same_json(left: Any, right: Any) -> bool:
    return json.dumps(
        left, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ) == json.dumps(right, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def validate_environment_binding(
    run: Path, environment_gate: dict[str, Any]
) -> dict[str, Any]:
    """Revalidate the immutable environment snapshot behind every run gate."""
    environment_path = (run / "environment.json").resolve()
    if not environment_path.is_file():
        raise FileNotFoundError(f"Environment report is missing: {environment_path}")
    environment = json.loads(environment_path.read_text(encoding="utf-8"))
    if not isinstance(environment, dict):
        raise ValueError("Environment report must be a JSON object")
    expected_checks = {
        "cuda_available",
        "cuda_build_recorded",
        "cuda_device_recorded",
        "torch_version_recorded",
        "pinned_upstream_commit",
        "environment_report_commit",
        "research_revision_marker",
        "python_source_inventory",
        "protocol_file_inventory",
        "rasterizer_extension_import",
        "simple_knn_extension_import",
    }
    if (
        environment_gate.get("schema") != "environment_cuda_gate_v1"
        or environment_gate.get("gate") != "P0_environment_and_cuda_dependencies"
        or environment_gate.get("passed") is not True
        or set(environment_gate.get("checks", {})) != expected_checks
        or any(value is not True for value in environment_gate["checks"].values())
    ):
        raise ValueError("Environment gate lacks the exact passed check set")
    if (
        Path(str(environment_gate.get("environment_report", ""))).expanduser().resolve()
        != environment_path
        or environment_gate.get("environment_report_sha256") != sha256_file(environment_path)
    ):
        raise ValueError("Environment gate is stale relative to environment.json")

    marker_path = Path(
        str(environment_gate.get("research_revision_marker", ""))
    ).expanduser().resolve()
    if not marker_path.is_file():
        raise FileNotFoundError(f"Research revision marker is missing: {marker_path}")
    if environment_gate.get("research_revision_marker_sha256") != sha256_file(marker_path):
        raise ValueError("Research revision marker hash is stale")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if not isinstance(marker, dict) or marker.get("revision") != "research-v2":
        raise ValueError("Research revision marker is not research-v2")

    details = environment_gate.get("details")
    if not isinstance(details, dict):
        raise ValueError("Environment gate lacks commit details")
    if (
        environment.get("upstream_commit") != PINNED_UPSTREAM_COMMIT
        or details.get("marker_upstream_commit") != PINNED_UPSTREAM_COMMIT
        or not details.get("current_commit")
        or environment.get("git_commit") != details.get("current_commit")
    ):
        raise ValueError(
            "Environment report/gate is not bound to the pinned upstream provenance and current revision"
        )

    recorded_hashes = environment.get("python_sha256")
    if not isinstance(recorded_hashes, dict) or not recorded_hashes:
        raise ValueError("Environment report lacks the Python source inventory")
    code_root = marker_path.parent.resolve()
    current_hashes: dict[str, str] = {}
    for source_path in code_root.rglob("*.py"):
        if ".git" in source_path.parts:
            continue
        relative = source_path.relative_to(code_root).as_posix()
        current_hashes[relative] = sha256_file(source_path)
    normalized_recorded = {
        str(relative).replace("\\", "/"): str(digest)
        for relative, digest in recorded_hashes.items()
    }
    if set(normalized_recorded) != set(current_hashes):
        raise ValueError("Environment Python source inventory changed after collection")
    for relative, digest in current_hashes.items():
        if normalized_recorded[relative] != digest:
            raise ValueError(f"Environment Python source hash is stale: {relative}")

    protocol_candidates = [
        code_root / "scripts" / "run_stage.sh",
        code_root / "configs" / "controlled_protocol.json",
        *sorted(code_root.glob("env/requirements-*.txt")),
    ]
    recorded_protocol = environment.get("protocol_sha256")
    if not isinstance(recorded_protocol, dict) or not recorded_protocol:
        raise ValueError("Environment report lacks the protocol-file inventory")
    current_protocol = {
        path.relative_to(code_root).as_posix(): sha256_file(path)
        for path in protocol_candidates
        if path.is_file()
    }
    normalized_protocol = {
        str(relative).replace("\\", "/"): str(digest)
        for relative, digest in recorded_protocol.items()
    }
    if normalized_protocol != current_protocol:
        raise ValueError("Environment protocol-file inventory changed after collection")
    return {
        "environment": environment_path,
        "environment_sha256": sha256_file(environment_path),
        "revision_marker": marker_path,
        "revision_marker_sha256": sha256_file(marker_path),
        "source_inventory_count": len(current_hashes),
        "protocol_inventory_count": len(current_protocol),
    }


def validate_track_binding(run: Path, track_gate: dict[str, Any]) -> tuple[Path, str]:
    track = (run / "tracks_source_only.h5").resolve()
    if not track.is_file():
        raise FileNotFoundError(f"Strict Track H5 is missing: {track}")
    track_sha256 = sha256_file(track)
    if (
        track_gate.get("gate") != "P0_strict_source_only_track_build_and_audit"
        or track_gate.get("track_sha256") != track_sha256
    ):
        raise ValueError("Track coverage gate is stale or has the wrong gate identity")
    return track, track_sha256


def validate_projection_binding(
    projection: dict[str, Any],
    *,
    run: Path,
    track: Path,
    track_sha256: str,
    scene: str,
) -> dict[str, Any]:
    if projection.get("gate") != "P0_H5_vs_live_Scene_projection_equivalence":
        raise ValueError("Projection report has the wrong gate identity")
    max_pixel_error = _finite_number(
        projection.get("max_pixel_error_threshold"), "projection pixel threshold"
    )
    max_matrix_error = _finite_number(
        projection.get("max_matrix_error_threshold"), "projection matrix threshold"
    )
    if not (0.0 < max_pixel_error <= 1e-3):
        raise ValueError("Projection pixel threshold is outside the precommitted range")
    if not (0.0 < max_matrix_error <= 1e-5):
        raise ValueError("Projection matrix threshold is outside the precommitted range")

    context = projection.get("run_stage_context")
    expected_context_fields = {
        "fingerprint",
        "schema",
        "scene",
        "n_views",
        "llff_holdout",
        "strict_source_only_geometry",
        "inputs",
    }
    if not isinstance(context, dict) or set(context) != expected_context_fields:
        raise ValueError("Projection report lacks the canonical run-stage context")
    if (
        context.get("schema") != 1
        or context.get("strict_source_only_geometry") is not True
        or not isinstance(context.get("n_views"), int)
        or int(context["n_views"]) <= 0
        or not isinstance(context.get("llff_holdout"), int)
        or int(context["llff_holdout"]) <= 0
    ):
        raise ValueError("Projection run-stage context has invalid protocol values")
    scene_path = Path(str(context.get("scene", ""))).expanduser().resolve()
    if scene_path.name != scene:
        raise ValueError("Projection context belongs to another scene")
    inputs = context.get("inputs")
    if not isinstance(inputs, dict) or not inputs:
        raise ValueError("Projection context has no immutable input inventory")

    fixed_paths = {
        "strict_track_h5": track,
        "environment_report": (run / "environment.json").resolve(),
        "environment_gate": (run / "environment_gate.json").resolve(),
        "source_image_list": (run / "split" / "source_images.txt").resolve(),
    }
    marker_record = inputs.get("research_revision_marker")
    if not isinstance(marker_record, dict) or not marker_record.get("path"):
        raise ValueError("Projection context lacks the research revision marker")
    code_root = Path(str(marker_record["path"])).expanduser().resolve().parent
    fixed_paths["research_revision_marker"] = code_root / ".research_revision_v2.json"
    for relative in PROJECTION_CODE_INPUTS:
        fixed_paths[f"code:{relative}"] = code_root / relative

    colmap_paths: dict[str, Path] = {}
    for stem in ("cameras", "images"):
        candidates = (
            scene_path / "sparse" / "0" / f"{stem}.bin",
            scene_path / "sparse" / "0" / f"{stem}.txt",
        )
        present = [candidate for candidate in candidates if candidate.is_file()]
        if not present:
            raise FileNotFoundError(
                f"Projection context lacks current COLMAP {stem}.bin/.txt"
            )
        for candidate in present:
            colmap_paths[f"colmap:{candidate.name}"] = candidate.resolve()

    source_list = fixed_paths["source_image_list"]
    if not source_list.is_file():
        raise FileNotFoundError(f"Projection source image list is missing: {source_list}")
    source_names = [
        line.strip()
        for line in source_list.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(source_names) != int(context["n_views"]) or len(source_names) != len(
        set(source_names)
    ):
        raise ValueError("Projection source image list disagrees with n_views")
    source_paths = {
        f"source_image:{index}:{name}": (scene_path / "images" / name).resolve()
        for index, name in enumerate(source_names)
    }
    expected_paths = {**fixed_paths, **colmap_paths, **source_paths}
    if set(inputs) != set(expected_paths):
        missing = sorted(set(expected_paths) - set(inputs))
        unexpected = sorted(set(inputs) - set(expected_paths))
        raise ValueError(
            "Projection input inventory is incomplete or non-canonical: "
            f"missing={missing}, unexpected={unexpected}"
        )
    for label, expected_path in expected_paths.items():
        descriptor = inputs[label]
        if not isinstance(descriptor, dict) or set(descriptor) != {"path", "sha256"}:
            raise ValueError(f"Projection input descriptor is invalid: {label}")
        actual_path = Path(str(descriptor["path"])).expanduser().resolve()
        if actual_path != expected_path.resolve():
            raise ValueError(f"Projection input path changed: {label}")
        if not actual_path.is_file() or descriptor["sha256"] != sha256_file(actual_path):
            raise ValueError(f"Projection input is missing or stale: {label}")
    if inputs["strict_track_h5"]["sha256"] != track_sha256:
        raise ValueError("Projection context binds another Track H5")
    source_inventory = canonical_source_image_inventory(
        [
            {
                "camera_name": name,
                "path": str(expected_path),
                "sha256": inputs[f"source_image:{index}:{name}"]["sha256"],
            }
            for index, name in enumerate(source_names)
            for expected_path in [source_paths[f"source_image:{index}:{name}"]]
        ]
    )
    source_inventory_digest = source_image_inventory_sha256(source_inventory)

    canonical_context = {
        field: context[field]
        for field in (
            "schema",
            "scene",
            "n_views",
            "llff_holdout",
            "strict_source_only_geometry",
            "inputs",
        )
    }
    fingerprint = hashlib.sha256(
        json.dumps(
            canonical_context, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
    ).hexdigest()
    if context["fingerprint"] != fingerprint:
        raise ValueError("Projection context fingerprint is not canonical")

    records = projection.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("Projection report has no camera measurements")
    resolutions_by_camera: dict[str, set[str]] = {}
    for index, record in enumerate(records):
        if not isinstance(record, dict) or record.get("passed") is not True:
            raise ValueError(f"Projection measurement {index} did not pass")
        name = str(record.get("camera_name", ""))
        kind = str(record.get("resolution_kind", ""))
        if not name or kind not in {"original", "training"}:
            raise ValueError(f"Projection measurement {index} has invalid identity")
        resolutions_by_camera.setdefault(name, set()).add(kind)
        valid_count = record.get("valid_point_count")
        if isinstance(valid_count, bool) or not isinstance(valid_count, int) or valid_count <= 0:
            raise ValueError(f"Projection measurement {index} has no valid points")
        pixel_values = [
            _finite_number(
                record.get(field), f"projection record {index} {field}"
            )
            for field in (
                "mean_pixel_difference",
                "median_pixel_difference",
                "max_pixel_difference",
            )
        ]
        if (
            any(value < 0.0 for value in pixel_values)
            or pixel_values[0] > pixel_values[2]
            or pixel_values[1] > pixel_values[2]
            or pixel_values[2] > max_pixel_error
        ):
            raise ValueError(f"Projection measurement {index} exceeds pixel tolerance")
        if (
            _finite_number(record.get("tolerance_px"), "record pixel tolerance")
            != max_pixel_error
            or _finite_number(record.get("matrix_tolerance"), "record matrix tolerance")
            != max_matrix_error
        ):
            raise ValueError(f"Projection measurement {index} changes its tolerance")
        resolution = record.get("resolution")
        if (
            not isinstance(resolution, list)
            or len(resolution) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in resolution
            )
        ):
            raise ValueError(f"Projection measurement {index} has invalid resolution")
        for field in (
            "rotation_max_abs_difference",
            "translation_max_abs_difference",
            "intrinsics_max_abs_difference",
        ):
            if _finite_number(record.get(field), f"projection record {index} {field}") > max_matrix_error:
                raise ValueError(f"Projection measurement {index} exceeds matrix tolerance")
    expected_cameras = set(normalized_camera_names(source_names))
    if (
        len(records) != 2 * len(expected_cameras)
        or set(resolutions_by_camera) != expected_cameras
        or any(
            kinds != {"original", "training"}
            for kinds in resolutions_by_camera.values()
        )
    ):
        raise ValueError("Projection report does not cover both resolutions of every source camera")
    return {
        **canonical_context,
        "fingerprint": fingerprint,
        "source_camera_names": normalized_camera_names(source_names),
        "source_camera_set_sha256": source_camera_set_sha256(source_names),
        "source_image_inventory": source_inventory,
        "source_image_inventory_sha256": source_inventory_digest,
    }


def validate_live_pseudo_binding(
    live: dict[str, Any],
    *,
    track: Path,
    track_sha256: str,
    projection_fingerprint: str,
    seed: int,
) -> tuple[list[str], str]:
    if (
        live.get("gate") != "P0_live_pseudo_camera_pose_intrinsics_provenance"
        or live.get("upstream_commit_expected") != PINNED_UPSTREAM_COMMIT
        or live.get("upstream_commit_current") != PINNED_UPSTREAM_COMMIT
        or int(live.get("seed", -1)) != seed
    ):
        raise ValueError("Live pseudo-camera gate has the wrong run identity")
    reported_track = Path(str(live.get("track_h5", ""))).expanduser().resolve()
    if reported_track != track or live.get("track_h5_sha256") != track_sha256:
        raise ValueError("Live pseudo-camera gate binds another Track H5")
    if live.get("projection_context_fingerprint") != projection_fingerprint:
        raise ValueError("Live pseudo-camera gate binds another projection context")
    source_names = live.get("source_camera_names")
    if not isinstance(source_names, list):
        raise ValueError("Live pseudo-camera gate lacks source camera names")
    normalized_sources = normalized_camera_names(source_names)
    heldout_names = live.get("heldout_camera_names")
    if not isinstance(heldout_names, list) or set(normalized_sources) & set(
        normalized_camera_names(heldout_names) if heldout_names else []
    ):
        raise ValueError("Live pseudo-camera source and held-out sets overlap")
    if (
        live.get("source_only_pose_claim") is not True
        or live.get("uses_heldout_pose_information") is not False
        or live.get("declared_pose_source_camera_names") != source_names
        or live.get("declared_heldout_pose_source_camera_names") != []
        or live.get("camera_sources") != ["source_camera_interpolation_only"]
    ):
        raise ValueError("Live pseudo-camera source-only declaration is inconsistent")
    records = live.get("records")
    if not isinstance(records, list) or int(live.get("record_count", -1)) != len(records):
        raise ValueError("Live pseudo-camera record count is invalid")
    fingerprints: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Live pseudo-camera record is not an object")
        fingerprint = str(record.get("camera_fingerprint", ""))
        if not fingerprint or fingerprint in fingerprints:
            raise ValueError("Live pseudo-camera fingerprints are empty or duplicated")
        fingerprints.add(fingerprint)
        if (
            record.get("camera_source") != "source_camera_interpolation_only"
            or record.get("uses_heldout_pose_information") is not False
            or normalized_camera_names(record.get("pose_source_camera_names", []))
            != normalized_sources
        ):
            raise ValueError("Live pseudo-camera record has inconsistent pose provenance")
    return normalized_sources, source_camera_set_sha256(normalized_sources)


def validate_geometry_smoke_binding(
    smoke: dict[str, Any],
    *,
    run: Path,
    track: Path,
    track_sha256: str,
    projection_fingerprint: str,
) -> Path:
    source_path = (run / "geometry_smoke" / "geometry_smoke.json").resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Geometry smoke source report is missing: {source_path}")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(source, dict) or source.get("passed") is not True:
        raise ValueError("Geometry smoke source report did not pass")
    if source.get("schema") != 1 or source.get("gate") != "P0_real_cuda_geometry_smoke":
        raise ValueError("Geometry smoke source report has the wrong schema or gate identity")
    for field, value in source.items():
        if not _same_json(smoke.get(field), value):
            raise ValueError(f"Geometry smoke gate changed source field {field}")
    if (
        Path(str(smoke.get("source_report", ""))).expanduser().resolve() != source_path
        or smoke.get("source_report_sha256") != sha256_file(source_path)
    ):
        raise ValueError("Geometry smoke source report path/hash binding is stale")
    if (
        Path(str(smoke.get("track_h5", ""))).expanduser().resolve() != track
        or smoke.get("track_h5_sha256") != track_sha256
    ):
        raise ValueError("Geometry smoke gate binds another Track H5")
    if smoke.get("projection_context_fingerprint") != projection_fingerprint:
        raise ValueError("Geometry smoke gate binds another projection context")
    renderer = smoke.get("renderer_probe")
    if not isinstance(renderer, dict) or any(
        renderer.get(field) is not True for field in ("passed", "cuda", "finite")
    ):
        raise ValueError("Geometry smoke lacks a passed finite CUDA renderer probe")
    expected_checks = {
        "source_passed_exact_true",
        "track_path_and_sha256_current",
        "finite_positive_camera_extent",
        "finite_nonzero_gradient",
        "finite_nonempty_geometry_loss",
        "nonempty_association",
        "finite_live_cuda_renderer",
        "projection_context_bound",
    }
    checks = smoke.get("validation_checks")
    if (
        not isinstance(checks, dict)
        or set(checks) != expected_checks
        or any(value is not True for value in checks.values())
    ):
        raise ValueError("Geometry smoke gate lacks complete exact-true validation checks")
    return source_path


def validate_pseudo_manifest_binding(
    pseudo_gate: dict[str, Any],
    *,
    run: Path,
    live_path: Path,
    live: dict[str, Any],
    track: Path,
    track_sha256: str,
    projection_fingerprint: str,
    source_camera_set_hash: str,
) -> tuple[Path, int]:
    manifest = (run / "pseudo" / "manifest.jsonl").resolve()
    if not manifest.is_file():
        raise FileNotFoundError(f"Pseudo-camera manifest is missing: {manifest}")
    if pseudo_gate.get("manifest_sha256") != sha256_file(manifest):
        raise ValueError("Pseudo-camera provenance gate has a stale manifest hash")
    recomputed = audit_pseudo_manifest(manifest)
    for field, value in recomputed.items():
        if not _same_json(pseudo_gate.get(field), value):
            raise ValueError(f"Pseudo-camera provenance gate is stale for {field}")
    if (
        pseudo_gate.get("live_pseudo_audit_sha256") != sha256_file(live_path)
        or pseudo_gate.get("projection_context_fingerprint")
        != projection_fingerprint
        or Path(str(pseudo_gate.get("track_h5", ""))).expanduser().resolve()
        != track
        or pseudo_gate.get("track_h5_sha256") != track_sha256
        or pseudo_gate.get("source_camera_set_sha256") != source_camera_set_hash
    ):
        raise ValueError("Pseudo-camera manifest gate is not bound to current upstream gates")

    raw_records: list[dict[str, Any]] = []
    with manifest.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"Pseudo manifest row is not an object: {line_number}")
            raw_records.append(record)
    keys = [str(record.get("key", "")) for record in raw_records]
    if len(raw_records) != 32 or len(set(keys)) != 32 or any(not key for key in keys):
        raise ValueError("Pseudo manifest must contain exactly 32 unique camera records")
    if int(pseudo_gate.get("record_count", -1)) != len(raw_records):
        raise ValueError("Pseudo manifest gate record count is stale")
    live_records = live.get("records")
    live_by_fingerprint = {
        str(record.get("camera_fingerprint")): record
        for record in live_records
        if isinstance(record, dict)
    }
    camera_fields = (
        "camera_fingerprint",
        "camera_source",
        "calibration_source_camera",
        "pose_source_camera_names",
        "uses_heldout_pose_information",
        "pose_generator",
        "pose_convention",
        "fx",
        "fy",
        "cx",
        "cy",
        "width",
        "height",
        "R",
        "t",
    )
    for record in raw_records:
        live_record = live_by_fingerprint.get(str(record["camera_fingerprint"]))
        if live_record is None:
            raise ValueError("Pseudo manifest camera is absent from the live audit")
        for field in camera_fields:
            if not _same_json(record.get(field), live_record.get(field)):
                raise ValueError(
                    "Pseudo manifest camera differs from its live audit for "
                    f"{record['camera_fingerprint']}:{field}"
                )
    return manifest, len(raw_records)


def _resolve_run_path(run_dir: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Controlled metadata contains an empty path")
    path = Path(value).expanduser()
    return (path if path.is_absolute() else run_dir / path).resolve()


def validate_recovery_binding(
    recovery: dict[str, Any],
    *,
    run: Path,
    track: Path,
    track_sha256: str,
    scene: str,
    seed: int,
    source_camera_names: list[str],
    source_camera_set_hash: str,
    camera_extent: float,
    pair_audit: dict[str, Any],
) -> dict[str, Any]:
    """Bind the real recovery result to the exact A0/projection protocol."""
    if (
        recovery.get("schema") != RECOVERY_SCHEMA
        or recovery.get("gate") != RECOVERY_GATE
        or recovery.get("mode") != "real"
        or recovery.get("passed") is not True
    ):
        raise ValueError("Recovery report is not a passed real recovery gate")
    if recovery.get("experiment_seed") != seed:
        raise ValueError("Recovery report uses another experiment seed")
    if recovery.get("optimization_steps") != RECOVERY_STEPS:
        raise ValueError("Recovery report changed the preregistered optimization steps")
    if recovery.get("requested_track_count") != RECOVERY_TRACK_COUNT:
        raise ValueError("Recovery report changed the preregistered track count")
    ratios = recovery.get("perturbation_ratios")
    if not isinstance(ratios, list) or tuple(float(value) for value in ratios) != RECOVERY_RATIOS:
        raise ValueError("Recovery report changed the fixed perturbation ratios")
    if float(recovery.get("minimum_reduction", float("nan"))) != RECOVERY_MINIMUM_REDUCTION:
        raise ValueError("Recovery report changed the preregistered reduction threshold")
    observed_extent = _finite_number(recovery.get("camera_extent"), "recovery camera extent")
    if observed_extent <= 0.0 or observed_extent != float(camera_extent):
        raise ValueError("Recovery report camera extent differs from the CUDA smoke context")

    metadata_paths = {
        "A1": (run / "A1" / "controlled_ab_metadata.json").resolve(),
        "B": (run / "B" / "controlled_ab_metadata.json").resolve(),
    }
    controlled: dict[str, dict[str, Any]] = {}
    for label, metadata_path in metadata_paths.items():
        if not metadata_path.is_file():
            raise FileNotFoundError(f"{label} controlled metadata is missing: {metadata_path}")
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"{label} controlled metadata is not an object")
        controlled[label] = payload
    for field in (
        "seed",
        "scene_source_path",
        "track_h5_path",
        "track_h5_sha256",
        "source_camera_names",
        "source_camera_set_sha256",
        "start_checkpoint",
        "start_checkpoint_sha256",
        "checkpoint_iteration",
        "checkpoint_state_format",
    ):
        if controlled["A1"].get(field) != controlled["B"].get(field):
            raise ValueError(f"A1/B controlled metadata differs for {field}")
    a1_meta = controlled["A1"]
    if int(a1_meta.get("seed", -1)) != seed or int(a1_meta.get("checkpoint_iteration", -1)) != 10000:
        raise ValueError("Recovery A0 metadata has the wrong seed or iteration")
    metadata_scene = _resolve_run_path(run / "A1", a1_meta.get("scene_source_path"))
    if metadata_scene.name != scene or metadata_scene != Path(
        str(pair_audit.get("scene_source_path", ""))
    ).expanduser().resolve():
        raise ValueError("Recovery scene path does not match the controlled-pair audit")
    reported_scene = Path(str(recovery.get("scene_source_path", ""))).expanduser().resolve()
    if reported_scene != metadata_scene:
        raise ValueError("Recovery report scene path is stale")

    metadata_track = _resolve_run_path(run / "A1", a1_meta.get("track_h5_path"))
    if metadata_track != track or a1_meta.get("track_h5_sha256") != track_sha256:
        raise ValueError("Recovery A0 metadata is bound to another Track H5")
    observed_names = normalized_camera_names(recovery.get("source_camera_names", []))
    if observed_names != source_camera_names or recovery.get("source_camera_set_sha256") != source_camera_set_hash:
        raise ValueError("Recovery source-camera set differs from the projection context")
    if normalized_camera_names(a1_meta.get("source_camera_names", [])) != source_camera_names:
        raise ValueError("Controlled continuation source cameras differ from projection context")
    if a1_meta.get("source_camera_set_sha256") != source_camera_set_hash:
        raise ValueError("Controlled continuation source-camera hash differs from projection context")

    a0_checkpoint = _resolve_run_path(run / "A1", a1_meta.get("start_checkpoint"))
    b_checkpoint = _resolve_run_path(run / "B", controlled["B"].get("start_checkpoint"))
    if a0_checkpoint != b_checkpoint or not a0_checkpoint.is_file():
        raise ValueError("Recovery A0 checkpoint path is not shared and current")
    a0_sha256 = sha256_file(a0_checkpoint)
    if a0_sha256 != a1_meta.get("start_checkpoint_sha256"):
        raise ValueError("Recovery A0 checkpoint hash differs from controlled metadata")
    checkpoint_summary = load_checkpoint_summary(
        a0_checkpoint,
        expected_iteration=10000,
        require_cuda_rng=True,
        require_controlled_provenance=True,
    )
    if (
        recovery.get("checkpoint")
        != str(a0_checkpoint)
        or recovery.get("checkpoint_sha256") != a0_sha256
        or recovery.get("checkpoint_sha256_after") != a0_sha256
        or recovery.get("checkpoint_modified") is not False
        or recovery.get("checkpoint_iteration") != 10000
        or recovery.get("checkpoint_state_format") != checkpoint_summary["format"]
        or recovery.get("checkpoint_gaussian_count") != checkpoint_summary["gaussian_count"]
        or recovery.get("checkpoint_render_state_schema")
        != checkpoint_summary["render_state_schema"]
        or recovery.get("checkpoint_render_state_sha256")
        != checkpoint_summary["render_state_sha256"]
        or recovery.get("checkpoint_controlled_provenance_sha256")
        != checkpoint_summary["controlled_provenance_sha256"]
    ):
        raise ValueError("Recovery report is not bound to the current A0 checkpoint state")
    if a1_meta.get("start_checkpoint_sha256") != a0_sha256:
        raise ValueError("A1 start checkpoint hash is stale")

    if (
        recovery.get("track_h5") != str(track)
        or recovery.get("track_h5_sha256") != track_sha256
    ):
        raise ValueError("Recovery report is not bound to the current Track H5")
    if (
        pair_audit.get("same_a0_hash") is not True
        or pair_audit.get("actual_a0_checkpoint_sha256") != a0_sha256
        or pair_audit.get("start_checkpoint_controlled_provenance_sha256")
        != checkpoint_summary["controlled_provenance_sha256"]
    ):
        raise ValueError("Controlled-pair audit does not prove the recovery A0 checkpoint")

    results = recovery.get("results")
    if not isinstance(results, list) or len(results) != len(RECOVERY_RATIOS):
        raise ValueError("Recovery report must contain one result for each fixed perturbation")
    required_result_fields = {
        "perturbation_scene_radius_ratio",
        "initial_pixel_reprojection_error",
        "final_pixel_reprojection_error",
        "initial_3d_anchor_distance",
        "final_3d_anchor_distance",
        "convergence_rate",
        "nan_free",
        "valid_associated_gaussian_count",
        "unique_gaussian_count",
        "mean_gradient_norm",
        "passed",
    }
    for index, (item, expected_ratio) in enumerate(zip(results, RECOVERY_RATIOS)):
        if not isinstance(item, dict) or not required_result_fields.issubset(item):
            raise ValueError(f"Recovery result {index} is incomplete")
        if float(item["perturbation_scene_radius_ratio"]) != expected_ratio:
            raise ValueError(f"Recovery result {index} has the wrong perturbation ratio")
        finite_values = [
            _finite_number(item[field], f"recovery result {index} {field}")
            for field in (
                "initial_pixel_reprojection_error",
                "final_pixel_reprojection_error",
                "initial_3d_anchor_distance",
                "final_3d_anchor_distance",
                "convergence_rate",
                "mean_gradient_norm",
            )
        ]
        if any(value < 0.0 for value in finite_values) or finite_values[0] <= 0.0 or finite_values[2] <= 0.0:
            raise ValueError(f"Recovery result {index} has invalid error values")
        if not 0.0 <= finite_values[4] <= 1.0 or finite_values[5] <= 0.0:
            raise ValueError(f"Recovery result {index} has invalid convergence evidence")
        for field in ("valid_associated_gaussian_count", "unique_gaussian_count"):
            count = item[field]
            if isinstance(count, bool) or not isinstance(count, int) or count != RECOVERY_TRACK_COUNT:
                raise ValueError(f"Recovery result {index} has an invalid {field}")
        if item["nan_free"] is not True or item["passed"] is not True:
            raise ValueError(f"Recovery result {index} did not pass its finite-recovery checks")
        if finite_values[1] > finite_values[0] * (1.0 - RECOVERY_MINIMUM_REDUCTION):
            raise ValueError(f"Recovery result {index} misses pixel-error reduction")
        if finite_values[3] > finite_values[2] * (1.0 - RECOVERY_MINIMUM_REDUCTION):
            raise ValueError(f"Recovery result {index} misses 3D-error reduction")
        if finite_values[4] < 0.8:
            raise ValueError(f"Recovery result {index} misses per-track convergence")
    if float(recovery.get("recovery_success_rate", float("nan"))) != 1.0:
        raise ValueError("Recovery success rate is not exactly one")
    return {
        "checkpoint": str(a0_checkpoint),
        "checkpoint_sha256": a0_sha256,
        "source_camera_names": source_camera_names,
        "source_camera_set_sha256": source_camera_set_hash,
        "result_count": len(results),
    }


def read_contrast_csv(
    path: Path,
    *,
    run: Path,
    dataset: str,
    scene: str,
    seed: int,
    role: str,
    baseline: str,
    treatment: str,
    audit_path: Path,
) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Measured contrast CSV is missing: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 2 or [row.get("method") for row in rows] != [baseline, treatment]:
        raise ValueError(
            f"{path} must contain exactly ordered {baseline}/{treatment} rows"
        )
    _require_row_bound_pair_audits(
        rows, baseline=baseline, treatment=treatment, csv_path=path
    )
    resolved_audit, audit = read_pair_audit(audit_path)
    if not audit_supports_contrast(audit, baseline, treatment):
        raise ValueError(f"Audit does not authorize {treatment}-{baseline}")
    expected_logs = {
        "A0": (run / "A0" / "train.log").resolve(),
        "A1": (run / "A1" / "train.log").resolve(),
        "SelfRender": (run / "SelfRender" / "train.log").resolve(),
        "B": (run / "B" / "train.log").resolve(),
    }
    expected_fields = {
        "dataset",
        "scene",
        "role",
        "seed",
        "method",
        "psnr",
        "ssim",
        "lpips",
        "pair_id",
        "pair_audit_path",
        "pair_audit_sha256",
        "geometry_error",
        "metric_log",
        "metric_log_sha256",
    }
    for row in rows:
        if (
            row.get("dataset") != dataset
            or row.get("scene") != scene
            or int(row.get("seed", -1)) != seed
            or row.get("role") != role
            or Path(row.get("pair_audit_path", "")).expanduser().resolve()
            != resolved_audit
        ):
            raise ValueError(f"Measured row is bound to another run: {path}")
        method = str(row.get("method", ""))
        expected_log = expected_logs.get(method)
        actual_log = Path(str(row.get("metric_log", ""))).expanduser().resolve()
        if expected_log is None or actual_log != expected_log:
            raise ValueError(f"Measured row uses the wrong training log: {path}")
        if set(row) != expected_fields:
            raise ValueError(f"Measured CSV schema is not exact: {path}")
        try:
            typed_row = {
                "dataset": str(row["dataset"]),
                "scene": str(row["scene"]),
                "role": str(row["role"]),
                "seed": int(row["seed"]),
                "method": method,
                "psnr": float(row["psnr"]),
                "ssim": float(row["ssim"]),
                "lpips": float(row["lpips"]),
                "pair_id": str(row["pair_id"]),
                "pair_audit_path": str(
                    Path(row["pair_audit_path"]).expanduser().resolve()
                ),
                "pair_audit_sha256": str(row["pair_audit_sha256"]),
                "geometry_error": str(row["geometry_error"]),
                "metric_log": str(actual_log),
                "metric_log_sha256": str(row["metric_log_sha256"]),
            }
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Measured CSV contains an invalid typed row: {path}") from exc
        expected_row = extract_metric_row(
            log=expected_log,
            dataset=dataset,
            scene=scene,
            seed=seed,
            method=method,
            pair_id=str(audit["pair_id"]),
            pair_audit=resolved_audit,
            iteration=12000,
        )
        if typed_row != expected_row:
            differing = sorted(
                field
                for field in expected_fields
                if typed_row.get(field) != expected_row.get(field)
            )
            raise ValueError(
                f"Measured CSV row is stale or edited: {path}; differing fields={differing}"
            )
    return rows


def build_integrity(
    run: Path, *, dataset: str, scene: str, seed: int
) -> dict[str, Any]:
    role = expected_scene_role(dataset, scene)
    json_artifacts = {
        "environment_gate": run / "environment_gate.json",
        "track_gate": run / "track_coverage.json",
        "projection_gate": run / "camera_projection_equivalence.json",
        "live_pseudo_gate": run / "pseudo_live_camera_audit.json",
        "geometry_smoke_gate": run / "geometry_smoke_gate.json",
        "pseudo_manifest_gate": run / "pseudo" / "provenance_audit.json",
        "recovery_gate": run / "recovery.json",
        "a1_selfrender_pair_audit": run / "a1_selfrender_pair_audit.json",
        "a1_b_pair_audit": run / "a1_b_pair_audit.json",
        "selfrender_b_pair_audit": run / "selfrender_b_pseudo_control_audit.json",
        "identity_association": run / "a1_b_identity_association.json",
    }
    loaded = {
        label: require_passed_json(path, label)
        for label, path in json_artifacts.items()
    }

    environment_context = validate_environment_binding(
        run, loaded["environment_gate"]
    )
    track_path, track_sha256 = validate_track_binding(
        run, loaded["track_gate"]
    )
    projection_context = validate_projection_binding(
        loaded["projection_gate"],
        run=run,
        track=track_path,
        track_sha256=track_sha256,
        scene=scene,
    )
    live_path = json_artifacts["live_pseudo_gate"].resolve()
    live_source_names, live_source_camera_set_sha256 = validate_live_pseudo_binding(
        loaded["live_pseudo_gate"],
        track=track_path,
        track_sha256=track_sha256,
        projection_fingerprint=projection_context["fingerprint"],
        seed=seed,
    )
    if live_source_names != projection_context["source_camera_names"]:
        raise ValueError(
            "Live pseudo-camera and projection gates use different source cameras"
        )
    smoke_source_path = validate_geometry_smoke_binding(
        loaded["geometry_smoke_gate"],
        run=run,
        track=track_path,
        track_sha256=track_sha256,
        projection_fingerprint=projection_context["fingerprint"],
    )
    pseudo_manifest_path, pseudo_record_count = validate_pseudo_manifest_binding(
        loaded["pseudo_manifest_gate"],
        run=run,
        live_path=live_path,
        live=loaded["live_pseudo_gate"],
        track=track_path,
        track_sha256=track_sha256,
        projection_fingerprint=projection_context["fingerprint"],
        source_camera_set_hash=live_source_camera_set_sha256,
    )

    pair_specs = (
        ("a1_selfrender_pair_audit", "A1", "SelfRender"),
        ("a1_b_pair_audit", "A1", "B"),
        ("selfrender_b_pair_audit", "SelfRender", "B"),
    )
    pair_ids: set[str] = set()
    pair_audits: dict[str, dict[str, Any]] = {}
    for label, baseline, treatment in pair_specs:
        _, audit = read_pair_audit(json_artifacts[label])
        pair_audits[label] = audit
        if not audit_supports_contrast(audit, baseline, treatment):
            raise ValueError(f"{label} does not authorize {treatment}-{baseline}")
        for method in (baseline, treatment):
            require_audit_binding(
                audit,
                dataset=dataset,
                scene=scene,
                seed=seed,
                method=method,
                pair_id=audit["pair_id"],
                role=role,
            )
        pair_ids.add(str(audit["pair_id"]))
    if len(pair_ids) != 1:
        raise ValueError("The three controlled contrasts do not share one run identity")
    pair_id = next(iter(pair_ids))

    recovery_context = validate_recovery_binding(
        loaded["recovery_gate"],
        run=run,
        track=track_path,
        track_sha256=track_sha256,
        scene=scene,
        seed=seed,
        source_camera_names=projection_context["source_camera_names"],
        source_camera_set_hash=projection_context["source_camera_set_sha256"],
        camera_extent=_finite_number(
            loaded["geometry_smoke_gate"].get("camera_extent"),
            "geometry smoke camera extent",
        ),
        pair_audit=pair_audits["a1_b_pair_audit"],
    )

    metrics = {
        "a1_b": (
            run / "measured_results_a1_b.csv",
            "A1",
            "B",
            json_artifacts["a1_b_pair_audit"],
        ),
        "a1_selfrender": (
            run / "measured_results_a1_selfrender.csv",
            "A1",
            "SelfRender",
            json_artifacts["a1_selfrender_pair_audit"],
        ),
        "selfrender_b": (
            run / "measured_results_selfrender_b.csv",
            "SelfRender",
            "B",
            json_artifacts["selfrender_b_pair_audit"],
        ),
    }
    for path, baseline, treatment, audit_path in metrics.values():
        rows = read_contrast_csv(
            path,
            run=run,
            dataset=dataset,
            scene=scene,
            seed=seed,
            role=role,
            baseline=baseline,
            treatment=treatment,
            audit_path=audit_path,
        )
        if {row["pair_id"] for row in rows} != {pair_id}:
            raise ValueError(f"Measured CSV has another controlled pair ID: {path}")

    paired_manifest = run / "paired_identity_manifest.jsonl"
    a1_binding = validate_paired_manifest_metadata(paired_manifest, "A1")
    b_binding = validate_paired_manifest_metadata(paired_manifest, "B")
    load_paired_target_manifest(paired_manifest, "A1")
    load_paired_target_manifest(paired_manifest, "B")
    for binding in (a1_binding, b_binding):
        if (
            binding["dataset"] != dataset
            or binding["scene"] != scene
            or int(binding["seed"]) != seed
            or binding["experiment_role"] != role
            or binding["controlled_pair_id"] != pair_id
            or Path(str(binding["track_h5"])).expanduser().resolve() != track_path
            or binding["track_h5_sha256"] != track_sha256
            or binding["source_camera_set_sha256"]
            != live_source_camera_set_sha256
            or Path(str(binding["scene_source_path"])).expanduser().resolve()
            != Path(str(projection_context["scene"])).expanduser().resolve()
            or Path(str(binding["source_images_dir"])).expanduser().resolve()
            != Path(str(projection_context["scene"])).expanduser().resolve() / "images"
            or binding["source_image_inventory"]
            != projection_context["source_image_inventory"]
            or binding["source_image_inventory_sha256"]
            != projection_context["source_image_inventory_sha256"]
        ):
            raise ValueError("Paired identity manifest belongs to another run")

    a1_report_path = run / "identity_a1" / "paired_report.json"
    b_report_path = run / "identity_b" / "paired_report.json"
    recomputed_identity = compare_reports(
        read_identity_report(a1_report_path),
        read_identity_report(b_report_path),
        a1_path=a1_report_path,
        b_path=b_report_path,
    )
    if loaded["identity_association"] != recomputed_identity:
        raise ValueError(
            "Stored A1/B identity association is stale; rerun identity-report"
        )
    if (
        recomputed_identity["dataset"] != dataset
        or recomputed_identity["scene"] != scene
        or int(recomputed_identity["seed"]) != seed
        or recomputed_identity["role"] != role
        or recomputed_identity["controlled_pair_id"] != pair_id
    ):
        raise ValueError("A1/B identity association belongs to another run")

    artifacts = {
        **json_artifacts,
        "paired_identity_manifest": paired_manifest,
        "paired_identity_metadata": paired_manifest.with_suffix(
            paired_manifest.suffix + ".metadata.json"
        ),
        "environment_report": run / "environment.json",
        "track_h5": track_path,
        "geometry_smoke_source": smoke_source_path,
        "pseudo_manifest": pseudo_manifest_path,
        "a1_identity_report": a1_report_path,
        "b_identity_report": b_report_path,
        **{f"metrics_{name}": spec[0] for name, spec in metrics.items()},
    }
    return {
        "schema": SCHEMA,
        "passed": True,
        "scope": "single_dataset_scene_seed",
        "dataset": dataset,
        "scene": scene,
        "seed": seed,
        "role": role,
        "controlled_pair_id": pair_id,
        "aggregate_performed": False,
        "validated_bindings": {
            "environment_report_sha256": environment_context["environment_sha256"],
            "research_revision_marker_sha256": environment_context[
                "revision_marker_sha256"
            ],
            "source_inventory_count": environment_context[
                "source_inventory_count"
            ],
            "protocol_inventory_count": environment_context[
                "protocol_inventory_count"
            ],
            "pair_audit_input_bindings": {
                label: {
                    "file_count": audit["audited_input_file_count"],
                    "fingerprint": audit["audited_input_fingerprint"],
                }
                for label, audit in pair_audits.items()
            },
            "track_h5_sha256": track_sha256,
            "projection_context_fingerprint": projection_context["fingerprint"],
            "source_camera_names": live_source_names,
            "source_camera_set_sha256": live_source_camera_set_sha256,
            "source_image_inventory_sha256": projection_context[
                "source_image_inventory_sha256"
            ],
            "pseudo_manifest_record_count": pseudo_record_count,
            "recovery_checkpoint_sha256": recovery_context["checkpoint_sha256"],
            "recovery_result_count": recovery_context["result_count"],
        },
        "artifacts": {
            label: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for label, path in artifacts.items()
        },
        "interpretation": (
            "All required artifacts for this one controlled run are present, "
            "current, and mutually bound. This is an integrity result, not an "
            "efficacy claim or a cross-scene aggregate."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = build_integrity(
            args.run.expanduser().resolve(),
            dataset=args.dataset,
            scene=args.scene,
            seed=args.seed,
        )
    except Exception as exc:
        result = {
            "schema": SCHEMA,
            "passed": False,
            "scope": "single_dataset_scene_seed",
            "dataset": args.dataset,
            "scene": args.scene,
            "seed": int(args.seed),
            "aggregate_performed": False,
            "failure": f"{type(exc).__name__}: {exc}",
        }
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["passed"] is not True:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
