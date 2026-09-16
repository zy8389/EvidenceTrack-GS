#!/usr/bin/env python3
"""Fail-closed audit of a controlled continuation pair recorded on a real server.

A passed audit proves that two continuations began from the same complete A0 state
and consumed the same real-view updates.  It does not establish reconstruction or
geometry efficacy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from diffusion_guidance.control_identity import (
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
    validate_preregistered_controlled_training_protocol,
)
from diffusion_guidance.checkpoint_state import load_checkpoint_summary
from diffusion_guidance.difix_provenance import sha256_file
from diffusion_guidance.experiment_registry import expected_scene_role
from diffusion_guidance.pseudo_manifest import load_pseudo_manifest
from diffusion_guidance.pseudo_schedule import (
    pseudo_call_trace_sha256,
    pseudo_camera_pool_sha256,
    validate_pseudo_usage,
)


REQUIRED_METADATA = (
    "seed",
    "start_checkpoint",
    "start_checkpoint_sha256",
    "checkpoint_iteration",
    "final_iteration",
    "real_view_sampler",
    "optimizer_state_restored",
    "checkpoint_state_format",
    "densification_state_restored",
    "scene_source_path",
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
    "controlled_pair_id",
    "strict_geometry_protocol",
)

CONTROLLED_PSEUDO_WINDOW = {
    "pseudo_rgb_start": 10000,
    "pseudo_rgb_end": 12000,
    "pseudo_rgb_interval": 50,
}

CANONICAL_RESULT_METHOD = {
    "A1": "A1",
    "SELFRENDER": "SelfRender",
    "B": "B",
}

RESULT_BINDING_SCHEMA = "controlled_result_binding_v5"


def _read_json(path: Path, expected_type: type = dict):
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, expected_type):
        raise ValueError(f"Expected {expected_type.__name__} in {path}")
    return value


def _fail(result: dict[str, Any], reason: str) -> dict[str, Any]:
    result["passed"] = False
    result.setdefault("failures", []).append(reason)
    return result


def _audited_input_fingerprint(records: list[dict[str, str]]) -> str:
    payload = json.dumps(
        records, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA256")
    return value


def _canonical_metadata_path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} is missing")
    path = Path(value).expanduser().resolve()
    if value != str(path):
        raise ValueError(f"{label} is not a canonical absolute path")
    return path


def _validate_controlled_input_contract(
    result: dict[str, Any], *, label: str, metadata: dict[str, Any]
) -> None:
    try:
        scene_path = _canonical_metadata_path(
            metadata.get("scene_source_path"), f"{label}.scene_source_path"
        )
        track_path = _canonical_metadata_path(
            metadata.get("track_h5_path"), f"{label}.track_h5_path"
        )
        checkpoint_path = _canonical_metadata_path(
            metadata.get("start_checkpoint"), f"{label}.start_checkpoint"
        )
        images_dir = _canonical_metadata_path(
            metadata.get("source_images_dir"), f"{label}.source_images_dir"
        )
        if not scene_path.is_dir():
            raise FileNotFoundError(scene_path)
        if not track_path.is_file():
            raise FileNotFoundError(track_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        if not images_dir.is_dir():
            raise FileNotFoundError(images_dir)
        if not isinstance(metadata.get("pseudo_rgb_strict_cache"), bool):
            raise ValueError("pseudo_rgb_strict_cache must be a boolean")
        try:
            images_dir.relative_to(scene_path)
        except ValueError as exc:
            raise ValueError("source image directory is outside the scene root") from exc

        source_names = normalized_camera_names(metadata.get("source_camera_names", []))
        if metadata.get("source_camera_names") != source_names:
            raise ValueError("source camera names are not canonical")
        source_set_digest = _require_sha256(
            metadata.get("source_camera_set_sha256"),
            f"{label}.source_camera_set_sha256",
        )
        if source_camera_set_sha256(source_names) != source_set_digest:
            raise ValueError("source camera names/hash mismatch")
        track_digest = _require_sha256(
            metadata.get("track_h5_sha256"), f"{label}.track_h5_sha256"
        )
        checkpoint_digest = _require_sha256(
            metadata.get("start_checkpoint_sha256"),
            f"{label}.start_checkpoint_sha256",
        )
        if sha256_file(track_path) != track_digest:
            raise ValueError("Track H5 hash mismatch")
        if sha256_file(checkpoint_path) != checkpoint_digest:
            raise ValueError("start checkpoint hash mismatch")

        image_inventory = canonical_source_image_inventory(
            metadata.get("source_image_inventory")
        )
        if metadata.get("source_image_inventory") != image_inventory:
            raise ValueError("source image inventory is not canonical")
        if [item["camera_name"] for item in image_inventory] != source_names:
            raise ValueError("source image inventory differs from source cameras")
        image_inventory_digest = _require_sha256(
            metadata.get("source_image_inventory_sha256"),
            f"{label}.source_image_inventory_sha256",
        )
        if source_image_inventory_sha256(image_inventory) != image_inventory_digest:
            raise ValueError("source image inventory hash mismatch")
        for item in image_inventory:
            image_path = Path(item["path"])
            try:
                image_path.relative_to(images_dir)
            except ValueError as exc:
                raise ValueError(
                    f"source image is outside the declared image directory: {image_path}"
                ) from exc
            if not image_path.is_file() or sha256_file(image_path) != item["sha256"]:
                raise ValueError(f"source image is missing or changed: {image_path}")

        camera_inventory = canonical_training_camera_inventory(
            metadata.get("source_training_camera_inventory")
        )
        if metadata.get("source_training_camera_inventory") != camera_inventory:
            raise ValueError("source training-camera inventory is not canonical")
        if [item["camera_name"] for item in camera_inventory] != source_names:
            raise ValueError("source training-camera inventory differs from source cameras")
        camera_inventory_digest = _require_sha256(
            metadata.get("source_training_camera_inventory_sha256"),
            f"{label}.source_training_camera_inventory_sha256",
        )
        if training_camera_inventory_sha256(camera_inventory) != camera_inventory_digest:
            raise ValueError("source training-camera inventory hash mismatch")

        protocol = validate_preregistered_controlled_training_protocol(
            metadata.get("controlled_training_protocol"),
            role=str(metadata.get("role", "")),
        )
        if metadata.get("controlled_training_protocol") != protocol:
            raise ValueError("controlled training protocol is not canonical")
        protocol_digest = _require_sha256(
            metadata.get("controlled_training_protocol_sha256"),
            f"{label}.controlled_training_protocol_sha256",
        )
        if controlled_training_protocol_sha256(protocol) != protocol_digest:
            raise ValueError("controlled training protocol hash mismatch")

        a0_provenance = canonical_controlled_checkpoint_provenance(
            metadata.get("start_checkpoint_controlled_provenance")
        )
        if metadata.get("start_checkpoint_controlled_provenance") != a0_provenance:
            raise ValueError("embedded A0 checkpoint provenance is not canonical")
        a0_provenance_digest = _require_sha256(
            metadata.get("start_checkpoint_controlled_provenance_sha256"),
            f"{label}.start_checkpoint_controlled_provenance_sha256",
        )
        if (
            controlled_checkpoint_provenance_sha256(a0_provenance)
            != a0_provenance_digest
        ):
            raise ValueError("embedded A0 checkpoint provenance hash mismatch")
        current_provenance = controlled_checkpoint_provenance_from_metadata(metadata)
        validate_a0_checkpoint_provenance(a0_provenance, current_provenance)

        model_protocol = protocol["model"]
        optimization_protocol = protocol["optimization"]
        runtime_protocol = protocol["runtime"]
        declared_images = model_protocol.get("images") or "images"
        expected_images_dir = (scene_path / str(declared_images)).resolve()
        protocol_checks = {
            "model.source_path": model_protocol.get("source_path") == str(scene_path),
            "model.track_path": model_protocol.get("track_path") == str(track_path),
            "model.images": expected_images_dir == images_dir,
            "optimization.iterations": optimization_protocol.get("iterations")
            == metadata.get("final_iteration"),
            "optimization.experiment_seed": optimization_protocol.get(
                "experiment_seed"
            )
            == metadata.get("seed"),
            "optimization.pseudo_rgb_weight": optimization_protocol.get(
                "pseudo_rgb_weight"
            )
            == metadata.get("pseudo_rgb_weight"),
            "optimization.pseudo_rgb_lambda_dssim": optimization_protocol.get(
                "pseudo_rgb_lambda_dssim"
            )
            == metadata.get("pseudo_rgb_lambda_dssim"),
            "optimization.pseudo_rgb_start": optimization_protocol.get(
                "pseudo_rgb_start"
            )
            == metadata.get("pseudo_rgb_start"),
            "optimization.pseudo_rgb_end": optimization_protocol.get(
                "pseudo_rgb_end"
            )
            == metadata.get("pseudo_rgb_end"),
            "optimization.pseudo_rgb_interval": optimization_protocol.get(
                "pseudo_rgb_interval"
            )
            == metadata.get("pseudo_rgb_interval"),
            "runtime.source_path": runtime_protocol.get("source_path")
            == str(scene_path),
            "runtime.track_path": runtime_protocol.get("track_path")
            == str(track_path),
            "runtime.start_checkpoint": runtime_protocol.get("start_checkpoint")
            == str(checkpoint_path),
            "runtime.iterations": runtime_protocol.get("iterations")
            == metadata.get("final_iteration"),
            "runtime.experiment_seed": runtime_protocol.get("experiment_seed")
            == metadata.get("seed"),
        }
        failed_checks = sorted(
            name for name, passed in protocol_checks.items() if not passed
        )
        if failed_checks:
            raise ValueError(
                f"controlled training protocol conflicts with metadata: {failed_checks}"
            )

        strict = metadata.get("strict_geometry_protocol")
        if not isinstance(strict, dict):
            raise ValueError("strict geometry protocol is not an object")
        strict_expected = {
            "strict_tracks": True,
            "strict_source_only_geometry": True,
            "densify_until_iter": 10000,
            "mixed_precision": False,
            "disable_legacy_pseudo_depth": True,
            "disable_depth_loss": True,
            "geometry_reg_enabled": False,
            "use_gt_dca": False,
            "strict_track_weight": 0.1,
        }
        mismatched_strict = sorted(
            key for key, expected in strict_expected.items()
            if strict.get(key) != expected
        )
        if mismatched_strict:
            raise ValueError(
                f"strict geometry protocol is invalid: {mismatched_strict}"
            )
        strict_protocol_checks = {
            "model.strict_tracks": model_protocol.get("strict_tracks") is True,
            "model.strict_source_only_geometry": model_protocol.get(
                "strict_source_only_geometry"
            )
            is True,
            "optimization.densify_until_iter": optimization_protocol.get(
                "densify_until_iter"
            )
            == 10000,
            "optimization.mixed_precision": optimization_protocol.get(
                "mixed_precision"
            )
            is False,
            "optimization.disable_legacy_pseudo_depth": optimization_protocol.get(
                "disable_legacy_pseudo_depth"
            )
            is True,
            "optimization.disable_depth_loss": optimization_protocol.get(
                "disable_depth_loss"
            )
            is True,
            "optimization.geometry_reg_enabled": optimization_protocol.get(
                "geometry_reg_enabled"
            )
            is False,
            "runtime.use_gt_dca": runtime_protocol.get("use_gt_dca") is False,
        }
        failed_strict_protocol_checks = sorted(
            name for name, passed in strict_protocol_checks.items() if not passed
        )
        if failed_strict_protocol_checks:
            raise ValueError(
                "controlled training protocol does not encode strict execution: "
                f"{failed_strict_protocol_checks}"
            )
    except Exception as exc:
        _fail(result, f"{label}_controlled_input_contract_invalid:{exc}")


def _manifest_input_paths(manifest: Path) -> set[Path]:
    """Enumerate every immutable file revalidated through a pseudo manifest."""
    manifest = manifest.expanduser().resolve()
    paths = {manifest}
    if not manifest.is_file():
        return paths
    has_difix_target = False
    with manifest.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {manifest}:{line_number}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected object at {manifest}:{line_number}")
            for field in (
                "input",
                "target",
                "mask",
                "a0_checkpoint",
                "live_pseudo_audit",
                "track_h5",
            ):
                value = record.get(field)
                if value not in (None, ""):
                    paths.add(_resolve_declared_path(manifest.parent, value))
            reference = record.get("reference_image", record.get("ref"))
            if reference not in (None, ""):
                paths.add(_resolve_declared_path(manifest.parent, reference))
            if (
                str(record.get("supervision_target_kind", "difix")).lower()
                == "difix"
            ):
                has_difix_target = True
                target = _resolve_declared_path(
                    manifest.parent, record.get("target")
                )
                paths.add(target.with_suffix(target.suffix + ".metadata.json"))
    if has_difix_target:
        paths.add(manifest.with_suffix(manifest.suffix + ".difix_metadata.json"))
    return paths


def _run_input_paths(run_dir: Path) -> set[Path]:
    """Collect the concrete files whose bytes a passed pair audit authorizes."""
    run_dir = run_dir.expanduser().resolve()
    metadata_path = run_dir / "controlled_ab_metadata.json"
    paths = {metadata_path, run_dir / "real_view_sequence.json"}
    metadata = _read_json(metadata_path)
    for item in canonical_source_image_inventory(
        metadata.get("source_image_inventory")
    ):
        paths.add(Path(item["path"]))
    for field in ("start_checkpoint", "track_h5_path"):
        value = metadata.get(field)
        if value not in (None, ""):
            paths.add(_resolve_declared_path(run_dir, value))
    final_iteration = metadata.get("final_iteration")
    if isinstance(final_iteration, bool) or not isinstance(final_iteration, int):
        raise ValueError(f"Controlled run has an invalid final_iteration: {run_dir}")
    paths.add(run_dir / f"chkpnt{final_iteration}.pth")
    role = str(metadata.get("role", "")).upper()
    if role in {"SELFRENDER", "B"}:
        paths.add(run_dir / "pseudo_supervision_usage.json")
        manifest_value = metadata.get("pseudo_manifest_path")
        if manifest_value in (None, ""):
            raise ValueError(f"{role} metadata lacks pseudo_manifest_path")
        paths.update(
            _manifest_input_paths(
                _resolve_declared_path(run_dir, manifest_value)
            )
        )
    return paths


def _seal_audited_inputs(result: dict[str, Any], *run_dirs: Path) -> None:
    """Bind a passed audit to all files it consumed, not only declared IDs."""
    try:
        paths: set[Path] = set()
        for run_dir in run_dirs:
            paths.update(_run_input_paths(run_dir))
        records = []
        for path in sorted(paths, key=lambda item: str(item)):
            if not path.is_file():
                raise FileNotFoundError(path)
            records.append({"path": str(path), "sha256": sha256_file(path)})
    except Exception as exc:
        _fail(result, f"audited_input_inventory_invalid:{exc}")
        records = []
    result["audited_input_files"] = records
    result["audited_input_file_count"] = len(records)
    result["audited_input_fingerprint"] = _audited_input_fingerprint(records)


def _state_evidence(
    metadata: dict[str, Any], checkpoint_summary: dict[str, Any] | None
) -> dict[str, bool]:
    """Return explicit state claims backed by a shared whole-checkpoint SHA.

    The v2 checkpoint serializes Gaussian tensors, Adam state, and all RNG states
    together.  Comparing the one immutable file digest is stronger than comparing
    selected in-memory fields after a restore.
    """
    complete = (
        bool(metadata.get("start_checkpoint_sha256"))
        and metadata.get("checkpoint_state_format") == "geotrack-research-v2"
        and metadata.get("optimizer_state_restored") is True
        and metadata.get("densification_state_restored") is True
        and checkpoint_summary is not None
        and checkpoint_summary.get("optimizer_state_present") is True
        and checkpoint_summary.get("densification_state_present") is True
        and checkpoint_summary.get("python_rng_present") is True
        and checkpoint_summary.get("numpy_rng_present") is True
        and checkpoint_summary.get("torch_rng_present") is True
        and checkpoint_summary.get("cuda_rng_present") is True
        and checkpoint_summary.get("controlled_provenance_present") is True
    )
    return {
        "same_serialized_gaussian_state": complete,
        "same_optimizer_state": complete,
        "same_rng_state": complete,
    }


def _final_checkpoint_binding(
    result: dict[str, Any],
    *,
    label: str,
    run_dir: Path,
    metadata: dict[str, Any],
) -> tuple[str, dict[str, Any]] | None:
    """Bind one exact final checkpoint to the metadata that governed its run."""
    try:
        role = str(metadata.get("role", "")).upper()
        method = CANONICAL_RESULT_METHOD.get(role)
        if method is None:
            raise ValueError(f"unsupported controlled role {role!r}")
        iteration = metadata.get("final_iteration")
        if isinstance(iteration, bool) or not isinstance(iteration, int):
            raise ValueError("final_iteration must be an integer")
        checkpoint = (run_dir / f"chkpnt{iteration}.pth").resolve()
        summary = load_checkpoint_summary(
            checkpoint,
            expected_iteration=iteration,
            require_cuda_rng=True,
            require_controlled_provenance=True,
        )
        expected_provenance = controlled_checkpoint_provenance_from_metadata(metadata)
        expected_digest = controlled_checkpoint_provenance_sha256(expected_provenance)
        if (
            summary["controlled_provenance"] != expected_provenance
            or summary["controlled_provenance_sha256"] != expected_digest
        ):
            raise ValueError(
                "serialized checkpoint provenance differs from controlled run metadata"
            )
        return method, {
            "path": str(checkpoint),
            "sha256": sha256_file(checkpoint),
            "iteration": int(summary["iteration"]),
            "state_format": summary["format"],
            "gaussian_count": int(summary["gaussian_count"]),
            "render_state_schema": summary["render_state_schema"],
            "render_state_sha256": summary["render_state_sha256"],
            "controlled_provenance": expected_provenance,
            "controlled_provenance_sha256": expected_digest,
        }
    except Exception as exc:
        _fail(result, f"{label}_final_checkpoint_invalid:{exc}")
        return None


def _attach_final_checkpoint_bindings(
    result: dict[str, Any],
    runs: list[tuple[str, Path, dict[str, Any]]],
) -> None:
    bindings: dict[str, dict[str, Any]] = {}
    for label, run_dir, metadata in runs:
        bound = _final_checkpoint_binding(
            result,
            label=label,
            run_dir=run_dir,
            metadata=metadata,
        )
        if bound is None:
            continue
        method, record = bound
        if method in bindings:
            _fail(result, f"duplicate_final_checkpoint_method:{method}")
            continue
        bindings[method] = record
    result["final_checkpoints"] = bindings


def _attach_result_binding(
    result: dict[str, Any],
    metadata: dict[str, Any],
    *,
    dataset: str | None,
    scene: str | None,
    audited_methods: list[str],
    audited_contrasts: list[list[str]],
) -> None:
    source_path = str(metadata.get("scene_source_path", ""))
    inferred_scene = Path(source_path).name if source_path else ""
    declared_scene = str(scene).strip() if scene is not None else inferred_scene
    declared_dataset = str(dataset).strip() if dataset is not None else ""
    if not declared_dataset:
        _fail(result, "missing_dataset_binding")
    if not declared_scene:
        _fail(result, "missing_scene_binding")
    elif inferred_scene and declared_scene.casefold() != inferred_scene.casefold():
        _fail(
            result,
            f"scene_binding_mismatch:declared={declared_scene},source_path={inferred_scene}",
        )
    try:
        role = expected_scene_role(declared_dataset, declared_scene)
    except Exception as exc:
        role = None
        _fail(result, f"scene_role_binding_invalid:{exc}")
    pair_id = metadata.get("controlled_pair_id")
    if not isinstance(pair_id, str) or not pair_id:
        _fail(result, "missing_or_invalid_controlled_pair_id")
    seed = metadata.get("seed")
    if isinstance(seed, bool):
        _fail(result, "invalid_seed_binding")
    else:
        try:
            int(seed)
        except (TypeError, ValueError):
            _fail(result, "invalid_seed_binding")
    result.update(
        {
            "audit_schema": RESULT_BINDING_SCHEMA,
            "pair_id": pair_id,
            "controlled_pair_id": pair_id,
            "dataset": declared_dataset,
            "scene": declared_scene,
            "role": role,
            "seed": metadata.get("seed"),
            "scene_source_path": source_path,
            "start_checkpoint_path": metadata.get("start_checkpoint"),
            "start_checkpoint_sha256": metadata.get(
                "start_checkpoint_sha256"
            ),
            "checkpoint_iteration": metadata.get("checkpoint_iteration"),
            "final_iteration": metadata.get("final_iteration"),
            "track_h5_path": metadata.get("track_h5_path"),
            "track_h5_sha256": metadata.get("track_h5_sha256"),
            "source_camera_names": metadata.get("source_camera_names"),
            "source_camera_set_sha256": metadata.get(
                "source_camera_set_sha256"
            ),
            "source_images_dir": metadata.get("source_images_dir"),
            "source_image_inventory": metadata.get("source_image_inventory"),
            "source_image_inventory_sha256": metadata.get(
                "source_image_inventory_sha256"
            ),
            "source_training_camera_inventory": metadata.get(
                "source_training_camera_inventory"
            ),
            "source_training_camera_inventory_sha256": metadata.get(
                "source_training_camera_inventory_sha256"
            ),
            "controlled_training_protocol": metadata.get(
                "controlled_training_protocol"
            ),
            "controlled_training_protocol_sha256": metadata.get(
                "controlled_training_protocol_sha256"
            ),
            "start_checkpoint_controlled_provenance": metadata.get(
                "start_checkpoint_controlled_provenance"
            ),
            "start_checkpoint_controlled_provenance_sha256": metadata.get(
                "start_checkpoint_controlled_provenance_sha256"
            ),
            "strict_geometry_protocol": metadata.get(
                "strict_geometry_protocol"
            ),
            "audited_methods": audited_methods,
            "audited_contrasts": audited_contrasts,
        }
    )


def _validate_real_sequence(
    result: dict[str, Any],
    *,
    label: str,
    sequence: list,
    metadata: dict[str, Any],
    expected_updates: int,
) -> bool:
    source_names = metadata.get("source_camera_names")
    if not isinstance(source_names, list) or not source_names:
        _fail(result, f"{label}_source_camera_names_invalid")
        return False
    source_set = {Path(str(name).replace("\\", "/")).stem for name in source_names}
    if len(source_set) != len(source_names):
        _fail(result, f"{label}_source_camera_names_not_unique")
        return False
    valid = True
    expected_iterations = range(
        int(metadata.get("checkpoint_iteration", -1)) + 1,
        int(metadata.get("final_iteration", -1)) + 1,
    )
    if len(sequence) != expected_updates:
        _fail(
            result,
            f"{label}_unexpected_real_update_count:{len(sequence)},expected={expected_updates}",
        )
        valid = False
    if len(sequence) != len(expected_iterations):
        _fail(result, f"{label}_real_sequence_iteration_span_mismatch")
        valid = False
    for offset, item in enumerate(sequence):
        if not isinstance(item, list) or len(item) != 2:
            _fail(result, f"{label}_real_sequence_record_invalid:index={offset}")
            valid = False
            continue
        iteration, camera_name = item
        if isinstance(iteration, bool):
            _fail(result, f"{label}_real_sequence_iteration_invalid:index={offset}")
            valid = False
            continue
        expected_iteration = int(metadata.get("checkpoint_iteration", -1)) + offset + 1
        try:
            observed_iteration = int(iteration)
        except (TypeError, ValueError):
            observed_iteration = -1
        if observed_iteration != expected_iteration:
            _fail(
                result,
                f"{label}_real_sequence_iteration_mismatch:index={offset}",
            )
            valid = False
        normalized = Path(str(camera_name).replace("\\", "/")).stem
        if normalized not in source_set:
            _fail(
                result,
                f"{label}_real_sequence_non_source_camera:index={offset}:{camera_name}",
            )
            valid = False
    return valid


def audit(
    baseline: Path,
    treatment: Path,
    *,
    require_pseudo: bool,
    expected_real_updates: int = 2000,
    expected_pseudo_updates: int = 39,
    expected_unique_pseudo_views: int = 32,
    dataset: str | None = None,
    scene: str | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "passed": True,
        "baseline": str(baseline.resolve()),
        "treatment": str(treatment.resolve()),
        "failures": [],
    }
    try:
        a = _read_json(baseline / "controlled_ab_metadata.json")
        b = _read_json(treatment / "controlled_ab_metadata.json")
    except Exception as exc:
        return _fail(result, f"metadata_unreadable: {exc}")

    for key in REQUIRED_METADATA:
        if key not in a or key not in b:
            _fail(result, f"missing_metadata:{key}")
            continue
        same = a[key] == b[key]
        result[f"same_{key}"] = same
        if not same:
            _fail(result, f"unmatched_metadata:{key}")

    for label, metadata in (("baseline", a), ("treatment", b)):
        try:
            expected_pair_id = controlled_pair_id(metadata)
            if metadata.get("controlled_pair_id") != expected_pair_id:
                _fail(result, f"{label}_controlled_pair_id_invalid")
        except Exception as exc:
            _fail(result, f"{label}_controlled_pair_identity_invalid:{exc}")
        _validate_controlled_input_contract(
            result, label=label, metadata=metadata
        )
    result.update(
        {
            "controlled_pair_id": a.get("controlled_pair_id"),
            "seed": a.get("seed"),
            "scene_source_path": a.get("scene_source_path"),
            "track_h5_sha256": a.get("track_h5_sha256"),
            "source_camera_set_sha256": a.get("source_camera_set_sha256"),
            "source_image_inventory_sha256": a.get(
                "source_image_inventory_sha256"
            ),
            "source_training_camera_inventory_sha256": a.get(
                "source_training_camera_inventory_sha256"
            ),
            "controlled_training_protocol_sha256": a.get(
                "controlled_training_protocol_sha256"
            ),
            "checkpoint_iteration": a.get("checkpoint_iteration"),
            "final_iteration": a.get("final_iteration"),
        }
    )

    start_hash = a.get("start_checkpoint_sha256")
    result["same_a0_hash"] = bool(start_hash and start_hash == b.get("start_checkpoint_sha256"))
    if not result["same_a0_hash"]:
        _fail(result, "start_checkpoint_hash_mismatch")
    checkpoint_summary = None
    try:
        baseline_checkpoint = _resolve_declared_path(
            baseline, a.get("start_checkpoint")
        )
        treatment_checkpoint = _resolve_declared_path(
            treatment, b.get("start_checkpoint")
        )
        result["same_a0_checkpoint_path"] = baseline_checkpoint == treatment_checkpoint
        if not result["same_a0_checkpoint_path"]:
            _fail(result, "start_checkpoint_path_mismatch")
        if not baseline_checkpoint.is_file():
            raise FileNotFoundError(baseline_checkpoint)
        actual_checkpoint_hash = sha256_file(baseline_checkpoint)
        result["actual_a0_checkpoint_sha256"] = actual_checkpoint_hash
        if actual_checkpoint_hash != start_hash or actual_checkpoint_hash != b.get(
            "start_checkpoint_sha256"
        ):
            _fail(result, "start_checkpoint_actual_hash_mismatch")
        checkpoint_summary = load_checkpoint_summary(
            baseline_checkpoint,
            expected_iteration=int(a.get("checkpoint_iteration", -1)),
            require_cuda_rng=True,
            require_controlled_provenance=True,
        )
        result["a0_checkpoint_structure"] = checkpoint_summary
        result["a0_checkpoint_controlled_provenance"] = checkpoint_summary.get(
            "controlled_provenance"
        )
        result["a0_checkpoint_controlled_provenance_sha256"] = checkpoint_summary.get(
            "controlled_provenance_sha256"
        )
        for label, metadata in (("baseline", a), ("treatment", b)):
            if checkpoint_summary.get("controlled_provenance") != metadata.get(
                "start_checkpoint_controlled_provenance"
            ):
                _fail(result, f"{label}_a0_checkpoint_provenance_mismatch")
            if checkpoint_summary.get(
                "controlled_provenance_sha256"
            ) != metadata.get("start_checkpoint_controlled_provenance_sha256"):
                _fail(result, f"{label}_a0_checkpoint_provenance_hash_mismatch")
    except Exception as exc:
        _fail(result, f"start_checkpoint_structure_invalid:{exc}")
    for label, metadata, run_dir in (
        ("baseline", a, baseline),
        ("treatment", b, treatment),
    ):
        try:
            track_path = _resolve_declared_path(
                run_dir, metadata.get("track_h5_path")
            )
            if not track_path.is_file():
                raise FileNotFoundError(track_path)
            if sha256_file(track_path) != metadata.get("track_h5_sha256"):
                raise ValueError("Track H5 SHA256 mismatch")
        except Exception as exc:
            _fail(result, f"{label}_track_h5_invalid:{exc}")
    result.update(_state_evidence(a, checkpoint_summary) if result["same_a0_hash"] else {
        "same_serialized_gaussian_state": False,
        "same_optimizer_state": False,
        "same_rng_state": False,
    })
    if not all(result[key] for key in ("same_serialized_gaussian_state", "same_optimizer_state", "same_rng_state")):
        _fail(result, "shared_checkpoint_does_not_prove_complete_state")
    for label, metadata in (("baseline", a), ("treatment", b)):
        if metadata.get("optimizer_state_restored") is not True:
            _fail(result, f"{label}_optimizer_state_not_restored")
        if metadata.get("densification_state_restored") is not True:
            _fail(result, f"{label}_densification_state_not_restored")

    try:
        baseline_sequence = _read_json(
            baseline / "real_view_sequence.json", list
        )
        treatment_sequence = _read_json(
            treatment / "real_view_sequence.json", list
        )
        result["same_source_sequence"] = baseline_sequence == treatment_sequence
        result["baseline_real_updates"] = len(baseline_sequence)
        result["treatment_real_updates"] = len(treatment_sequence)
        if not result["same_source_sequence"]:
            _fail(result, "real_view_sequence_mismatch")
        _validate_real_sequence(
            result,
            label="baseline",
            sequence=baseline_sequence,
            metadata=a,
            expected_updates=expected_real_updates,
        )
        _validate_real_sequence(
            result,
            label="treatment",
            sequence=treatment_sequence,
            metadata=b,
            expected_updates=expected_real_updates,
        )
    except Exception as exc:
        _fail(result, f"real_view_sequence_unreadable: {exc}")

    baseline_role = str(a.get("role", "")).upper()
    treatment_role = str(b.get("role", "")).upper()
    result["baseline_role"] = baseline_role
    result["treatment_role"] = treatment_role
    if baseline_role != "A1":
        _fail(result, f"unexpected_baseline_role:{baseline_role}")
    if treatment_role not in {"B", "SELFRENDER"}:
        _fail(result, f"unexpected_treatment_role:{treatment_role}")
    if treatment_role in {"B", "SELFRENDER"} and b.get(
        "pseudo_rgb_strict_cache"
    ) is not True:
        _fail(result, "treatment_pseudo_cache_not_strict")
    treatment_method = CANONICAL_RESULT_METHOD.get(treatment_role, treatment_role)
    _attach_result_binding(
        result,
        a,
        dataset=dataset,
        scene=scene,
        audited_methods=["A0", "A1", treatment_method],
        audited_contrasts=[["A1", treatment_method]],
    )

    # A controlled B/SelfRender audit is incomplete without validating its
    # realized 39/32 pseudo schedule and immutable manifest.  Keep the CLI flag
    # for an explicit invocation record, but fail closed when it is omitted.
    result["pseudo_validated_methods"] = (
        [treatment_method] if require_pseudo else []
    )
    if not require_pseudo:
        _fail(result, "strict_pseudo_validation_was_not_requested")

    if require_pseudo:
        try:
            usage = _read_json(treatment / "pseudo_supervision_usage.json")
            calls = int(usage.get("pseudo_supervision_calls", -1))
            unique = int(usage.get("unique_pseudo_views_used", -1))
            result["treatment_pseudo_updates"] = calls
            result["treatment_unique_pseudo_views"] = unique
            if calls != expected_pseudo_updates:
                _fail(result, f"unexpected_pseudo_update_count:{calls}")
            if unique != expected_unique_pseudo_views:
                _fail(result, f"unexpected_unique_pseudo_view_count:{unique}")
            for failure in validate_pseudo_usage(
                usage,
                expected_calls=expected_pseudo_updates,
                expected_unique_views=expected_unique_pseudo_views,
            ):
                _fail(result, failure)
            _validate_controlled_pseudo_window(
                result, label="treatment", usage=usage
            )
            result["treatment_pseudo_camera_pool_sha256"] = usage.get(
                "pseudo_camera_pool_sha256"
            )
            result["treatment_pseudo_call_trace_sha256"] = usage.get(
                "pseudo_call_trace_sha256"
            )
            for key in (
                "pseudo_manifest_path",
                "pseudo_manifest_sha256",
                "pseudo_camera_pool_sha256",
                "pseudo_target_kind",
            ):
                if not b.get(key):
                    _fail(result, f"missing_treatment_pseudo_provenance:{key}")
            if b.get("pseudo_camera_pool_sha256") != usage.get(
                "pseudo_camera_pool_sha256"
            ):
                _fail(result, "treatment_pseudo_manifest_pool_hash_mismatch")
            target_kind = str(b.get("pseudo_target_kind", "")).lower()
            if target_kind not in {"difix", "self_render_a0"}:
                _fail(result, "treatment_pseudo_target_kind_invalid")
            else:
                manifest_path = _resolve_declared_path(
                    treatment, b["pseudo_manifest_path"]
                )
                strict_records = _validate_declared_manifest(
                    result,
                    label="treatment",
                    metadata=b,
                    manifest=manifest_path,
                    run_dir=treatment,
                    expected_target_kind=target_kind,
                    expected_records=expected_unique_pseudo_views,
                )
                strict_keys = [str(record.get("key", "")) for record in strict_records]
                if strict_keys and usage.get("pseudo_camera_keys") != strict_keys:
                    _fail(result, "treatment_pseudo_usage_manifest_camera_keys_mismatch")
        except Exception as exc:
            _fail(result, f"pseudo_usage_unreadable: {exc}")
    else:
        result["treatment_pseudo_updates"] = 0
        result["treatment_unique_pseudo_views"] = 0

    if bool(a.get("pseudo_supervision_enabled", a.get("difix_enabled", False))):
        _fail(result, "baseline_has_pseudo_supervision")
    if require_pseudo and not bool(b.get("pseudo_supervision_enabled", b.get("difix_enabled", False))):
        _fail(result, "treatment_missing_pseudo_supervision")
    _attach_final_checkpoint_bindings(
        result,
        [
            ("baseline", baseline, a),
            ("treatment", treatment, b),
        ],
    )
    _seal_audited_inputs(result, baseline, treatment)
    result["interpretation"] = (
        "Valid paired experiment configuration; NOT a positive efficacy finding."
        if result["passed"]
        else "INVALID controlled comparison; do not aggregate or interpret this pair."
    )
    return result


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            records.append(value)
    if not records:
        raise ValueError(f"Empty JSONL manifest: {path}")
    return records


def _load_strict_manifest(
    path: Path, *, label: str, result: dict[str, Any]
) -> list[dict[str, Any]]:
    """Load and revalidate every immutable asset using manifest-relative paths.

    ``load_pseudo_manifest(..., strict_targets=True)`` resolves relative paths
    against the manifest itself and calls the production SelfRender/Difix target
    validators.  The audit retains raw records separately only to compare the
    two manifests field-by-field.
    """
    try:
        records = load_pseudo_manifest(path, strict_targets=True)
    except Exception as exc:
        _fail(result, f"{label}_strict_manifest_validation_failed:{exc}")
        return []
    return [record.raw for record in records]


def _validate_controlled_pseudo_window(
    result: dict[str, Any], *, label: str, usage: dict[str, Any]
) -> None:
    """Reject a self-consistent schedule that changed the preregistered window."""
    for key, expected in CONTROLLED_PSEUDO_WINDOW.items():
        try:
            observed = int(usage.get(key))
        except (TypeError, ValueError):
            _fail(result, f"{label}_pseudo_window_invalid:{key}")
            continue
        if observed != expected:
            _fail(
                result,
                f"{label}_pseudo_window_mismatch:{key}={observed},expected={expected}",
            )


def _resolve_declared_path(base_dir: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("missing declared path value")
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = base_dir / candidate
    return candidate.resolve()


def _validate_declared_manifest(
    result: dict[str, Any],
    *,
    label: str,
    metadata: dict[str, Any],
    manifest: Path,
    run_dir: Path,
    expected_target_kind: str,
    expected_records: int,
) -> list[dict[str, Any]]:
    """Bind one train artifact to its current immutable pseudo manifest.

    This check is intentionally repeated by both the individual continuation
    audit and the SelfRender/B control audit.  A metadata field alone is not
    evidence that the manifest and target bytes still match it.
    """
    manifest = manifest.expanduser().resolve()
    try:
        declared_path = _resolve_declared_path(
            run_dir, metadata.get("pseudo_manifest_path")
        )
        if declared_path != manifest:
            _fail(result, f"{label}_pseudo_manifest_path_mismatch")
        if not manifest.is_file():
            _fail(result, f"{label}_pseudo_manifest_missing")
            return []
        actual_sha256 = sha256_file(manifest)
        if metadata.get("pseudo_manifest_sha256") != actual_sha256:
            _fail(result, f"{label}_pseudo_manifest_sha256_mismatch")
        declared_kind = str(metadata.get("pseudo_target_kind", "")).lower()
        declared_supervision_kind = str(
            metadata.get("pseudo_supervision_target_kind", "")
        ).lower()
        if declared_kind != expected_target_kind:
            _fail(result, f"{label}_pseudo_target_kind_mismatch")
        if declared_supervision_kind != expected_target_kind:
            _fail(result, f"{label}_pseudo_supervision_target_kind_mismatch")
        strict_records = _load_strict_manifest(
            manifest, label=label, result=result
        )
        if len(strict_records) != expected_records:
            _fail(
                result,
                f"{label}_pseudo_manifest_record_count:{len(strict_records)},expected={expected_records}",
            )
        keys = [str(record.get("key", "")) for record in strict_records]
        try:
            pool_sha256 = pseudo_camera_pool_sha256(keys)
        except Exception as exc:
            _fail(result, f"{label}_pseudo_manifest_camera_pool_invalid:{exc}")
        else:
            if metadata.get("pseudo_camera_pool_sha256") != pool_sha256:
                _fail(result, f"{label}_pseudo_manifest_camera_pool_sha256_mismatch")
        return strict_records
    except Exception as exc:
        _fail(result, f"{label}_pseudo_manifest_validation_failed:{exc}")
        return []


def audit_pseudo_control(
    self_render: Path,
    b_run: Path,
    self_manifest: Path,
    b_manifest: Path,
    *,
    expected_real_updates: int = 2000,
    expected_pseudo_updates: int = 39,
    expected_unique_pseudo_views: int = 32,
    dataset: str | None = None,
    scene: str | None = None,
) -> dict[str, Any]:
    """Audit that SelfRender and B differ only in immutable pseudo target kind."""
    result: dict[str, Any] = {
        "passed": True,
        "self_render": str(self_render.resolve()),
        "b": str(b_run.resolve()),
        "self_render_manifest": str(self_manifest.resolve()),
        "b_manifest": str(b_manifest.resolve()),
        "failures": [],
    }
    try:
        self_meta = _read_json(self_render / "controlled_ab_metadata.json")
        b_meta = _read_json(b_run / "controlled_ab_metadata.json")
        self_usage = _read_json(self_render / "pseudo_supervision_usage.json")
        b_usage = _read_json(b_run / "pseudo_supervision_usage.json")
        self_sequence = _read_json(
            self_render / "real_view_sequence.json", list
        )
        b_sequence = _read_json(b_run / "real_view_sequence.json", list)
    except Exception as exc:
        return _fail(result, f"pseudo_control_inputs_unreadable:{exc}")

    if str(self_meta.get("role", "")).upper() != "SELFRENDER":
        _fail(result, "self_render_role_invalid")
    if str(b_meta.get("role", "")).upper() != "B":
        _fail(result, "b_role_invalid")
    matched_metadata = (
        "seed",
        "start_checkpoint",
        "start_checkpoint_sha256",
        "checkpoint_iteration",
        "checkpoint_state_format",
        "optimizer_state_restored",
        "densification_state_restored",
        "final_iteration",
        "real_view_sampler",
        "pseudo_view_sampler",
        "pseudo_rgb_weight",
        "pseudo_rgb_lambda_dssim",
        "pseudo_rgb_start",
        "pseudo_rgb_end",
        "pseudo_rgb_interval",
        "required_pseudo_view_pool_size",
        "pseudo_camera_pool_sha256",
        "scene_source_path",
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
        "pseudo_rgb_strict_cache",
        "controlled_pair_id",
        "strict_geometry_protocol",
    )
    for key in matched_metadata:
        if key not in self_meta or key not in b_meta:
            _fail(result, f"pseudo_control_missing_metadata:{key}")
        elif self_meta[key] != b_meta[key]:
            _fail(result, f"pseudo_control_metadata_mismatch:{key}")
        result[f"same_{key}"] = self_meta.get(key) == b_meta.get(key)

    for label, metadata in (("self_render", self_meta), ("b", b_meta)):
        try:
            if metadata.get("controlled_pair_id") != controlled_pair_id(metadata):
                _fail(result, f"{label}_controlled_pair_id_invalid")
        except Exception as exc:
            _fail(result, f"{label}_controlled_pair_identity_invalid:{exc}")
        _validate_controlled_input_contract(
            result, label=label, metadata=metadata
        )
        if metadata.get("pseudo_rgb_strict_cache") is not True:
            _fail(result, f"{label}_pseudo_cache_not_strict")
    result.update(
        {
            "controlled_pair_id": self_meta.get("controlled_pair_id"),
            "seed": self_meta.get("seed"),
            "scene_source_path": self_meta.get("scene_source_path"),
            "track_h5_sha256": self_meta.get("track_h5_sha256"),
            "source_camera_set_sha256": self_meta.get(
                "source_camera_set_sha256"
            ),
            "source_image_inventory_sha256": self_meta.get(
                "source_image_inventory_sha256"
            ),
            "source_training_camera_inventory_sha256": self_meta.get(
                "source_training_camera_inventory_sha256"
            ),
            "controlled_training_protocol_sha256": self_meta.get(
                "controlled_training_protocol_sha256"
            ),
        }
    )
    _attach_result_binding(
        result,
        self_meta,
        dataset=dataset,
        scene=scene,
        audited_methods=["SelfRender", "B"],
        audited_contrasts=[["SelfRender", "B"]],
    )
    result["pseudo_validated_methods"] = ["B", "SelfRender"]

    result["same_source_sequence"] = self_sequence == b_sequence
    result["self_render_real_updates"] = len(self_sequence)
    result["b_real_updates"] = len(b_sequence)
    if not result["same_source_sequence"]:
        _fail(result, "selfrender_b_real_view_sequence_mismatch")
    _validate_real_sequence(
        result,
        label="self_render",
        sequence=self_sequence,
        metadata=self_meta,
        expected_updates=expected_real_updates,
    )
    _validate_real_sequence(
        result,
        label="b",
        sequence=b_sequence,
        metadata=b_meta,
        expected_updates=expected_real_updates,
    )
    try:
        self_checkpoint = _resolve_declared_path(
            self_render, self_meta.get("start_checkpoint")
        )
        b_checkpoint = _resolve_declared_path(b_run, b_meta.get("start_checkpoint"))
        if self_checkpoint != b_checkpoint:
            _fail(result, "selfrender_b_start_checkpoint_path_mismatch")
        actual_hash = sha256_file(self_checkpoint)
        if actual_hash != self_meta.get("start_checkpoint_sha256") or actual_hash != b_meta.get(
            "start_checkpoint_sha256"
        ):
            _fail(result, "selfrender_b_start_checkpoint_actual_hash_mismatch")
        checkpoint_summary = load_checkpoint_summary(
            self_checkpoint,
            expected_iteration=int(self_meta.get("checkpoint_iteration", -1)),
            require_cuda_rng=True,
            require_controlled_provenance=True,
        )
        result["a0_checkpoint_structure"] = checkpoint_summary
        result["a0_checkpoint_controlled_provenance"] = checkpoint_summary[
            "controlled_provenance"
        ]
        result["a0_checkpoint_controlled_provenance_sha256"] = checkpoint_summary[
            "controlled_provenance_sha256"
        ]
        for label, metadata in (("self_render", self_meta), ("b", b_meta)):
            if checkpoint_summary["controlled_provenance"] != metadata.get(
                "start_checkpoint_controlled_provenance"
            ):
                _fail(result, f"{label}_a0_checkpoint_provenance_mismatch")
            if checkpoint_summary[
                "controlled_provenance_sha256"
            ] != metadata.get("start_checkpoint_controlled_provenance_sha256"):
                _fail(result, f"{label}_a0_checkpoint_provenance_hash_mismatch")
        state_evidence = _state_evidence(self_meta, checkpoint_summary)
        result.update(state_evidence)
        if not all(state_evidence.values()):
            _fail(result, "shared_checkpoint_does_not_prove_complete_state")
    except Exception as exc:
        _fail(result, f"selfrender_b_start_checkpoint_structure_invalid:{exc}")
    for label, metadata, run_dir in (
        ("self_render", self_meta, self_render),
        ("b", b_meta, b_run),
    ):
        if metadata.get("optimizer_state_restored") is not True:
            _fail(result, f"{label}_optimizer_state_not_restored")
        if metadata.get("densification_state_restored") is not True:
            _fail(result, f"{label}_densification_state_not_restored")
        try:
            track_path = _resolve_declared_path(
                run_dir, metadata.get("track_h5_path")
            )
            if sha256_file(track_path) != metadata.get("track_h5_sha256"):
                raise ValueError("Track H5 SHA256 mismatch")
        except Exception as exc:
            _fail(result, f"{label}_track_h5_invalid:{exc}")

    for label, usage in (("self_render", self_usage), ("b", b_usage)):
        for failure in validate_pseudo_usage(
            usage,
            expected_calls=expected_pseudo_updates,
            expected_unique_views=expected_unique_pseudo_views,
        ):
            _fail(result, f"{label}:{failure}")
        _validate_controlled_pseudo_window(result, label=label, usage=usage)

    self_records = _validate_declared_manifest(
        result,
        label="self_render",
        metadata=self_meta,
        manifest=self_manifest.expanduser().resolve(),
        run_dir=self_render,
        expected_target_kind="self_render_a0",
        expected_records=expected_unique_pseudo_views,
    )
    b_records = _validate_declared_manifest(
        result,
        label="b",
        metadata=b_meta,
        manifest=b_manifest.expanduser().resolve(),
        run_dir=b_run,
        expected_target_kind="difix",
        expected_records=expected_unique_pseudo_views,
    )

    self_keys = [str(record.get("camera_fingerprint", record.get("key", ""))) for record in self_records]
    b_keys = [str(record.get("camera_fingerprint", record.get("key", ""))) for record in b_records]
    if len(self_records) != expected_unique_pseudo_views or len(b_records) != expected_unique_pseudo_views:
        _fail(result, "pseudo_manifest_pool_size_mismatch")
    result["same_pseudo_camera_key_sequence"] = self_keys == b_keys
    if not result["same_pseudo_camera_key_sequence"]:
        _fail(result, "selfrender_b_pseudo_camera_key_sequence_mismatch")
    camera_fields = (
        "camera",
        "fx",
        "fy",
        "cx",
        "cy",
        "width",
        "height",
        "R",
        "t",
        "camera_source",
        "calibration_source_camera",
        "pose_source_camera_names",
        "uses_heldout_pose_information",
        "pose_generator",
    )
    for index, (left, right) in enumerate(zip(self_records, b_records)):
        mismatched = [key for key in camera_fields if left.get(key) != right.get(key)]
        if mismatched:
            _fail(result, f"pseudo_camera_metadata_mismatch:index={index}:{mismatched}")
        if str(left.get("supervision_target_kind", "")).lower() != "self_render_a0":
            _fail(result, f"self_render_target_kind_invalid:index={index}")
        if left.get("target") != left.get("input"):
            _fail(result, f"self_render_target_is_not_a0_render:index={index}")
        for field in ("input_sha256", "target_sha256", "a0_checkpoint_sha256"):
            if not left.get(field):
                _fail(result, f"self_render_missing_immutable_provenance:index={index}:{field}")
        try:
            self_input = _resolve_declared_path(self_manifest.parent, left["input"])
            self_target = _resolve_declared_path(self_manifest.parent, left["target"])
            if not self_input.is_file() or left.get("input_sha256") != sha256_file(self_input):
                _fail(result, f"self_render_a0_input_hash_mismatch:index={index}")
            if not self_target.is_file() or left.get("target_sha256") != sha256_file(self_target):
                _fail(result, f"self_render_a0_target_hash_mismatch:index={index}")
            if left.get("target_sha256") != left.get("input_sha256"):
                _fail(result, f"self_render_target_hash_not_equal_input:index={index}")
        except Exception as exc:
            _fail(result, f"self_render_a0_input_unreadable:index={index}:{exc}")
        if str(right.get("supervision_target_kind", "difix")).lower() != "difix":
            _fail(result, f"b_target_kind_invalid:index={index}")
        if right.get("target") == right.get("input"):
            _fail(result, f"b_target_is_self_render:index={index}")
        for field in (
            "input_sha256",
            "reference_image_sha256",
            "a0_checkpoint_sha256",
        ):
            if not right.get(field):
                _fail(result, f"b_missing_a0_provenance:index={index}:{field}")
        for field in ("input_sha256", "a0_checkpoint_sha256"):
            if left.get(field) != right.get(field):
                _fail(result, f"selfrender_b_a0_provenance_mismatch:index={index}:{field}")
        if left.get("reference_image_sha256") != right.get("reference_image_sha256"):
            _fail(result, f"selfrender_b_reference_provenance_mismatch:index={index}")

    try:
        pool_hash = pseudo_camera_pool_sha256(self_keys)
        result["pseudo_camera_pool_sha256"] = pool_hash
        if self_usage.get("pseudo_camera_keys") != self_keys:
            _fail(result, "self_render_usage_manifest_camera_keys_mismatch")
        if b_usage.get("pseudo_camera_keys") != b_keys:
            _fail(result, "b_usage_manifest_camera_keys_mismatch")
        if self_usage.get("pseudo_camera_pool_sha256") != pool_hash:
            _fail(result, "self_render_usage_manifest_pool_hash_mismatch")
        if b_usage.get("pseudo_camera_pool_sha256") != pool_hash:
            _fail(result, "b_usage_manifest_pool_hash_mismatch")
    except Exception as exc:
        _fail(result, f"pseudo_camera_pool_invalid:{exc}")

    self_trace = self_usage.get("pseudo_call_trace", [])
    b_trace = b_usage.get("pseudo_call_trace", [])
    result["same_pseudo_call_trace"] = self_trace == b_trace
    if not result["same_pseudo_call_trace"]:
        _fail(result, "selfrender_b_pseudo_call_trace_mismatch")
    result["pseudo_call_trace_sha256"] = pseudo_call_trace_sha256(self_trace)
    if self_usage.get("pseudo_call_trace_sha256") != result["pseudo_call_trace_sha256"]:
        _fail(result, "self_render_trace_hash_mismatch")
    if b_usage.get("pseudo_call_trace_sha256") != result["pseudo_call_trace_sha256"]:
        _fail(result, "b_trace_hash_mismatch")

    self_kind = str(self_meta.get("pseudo_supervision_target_kind", "")).lower()
    b_kind = str(b_meta.get("pseudo_supervision_target_kind", "")).lower()
    if self_kind != "self_render_a0" or b_kind != "difix":
        _fail(result, "controlled_pseudo_target_kinds_invalid")
    _attach_final_checkpoint_bindings(
        result,
        [
            ("self_render", self_render, self_meta),
            ("b", b_run, b_meta),
        ],
    )
    _seal_audited_inputs(result, self_render, b_run)
    result["only_declared_target_difference"] = bool(
        result["passed"] and self_kind != b_kind
    )
    result["interpretation"] = (
        "Valid SelfRender/B pseudo control; target generation is the declared controlled difference."
        if result["passed"]
        else "INVALID SelfRender/B pseudo control; do not attribute B-SelfRender differences to the Difix-target replacement."
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", "--a1", dest="baseline", type=Path, required=True)
    parser.add_argument("--treatment", "--b", dest="treatment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-pseudo", action="store_true")
    parser.add_argument("--expected-real-updates", type=int, default=2000)
    parser.add_argument("--expected-pseudo-updates", type=int, default=39)
    parser.add_argument("--expected-unique-pseudo-views", type=int, default=32)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--scene", required=True)
    args = parser.parse_args()
    result = audit(
        args.baseline,
        args.treatment,
        require_pseudo=args.require_pseudo,
        expected_real_updates=args.expected_real_updates,
        expected_pseudo_updates=args.expected_pseudo_updates,
        expected_unique_pseudo_views=args.expected_unique_pseudo_views,
        dataset=args.dataset,
        scene=args.scene,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
