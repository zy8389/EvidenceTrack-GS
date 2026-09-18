#!/usr/bin/env python3
"""Fail-closed provenance audits for exported and live pseudo cameras."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image

from diffusion_guidance.calibration_guard import (
    PSEUDO_CAMERA_SOURCE,
    intrinsic_projection_matrix,
)
from diffusion_guidance.camera_utils import (
    camera_fingerprint_from_payload,
    canonical_camera_payload,
    pseudo_camera_manifest_fields,
)
from diffusion_guidance.control_identity import (
    normalized_camera_names,
    source_camera_set_sha256,
)
from diffusion_guidance.checkpoint_state import (
    RENDER_STATE_SCHEMA,
    load_checkpoint_summary,
)
from diffusion_guidance.difix_provenance import A0_PSEUDO_MANIFEST_SCHEMA
from diffusion_guidance.pseudo_manifest import (
    load_pseudo_manifest,
    validate_a0_source_provenance,
)
from geometric_constraints.strict_track_store import normalize_image_name


CAMERA_REQUIRED = (
    "index",
    "key",
    "camera_fingerprint",
    "camera",
    "camera_id",
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
ASSET_REQUIRED = (
    "manifest_schema",
    "input",
    "target",
    "reference_image",
    "input_sha256",
    "reference_image_sha256",
    "a0_checkpoint",
    "a0_checkpoint_sha256",
    "live_pseudo_audit",
    "live_pseudo_audit_sha256",
    "projection_context_fingerprint",
    "a0_checkpoint_iteration",
    "a0_checkpoint_state_format",
    "a0_checkpoint_gaussian_count",
    "a0_checkpoint_render_state_schema",
    "a0_checkpoint_render_state_sha256",
    "a0_checkpoint_controlled_provenance_sha256",
    "a0_render_state_source",
    "track_h5",
    "track_h5_sha256",
    "source_camera_names",
    "source_camera_set_sha256",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_asset(manifest: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (manifest.parent / path).resolve()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def _same_numeric(left: Any, right: Any, *, atol: float = 1e-7) -> bool:
    try:
        return bool(
            np.allclose(
                np.asarray(left, dtype=np.float64),
                np.asarray(right, dtype=np.float64),
                rtol=0.0,
                atol=atol,
            )
        )
    except (TypeError, ValueError):
        return False


def audit_records(
    raw_records: Iterable[dict[str, Any]],
    *,
    origin: str,
    require_assets: bool,
) -> dict[str, Any]:
    records = []
    keys: set[str] = set()
    camera_ids: set[str] = set()
    # ``audit_records`` is also a pure CPU metadata helper.  Byte-level asset
    # checks live in ``audit(path)`` because only it has a manifest base path.
    # The public file audit still requires every immutable asset field.
    expected_fields = CAMERA_REQUIRED + (
        ("input", "target", "reference_image") if require_assets else ()
    )
    for line_number, raw in enumerate(raw_records, start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"Pseudo record is not an object at {origin}:{line_number}")
        missing = [key for key in expected_fields if key not in raw]
        if missing:
            raise ValueError(
                f"Missing pseudo provenance fields at {origin}:{line_number}: {missing}"
            )
        camera = canonical_camera_payload(raw["camera"])
        if "intrinsics" not in camera:
            raise ValueError(
                f"Pseudo camera lacks explicit fx/fy/cx/cy at {origin}:{line_number}"
            )
        fingerprint = camera_fingerprint_from_payload(camera)
        if raw["key"] != fingerprint or raw["camera_fingerprint"] != fingerprint:
            raise ValueError(f"Camera fingerprint mismatch at {origin}:{line_number}")
        if fingerprint in keys:
            raise ValueError(
                f"Duplicate pseudo camera fingerprint at {origin}:{line_number}"
            )
        keys.add(fingerprint)
        camera_id = str(raw["camera_id"])
        if not camera_id or camera_id in camera_ids:
            raise ValueError(f"Invalid or duplicate camera_id at {origin}:{line_number}")
        camera_ids.add(camera_id)

        flat_pairs = {
            "fx": camera["intrinsics"]["fx"],
            "fy": camera["intrinsics"]["fy"],
            "cx": camera["intrinsics"]["cx"],
            "cy": camera["intrinsics"]["cy"],
            "width": camera["width"],
            "height": camera["height"],
            "R": camera["R"],
            "t": camera["T"],
        }
        mismatched = [
            name
            for name, expected in flat_pairs.items()
            if not _same_numeric(raw[name], expected)
        ]
        if mismatched:
            raise ValueError(
                f"Flat/nested camera metadata mismatch at {origin}:{line_number}: "
                f"{mismatched}"
            )

        pose_sources = [
            normalize_image_name(str(name))
            for name in raw["pose_source_camera_names"]
        ]
        if not pose_sources or len(pose_sources) != len(set(pose_sources)):
            raise ValueError(f"Invalid pose source camera list at {origin}:{line_number}")
        calibration_source = normalize_image_name(
            str(raw["calibration_source_camera"])
        )
        if calibration_source not in pose_sources:
            raise ValueError(
                f"Calibration source is absent from pose source set at {origin}:{line_number}"
            )
        source = str(raw["camera_source"])
        uses_heldout = raw["uses_heldout_pose_information"]
        if not isinstance(uses_heldout, bool):
            raise ValueError(
                f"uses_heldout_pose_information must be boolean at {origin}:{line_number}"
            )
        if source == PSEUDO_CAMERA_SOURCE and uses_heldout:
            raise ValueError(
                f"Contradictory source-only/held-out declaration at {origin}:{line_number}"
            )
        if not str(raw["pose_generator"]) or not str(raw["pose_convention"]):
            raise ValueError(f"Incomplete pose provenance at {origin}:{line_number}")

        records.append(
            {
                "index": int(raw["index"]),
                "camera_id": camera_id,
                "camera_source": source,
                "camera_fingerprint": fingerprint,
                "calibration_source_camera": calibration_source,
                "pose_source_camera_names": pose_sources,
                "uses_heldout_pose_information": uses_heldout,
                "pose_generator": str(raw["pose_generator"]),
                "pose_convention": str(raw["pose_convention"]),
                **flat_pairs,
            }
        )
    if not records:
        raise ValueError("Pseudo camera collection is empty")
    if [record["index"] for record in records] != list(range(len(records))):
        raise ValueError("Pseudo camera indices must be contiguous and ordered from zero")

    sources = sorted({record["camera_source"] for record in records})
    uses_heldout = any(
        record["uses_heldout_pose_information"] for record in records
    )
    source_only_pose = (
        sources == [PSEUDO_CAMERA_SOURCE]
        and not uses_heldout
        and all(
            record["pose_source_camera_names"]
            == records[0]["pose_source_camera_names"]
            for record in records
        )
    )
    return {
        "origin": origin,
        "record_count": len(records),
        "camera_sources": sources,
        "source_only_pose_claim": source_only_pose,
        "uses_heldout_pose_information": uses_heldout,
        "claim_boundary": (
            "pseudo poses declared source-camera-generated; pseudo views are not independent physical observations"
            if source_only_pose
            else "source-RGB-only geometry conditional on supplied/full calibration; pseudo trajectory is not source-only"
        ),
        "records": records,
        "passed": source_only_pose,
    }


def audit(path: Path) -> dict[str, Any]:
    raw_records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw_records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
    result = audit_records(
        raw_records, origin=str(path.resolve()), require_assets=True
    )
    for line_number, raw in enumerate(raw_records, start=1):
        missing = [field for field in ASSET_REQUIRED if not raw.get(field)]
        if missing:
            raise ValueError(
                f"Missing immutable pseudo asset provenance at {path}:{line_number}: {missing}"
            )
    a0_hashes: set[str] = set()
    a0_paths: set[str] = set()
    a0_provenance_hashes: set[str] = set()
    live_audit_hashes: set[str] = set()
    live_projection_fingerprints: set[str] = set()
    track_hashes: set[str] = set()
    track_paths: set[str] = set()
    source_camera_hashes: set[str] = set()
    source_camera_sets: set[tuple[str, ...]] = set()
    for line_number, raw in enumerate(raw_records, start=1):
        if raw["manifest_schema"] != A0_PSEUDO_MANIFEST_SCHEMA:
            raise ValueError(
                f"Pseudo supervision schema mismatch at {path}:{line_number}: "
                f"expected={A0_PSEUDO_MANIFEST_SCHEMA!r}, "
                f"observed={raw['manifest_schema']!r}"
            )
        input_path = _resolve_asset(path, str(raw["input"]))
        reference_path = _resolve_asset(path, str(raw["reference_image"]))
        checkpoint_path = _resolve_asset(path, str(raw["a0_checkpoint"]))
        live_audit_path = _resolve_asset(path, str(raw["live_pseudo_audit"]))
        track_path = _resolve_asset(path, str(raw["track_h5"]))
        for label, asset in (
            ("A0 pseudo input", input_path),
            ("source reference", reference_path),
            ("A0 checkpoint", checkpoint_path),
            ("live pseudo-camera audit", live_audit_path),
            ("strict Track H5", track_path),
        ):
            if not asset.is_file():
                raise FileNotFoundError(
                    f"Missing {label} at {path}:{line_number}: {asset}"
                )
        if raw["input_sha256"] != _sha256_file(input_path):
            raise ValueError(f"A0 pseudo input hash mismatch at {path}:{line_number}")
        if raw["reference_image_sha256"] != _sha256_file(reference_path):
            raise ValueError(f"Pseudo source reference hash mismatch at {path}:{line_number}")
        track_sha256 = _sha256_file(track_path)
        if raw["track_h5_sha256"] != track_sha256:
            raise ValueError(f"Pseudo Track H5 hash mismatch at {path}:{line_number}")
        source_names_raw = raw["source_camera_names"]
        if not isinstance(source_names_raw, list):
            raise ValueError(
                f"Pseudo source_camera_names must be a list at {path}:{line_number}"
            )
        source_names = normalized_camera_names(source_names_raw)
        if source_names_raw != source_names:
            raise ValueError(
                f"Pseudo source_camera_names are not canonical at {path}:{line_number}"
            )
        source_hash = source_camera_set_sha256(source_names)
        if raw["source_camera_set_sha256"] != source_hash:
            raise ValueError(
                f"Pseudo source-camera set hash mismatch at {path}:{line_number}"
            )
        checkpoint_sha256 = _sha256_file(checkpoint_path)
        if raw["a0_checkpoint_sha256"] != checkpoint_sha256:
            raise ValueError(f"A0 checkpoint hash mismatch at {path}:{line_number}")
        checkpoint_summary = load_checkpoint_summary(
            checkpoint_path,
            expected_iteration=10000,
            require_cuda_rng=True,
            require_controlled_provenance=True,
        )
        if (
            int(raw["a0_checkpoint_iteration"]) != 10000
            or raw["a0_checkpoint_state_format"] != "geotrack-research-v2"
        ):
            raise ValueError(f"Invalid A0 checkpoint iteration state at {path}:{line_number}")
        if int(raw["a0_checkpoint_gaussian_count"]) <= 0:
            raise ValueError(f"Invalid A0 Gaussian-count provenance at {path}:{line_number}")
        if raw["a0_checkpoint_render_state_schema"] != RENDER_STATE_SCHEMA:
            raise ValueError(f"Invalid A0 render-state schema at {path}:{line_number}")
        if not _is_sha256(raw["a0_checkpoint_render_state_sha256"]):
            raise ValueError(f"Invalid A0 render-state hash at {path}:{line_number}")
        if int(raw["a0_checkpoint_gaussian_count"]) != checkpoint_summary[
            "gaussian_count"
        ]:
            raise ValueError(f"A0 Gaussian-count mismatch at {path}:{line_number}")
        if raw["a0_checkpoint_render_state_sha256"] != checkpoint_summary[
            "render_state_sha256"
        ]:
            raise ValueError(f"A0 render-state tensor hash mismatch at {path}:{line_number}")
        if (
            not _is_sha256(raw["a0_checkpoint_controlled_provenance_sha256"])
            or raw["a0_checkpoint_controlled_provenance_sha256"]
            != checkpoint_summary["controlled_provenance_sha256"]
        ):
            raise ValueError(
                f"A0 controlled-provenance hash mismatch at {path}:{line_number}"
            )
        if (
            raw["a0_render_state_source"]
            != "complete_checkpoint_restore_after_scene_ply_load"
        ):
            raise ValueError(f"A0 render-state provenance mismatch at {path}:{line_number}")
        live_audit_sha256 = _sha256_file(live_audit_path)
        if raw["live_pseudo_audit_sha256"] != live_audit_sha256:
            raise ValueError(f"Live pseudo-camera audit hash mismatch at {path}:{line_number}")
        live_audit = json.loads(live_audit_path.read_text(encoding="utf-8"))
        if live_audit.get("passed") is not True:
            raise ValueError(f"Live pseudo-camera audit is not passed at {path}:{line_number}")
        live_track = Path(str(live_audit.get("track_h5", ""))).expanduser().resolve()
        if (
            live_track != track_path
            or live_audit.get("track_h5_sha256") != track_sha256
        ):
            raise ValueError(
                f"Live pseudo-camera audit binds another Track H5 at {path}:{line_number}"
            )
        live_source_names = live_audit.get("source_camera_names")
        if (
            not isinstance(live_source_names, list)
            or source_camera_set_sha256(live_source_names) != source_hash
        ):
            raise ValueError(
                f"Live pseudo-camera audit binds another source-camera set at "
                f"{path}:{line_number}"
            )
        if live_audit.get("projection_context_fingerprint") != raw[
            "projection_context_fingerprint"
        ]:
            raise ValueError(
                f"Live pseudo-camera projection context mismatch at {path}:{line_number}"
            )
        a0_hashes.add(checkpoint_sha256)
        a0_paths.add(str(checkpoint_path))
        a0_provenance_hashes.add(
            str(raw["a0_checkpoint_controlled_provenance_sha256"])
        )
        live_audit_hashes.add(live_audit_sha256)
        live_projection_fingerprints.add(str(raw["projection_context_fingerprint"]))
        track_hashes.add(track_sha256)
        track_paths.add(str(track_path))
        source_camera_hashes.add(source_hash)
        source_camera_sets.add(tuple(source_names))
        with Image.open(input_path) as image:
            input_size = tuple(map(int, image.size))
        with Image.open(reference_path) as image:
            reference_size = tuple(map(int, image.size))
        camera_size = (int(raw["width"]), int(raw["height"]))
        if input_size != camera_size:
            raise ValueError(
                f"A0 pseudo input resolution mismatch at {path}:{line_number}: "
                f"input={input_size}, camera={camera_size}"
            )
        if reference_size[0] <= 0 or reference_size[1] <= 0:
            raise ValueError(f"Invalid pseudo source reference resolution at {path}:{line_number}")
    if len(a0_hashes) != 1:
        raise ValueError("Pseudo manifest mixes A0 checkpoint hashes")
    if len(a0_paths) != 1:
        raise ValueError("Pseudo manifest mixes A0 checkpoint paths")
    if len(a0_provenance_hashes) != 1:
        raise ValueError("Pseudo manifest mixes A0 checkpoint provenance hashes")
    if len(live_audit_hashes) != 1 or len(live_projection_fingerprints) != 1:
        raise ValueError("Pseudo manifest mixes live pseudo-camera audit contexts")
    if len(track_hashes) != 1 or len(track_paths) != 1:
        raise ValueError("Pseudo manifest mixes strict Track H5 inputs")
    if len(source_camera_hashes) != 1 or len(source_camera_sets) != 1:
        raise ValueError("Pseudo manifest mixes source-camera sets")
    result["manifest"] = str(path.resolve())
    result["manifest_schema"] = A0_PSEUDO_MANIFEST_SCHEMA
    result["a0_checkpoint_sha256"] = next(iter(a0_hashes))
    result["a0_checkpoint"] = next(iter(a0_paths))
    result["a0_checkpoint_controlled_provenance_sha256"] = next(
        iter(a0_provenance_hashes)
    )
    result["live_pseudo_audit_sha256"] = next(iter(live_audit_hashes))
    result["projection_context_fingerprint"] = next(
        iter(live_projection_fingerprints)
    )
    result["track_h5"] = next(iter(track_paths))
    result["track_h5_sha256"] = next(iter(track_hashes))
    result["source_camera_names"] = list(next(iter(source_camera_sets)))
    result["source_camera_set_sha256"] = next(iter(source_camera_hashes))
    # Reuse the exact strict-loader validation used by B/SelfRender training so
    # the post-A0 audit and actual consumption cannot drift.
    strict_records = load_pseudo_manifest(path, strict_targets=False)
    if len(strict_records) != len(raw_records):
        raise ValueError("Pseudo manifest loader record count disagrees with file audit")
    for record in strict_records:
        validate_a0_source_provenance(record)
    return result


def audit_live_scene(scene) -> dict[str, Any]:
    source_names = [
        normalize_image_name(camera.image_name)
        for camera in scene.getTrainCameras()
    ]
    heldout_names = [
        normalize_image_name(camera.image_name)
        for camera in scene.getTestCameras()
    ]
    if not source_names or len(source_names) != len(set(source_names)):
        raise ValueError("Live Scene source camera set is empty or contains duplicates")
    if set(source_names) & set(heldout_names):
        raise ValueError("Live Scene source and held-out camera sets overlap")
    if list(getattr(scene, "pseudo_camera_pose_source_names", [])) != source_names:
        raise ValueError("Live Scene pseudo pose-source declaration differs from train cameras")
    if list(getattr(scene, "pseudo_camera_heldout_pose_source_names", [])):
        raise ValueError("Live Scene declares held-out pseudo pose sources")
    if bool(getattr(scene, "pseudo_camera_uses_heldout_pose_information", True)):
        raise ValueError("Live Scene pseudo pose provenance is not source-only")

    raw_records = []
    for index, camera in enumerate(scene.getPseudoCameras()):
        fields = pseudo_camera_manifest_fields(camera)
        fields["index"] = index
        expected_projection = intrinsic_projection_matrix(
            camera.research_intrinsics,
            camera.znear,
            camera.zfar,
            device=camera.projection_matrix.device,
            dtype=camera.projection_matrix.dtype,
        ).T.contiguous()
        if not torch.allclose(
            camera.projection_matrix, expected_projection, rtol=0.0, atol=1e-7
        ):
            raise ValueError(f"Live pseudo projection matrix mismatch at index {index}")
        expected_full = camera.world_view_transform @ camera.projection_matrix
        if not torch.allclose(
            camera.full_proj_transform, expected_full, rtol=0.0, atol=1e-7
        ):
            raise ValueError(f"Live pseudo full projection mismatch at index {index}")
        raw_records.append(fields)

    result = audit_records(
        raw_records, origin="live_scene.getPseudoCameras", require_assets=False
    )
    result.update(
        {
            "gate": "P0_live_pseudo_camera_pose_intrinsics_provenance",
            "source_camera_names": source_names,
            "heldout_camera_names": heldout_names,
            "declared_pose_source_camera_names": list(
                scene.pseudo_camera_pose_source_names
            ),
            "declared_heldout_pose_source_camera_names": list(
                scene.pseudo_camera_heldout_pose_source_names
            ),
            "pose_generator": str(scene.pseudo_camera_pose_generator),
        }
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.manifest.expanduser().resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
