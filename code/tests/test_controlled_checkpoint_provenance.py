from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest

from diffusion_guidance.control_identity import (
    PREREGISTERED_CONTROLLED_OPTIMIZATION,
    controlled_checkpoint_provenance_from_metadata,
    controlled_checkpoint_provenance_sha256,
    controlled_training_protocol_sha256,
    source_camera_set_sha256,
    source_image_inventory_sha256,
    training_camera_inventory_sha256,
    validate_a0_checkpoint_provenance,
    validate_preregistered_controlled_training_protocol,
)
from diffusion_guidance.result_binding import require_final_checkpoint_binding


def _protocol(role: str, root: Path) -> dict:
    iterations = 10000 if role == "A0" else 12000
    optimization = {
        **PREREGISTERED_CONTROLLED_OPTIMIZATION,
        "iterations": iterations,
        "experiment_seed": 1,
        "position_lr_init": 0.00016,
    }
    runtime = {
        **PREREGISTERED_CONTROLLED_OPTIMIZATION,
        "iterations": iterations,
        "source_path": str(root),
        "track_path": str(root / "tracks.h5"),
        "experiment_seed": 1,
        "start_checkpoint": None if role == "A0" else str(root / "chkpnt10000.pth"),
        "test_iterations": [iterations],
        "save_iterations": [iterations],
        "checkpoint_iterations": [iterations],
        "quiet": False,
    }
    return {
        "schema": "controlled_training_protocol_v1",
        "model": {
            "source_path": str(root),
            "track_path": str(root / "tracks.h5"),
            "images": "images",
            "strict_tracks": True,
            "strict_source_only_geometry": True,
        },
        "optimization": optimization,
        "pipeline": {"debug": False},
        "runtime": runtime,
    }


def _metadata(tmp_path: Path, role: str) -> dict:
    scene = (tmp_path / "fern").resolve()
    images_dir = scene / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    image = images_dir / "000.png"
    image.write_bytes(b"source")
    track = scene / "tracks.h5"
    track.write_bytes(b"track")
    names = ["000"]
    image_inventory = [
        {
            "camera_name": "000",
            "path": str(image),
            "sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
        }
    ]
    camera_inventory = [
        {
            "camera_name": "000",
            "camera": {
                "R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                "T": [0.0, 0.0, 0.0],
                "FoVx": 1.0,
                "FoVy": 1.0,
                "width": 8,
                "height": 6,
                "intrinsics": {
                    "fx": 4.0,
                    "fy": 4.0,
                    "cx": 3.5,
                    "cy": 2.5,
                    "width": 8,
                    "height": 6,
                },
            },
        }
    ]
    protocol = _protocol(role, scene)
    strict = {
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
    start_checkpoint = scene / "chkpnt10000.pth"
    if role != "A0":
        start_checkpoint.write_bytes(b"complete-a0-checkpoint")
    pseudo_manifest = scene / f"pseudo_{role.lower()}.jsonl"
    if role in {"SELFRENDER", "B"}:
        pseudo_manifest.write_text("{}\n", encoding="utf-8")
    metadata = {
        "role": role,
        "seed": 1,
        "final_iteration": 10000 if role == "A0" else 12000,
        "start_checkpoint": str(start_checkpoint) if role != "A0" else None,
        "start_checkpoint_sha256": (
            hashlib.sha256(start_checkpoint.read_bytes()).hexdigest()
            if role != "A0"
            else None
        ),
        "scene_source_path": str(scene),
        "track_h5_path": str(track),
        "track_h5_sha256": hashlib.sha256(track.read_bytes()).hexdigest(),
        "source_camera_names": names,
        "source_camera_set_sha256": source_camera_set_sha256(names),
        "source_images_dir": str(images_dir),
        "source_image_inventory": image_inventory,
        "source_image_inventory_sha256": source_image_inventory_sha256(
            image_inventory
        ),
        "source_training_camera_inventory": camera_inventory,
        "source_training_camera_inventory_sha256": training_camera_inventory_sha256(
            camera_inventory
        ),
        "controlled_training_protocol": protocol,
        "controlled_training_protocol_sha256": controlled_training_protocol_sha256(
            protocol
        ),
        "strict_geometry_protocol": strict,
        "pseudo_manifest_path": (
            str(pseudo_manifest) if role in {"SELFRENDER", "B"} else None
        ),
        "pseudo_manifest_sha256": (
            hashlib.sha256(pseudo_manifest.read_bytes()).hexdigest()
            if role in {"SELFRENDER", "B"}
            else None
        ),
        "pseudo_camera_pool_sha256": (
            "f" * 64 if role in {"SELFRENDER", "B"} else None
        ),
        "pseudo_target_kind": (
            "self_render_a0"
            if role == "SELFRENDER"
            else "difix" if role == "B" else None
        ),
        "pseudo_rgb_strict_cache": role in {"SELFRENDER", "B"},
    }
    if role == "A0":
        metadata["start_checkpoint_controlled_provenance_sha256"] = None
    else:
        a0_provenance = controlled_checkpoint_provenance_from_metadata(
            _metadata(tmp_path, "A0")
        )
        metadata["start_checkpoint_controlled_provenance_sha256"] = (
            controlled_checkpoint_provenance_sha256(a0_provenance)
        )
    return metadata


def test_preregistered_weights_cannot_be_consistently_changed(tmp_path: Path):
    protocol = _protocol("B", tmp_path.resolve())
    protocol["optimization"]["pseudo_rgb_weight"] = 0.3
    protocol["runtime"]["pseudo_rgb_weight"] = 0.3
    with pytest.raises(ValueError, match="preregistered"):
        validate_preregistered_controlled_training_protocol(protocol, role="B")


def test_a0_checkpoint_provenance_allows_only_phase_transition_fields(
    tmp_path: Path,
):
    a0 = controlled_checkpoint_provenance_from_metadata(_metadata(tmp_path, "A0"))
    continuation = controlled_checkpoint_provenance_from_metadata(
        _metadata(tmp_path, "A1")
    )
    assert validate_a0_checkpoint_provenance(a0, continuation) == a0
    assert len(controlled_checkpoint_provenance_sha256(a0)) == 64


def test_a0_checkpoint_provenance_rejects_changed_common_parameter(
    tmp_path: Path,
):
    a0 = controlled_checkpoint_provenance_from_metadata(_metadata(tmp_path, "A0"))
    changed = _metadata(tmp_path, "A1")
    changed["controlled_training_protocol"]["optimization"][
        "position_lr_init"
    ] = 0.0002
    changed["controlled_training_protocol"]["runtime"][
        "position_lr_init"
    ] = 0.0002
    changed["controlled_training_protocol_sha256"] = controlled_training_protocol_sha256(
        changed["controlled_training_protocol"]
    )
    continuation = controlled_checkpoint_provenance_from_metadata(changed)
    with pytest.raises(ValueError, match="authorized phase transition"):
        validate_a0_checkpoint_provenance(a0, continuation)


def test_final_checkpoint_binding_rejects_a_swapped_controlled_role(
    tmp_path: Path,
):
    a1_metadata = _metadata(tmp_path, "A1")
    a1_provenance = controlled_checkpoint_provenance_from_metadata(a1_metadata)
    checkpoint = tmp_path / "final_a1.pth"
    checkpoint.write_bytes(b"final-a1-checkpoint")
    summary = {
        "iteration": 12000,
        "format": "geotrack-research-v2",
        "gaussian_count": 17,
        "render_state_schema": "renderer-visible-state-v1",
        "render_state_sha256": "c" * 64,
        "controlled_provenance": a1_provenance,
        "controlled_provenance_sha256": controlled_checkpoint_provenance_sha256(
            a1_provenance
        ),
    }
    audit = {
        **a1_metadata,
        "start_checkpoint_path": a1_metadata["start_checkpoint"],
        "audited_methods": ["A1", "B"],
        "final_checkpoints": {
            "A1": {
                "path": str(checkpoint.resolve()),
                "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "iteration": summary["iteration"],
                "state_format": summary["format"],
                "gaussian_count": summary["gaussian_count"],
                "render_state_schema": summary["render_state_schema"],
                "render_state_sha256": summary["render_state_sha256"],
                "controlled_provenance": a1_provenance,
                "controlled_provenance_sha256": summary[
                    "controlled_provenance_sha256"
                ],
            }
        },
    }
    assert require_final_checkpoint_binding(
        summary,
        audit,
        method="A1",
        checkpoint_path=checkpoint,
    ) == a1_provenance

    b_metadata = _metadata(tmp_path, "B")
    b_provenance = controlled_checkpoint_provenance_from_metadata(b_metadata)
    summary["controlled_provenance"] = b_provenance
    summary["controlled_provenance_sha256"] = (
        controlled_checkpoint_provenance_sha256(b_provenance)
    )
    with pytest.raises(ValueError, match="differs from its pair audit"):
        require_final_checkpoint_binding(
            summary,
            audit,
            method="A1",
            checkpoint_path=checkpoint,
        )
