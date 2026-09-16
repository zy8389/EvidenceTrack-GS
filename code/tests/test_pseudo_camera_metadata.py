from __future__ import annotations

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest

from diffusion_guidance.calibration_guard import (
    PSEUDO_CAMERA_SOURCE,
    PSEUDO_POSE_GENERATOR,
    intrinsic_projection_matrix,
)
from diffusion_guidance.camera_utils import pseudo_camera_manifest_fields
from tools.audit_pseudo_camera_manifest import audit_records


def make_camera(index: int = 0):
    camera = SimpleNamespace(
        uid=f"pseudo_{index:05d}",
        R=torch.eye(3, dtype=torch.float64).numpy(),
        T=torch.tensor([0.1 * index, 0.0, 0.0], dtype=torch.float64).numpy(),
        FoVx=1.0,
        FoVy=0.8,
        image_width=80,
        image_height=60,
        znear=0.01,
        zfar=100.0,
        world_view_transform=torch.eye(4, dtype=torch.float64),
        camera_source=PSEUDO_CAMERA_SOURCE,
        calibration_source_camera="source_000",
        pose_source_camera_names=["source_000", "source_001"],
        pose_uses_heldout_camera_information=False,
        pose_generator=PSEUDO_POSE_GENERATOR,
        research_intrinsics={
            "fx": 50.0,
            "fy": 49.0,
            "cx": 39.5,
            "cy": 29.5,
            "width": 80,
            "height": 60,
        },
    )
    camera.projection_matrix = intrinsic_projection_matrix(
        camera.research_intrinsics,
        camera.znear,
        camera.zfar,
        dtype=torch.float64,
    ).T.contiguous()
    camera.full_proj_transform = camera.world_view_transform @ camera.projection_matrix
    return camera


def manifest_record(camera, index=0):
    record = pseudo_camera_manifest_fields(camera)
    record.update(
        {
            "index": index,
            "input": "input.png",
            "target": "target.png",
            "reference_image": "reference.png",
        }
    )
    return record


def test_manifest_contains_explicit_pose_intrinsics_and_fingerprint():
    record = manifest_record(make_camera())
    result = audit_records([record], origin="fixture", require_assets=True)
    assert result["passed"] is True
    for key in (
        "camera_id",
        "camera_source",
        "fx",
        "fy",
        "cx",
        "cy",
        "width",
        "height",
        "R",
        "t",
        "camera_fingerprint",
    ):
        assert key in record


def test_flat_camera_metadata_tampering_fails_closed():
    record = manifest_record(make_camera())
    record["cx"] += 1.0
    with pytest.raises(ValueError, match="Flat/nested"):
        audit_records([record], origin="fixture", require_assets=True)


def test_heldout_pose_claim_cannot_be_labeled_source_only():
    record = manifest_record(make_camera())
    record["uses_heldout_pose_information"] = True
    with pytest.raises(ValueError, match="source-only"):
        audit_records([record], origin="fixture", require_assets=True)
