#!/usr/bin/env python3
"""P0 fail-closed equivalence gate for H5 and live-upstream camera projections.

Path A is H5 K[R|t]. Path B is the exact `project_points_pixel` path used by
StrictGeometryManager, including the live upstream world_view_transform.  The
script checks both native H5 resolution and the camera's training resolution.
It is a server integration gate, not a substitute for the CPU fake-camera test.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from diffusion_guidance.calibration_guard import resized_intrinsics
from diffusion_guidance.calibration_guard import calibrate_scene
from geometric_constraints.repaired_geometry import project_points_pixel
from geometric_constraints.strict_track_store import StrictTrackStore, normalize_image_name


def direct_projection(points: torch.Tensor, calibration: dict, width: int, height: int):
    rotation = torch.as_tensor(calibration["rotation_w2c"], device=points.device, dtype=points.dtype)
    translation = torch.as_tensor(calibration["translation_w2c"], device=points.device, dtype=points.dtype)
    camera = points @ rotation.transpose(0, 1) + translation
    depth = camera[:, 2]
    sx, sy = width / float(calibration["width"]), height / float(calibration["height"])
    fx, fy = float(calibration["fx"]) * sx, float(calibration["fy"]) * sy
    cx = (float(calibration["cx"]) + 0.5) * sx - 0.5
    cy = (float(calibration["cy"]) + 0.5) * sy - 0.5
    safe = torch.clamp(depth, min=torch.finfo(points.dtype).eps)
    return torch.stack([fx * camera[:, 0] / safe + cx, fy * camera[:, 1] / safe + cy], dim=-1), depth


def live_proxy(camera, width: int, height: int):
    if not hasattr(camera, "world_view_transform"):
        raise RuntimeError(f"Live camera {camera.image_name} lacks world_view_transform")
    if not hasattr(camera, "research_intrinsics"):
        raise RuntimeError(f"Live camera {camera.image_name} lacks research_intrinsics")
    return SimpleNamespace(
        image_width=width,
        image_height=height,
        world_view_transform=camera.world_view_transform,
        research_intrinsics=resized_intrinsics(
            camera.research_intrinsics, width, height
        ),
    )


def compare_camera(
    anchors,
    camera,
    calibration,
    tolerance: float,
    matrix_tolerance: float = 1e-6,
):
    name = normalize_image_name(camera.image_name)
    live_world_to_camera = camera.world_view_transform.transpose(0, 1).to(
        device=anchors.device, dtype=anchors.dtype
    )
    h5_rotation = torch.as_tensor(
        calibration["rotation_w2c"], device=anchors.device, dtype=anchors.dtype
    )
    h5_translation = torch.as_tensor(
        calibration["translation_w2c"], device=anchors.device, dtype=anchors.dtype
    )
    expected_homogeneous_row = torch.zeros(
        4, device=anchors.device, dtype=anchors.dtype
    )
    expected_homogeneous_row[3] = 1.0
    rotation_difference = float(
        torch.max(torch.abs(live_world_to_camera[:3, :3] - h5_rotation)).item()
    )
    translation_difference = float(
        torch.max(torch.abs(live_world_to_camera[:3, 3] - h5_translation)).item()
    )
    homogeneous_difference = float(
        torch.max(
            torch.abs(live_world_to_camera[3, :] - expected_homogeneous_row)
        ).item()
    )
    resolutions = [
        ("original", int(calibration["width"]), int(calibration["height"])),
        ("training", int(camera.image_width), int(camera.image_height)),
    ]
    records = []
    for label, width, height in resolutions:
        expected_intrinsics = resized_intrinsics(calibration, width, height)
        live_intrinsics = resized_intrinsics(
            camera.research_intrinsics, width, height
        )
        intrinsic_difference = max(
            abs(float(expected_intrinsics[key]) - float(live_intrinsics[key]))
            for key in ("fx", "fy", "cx", "cy")
        )
        direct, direct_depth = direct_projection(anchors, calibration, width, height)
        live, live_depth = project_points_pixel(anchors, live_proxy(camera, width, height), calibration)
        valid = (direct_depth > 1e-8) & (live_depth > 1e-8) & torch.isfinite(direct).all(dim=1) & torch.isfinite(live).all(dim=1)
        if not bool(valid.any()):
            raise RuntimeError(f"No common positive-depth H5 anchors for {name}/{label}")
        delta = torch.linalg.norm(direct[valid] - live[valid], dim=1).detach().cpu().numpy()
        record = {
            "camera_name": name,
            "resolution_kind": label,
            "resolution": [width, height],
            "valid_point_count": int(valid.sum().item()),
            "mean_pixel_difference": float(delta.mean()),
            "median_pixel_difference": float(np.median(delta)),
            "max_pixel_difference": float(delta.max()),
            "rotation_max_abs_difference": rotation_difference,
            "translation_max_abs_difference": translation_difference,
            "homogeneous_row_max_abs_difference": homogeneous_difference,
            "intrinsics_max_abs_difference": intrinsic_difference,
            "tolerance_px": tolerance,
            "matrix_tolerance": matrix_tolerance,
            "passed": bool(
                delta.max() <= tolerance
                and rotation_difference <= matrix_tolerance
                and translation_difference <= matrix_tolerance
                and homogeneous_difference <= matrix_tolerance
                and intrinsic_difference <= matrix_tolerance
            ),
            "path_a": "H5 world_to_camera: R @ X + t; row-vector implementation X @ R.T + t",
            "path_b": (
                "live upstream world_view_transform.T plus camera.research_intrinsics "
                "used by project_points_pixel"
            ),
        }
        records.append(record)
    return records


def run(args):
    from arguments import ModelParams
    from scene import GaussianModel, Scene
    from utils.general_utils import safe_state

    safe_state(args.quiet, args.seed)
    store = StrictTrackStore.load(args.track_h5)
    gaussians = GaussianModel(args)
    scene = Scene(args, gaussians, shuffle=False)
    # This is the same explicit calibration attachment used by training after
    # Scene construction.  Without it the gate could compare H5 against an
    # uncalibrated upstream camera rather than the live loss path.
    calibrate_scene(scene, args.track_h5)
    live = {normalize_image_name(camera.image_name): camera for camera in scene.getTrainCameras()}
    if set(live) != set(store.source_images):
        raise RuntimeError(f"Source camera mismatch: H5={sorted(store.source_images)}, Scene={sorted(live)}")
    anchors = torch.as_tensor(store.xyz, device=gaussians.get_xyz.device, dtype=gaussians.get_xyz.dtype)
    records = []
    for name in sorted(store.source_images):
        records.extend(
            compare_camera(
                anchors,
                live[name],
                store.camera_calibration(name),
                args.max_pixel_error,
                args.max_matrix_error,
            )
        )
    return {
        "gate": "P0_H5_vs_live_Scene_projection_equivalence",
        "convention": {
            "h5_rotation": "world_to_camera",
            "h5_translation": "world_to_camera translation in R @ X + t",
            "row_column_mapping": "row-vector tensors use X @ R.T + t",
            "live_path": "world_view_transform is transposed in project_points_pixel before row-vector projection",
        },
        "max_pixel_error_threshold": args.max_pixel_error,
        "max_matrix_error_threshold": args.max_matrix_error,
        "records": records,
        "passed": all(record["passed"] for record in records),
    }


def main():
    parser = argparse.ArgumentParser()
    ModelParams = None
    parser.add_argument("--track-h5", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-pixel-error", type=float, default=1e-4)
    parser.add_argument("--max-matrix-error", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--quiet", action="store_true")
    # Scene consumes these run-stage dataset selectors directly. They are not
    # part of upstream ModelParams, so every Scene-based gate must declare them.
    parser.add_argument("--data_type", default="colmap", choices=["colmap", "360"])
    parser.add_argument("--llff_holdout", type=int, default=8)
    known, _ = parser.parse_known_args()
    if known.max_pixel_error <= 0 or known.max_pixel_error > 1e-3:
        parser.error("--max-pixel-error must be a precommitted positive numerical tolerance <= 1e-3 px")
    if known.max_matrix_error <= 0 or known.max_matrix_error > 1e-5:
        parser.error("--max-matrix-error must be a precommitted positive tolerance <= 1e-5")
    from arguments import ModelParams as UpstreamModelParams, PipelineParams
    UpstreamModelParams(parser)
    PipelineParams(parser)
    parser.add_argument("--train_bg", action="store_true")
    args = parser.parse_args()
    if not args.strict_source_only_geometry:
        parser.error("P0 gate requires --strict_source_only_geometry")
    if Path(args.track_path).expanduser().resolve() != args.track_h5.expanduser().resolve():
        parser.error(
            "--track_path and --track-h5 must resolve to the same immutable H5; "
            "Scene calibration and projection audit cannot use different files"
        )
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit("P0 projection equivalence failed; STOP before geometry smoke/A0")


if __name__ == "__main__":
    main()
