#!/usr/bin/env python3
"""Functional recovery test for real Track-associated Gaussian geometry.

The checkpoint is loaded read-only.  In-memory XYZ values are always restored
in a finally block and no checkpoint or formal training result is written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from geometric_constraints.repaired_geometry import (
    StrictGeometryManager,
    project_points_pixel,
)
from diffusion_guidance.checkpoint_state import validate_complete_state
from diffusion_guidance.camera_utils import camera_payload
from diffusion_guidance.control_identity import (
    normalized_camera_names,
    source_camera_set_sha256,
    validate_controlled_checkpoint_provenance_inputs,
)


REAL_RECOVERY_SCHEMA = "real_gaussian_geometry_recovery_v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mean_anchor_distance(manager, xyz, track_indices):
    tracks = torch.as_tensor(track_indices, device=xyz.device, dtype=torch.long)
    gaussian_ids = manager.associated_gaussian_ids.index_select(0, tracks)
    return torch.linalg.norm(
        xyz.index_select(0, gaussian_ids) - manager.anchors.index_select(0, tracks),
        dim=-1,
    )


def _unique_best_tracks(manager, count: int):
    order = sorted(
        manager.active_track_indices.tolist(),
        key=lambda index: float(manager.store.quality[index]),
        reverse=True,
    )
    chosen, used_gaussians = [], set()
    for track_index in order:
        gaussian_id = int(manager.associated_gaussian_ids[track_index].item())
        if gaussian_id < 0 or gaussian_id in used_gaussians:
            continue
        used_gaussians.add(gaussian_id)
        chosen.append(track_index)
        if len(chosen) >= count:
            break
    if not chosen:
        raise RuntimeError("No uniquely associated active tracks are available")
    return chosen


def run_real(args) -> dict:
    from arguments import ModelParams, OptimizationParams
    from scene import GaussianModel, Scene
    from utils.general_utils import safe_state

    ratios = tuple(float(value) for value in args.perturbation_ratios)
    if ratios != (0.005, 0.01, 0.02):
        raise ValueError(
            "The preregistered recovery gate requires perturbation ratios "
            "0.005, 0.01, and 0.02 in that order"
        )
    if int(args.steps) != 100:
        raise ValueError("The preregistered real recovery gate requires exactly 100 steps")
    if int(args.track_count) != 32:
        raise ValueError("The preregistered real recovery gate requires exactly 32 tracks")
    if float(args.minimum_reduction) != 0.10:
        raise ValueError(
            "The preregistered real recovery gate requires minimum_reduction=0.10"
        )
    safe_state(args.quiet, args.experiment_seed)
    gaussians = GaussianModel(args)
    scene = Scene(args, gaussians, shuffle=False)
    from diffusion_guidance.calibration_guard import calibrate_scene
    calibrate_scene(scene, args.track_path)
    optimization = OptimizationParams(argparse.ArgumentParser()).extract(args)
    gaussians.training_setup(optimization)
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    checkpoint_sha256_before = _sha256_file(checkpoint_path)
    loaded = torch.load(checkpoint_path, weights_only=False)
    if not isinstance(loaded, tuple) or len(loaded) != 2:
        raise ValueError("Recovery requires exactly (complete_state, iteration)")
    model_params, checkpoint_iteration = loaded
    if isinstance(checkpoint_iteration, bool) or int(checkpoint_iteration) != 10000:
        raise ValueError("Recovery requires the complete A0 checkpoint at iteration 10000")
    checkpoint_summary = validate_complete_state(
        model_params,
        require_cuda_rng=True,
        require_controlled_provenance=True,
    )
    source_cameras = list(scene.getTrainCameras())
    source_camera_names = normalized_camera_names(
        [camera.image_name for camera in source_cameras]
    )
    validate_controlled_checkpoint_provenance_inputs(
        checkpoint_summary["controlled_provenance"],
        scene_source_path=args.source_path,
        track_h5_path=args.track_path,
        source_images_dir=(
            Path(args.source_path).expanduser().resolve()
            / (args.images if getattr(args, "images", None) else "images")
        ),
        source_camera_names=source_camera_names,
        source_training_camera_inventory=[
            {"camera_name": camera.image_name, "camera": camera_payload(camera)}
            for camera in source_cameras
        ],
        seed=int(args.experiment_seed),
    )
    gaussians.restore(model_params, optimization)
    if not bool(getattr(gaussians, "optimizer_state_restored", False)):
        raise RuntimeError("Recovery checkpoint did not restore optimizer state")
    if not bool(getattr(gaussians, "densification_state_restored", False)):
        raise RuntimeError("Recovery checkpoint did not restore densification state")
    manager = StrictGeometryManager.from_path(
        args.track_path,
        scene.getTrainCameras(),
        device=gaussians.get_xyz.device,
        dtype=gaussians.get_xyz.dtype,
        min_length=args.strict_track_min_length,
        min_quality=args.strict_track_min_quality,
        max_tracks=args.strict_track_max_tracks,
        huber_delta=args.strict_track_huber_delta,
        association_chunk_size=args.strict_track_association_chunk_size,
        max_association_distance_ratio=args.strict_track_max_association_distance_ratio,
        collision_warning_rate=args.strict_track_collision_warning_rate,
    )
    manager.associate(gaussians.get_xyz, scene.cameras_extent)
    selected_tracks = _unique_best_tracks(manager, args.track_count)
    if len(selected_tracks) != int(args.track_count):
        raise RuntimeError(
            "Real recovery requires 32 uniquely associated active tracks; "
            f"only {len(selected_tracks)} are available"
        )
    selected_gaussians = torch.as_tensor(
        [int(manager.associated_gaussian_ids[index]) for index in selected_tracks],
        device=gaussians.get_xyz.device,
        dtype=torch.long,
    )
    original_xyz = gaussians.get_xyz.detach().clone()
    direction = torch.tensor(
        [1.0, -0.5, 0.25],
        device=gaussians.get_xyz.device,
        dtype=gaussians.get_xyz.dtype,
    )
    direction = direction / direction.norm()
    results = []
    try:
        for ratio in args.perturbation_ratios:
            with torch.no_grad():
                gaussians.get_xyz.copy_(original_xyz)
                gaussians.get_xyz[selected_gaussians] += (
                    float(ratio) * scene.cameras_extent * direction
                )
            initial = manager.compute_loss(
                gaussians.get_xyz, track_indices=selected_tracks
            )
            initial_3d_per_track = _mean_anchor_distance(
                manager, gaussians.get_xyz, selected_tracks
            ).detach()
            optimizer = torch.optim.Adam(
                [gaussians._xyz],
                lr=max(float(ratio) * scene.cameras_extent / 10.0, 1e-7),
            )
            for _ in range(args.steps):
                optimizer.zero_grad(set_to_none=True)
                loss = manager.compute_loss(
                    gaussians.get_xyz, track_indices=selected_tracks
                )["loss"]
                if not torch.isfinite(loss):
                    raise RuntimeError("Geometry recovery produced a non-finite loss")
                loss.backward()
                optimizer.step()
            final = manager.compute_loss(
                gaussians.get_xyz, track_indices=selected_tracks
            )
            final_3d_per_track = _mean_anchor_distance(
                manager, gaussians.get_xyz, selected_tracks
            ).detach()
            initial_error = float(initial["mean_error"].detach().item())
            final_error = float(final["mean_error"].detach().item())
            initial_3d = float(initial_3d_per_track.mean().item())
            final_3d = float(final_3d_per_track.mean().item())
            convergence_rate = float(
                (final_3d_per_track < initial_3d_per_track).float().mean().item()
            )
            finite = all(
                math.isfinite(value)
                for value in (initial_error, final_error, initial_3d, final_3d)
            )
            gradient_norm = float(
                gaussians.get_xyz.grad.index_select(0, selected_gaussians)
                .norm(dim=-1).mean().item()
            ) if gaussians.get_xyz.grad is not None else 0.0
            passed = (
                finite
                and gradient_norm > 0.0
                and final_error <= initial_error * (1.0 - args.minimum_reduction)
                and final_3d <= initial_3d * (1.0 - args.minimum_reduction)
                and convergence_rate >= 0.8
            )
            results.append(
                {
                    "perturbation_scene_radius_ratio": float(ratio),
                    "initial_pixel_reprojection_error": initial_error,
                    "final_pixel_reprojection_error": final_error,
                    "initial_3d_anchor_distance": initial_3d,
                    "final_3d_anchor_distance": final_3d,
                    "convergence_rate": convergence_rate,
                    "nan_free": finite,
                    "valid_associated_gaussian_count": len(selected_tracks),
                    "unique_gaussian_count": len(torch.unique(selected_gaussians)),
                    "mean_gradient_norm": gradient_norm,
                    "passed": passed,
                }
            )
    finally:
        with torch.no_grad():
            gaussians.get_xyz.copy_(original_xyz)
        if gaussians.get_xyz.grad is not None:
            gaussians.get_xyz.grad.zero_()
    checkpoint_sha256_after = _sha256_file(checkpoint_path)
    report = {
        "schema": REAL_RECOVERY_SCHEMA,
        "gate": "P0_real_gaussian_geometry_recovery",
        "mode": "real",
        "scene_source_path": str(Path(args.source_path).expanduser().resolve()),
        "experiment_seed": int(args.experiment_seed),
        "optimization_steps": int(args.steps),
        "minimum_reduction": float(args.minimum_reduction),
        "requested_track_count": int(args.track_count),
        "perturbation_ratios": list(ratios),
        "camera_extent": float(scene.cameras_extent),
        "source_camera_names": source_camera_names,
        "source_camera_set_sha256": source_camera_set_sha256(source_camera_names),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256_before,
        "checkpoint_iteration": int(checkpoint_iteration),
        "checkpoint_state_format": checkpoint_summary["format"],
        "checkpoint_gaussian_count": checkpoint_summary["gaussian_count"],
        "checkpoint_render_state_schema": checkpoint_summary[
            "render_state_schema"
        ],
        "checkpoint_render_state_sha256": checkpoint_summary[
            "render_state_sha256"
        ],
        "checkpoint_controlled_provenance_sha256": checkpoint_summary[
            "controlled_provenance_sha256"
        ],
        "checkpoint_modified": checkpoint_sha256_after != checkpoint_sha256_before,
        "checkpoint_sha256_after": checkpoint_sha256_after,
        "track_h5": str(Path(args.track_path).expanduser().resolve()),
        "track_h5_sha256": _sha256_file(
            Path(args.track_path).expanduser().resolve()
        ),
        "results": results,
        "recovery_success_rate": float(
            sum(item["passed"] for item in results) / len(results)
        ) if results else 0.0,
        "passed": (
            bool(results)
            and all(item["passed"] for item in results)
            and checkpoint_sha256_after == checkpoint_sha256_before
        ),
    }
    return report


def run_synthetic(steps: int = 100) -> dict:
    anchor = torch.tensor([[0.1, -0.05, 2.5]], dtype=torch.float64)
    xyz = torch.nn.Parameter(anchor + torch.tensor([[0.04, -0.03, 0.02]], dtype=torch.float64))
    calibrations = []
    cameras = []
    for center_x in (-0.4, 0.0, 0.4):
        rotation = np.eye(3)
        translation = np.asarray([-center_x, 0.0, 0.0])
        calibration = {
            "width": 640,
            "height": 480,
            "fx": 500.0,
            "fy": 500.0,
            "cx": 320.0,
            "cy": 240.0,
            "rotation_w2c": rotation,
            "translation_w2c": translation,
        }
        calibrations.append(calibration)
        cameras.append(SimpleNamespace(image_width=640, image_height=480))
    targets = [project_points_pixel(anchor, camera, calibration)[0].detach() for camera, calibration in zip(cameras, calibrations)]

    def loss_and_error():
        predictions = [project_points_pixel(xyz, camera, calibration)[0] for camera, calibration in zip(cameras, calibrations)]
        errors = torch.cat(
            [torch.linalg.norm(prediction - target, dim=-1) for prediction, target in zip(predictions, targets)]
        )
        return errors.square().mean(), errors.mean()

    initial_loss, initial_error = loss_and_error()
    initial_3d = torch.linalg.norm(xyz.detach() - anchor).item()
    optimizer = torch.optim.Adam([xyz], lr=0.01)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = loss_and_error()
        loss.backward()
        optimizer.step()
    _, final_error = loss_and_error()
    final_3d = torch.linalg.norm(xyz.detach() - anchor).item()
    report = {
        "initial_pixel_reprojection_error": float(initial_error.item()),
        "final_pixel_reprojection_error": float(final_error.item()),
        "initial_3d_anchor_distance": float(initial_3d),
        "final_3d_anchor_distance": float(final_3d),
    }
    report["passed"] = (
        report["final_pixel_reprojection_error"] < 0.1 * report["initial_pixel_reprojection_error"]
        and report["final_3d_anchor_distance"] < 0.1 * report["initial_3d_anchor_distance"]
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic-smoke", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, default=Path("geometry_recovery.json"))
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--perturbation-ratios", nargs="+", type=float, default=[0.005, 0.01, 0.02])
    parser.add_argument("--track-count", type=int, default=32)
    parser.add_argument("--minimum-reduction", type=float, default=0.10)
    parser.add_argument("--quiet", action="store_true")
    # Real-mode arguments are added lazily to keep synthetic CPU smoke independent.
    known, _ = parser.parse_known_args()
    if known.synthetic_smoke:
        report = run_synthetic(known.steps)
    else:
        from arguments import ModelParams, OptimizationParams

        ModelParams(parser)
        OptimizationParams(parser)
        parser.add_argument("--data_type", default="colmap")
        parser.add_argument("--llff_holdout", type=int, default=8)
        args = parser.parse_args()
        if args.checkpoint is None:
            parser.error("--checkpoint is required in real mode")
        if not args.strict_source_only_geometry:
            parser.error("real mode requires --strict_source_only_geometry")
        report = run_real(args)
    known.output.parent.mkdir(parents=True, exist_ok=True)
    known.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
