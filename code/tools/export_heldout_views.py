#!/usr/bin/env python3
"""Export GS/Difix/real held-out targets and manifests for Evidence diagnostic."""

from __future__ import annotations

from argparse import ArgumentParser
import hashlib
import json
from pathlib import Path

import torch
import torchvision

from arguments import ModelParams, OptimizationParams, PipelineParams, get_combined_args
from diffusion_guidance.camera_utils import (
    camera_fingerprint,
    camera_payload,
    nearest_reference_camera,
)
from diffusion_guidance.calibration_guard import calibrate_scene
from diffusion_guidance.checkpoint_state import (
    renderer_state_sha256,
    validate_complete_state,
)
from diffusion_guidance.control_identity import (
    normalized_camera_names,
    source_camera_set_sha256,
)
from diffusion_guidance.difix_provenance import (
    HELDOUT_DIAGNOSTIC_MANIFEST_SCHEMA,
)
from diffusion_guidance.experiment_registry import expected_scene_role
from diffusion_guidance.result_binding import (
    read_pair_audit,
    require_audit_binding,
    require_final_checkpoint_binding,
)
from gaussian_renderer import render
from scene import GaussianModel, Scene
from utils.general_utils import safe_state


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def restore_final_checkpoint(
    gaussians: GaussianModel,
    checkpoint: Path,
    optimization,
    iteration: int,
    *,
    pair_audit: dict,
    arm: str,
) -> dict:
    """Bind final diagnostic renders to the complete saved continuation state."""
    try:
        loaded = torch.load(checkpoint, weights_only=False)
    except Exception as exc:
        raise RuntimeError(f"Cannot read final checkpoint {checkpoint}: {exc}") from exc
    if not isinstance(loaded, tuple) or len(loaded) != 2:
        raise ValueError("Final checkpoint must contain exactly (state, iteration)")
    state, stored_iteration = loaded
    if int(stored_iteration) != int(iteration):
        raise ValueError(
            f"Final checkpoint iteration mismatch: checkpoint={stored_iteration}, requested={iteration}"
        )
    if not isinstance(state, dict) or state.get("format") != "geotrack-research-v2":
        raise ValueError("Held-out export requires a complete geotrack-research-v2 checkpoint")
    checkpoint_summary = {
        "iteration": int(stored_iteration),
        **validate_complete_state(
            state,
            require_cuda_rng=True,
            require_controlled_provenance=True,
        ),
    }
    require_final_checkpoint_binding(
        checkpoint_summary,
        pair_audit,
        method=arm,
        checkpoint_path=checkpoint,
    )
    gaussians.restore(state, optimization)
    if not bool(getattr(gaussians, "optimizer_state_restored", False)):
        raise RuntimeError("Final checkpoint restore did not restore optimizer state")
    if not bool(getattr(gaussians, "densification_state_restored", False)):
        raise RuntimeError("Final checkpoint restore did not restore densification state")
    if not bool(torch.isfinite(gaussians.get_xyz).all()):
        raise RuntimeError("Final checkpoint contains non-finite Gaussian centers")
    restored_render_sha256 = renderer_state_sha256(gaussians)
    if restored_render_sha256 != checkpoint_summary["render_state_sha256"]:
        raise RuntimeError(
            "Restored final renderer state differs from the complete checkpoint"
        )
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_iteration": int(stored_iteration),
        "checkpoint_state_format": state["format"],
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
    }


def main() -> None:
    parser = ArgumentParser()
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    optimization = OptimizationParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Immutable complete final checkpoint used for this evidence render",
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--pair-audit",
        type=Path,
        required=True,
        help="Passed A1/B audit authorizing this held-out evidence export",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data_type", default="colmap")
    parser.add_argument("--llff_holdout", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--use_gt_dca", action="store_true")
    args = get_combined_args(parser)
    if not hasattr(args, "gt_dca_config"):
        args.gt_dca_config = None
    safe_state(args.quiet, args.seed)
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Final checkpoint is missing: {checkpoint}")
    if int(args.iteration) <= 0:
        raise ValueError("Held-out export requires an explicit positive --iteration")
    if not str(getattr(args, "model_path", "")).strip():
        raise ValueError("Held-out export requires -m/--model_path")
    model_path = Path(args.model_path).expanduser().resolve()
    if checkpoint.parent != model_path:
        raise ValueError("--checkpoint must belong to the same -m/--model_path run")
    controlled_metadata_path = model_path / "controlled_ab_metadata.json"
    if not controlled_metadata_path.is_file():
        raise FileNotFoundError(
            f"Held-out export lacks controlled run metadata: {controlled_metadata_path}"
        )
    controlled_metadata = json.loads(
        controlled_metadata_path.read_text(encoding="utf-8")
    )
    if not isinstance(controlled_metadata, dict):
        raise ValueError("Controlled run metadata must be a JSON object")
    arm = str(controlled_metadata.get("role", "")).upper()
    if arm not in {"A1", "B"}:
        raise ValueError("Held-out identity export is restricted to A1 or B")
    scene_source_path = Path(args.source_path).expanduser().resolve()
    scene_name = scene_source_path.name
    experiment_role = expected_scene_role(args.dataset, scene_name)
    pair_audit_path, pair_audit = read_pair_audit(args.pair_audit)
    require_audit_binding(
        pair_audit,
        dataset=args.dataset,
        scene=scene_name,
        seed=int(args.seed),
        method=arm,
        pair_id=str(controlled_metadata.get("controlled_pair_id", "")),
        role=experiment_role,
    )
    if Path(pair_audit["scene_source_path"]).expanduser().resolve() != scene_source_path:
        raise ValueError("Held-out export scene path differs from the controlled-pair audit")
    gaussians = GaussianModel(args)
    scene = Scene(args, gaussians, load_iteration=args.iteration, shuffle=False)
    if not args.strict_source_only_geometry:
        raise RuntimeError(
            "Held-out evidence export requires --strict_source_only_geometry "
            "to bind camera calibration to the audited Track H5"
        )
    scene_ply_count = int(gaussians.get_xyz.shape[0])
    scene_ply_state_sha256 = renderer_state_sha256(gaussians)
    restored = restore_final_checkpoint(
        gaussians,
        checkpoint,
        optimization.extract(args),
        int(args.iteration),
        pair_audit=pair_audit,
        arm=arm,
    )
    calibrate_scene(scene, args.track_path)
    if int(getattr(scene, "loaded_iter", -1)) != int(args.iteration):
        raise RuntimeError(
            "Scene did not load the requested point-cloud iteration for final evidence export"
        )
    if int(gaussians.get_xyz.shape[0]) != restored["checkpoint_gaussian_count"]:
        raise RuntimeError(
            "Final Scene point cloud/checkpoint Gaussian count differs; refuse ambiguous evidence export"
        )
    if scene_ply_count != restored["checkpoint_gaussian_count"]:
        raise RuntimeError(
            "Final Scene PLY/checkpoint Gaussian count differs; refuse ambiguous evidence export"
        )
    if scene_ply_state_sha256 != restored["checkpoint_render_state_sha256"]:
        raise RuntimeError(
            "Final Scene PLY render state differs from complete checkpoint; "
            "refuse ambiguous evidence export"
        )
    restored.update(
        {
            "scene_ply_gaussian_count": scene_ply_count,
            "scene_ply_checkpoint_count_match": True,
            "scene_ply_render_state_schema": restored[
                "checkpoint_render_state_schema"
            ],
            "scene_ply_render_state_sha256": scene_ply_state_sha256,
            "scene_ply_checkpoint_render_state_match": True,
        }
    )
    dataset = model.extract(args)
    pipe = pipeline.extract(args)
    background = torch.tensor(
        [1.0, 1.0, 1.0] if dataset.white_background else [0.0, 0.0, 0.0],
        dtype=torch.float32,
        device="cuda",
    )
    output = Path(args.output_dir).expanduser().resolve()
    gs_dir = output / "gs"
    real_dir = output / "real"
    references_dir = output / "references"
    difix_dir = output / "difix"
    for directory in (gs_dir, real_dir, references_dir, difix_dir):
        directory.mkdir(parents=True, exist_ok=True)
    source_cameras = scene.getTrainCameras()
    heldout_cameras = scene.getTestCameras()
    if not heldout_cameras:
        raise RuntimeError("No held-out cameras are available")
    source_camera_names = normalized_camera_names(
        [camera.image_name for camera in source_cameras]
    )
    source_camera_hash = source_camera_set_sha256(source_camera_names)
    track_path = Path(args.track_path).expanduser().resolve()
    if not track_path.is_file():
        raise FileNotFoundError(f"Strict Track H5 is missing: {track_path}")
    track_hash = sha256_file(track_path)
    expected_controlled = {
        "seed": int(args.seed),
        "scene_source_path": str(scene_source_path),
        "track_h5_path": str(track_path),
        "track_h5_sha256": track_hash,
        "source_camera_names": source_camera_names,
        "source_camera_set_sha256": source_camera_hash,
        "final_iteration": int(args.iteration),
    }
    for field, expected in expected_controlled.items():
        if controlled_metadata.get(field) != expected:
            raise ValueError(
                f"Controlled metadata mismatch for held-out export {field}: "
                f"metadata={controlled_metadata.get(field)!r}, current={expected!r}"
            )
    binding = {
        "manifest_schema": HELDOUT_DIAGNOSTIC_MANIFEST_SCHEMA,
        "supervision_target_kind": "difix",
        "dataset": str(args.dataset),
        "scene": scene_name,
        "seed": int(args.seed),
        "experiment_role": experiment_role,
        "scene_source_path": str(scene_source_path),
        "paired_identity_arm": arm,
        "controlled_pair_id": controlled_metadata["controlled_pair_id"],
        "pair_audit": str(pair_audit_path),
        "pair_audit_sha256": sha256_file(pair_audit_path),
        "track_h5": str(track_path),
        "track_h5_sha256": track_hash,
        "source_camera_names": source_camera_names,
        "source_camera_set_sha256": source_camera_hash,
    }

    difix_records, evidence_records = [], []
    with torch.inference_mode():
        for camera in heldout_cameras:
            key = camera_fingerprint(camera, prefix="heldout")
            gs_path = gs_dir / f"{camera.image_name}.png"
            real_path = real_dir / f"{camera.image_name}.png"
            reference_path = references_dir / f"{camera.image_name}.png"
            difix_path = difix_dir / f"{camera.image_name}.png"
            reference = nearest_reference_camera(camera, source_cameras)
            torchvision.utils.save_image(
                render(camera, gaussians, pipe, background)["render"].clamp(0.0, 1.0),
                gs_path,
            )
            torchvision.utils.save_image(
                camera.original_image[:3].clamp(0.0, 1.0), real_path
            )
            torchvision.utils.save_image(
                reference.original_image[:3].clamp(0.0, 1.0), reference_path
            )
            payload = camera_payload(camera)
            difix_records.append(
                {
                    **binding,
                    "key": key,
                    "camera": payload,
                    "camera_id": str(getattr(camera, "uid", camera.image_name)),
                    "camera_source": "heldout_evaluation_camera",
                    "input": str(gs_path),
                    "reference_image": str(reference_path),
                    "target": str(difix_path),
                    "input_sha256": sha256_file(gs_path),
                    "reference_image_sha256": sha256_file(reference_path),
                    "reference_image_name": reference.image_name,
                    **restored,
                }
            )
            evidence_records.append(
                {
                    **binding,
                    "image_name": camera.image_name,
                    "camera_fingerprint": key,
                    "camera": payload,
                    "camera_source": "heldout_evaluation_camera",
                    "gs_render": str(gs_path),
                    "reference_image": str(reference_path),
                    "real_target": str(real_path),
                    "difix_output": str(difix_path),
                    **restored,
                    "gs_render_sha256": sha256_file(gs_path),
                    "reference_image_sha256": sha256_file(reference_path),
                    "real_target_sha256": sha256_file(real_path),
                }
            )
    with (output / "difix_manifest.jsonl").open("w", encoding="utf-8") as handle:
        for record in difix_records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    with (output / "evidence_manifest.jsonl").open("w", encoding="utf-8") as handle:
        for record in evidence_records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    print(f"Held-out Difix manifest: {output / 'difix_manifest.jsonl'}")
    print(f"Evidence manifest: {output / 'evidence_manifest.jsonl'}")


if __name__ == "__main__":
    main()
