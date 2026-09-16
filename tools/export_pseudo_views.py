#!/usr/bin/env python3
"""Export a fixed pseudo-camera manifest from the shared 10k checkpoint."""

from __future__ import annotations

from argparse import ArgumentParser
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torchvision

from arguments import (
    ModelParams,
    OptimizationParams,
    PipelineParams,
    get_combined_args,
)
from diffusion_guidance.camera_utils import (
    camera_payload,
    nearest_reference_camera,
    pseudo_camera_manifest_fields,
)
from diffusion_guidance.control_identity import (
    normalized_camera_names,
    source_camera_set_sha256,
    validate_controlled_checkpoint_provenance_inputs,
)
from diffusion_guidance.difix_provenance import A0_PSEUDO_MANIFEST_SCHEMA
from diffusion_guidance.calibration_guard import calibrate_scene
from diffusion_guidance.checkpoint_state import (
    renderer_state_sha256,
    validate_complete_state,
)
from gaussian_renderer import render
from scene import GaussianModel, Scene
from utils.general_utils import safe_state


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def even_subset(values, max_count: int):
    if max_count <= 0 or len(values) <= max_count:
        return list(values)
    indices = np.linspace(0, len(values) - 1, max_count, dtype=int)
    return [values[int(index)] for index in indices]


def restore_a0_render_state(
    gaussians: GaussianModel,
    checkpoint: Path,
    optimization,
    *,
    expected_iteration: int,
) -> dict:
    """Restore the complete A0 state that supplies every exported pseudo render.

    The Scene loader supplies camera context from its saved point cloud, while
    the pseudo RGB itself is rendered only after this complete checkpoint
    restore. Thus the immutable artifact governing the exported pixels is the
    A0 checkpoint rather than a same-path PLY replacement.
    """
    try:
        loaded = torch.load(checkpoint, weights_only=False)
    except Exception as exc:
        raise RuntimeError(f"Cannot read complete A0 checkpoint {checkpoint}: {exc}") from exc
    if not isinstance(loaded, tuple) or len(loaded) != 2:
        raise ValueError("A0 checkpoint must contain exactly (complete_state, iteration)")
    checkpoint_state, checkpoint_iteration = loaded
    if isinstance(checkpoint_iteration, bool):
        raise ValueError("A0 checkpoint iteration must be an integer")
    try:
        checkpoint_iteration = int(checkpoint_iteration)
    except (TypeError, ValueError) as exc:
        raise ValueError("A0 checkpoint iteration must be an integer") from exc
    if checkpoint_iteration != int(expected_iteration):
        raise ValueError(
            "A0 checkpoint iteration does not match --iteration: "
            f"checkpoint={checkpoint_iteration}, requested={expected_iteration}"
        )
    if not isinstance(checkpoint_state, dict):
        raise ValueError("A0 checkpoint does not contain the complete research-v2 state")
    state_format = checkpoint_state.get("format")
    if state_format != "geotrack-research-v2":
        raise ValueError(
            "Pseudo export requires a complete geotrack-research-v2 A0 checkpoint; "
            f"got {state_format!r}"
        )
    checkpoint_summary = validate_complete_state(
        checkpoint_state,
        require_cuda_rng=True,
        require_controlled_provenance=True,
    )
    try:
        gaussians.restore(checkpoint_state, optimization)
    except Exception as exc:
        raise RuntimeError(f"A0 checkpoint restore failed: {exc}") from exc
    if not bool(getattr(gaussians, "optimizer_state_restored", False)):
        raise RuntimeError("A0 checkpoint restore did not confirm optimizer-state restoration")
    if not bool(getattr(gaussians, "densification_state_restored", False)):
        raise RuntimeError(
            "A0 checkpoint restore did not confirm densification-state restoration"
        )
    xyz = gaussians.get_xyz
    if xyz.ndim != 2 or xyz.shape[1] != 3 or int(xyz.shape[0]) <= 0:
        raise RuntimeError("Restored A0 checkpoint has an invalid Gaussian state")
    if not bool(torch.isfinite(xyz).all()):
        raise RuntimeError("Restored A0 checkpoint contains non-finite Gaussian centers")
    restored_render_sha256 = renderer_state_sha256(gaussians)
    if restored_render_sha256 != checkpoint_summary["render_state_sha256"]:
        raise RuntimeError(
            "Restored A0 renderer state differs from the complete checkpoint"
        )
    return {
        "a0_checkpoint_iteration": checkpoint_iteration,
        "a0_checkpoint_state_format": state_format,
        "a0_checkpoint_gaussian_count": checkpoint_summary["gaussian_count"],
        "a0_checkpoint_render_state_schema": checkpoint_summary[
            "render_state_schema"
        ],
        "a0_checkpoint_render_state_sha256": checkpoint_summary[
            "render_state_sha256"
        ],
        "a0_checkpoint_controlled_provenance_sha256": checkpoint_summary[
            "controlled_provenance_sha256"
        ],
        "_a0_checkpoint_controlled_provenance": checkpoint_summary[
            "controlled_provenance"
        ],
        "a0_render_state_source": "complete_checkpoint_restore_after_scene_ply_load",
    }


def main() -> None:
    parser = ArgumentParser()
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    optimization = OptimizationParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument(
        "--a0-checkpoint",
        type=Path,
        required=True,
        help="Immutable shared A0 checkpoint used to bind this pseudo pool",
    )
    parser.add_argument(
        "--live-pseudo-audit",
        type=Path,
        required=True,
        help="Current passed pre-A0 live pseudo-camera audit bound to this projection context",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-pseudo-views", default=32, type=int)
    parser.add_argument("--data_type", default="colmap", choices=["colmap", "360", "blender"])
    parser.add_argument("--llff_holdout", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--use_gt_dca", action="store_true")
    args = get_combined_args(parser)
    if not hasattr(args, "gt_dca_config"):
        args.gt_dca_config = None
    checkpoint = args.a0_checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Shared A0 checkpoint is missing: {checkpoint}")
    a0_checkpoint_sha256 = sha256_file(checkpoint)
    if int(args.iteration) <= 0:
        raise ValueError(
            "Pseudo export requires an explicit positive --iteration matching "
            "the complete A0 checkpoint"
        )
    if not str(getattr(args, "model_path", "")).strip():
        raise ValueError("Pseudo export requires -m/--model_path for the shared A0 run")
    a0_model_path = Path(args.model_path).expanduser().resolve()
    if checkpoint.parent != a0_model_path:
        raise ValueError(
            "--a0-checkpoint must be the checkpoint inside the -m/--model_path "
            "A0 run; refusing to combine a checkpoint with another Scene PLY"
        )
    live_audit_path = args.live_pseudo_audit.expanduser().resolve()
    if not live_audit_path.is_file():
        raise FileNotFoundError(f"Live pseudo-camera audit is missing: {live_audit_path}")
    live_audit = json.loads(live_audit_path.read_text(encoding="utf-8"))
    if not isinstance(live_audit, dict) or live_audit.get("passed") is not True:
        raise ValueError("Pseudo export requires a passed live pseudo-camera audit")
    live_projection_fingerprint = live_audit.get("projection_context_fingerprint")
    if not isinstance(live_projection_fingerprint, str) or not live_projection_fingerprint:
        raise ValueError("Live pseudo-camera audit lacks its projection-context fingerprint")
    safe_state(args.quiet, args.seed)
    gaussians = GaussianModel(args)
    scene = Scene(args, gaussians, load_iteration=args.iteration, shuffle=False)
    if not args.strict_source_only_geometry:
        raise RuntimeError(
            "Pseudo export requires --strict_source_only_geometry so camera "
            "calibration and source-only pose provenance are explicit"
        )
    if int(getattr(scene, "loaded_iter", -1)) != int(args.iteration):
        raise RuntimeError(
            "Scene did not load the requested A0 point-cloud iteration: "
            f"requested={args.iteration}, loaded={getattr(scene, 'loaded_iter', None)}"
        )
    a0_render_state = restore_a0_render_state(
        gaussians,
        checkpoint,
        optimization.extract(args),
        expected_iteration=int(args.iteration),
    )
    a0_controlled_provenance = a0_render_state.pop(
        "_a0_checkpoint_controlled_provenance"
    )
    # Keep the production calibration call literal so the guarded assembler can
    # verify exactly one overlay-owned calibration attachment.
    calibrate_scene(scene, args.track_path)
    pipe = pipeline.extract(args)
    dataset = model.extract(args)
    background = torch.tensor(
        [1.0, 1.0, 1.0] if dataset.white_background else [0.0, 0.0, 0.0],
        dtype=torch.float32,
        device="cuda",
    )

    output = Path(args.output_dir).expanduser().resolve()
    inputs = output / "inputs"
    references = output / "references"
    targets = output / "targets"
    for directory in (inputs, references, targets):
        directory.mkdir(parents=True, exist_ok=True)
    pseudo_cameras = even_subset(scene.getPseudoCameras(), args.max_pseudo_views)
    source_cameras = scene.getTrainCameras()
    source_camera_names = normalized_camera_names(
        [camera.image_name for camera in source_cameras]
    )
    source_camera_hash = source_camera_set_sha256(source_camera_names)
    track_path = Path(args.track_path).expanduser().resolve()
    if not track_path.is_file():
        raise FileNotFoundError(f"Strict Track H5 is missing: {track_path}")
    source_images_dir = (
        Path(dataset.source_path) / (dataset.images if dataset.images else "images")
    ).resolve()
    validate_controlled_checkpoint_provenance_inputs(
        a0_controlled_provenance,
        scene_source_path=dataset.source_path,
        track_h5_path=track_path,
        source_images_dir=source_images_dir,
        source_camera_names=source_camera_names,
        source_training_camera_inventory=[
            {"camera_name": camera.image_name, "camera": camera_payload(camera)}
            for camera in source_cameras
        ],
        seed=int(args.seed),
    )
    pseudo_source = str(
        getattr(scene, "pseudo_camera_provenance", "upstream_scene_getPseudoCameras_unverified")
    )
    if not pseudo_cameras or not source_cameras:
        raise RuntimeError("Pseudo and source cameras must both be non-empty")
    if pseudo_source != "source_camera_interpolation_only":
        raise RuntimeError(
            "Pseudo-camera provenance is not verified as source-camera interpolation; "
            "refuse to create a controlled source-conditioned pseudo manifest"
        )
    live_records = live_audit.get("records")
    if not isinstance(live_records, list) or not live_records:
        raise ValueError("Live pseudo-camera audit lacks camera records")
    live_by_fingerprint = {
        str(record.get("camera_fingerprint")): record for record in live_records
    }
    if len(live_by_fingerprint) != len(live_records):
        raise ValueError("Live pseudo-camera audit contains duplicate fingerprints")

    records = []
    with torch.inference_mode():
        for index, pseudo_camera in enumerate(pseudo_cameras):
            camera_fields = pseudo_camera_manifest_fields(pseudo_camera)
            if camera_fields["camera_source"] != pseudo_source:
                raise RuntimeError("Scene and pseudo-camera provenance declarations differ")
            key = camera_fields["camera_fingerprint"]
            live_record = live_by_fingerprint.get(key)
            if live_record is None:
                raise RuntimeError(
                    "Post-A0 pseudo camera is absent from the passed pre-A0 live audit: "
                    f"{key}"
                )
            for field in (
                "camera_fingerprint",
                "camera_source",
                "calibration_source_camera",
                "pose_source_camera_names",
                "uses_heldout_pose_information",
                "fx",
                "fy",
                "cx",
                "cy",
                "width",
                "height",
                "R",
                "t",
            ):
                if camera_fields.get(field) != live_record.get(field):
                    raise RuntimeError(
                        "Post-A0 pseudo camera differs from the passed live audit: "
                        f"{key}:{field}"
                    )
            input_path = inputs / f"{key}.png"
            reference_path = references / f"{key}.png"
            target_path = targets / f"{key}.png"
            reference_camera = nearest_reference_camera(pseudo_camera, source_cameras)
            rendered = render(pseudo_camera, gaussians, pipe, background)["render"]
            torchvision.utils.save_image(rendered.clamp(0.0, 1.0), input_path)
            torchvision.utils.save_image(
                reference_camera.original_image[:3].clamp(0.0, 1.0), reference_path
            )
            records.append(
                {
                    **camera_fields,
                    "manifest_schema": A0_PSEUDO_MANIFEST_SCHEMA,
                    "index": index,
                    "input": str(input_path),
                    "input_sha256": sha256_file(input_path),
                    "target": str(target_path),
                    "supervision_target_kind": "difix",
                    "reference_image": str(reference_path),
                    "reference_image_sha256": sha256_file(reference_path),
                    "reference_image_name": reference_camera.image_name,
                    "a0_checkpoint": str(checkpoint),
                    "a0_checkpoint_sha256": a0_checkpoint_sha256,
                    **a0_render_state,
                    "live_pseudo_audit": str(live_audit_path),
                    "live_pseudo_audit_sha256": sha256_file(live_audit_path),
                    "projection_context_fingerprint": live_projection_fingerprint,
                    "track_h5": str(track_path),
                    "track_h5_sha256": sha256_file(track_path),
                    "source_camera_names": source_camera_names,
                    "source_camera_set_sha256": source_camera_hash,
                    "export_seed": args.seed,
                }
            )
    manifest = output / "manifest.jsonl"
    with manifest.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"Exported {len(records)} fixed pseudo views to {manifest}")


if __name__ == "__main__":
    main()
