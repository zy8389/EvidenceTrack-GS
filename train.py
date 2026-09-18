# -*- coding: utf-8 -*-
#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

import numpy as np
import os
import hashlib
import json
import random
import matplotlib.pyplot as plt
import torch
from torchmetrics import PearsonCorrCoef
from torchmetrics.functional.regression import pearson_corrcoef
from utils.loss_utils import l1_loss, l1_loss_mask, l2_loss, ssim
def estimate_depth(*args, **kwargs):
    from utils.depth_utils import estimate_depth as implementation
    return implementation(*args, **kwargs)
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from lpipsPyTorch import lpips

from utils.config_utils import setup_geometric_constraints_config, print_geometric_constraints_summary

# 导入几何正则化模块
from utils.geometry_regularization import create_geometry_regularizer
from diffusion_guidance.losses import pseudo_rgb_loss
from diffusion_guidance.pseudo_manifest import (
    build_pseudo_camera,
    load_pseudo_manifest,
    load_pseudo_target,
    pseudo_view_index,
)
from diffusion_guidance.camera_utils import camera_payload
from diffusion_guidance.control_identity import (
    CONTROLLED_TRAINING_PROTOCOL_EXCLUDED_FIELDS,
    PREREGISTERED_CONTROLLED_OPTIMIZATION,
    build_source_image_inventory,
    controlled_checkpoint_provenance_from_metadata,
    controlled_checkpoint_provenance_sha256,
    canonical_controlled_training_protocol,
    canonical_training_camera_inventory,
    controlled_pair_id,
    controlled_training_protocol_sha256,
    normalized_camera_names,
    source_camera_set_sha256,
    source_image_inventory_sha256,
    training_camera_inventory_sha256,
    validate_a0_checkpoint_provenance,
    validate_preregistered_controlled_training_protocol,
)
from diffusion_guidance.pseudo_schedule import (
    pseudo_call_trace_sha256,
    pseudo_camera_pool_sha256,
)
from diffusion_guidance.run_status import (
    fail_closed,
    mark_training_invalid,
    write_training_status,
)
from geometric_constraints.repaired_geometry import StrictGeometryManager


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


_CONTROLLED_PROTOCOL_PATH_FIELDS = frozenset(
    {"source_path", "track_path", "start_checkpoint", "constraint_config_path"}
)


def _controlled_protocol_namespace(value):
    result = {}
    for key, item in sorted(vars(value).items()):
        if key in CONTROLLED_TRAINING_PROTOCOL_EXCLUDED_FIELDS:
            continue
        if isinstance(item, os.PathLike):
            item = os.fspath(item)
        if key in _CONTROLLED_PROTOCOL_PATH_FIELDS and item not in (None, ""):
            item = os.path.realpath(item)
        result[key] = item
    return result


def _controlled_training_protocol(dataset, opt, pipe, args):
    protocol = canonical_controlled_training_protocol(
        {
            "schema": "controlled_training_protocol_v1",
            "model": _controlled_protocol_namespace(dataset),
            "optimization": _controlled_protocol_namespace(opt),
            "pipeline": _controlled_protocol_namespace(pipe),
            "runtime": _controlled_protocol_namespace(args),
        }
    )
    return validate_preregistered_controlled_training_protocol(
        protocol, role=str(getattr(opt, "controlled_ab_role", ""))
    )


def _live_cuda_renderer_probe(scene, gaussians, pipe, dataset):
    """Exercise the pinned CUDA renderer on one actual source Scene camera.

    This belongs only to ``--geometry_smoke_only``.  The Track gradient probe
    checks the differentiable source-track objective; this probe makes the
    required renderer/Scene/Gaussian part of that same P0 gate explicit.
    """
    source_cameras = list(scene.getTrainCameras())
    if not source_cameras:
        raise RuntimeError("Geometry smoke has no source Scene camera to render")
    if not torch.cuda.is_available():
        raise RuntimeError("Geometry smoke requires CUDA")
    if gaussians.get_xyz.device.type != "cuda":
        raise RuntimeError("Geometry smoke Gaussians are not resident on CUDA")

    camera = source_cameras[0]
    background = torch.tensor(
        [1, 1, 1] if dataset.white_background else [0, 0, 0],
        dtype=gaussians.get_xyz.dtype,
        device=gaussians.get_xyz.device,
    )
    rendered = render(camera, gaussians, pipe, background).get("render")
    if not isinstance(rendered, torch.Tensor):
        raise RuntimeError("CUDA renderer did not return a render tensor")
    if rendered.device.type != "cuda":
        raise RuntimeError("CUDA renderer returned a non-CUDA render tensor")
    torch.cuda.synchronize(rendered.device)
    finite = torch.isfinite(rendered)
    pixel_count = int(rendered.numel())
    finite_pixel_count = int(finite.sum().item())
    camera_name = str(getattr(camera, "image_name", ""))
    probe_passed = bool(
        pixel_count > 0
        and finite_pixel_count == pixel_count
        and int(gaussians.get_xyz.shape[0]) > 0
        and bool(camera_name)
    )
    return {
        "cuda": True,
        "finite": finite_pixel_count == pixel_count,
        "pixel_count": pixel_count,
        "finite_pixel_count": finite_pixel_count,
        "gaussian_count": int(gaussians.get_xyz.shape[0]),
        "source_camera_count": len(source_cameras),
        "camera_name": camera_name,
        "camera_uid": int(getattr(camera, "uid", -1)),
        "resolution": [
            int(getattr(camera, "image_width", 0)),
            int(getattr(camera, "image_height", 0)),
        ],
        "passed": probe_passed,
    }


def _write_pseudo_usage(
    path,
    *,
    calls,
    unique_indices,
    pool_size,
    camera_keys,
    call_trace,
    interval,
    start_iteration,
    end_iteration,
    last_iteration,
):
    keys = [str(key) for key in camera_keys]
    trace = list(call_trace)
    if len(keys) != int(pool_size):
        raise ValueError(
            "Pseudo usage cannot serialize a camera pool whose key count differs "
            "from its declared pool size"
        )
    if len(trace) != int(calls):
        raise ValueError(
            "Pseudo usage cannot serialize a call trace whose length differs "
            "from its declared call count"
        )
    payload = {
        "pseudo_supervision_calls": int(calls),
        "unique_pseudo_views_used": len(unique_indices),
        "pseudo_view_pool_size": int(pool_size),
        "pseudo_pool_coverage": (
            len(unique_indices) / float(pool_size) if pool_size else 0.0
        ),
        "pseudo_view_indices_used": sorted(map(int, unique_indices)),
        "pseudo_camera_keys": keys,
        "pseudo_camera_pool_sha256": pseudo_camera_pool_sha256(keys),
        "pseudo_call_trace": trace,
        "pseudo_call_trace_sha256": pseudo_call_trace_sha256(trace),
        "pseudo_rgb_interval": int(interval),
        "pseudo_rgb_start": int(start_iteration),
        "pseudo_rgb_end": int(end_iteration),
        "last_supervision_iteration": (
            int(last_iteration) if last_iteration is not None else None
        ),
    }
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(temporary, path)


def _validate_controlled_resume(args, opt, gaussians, first_iter):
    role = str(getattr(opt, "controlled_ab_role", "")).upper()
    if role not in {"", "A0", "A1", "SELFRENDER", "B"}:
        raise ValueError("--controlled_ab_role must be empty, A0, A1, SelfRender, or B")
    if role and not (
        bool(getattr(args, "strict_tracks", False))
        and bool(getattr(args, "strict_source_only_geometry", False))
    ):
        raise RuntimeError(
            f"Controlled {role} requires strict source-only Track geometry"
        )
    if role and not opt.disable_legacy_pseudo_depth:
        raise RuntimeError(
            f"Controlled {role} must use --disable_legacy_pseudo_depth"
        )
    if role:
        mismatched = sorted(
            key
            for key, expected in PREREGISTERED_CONTROLLED_OPTIMIZATION.items()
            if getattr(opt, key, None) != expected
        )
        if mismatched:
            raise RuntimeError(
                "Controlled first-run optimization values differ from the "
                f"preregistered protocol: {mismatched}"
            )
    if role == "A0":
        if args.start_checkpoint or int(first_iter) != 0:
            raise RuntimeError("Controlled A0 must start from iteration 0 without a checkpoint")
        if opt.enable_diffusion_pseudo_rgb:
            raise RuntimeError("Controlled A0 cannot use pseudo-RGB supervision")
        if int(opt.iterations) != int(opt.controlled_ab_checkpoint_iteration):
            raise RuntimeError("Controlled A0 must end at the registered 10k checkpoint")
    if role in {"A1", "SELFRENDER", "B"}:
        expected = int(opt.controlled_ab_checkpoint_iteration)
        if not args.start_checkpoint:
            raise ValueError(f"Controlled {role} requires --start_checkpoint")
        if int(first_iter) != expected:
            raise RuntimeError(
                f"Controlled {role} expected checkpoint iteration {expected}, got {first_iter}"
            )
        if not gaussians.optimizer_state_restored:
            raise RuntimeError(f"Controlled {role} did not restore optimizer state")
        if role == "A1" and opt.enable_diffusion_pseudo_rgb:
            raise RuntimeError("A1 must continue normally with no Difix pseudo RGB")
        if role in {"B", "SELFRENDER"} and not opt.enable_diffusion_pseudo_rgb:
            raise RuntimeError(f"{role} must enable frozen pseudo RGB supervision")
        if role in {"B", "SELFRENDER"} and not opt.pseudo_rgb_strict_cache:
            raise RuntimeError(
                f"Controlled {role} requires --pseudo_rgb_strict_cache"
            )
        if role in {"B", "SELFRENDER"} and int(opt.pseudo_rgb_interval) != 50:
            raise RuntimeError(
                "First controlled Difix experiment requires --pseudo_rgb_interval 50"
            )
        if int(opt.iterations) != expected + 2000:
            raise RuntimeError(
                f"Controlled {role} must continue exactly 2000 iterations"
            )
        if role in {"B", "SELFRENDER"} and (
            int(opt.pseudo_rgb_start) != expected
            or int(opt.pseudo_rgb_end) != expected + 2000
        ):
            raise RuntimeError(
                "First controlled Difix experiment requires pseudo RGB over "
                "the 10k-to-12k continuation"
            )
    return role


def training(dataset, opt, pipe, args):
    testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from = args.test_iterations, \
        args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from
    first_iter = 0
    if str(getattr(opt, "controlled_ab_role", "")).upper() in {"A0", "A1", "SELFRENDER", "B"}:
        # One identifiable source RGB + repaired track objective, no extra legacy module.
        args.geometry_reg_enabled = False
        opt.geometry_reg_enabled = False
        opt.disable_depth_loss = True
        args.disable_depth_loss = True
        if getattr(opt, "mixed_precision", False):
            raise RuntimeError("Controlled revision v2 requires full precision")
        if int(opt.densify_until_iter) != 10000:
            raise RuntimeError("Controlled v2 freezes topology at 10k")
    tb_writer = prepare_output_and_logger(args)
    write_training_status(args.model_path, status="RUNNING", stage="initialization")
    gaussians = GaussianModel(args)
    strict_geometry_enabled = bool(
        getattr(args, "strict_tracks", False)
        or getattr(args, "strict_source_only_geometry", False)
    )
    if getattr(args, "strict_source_only_geometry", False):
        args.strict_tracks = True
    if strict_geometry_enabled and (
        getattr(args, "enable_geometric_constraints", False)
        or getattr(args, "use_gt_dca", False)
    ):
        raise RuntimeError(
            "Strict repaired geometry cannot be combined with the legacy "
            "geometric-constraint or GT-DCA paths"
        )
    if getattr(args, "strict_tracks", False) and not getattr(
        args, "strict_source_only_geometry", False
    ):
        raise RuntimeError(
            "Strict Track training requires --strict_source_only_geometry; "
            "otherwise Gaussian initialisation provenance is not strict"
        )

    # 初始化混合精度训练
    scaler = None
    if getattr(opt, 'mixed_precision', False):
        from torch.amp import GradScaler, autocast
        scaler = GradScaler('cuda')
        amp_dtype = getattr(opt, 'amp_dtype', 'fp16')
        dtype = torch.float16 if amp_dtype == 'fp16' else torch.bfloat16
        print(f"[AMP] 启用混合精度训练，精度类型: {amp_dtype}")
    else:
        # 即使不启用AMP也要导入，避免后续代码报错
        from torch.amp import autocast
        dtype = torch.float32
    # 如果命令行参数启用了 GT-DCA，则显式启用集成（确保训练阶段使用增强外观特征）
    if getattr(args, 'use_gt_dca', False):
        try:
            gaussians.enable_gt_dca()
        except Exception as e:
            print(f"[GT-DCA] 启用失败，将继续使用标准 SH 特征: {e}")
    scene = Scene(args, gaussians, shuffle=False)
    if getattr(args, "strict_source_only_geometry", False):
        from diffusion_guidance.calibration_guard import calibrate_scene
        calibrate_scene(scene, args.track_path)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint, weights_only=False)
        gaussians.restore(model_params, opt)

    controlled_role = _validate_controlled_resume(
        args, opt, gaussians, first_iter
    )
    source_cameras = list(scene.getTrainCameras()) if controlled_role else []
    source_camera_names = (
        normalized_camera_names([camera.image_name for camera in source_cameras])
        if controlled_role
        else []
    )
    source_images_dir = (
        os.path.realpath(
            os.path.join(
                args.source_path,
                dataset.images if dataset.images is not None else "images",
            )
        )
        if controlled_role
        else None
    )
    source_image_inventory = None
    source_training_camera_inventory = None
    controlled_training_protocol = None
    if controlled_role:
        source_image_inventory = build_source_image_inventory(
            source_images_dir, source_camera_names
        )
        source_training_camera_inventory = canonical_training_camera_inventory(
            [
                {
                    "camera_name": camera.image_name,
                    "camera": camera_payload(camera),
                }
                for camera in source_cameras
            ]
        )
        controlled_training_protocol = _controlled_training_protocol(
            dataset, opt, pipe, args
        )
    controlled_metadata = {
        "role": controlled_role or None,
        "seed": int(opt.experiment_seed),
        "start_checkpoint": os.path.realpath(checkpoint) if checkpoint else None,
        "start_checkpoint_sha256": _sha256_file(checkpoint) if checkpoint else None,
        "checkpoint_iteration": int(first_iter),
        "checkpoint_state_format": (
            model_params.get("format") if checkpoint and isinstance(model_params, dict)
            else None
        ),
        "optimizer_state_restored": bool(gaussians.optimizer_state_restored),
        "densification_state_restored": bool(getattr(gaussians, "densification_state_restored", False)),
        "final_iteration": int(opt.iterations),
        "real_view_sampler": "independent_seeded_cycle_shuffle_v1",
        "pseudo_view_sampler": "deterministic_round_robin_call_count_v1",
        "difix_enabled": bool(opt.enable_diffusion_pseudo_rgb),
        "pseudo_supervision_enabled": bool(opt.enable_diffusion_pseudo_rgb),
        "pseudo_supervision_target_kind": (
            "self_render_a0" if controlled_role == "SELFRENDER"
            else "difix" if opt.enable_diffusion_pseudo_rgb else "none"
        ),
        "pseudo_rgb_interval": int(opt.pseudo_rgb_interval),
        "pseudo_rgb_start": int(opt.pseudo_rgb_start),
        "pseudo_rgb_end": int(opt.pseudo_rgb_end),
        "required_pseudo_view_pool_size": 32 if controlled_role in {"B", "SELFRENDER"} else None,
        "pseudo_rgb_weight": float(opt.pseudo_rgb_weight),
        "pseudo_rgb_lambda_dssim": float(opt.pseudo_rgb_lambda_dssim),
        "pseudo_rgb_strict_cache": bool(opt.pseudo_rgb_strict_cache),
        "scene_source_path": (
            os.path.realpath(args.source_path) if controlled_role else None
        ),
        "track_h5_path": (
            os.path.realpath(args.track_path) if controlled_role else None
        ),
        "track_h5_sha256": (
            _sha256_file(args.track_path) if controlled_role else None
        ),
        "source_camera_names": source_camera_names,
        "source_camera_set_sha256": (
            source_camera_set_sha256(source_camera_names)
            if controlled_role
            else None
        ),
        "source_images_dir": source_images_dir,
        "source_image_inventory": source_image_inventory,
        "source_image_inventory_sha256": (
            source_image_inventory_sha256(source_image_inventory)
            if source_image_inventory is not None
            else None
        ),
        "source_training_camera_inventory": source_training_camera_inventory,
        "source_training_camera_inventory_sha256": (
            training_camera_inventory_sha256(source_training_camera_inventory)
            if source_training_camera_inventory is not None
            else None
        ),
        "controlled_training_protocol": controlled_training_protocol,
        "controlled_training_protocol_sha256": (
            controlled_training_protocol_sha256(controlled_training_protocol)
            if controlled_training_protocol is not None
            else None
        ),
        "strict_geometry_protocol": {
            "strict_tracks": bool(getattr(args, "strict_tracks", False)),
            "strict_source_only_geometry": bool(
                getattr(args, "strict_source_only_geometry", False)
            ),
            "strict_track_weight": float(opt.strict_track_weight),
            "strict_track_start": int(opt.strict_track_start),
            "strict_track_warmup": int(opt.strict_track_warmup),
            "strict_track_min_length": int(opt.strict_track_min_length),
            "strict_track_min_quality": float(opt.strict_track_min_quality),
            "strict_track_max_tracks": int(opt.strict_track_max_tracks),
            "strict_track_huber_delta": float(opt.strict_track_huber_delta),
            "strict_track_max_association_distance_ratio": float(
                opt.strict_track_max_association_distance_ratio
            ),
            "strict_track_collision_warning_rate": float(
                opt.strict_track_collision_warning_rate
            ),
            "densify_until_iter": int(opt.densify_until_iter),
            "mixed_precision": bool(getattr(opt, "mixed_precision", False)),
            "disable_legacy_pseudo_depth": bool(
                opt.disable_legacy_pseudo_depth
            ),
            "disable_depth_loss": bool(getattr(args, "disable_depth_loss", False)),
            "geometry_reg_enabled": bool(
                getattr(opt, "geometry_reg_enabled", False)
            ),
            "use_gt_dca": bool(getattr(args, "use_gt_dca", False)),
        },
    }
    pseudo_records = []
    pseudo_cameras = []
    if opt.enable_diffusion_pseudo_rgb:
        if not opt.pseudo_rgb_manifest:
            raise ValueError("--pseudo_rgb_manifest is required for frozen pseudo supervision")
        pseudo_records = load_pseudo_manifest(
            opt.pseudo_rgb_manifest,
            strict_targets=opt.pseudo_rgb_strict_cache,
        )
        controlled_metadata["pseudo_manifest_path"] = os.path.realpath(
            opt.pseudo_rgb_manifest
        )
        controlled_metadata["pseudo_manifest_sha256"] = _sha256_file(
            opt.pseudo_rgb_manifest
        )
        controlled_metadata["pseudo_camera_pool_sha256"] = (
            pseudo_camera_pool_sha256([record.key for record in pseudo_records])
        )
        controlled_metadata["pseudo_target_kind"] = str(
            pseudo_records[0].raw.get("supervision_target_kind", "difix")
        ).lower()
        if any(
            str(record.raw.get("supervision_target_kind", "difix")).lower()
            != controlled_metadata["pseudo_target_kind"]
            for record in pseudo_records
        ):
            raise RuntimeError("Controlled pseudo manifest mixes target kinds")
        if controlled_role in {"B", "SELFRENDER"}:
            if len(pseudo_records) != 32:
                raise RuntimeError(
                    "First controlled pseudo-supervision experiment requires exactly "
                    f"32 pseudo views; manifest contains {len(pseudo_records)}"
                )
            expected_target_kind = (
                "self_render_a0" if controlled_role == "SELFRENDER" else "difix"
            )
            if controlled_metadata["pseudo_target_kind"] != expected_target_kind:
                raise RuntimeError(
                    f"Controlled {controlled_role} requires {expected_target_kind!r} "
                    "pseudo targets"
                )
        pseudo_cameras = [build_pseudo_camera(record) for record in pseudo_records]
        print(f"[Difix] Loaded fixed pseudo manifest with {len(pseudo_records)} views")

    if controlled_role:
        if controlled_role in {"A1", "SELFRENDER", "B"}:
            restored_provenance = getattr(
                gaussians, "restored_controlled_provenance", None
            )
            controlled_metadata["start_checkpoint_controlled_provenance"] = (
                restored_provenance
            )
            controlled_metadata[
                "start_checkpoint_controlled_provenance_sha256"
            ] = controlled_checkpoint_provenance_sha256(restored_provenance)
        else:
            restored_provenance = None
            controlled_metadata["start_checkpoint_controlled_provenance"] = None
            controlled_metadata[
                "start_checkpoint_controlled_provenance_sha256"
            ] = None
        current_checkpoint_provenance = (
            controlled_checkpoint_provenance_from_metadata(controlled_metadata)
        )
        if restored_provenance is not None:
            validate_a0_checkpoint_provenance(
                restored_provenance, current_checkpoint_provenance
            )
        gaussians.controlled_provenance = current_checkpoint_provenance
    if controlled_role in {"A1", "SELFRENDER", "B"}:
        controlled_metadata["controlled_pair_id"] = controlled_pair_id(
            controlled_metadata
        )
    if controlled_role:
        with open(
            os.path.join(scene.model_path, "controlled_ab_metadata.json"),
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(controlled_metadata, handle, indent=2, sort_keys=True)

    strict_geometry = None
    if strict_geometry_enabled:
        strict_geometry = fail_closed(
            args.model_path,
            stage="strict_geometry_initialization",
            operation=lambda: StrictGeometryManager.from_path(
                args.track_path,
                scene.getTrainCameras(),
                device=gaussians.get_xyz.device,
                dtype=gaussians.get_xyz.dtype,
                min_length=opt.strict_track_min_length,
                min_quality=opt.strict_track_min_quality,
                max_tracks=opt.strict_track_max_tracks,
                huber_delta=opt.strict_track_huber_delta,
                association_chunk_size=opt.strict_track_association_chunk_size,
                max_association_distance_ratio=(
                    opt.strict_track_max_association_distance_ratio
                ),
                collision_warning_rate=opt.strict_track_collision_warning_rate,
            ),
        )
        strict_geometry.store.leakage_audit().emit()
        association_statistics = fail_closed(
            args.model_path,
            stage="strict_geometry_association",
            operation=lambda: strict_geometry.associate(
                gaussians.get_xyz, scene.cameras_extent
            ),
        )
        gradient_probe = fail_closed(
            args.model_path,
            stage="strict_geometry_gradient_probe",
            operation=lambda: strict_geometry.gradient_probe(
                gaussians.get_xyz,
                count=opt.strict_track_gradient_test_count,
                offset=(
                    opt.strict_track_gradient_probe_offset_ratio
                    * scene.cameras_extent
                ),
            ),
        )
        print(
            "[Strict Geometry] Track gradient coverage: "
            f"{gradient_probe['coverage']:.6f}"
        )
        print(
            "[Strict Geometry] Mean gradient norm: "
            f"{gradient_probe['mean_gradient_norm']:.9g}"
        )
        print(
            "[Strict Geometry] Zero-gradient ratio: "
            f"{gradient_probe['zero_gradient_ratio']:.6f}"
        )
        renderer_probe = (
            _live_cuda_renderer_probe(scene, gaussians, pipe, dataset)
            if opt.geometry_smoke_only
            else None
        )
        smoke_passed = bool(
            np.isfinite(gradient_probe["mean_gradient_norm"])
            and association_statistics.associated_tracks > 0
            and association_statistics.unique_associated_gaussians > 0
            and gradient_probe["coverage"] >= opt.strict_track_min_gradient_coverage
            and gradient_probe["mean_gradient_norm"] > 0.0
            and (
                renderer_probe is None
                or renderer_probe["passed"] is True
            )
        )
        if opt.geometry_smoke_only:
            smoke_report = {
                "schema": 1,
                "gate": "P0_real_cuda_geometry_smoke",
                "track_h5": os.path.realpath(args.track_path),
                "track_h5_sha256": _sha256_file(args.track_path),
                "camera_extent": float(scene.cameras_extent),
                "gradient_probe": gradient_probe,
                "association": dict(vars(association_statistics)),
                "renderer_probe": renderer_probe,
                "thresholds": {
                    "minimum_gradient_coverage": float(
                        opt.strict_track_min_gradient_coverage
                    ),
                    "positive_finite_mean_gradient_norm": True,
                    "nonempty_association": True,
                    "live_cuda_renderer": True,
                },
                "passed": smoke_passed,
            }
            smoke_path = os.path.join(scene.model_path, "geometry_smoke.json")
            temporary = smoke_path + ".tmp"
            with open(temporary, "w", encoding="utf-8") as handle:
                json.dump(smoke_report, handle, indent=2, sort_keys=True)
            os.replace(temporary, smoke_path)
        if not smoke_passed:
            raise RuntimeError("Strict geometry gradient integrity gate failed")
        if opt.geometry_smoke_only:
            write_training_status(
                scene.model_path,
                status="COMPLETED",
                stage="geometry_smoke_complete",
                iteration=int(opt.iterations),
            )
            print(f"STRICT GEOMETRY SMOKE: PASS ({smoke_path})")
            return

    # Legacy geometry constraints are retained for compatibility only.  They
    # are fail-closed when explicitly enabled and are not used by the strict
    # controlled protocol.
    constraint_system = None
    trajectory_manager = None
    reprojection_validator = None

    if (
        not strict_geometry_enabled
        and hasattr(args, 'enable_geometric_constraints')
        and args.enable_geometric_constraints
    ):
        def _initialize_legacy_constraints():
            from geometric_constraints import (
                ConstraintConfig,
                TrajectoryManagerImpl,
                EnhancedConstraintEngine,
                EnhancedReprojectionValidator,
            )
            # 加载约束配置
            config_path = getattr(args, 'constraint_config_path', None)
            if config_path and os.path.exists(config_path):
                constraint_config = ConstraintConfig.from_file(config_path)
            else:
                constraint_config = ConstraintConfig()

            # 初始化轨迹管理器
            if not getattr(args, "track_path", None) or not os.path.exists(args.track_path):
                raise FileNotFoundError(
                    f"Enabled geometric constraints require an existing track file: {args.track_path}"
                )
            trajectory_manager = TrajectoryManagerImpl(constraint_config)
            trajectories = trajectory_manager.load_trajectories(args.track_path)
            if not trajectories:
                raise RuntimeError(
                    f"Enabled geometric constraints loaded no trajectories from {args.track_path}"
                )
            print(f"[EvidenceTrack-GS] Loaded {len(trajectories)} trajectories from {args.track_path}")

            # 初始化约束引擎
            constraint_system = EnhancedConstraintEngine(constraint_config)

            # 初始化验证器 (已修复: 传入 constraint_system 而不是 config)
            reprojection_validator = EnhancedReprojectionValidator(
                constraint_engine=constraint_system,
                report_dir=os.path.join(args.model_path, "validation_reports")
            )

            print("[EvidenceTrack-GS] Geometric constraint system initialized successfully")
            return constraint_system, trajectory_manager, reprojection_validator

        constraint_system, trajectory_manager, reprojection_validator = fail_closed(
            args.model_path,
            stage="legacy_constraint_initialization",
            operation=_initialize_legacy_constraints,
        )

    # 初始化几何正则化器
    geometry_regularizer = None
    if opt.geometry_reg_enabled:
        geometry_regularizer = fail_closed(
            args.model_path,
            stage="geometry_regularizer_initialization",
            operation=lambda: create_geometry_regularizer(opt),
        )
        print(f"[Geometry Regularization] Initialized with weight={opt.geometry_reg_weight}, k_neighbors={opt.geometry_reg_k_neighbors}")

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")

    real_view_audit = []
    viewpoint_stack, pseudo_stack = None, None
    real_view_rng = random.Random(int(opt.experiment_seed))
    legacy_pseudo_rng = random.Random(int(opt.experiment_seed) + 1000003)
    pseudo_supervision_calls = 0
    pseudo_unique_view_indices = set()
    pseudo_last_supervision_iteration = None
    pseudo_camera_keys = [record.key for record in pseudo_records]
    pseudo_call_trace = []
    pseudo_usage_path = os.path.join(
        scene.model_path, "pseudo_supervision_usage.json"
    )
    if opt.enable_diffusion_pseudo_rgb:
        _write_pseudo_usage(
            pseudo_usage_path,
            calls=0,
            unique_indices=pseudo_unique_view_indices,
            pool_size=len(pseudo_records),
            camera_keys=pseudo_camera_keys,
            call_trace=pseudo_call_trace,
            interval=opt.pseudo_rgb_interval,
            start_iteration=opt.pseudo_rgb_start,
            end_iteration=opt.pseudo_rgb_end,
            last_iteration=None,
        )
    ema_loss_for_log = 0.0
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        pseudo_rgb_loss_value = torch.tensor(0.0, device="cuda")
        pseudo_rgb_cache_hit = 0.0
        pseudo_rgb_ramp = 0.0
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        # Every 500 its we increase the levels of SH up to a maximum degree
        if iteration % 500 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            real_view_rng.shuffle(viewpoint_stack)

        viewpoint_cam = viewpoint_stack.pop()
        real_view_audit.append([int(iteration), str(viewpoint_cam.image_name)])

        # Task: 修改渲染流程以使用增强的外观特征
        # Task: 确保与现有SH系数的兼容性
        # Requirements: 3.3 - Enhanced rendering with GT-DCA features

        # Update GT-DCA cache if needed (for training efficiency)
        if hasattr(gaussians, 'is_gt_dca_enabled') and gaussians.is_gt_dca_enabled():
            # Invalidate cache periodically to ensure fresh features
            if iteration % 500 == 0:
                gaussians.invalidate_gt_dca_cache()

        # 使用混合精度进行前向计算
        with autocast('cuda', enabled=scaler is not None, dtype=dtype):
            render_pkg = render(viewpoint_cam, gaussians, pipe, background)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

            # Loss
            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 =  l1_loss_mask(image, gt_image)
            loss = ((1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)))

            # Required geometry objectives must never silently become zero.
            geometric_constraint_loss = torch.tensor(0.0, device="cuda")
            constraint_result = None # 确保变量存在
            active_trajectories = [] # 确保变量存在

            strict_geometry_result = None
            if strict_geometry is not None and iteration >= opt.strict_track_start:
                strict_geometry_result = fail_closed(
                    args.model_path,
                    stage="strict_geometry_forward",
                    iteration=iteration,
                    operation=lambda: strict_geometry.compute_loss(
                        gaussians.get_xyz
                    ),
                )
                if opt.strict_track_warmup > 0:
                    strict_ramp = min(
                        1.0,
                        max(
                            0.0,
                            (iteration - opt.strict_track_start)
                            / float(opt.strict_track_warmup),
                        ),
                    )
                else:
                    strict_ramp = 1.0
                geometric_constraint_loss = (
                    strict_geometry_result["loss"]
                    * opt.strict_track_weight
                    * strict_ramp
                )
            elif constraint_system is not None and trajectory_manager is not None:
                def _compute_legacy_constraint_loss():
                    # 获取活跃轨迹
                    active_trajectories = trajectory_manager.get_active_trajectories()
                    if not active_trajectories:
                        raise RuntimeError(
                            "Enabled geometric constraints have no active trajectories"
                        )
                    cameras = scene.getTrainCameras()
                    gaussian_points = gaussians.get_xyz
                    constraint_system.compute_adaptive_weights(
                        active_trajectories,
                        image_regions=image,
                        iteration=iteration,
                    )
                    constraint_result = constraint_system.compute_reprojection_constraints(
                        active_trajectories, cameras, gaussian_points
                    )
                    multiscale_result = constraint_system.compute_multiscale_constraints(
                        active_trajectories, cameras, scales=[1.0, 0.5, 0.25]
                    )
                    constraint_weight = getattr(args, 'geometric_constraint_weight', 0.1)
                    if iteration < 1000:
                        constraint_weight *= 0.1
                    elif iteration < 5000:
                        constraint_weight *= (0.1 + 0.9 * (iteration - 1000) / 4000)
                    return (
                        (constraint_result.loss_value + multiscale_result.loss_value)
                        * constraint_weight,
                        constraint_result,
                        active_trajectories,
                    )

                (
                    geometric_constraint_loss,
                    constraint_result,
                    active_trajectories,
                ) = fail_closed(
                    args.model_path,
                    stage="legacy_constraint_forward",
                    iteration=iteration,
                    operation=_compute_legacy_constraint_loss,
                )

            # 深度损失计算
            rendered_depth = render_pkg["depth"][0]
            midas_depth = torch.tensor(viewpoint_cam.depth_image).cuda()
            rendered_depth = rendered_depth.reshape(-1, 1)
            midas_depth = midas_depth.reshape(-1, 1)

            depth_loss = min(
                                 (1 - pearson_corrcoef( - midas_depth, rendered_depth)),
                                 (1 - pearson_corrcoef(1 / (midas_depth + 200.), rendered_depth))
            )

            # The legacy depth loss remains an explicit ablation switch.
            if not getattr(args, 'disable_depth_loss', False):
                loss += opt.depth_weight * depth_loss

            # Add the required geometry objective.
            loss += geometric_constraint_loss

            # 添加几何正则化损失
            geometry_reg_loss = torch.tensor(0.0, device="cuda")
            if geometry_regularizer is not None:
                geometry_reg_loss = fail_closed(
                    args.model_path,
                    stage="geometry_regularizer_forward",
                    iteration=iteration,
                    operation=lambda: geometry_regularizer.compute_anisotropic_regularization_loss(
                        xyz=gaussians.get_xyz,
                        scaling=gaussians.get_scaling,
                        rotation=gaussians.get_rotation,
                        iteration=iteration
                    ),
                )
                loss += geometry_reg_loss

            # 伪相机渲染损失
            if (
                not opt.disable_legacy_pseudo_depth
                and iteration % args.sample_pseudo_interval == 0
                and iteration > args.start_sample_pseudo
                and iteration < args.end_sample_pseudo
            ):
                if not pseudo_stack:
                    pseudo_stack = scene.getPseudoCameras().copy()
                    legacy_pseudo_rng.shuffle(pseudo_stack)
                pseudo_cam = pseudo_stack.pop() if pseudo_stack else None

                if pseudo_cam is not None:
                    render_pkg_pseudo = render(pseudo_cam, gaussians, pipe, background)
                    rendered_depth_pseudo = render_pkg_pseudo["depth"][0]
                    midas_depth_pseudo = estimate_depth(render_pkg_pseudo["render"], mode='train')

                    rendered_depth_pseudo = rendered_depth_pseudo.reshape(-1, 1)
                    midas_depth_pseudo = midas_depth_pseudo.reshape(-1, 1)
                    depth_loss_pseudo = (1 - pearson_corrcoef(rendered_depth_pseudo, -midas_depth_pseudo)).mean()

                    if torch.isnan(depth_loss_pseudo).sum() == 0:
                        loss_scale = min((iteration - args.start_sample_pseudo) / 500., 1)
                        loss += loss_scale * opt.depth_pseudo_weight * depth_loss_pseudo

            if (
                opt.enable_diffusion_pseudo_rgb
                and iteration > opt.pseudo_rgb_start
                and iteration < opt.pseudo_rgb_end
                and (iteration - opt.pseudo_rgb_start) % opt.pseudo_rgb_interval == 0
            ):
                pseudo_index = pseudo_view_index(
                    pseudo_supervision_calls, len(pseudo_records)
                )
                pseudo_camera = pseudo_cameras[pseudo_index]
                pseudo_record = pseudo_records[pseudo_index]
                pseudo_render = render(
                    pseudo_camera, gaussians, pipe, background
                )["render"]
                pseudo_target, pseudo_mask = load_pseudo_target(
                    pseudo_record,
                    size=tuple(pseudo_render.shape[-2:]),
                    device=pseudo_render.device,
                    dtype=pseudo_render.dtype,
                )
                pseudo_terms = pseudo_rgb_loss(
                    pseudo_render,
                    pseudo_target,
                    lambda_dssim=opt.pseudo_rgb_lambda_dssim,
                    ssim_fn=ssim,
                    mask=pseudo_mask,
                )
                pseudo_rgb_ramp = min(
                    1.0,
                    (iteration - opt.pseudo_rgb_start) / 500.0,
                )
                pseudo_rgb_loss_value = pseudo_terms["total"]
                loss += (
                    pseudo_rgb_ramp
                    * opt.pseudo_rgb_weight
                    * pseudo_rgb_loss_value
                )
                pseudo_rgb_cache_hit = 1.0
                pseudo_supervision_calls += 1
                pseudo_unique_view_indices.add(pseudo_index)
                pseudo_last_supervision_iteration = iteration
                pseudo_call_trace.append(
                    {
                        "call_index": pseudo_supervision_calls - 1,
                        "iteration": int(iteration),
                        "camera_index": int(pseudo_index),
                        "camera_key": pseudo_camera_keys[pseudo_index],
                    }
                )

        # 动态调整深度权重
        if iteration > args.end_sample_pseudo:
            opt.depth_weight = 0.001

        # 使用scaler进行反向传播
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            if tb_writer and opt.enable_diffusion_pseudo_rgb:
                tb_writer.add_scalar(
                    "diffusion_pseudo/cache_hit", pseudo_rgb_cache_hit, iteration
                )
                tb_writer.add_scalar(
                    "diffusion_pseudo/loss", pseudo_rgb_loss_value.item(), iteration
                )
                tb_writer.add_scalar(
                    "diffusion_pseudo/ramp", pseudo_rgb_ramp, iteration
                )
                tb_writer.add_scalar(
                    "diffusion_pseudo/supervision_calls",
                    pseudo_supervision_calls,
                    iteration,
                )
                tb_writer.add_scalar(
                    "diffusion_pseudo/unique_views_used",
                    len(pseudo_unique_view_indices),
                    iteration,
                )
                tb_writer.add_scalar(
                    "diffusion_pseudo/pool_coverage",
                    len(pseudo_unique_view_indices) / float(len(pseudo_records)),
                    iteration,
                )

            if pseudo_rgb_cache_hit > 0.0:
                _write_pseudo_usage(
                    pseudo_usage_path,
                    calls=pseudo_supervision_calls,
                    unique_indices=pseudo_unique_view_indices,
                    pool_size=len(pseudo_records),
                    camera_keys=pseudo_camera_keys,
                    call_trace=pseudo_call_trace,
                    interval=opt.pseudo_rgb_interval,
                    start_iteration=opt.pseudo_rgb_start,
                    end_iteration=opt.pseudo_rgb_end,
                    last_iteration=pseudo_last_supervision_iteration,
                )
                print(
                    "[Difix] pseudo supervision calls="
                    f"{pseudo_supervision_calls}, unique views="
                    f"{len(pseudo_unique_view_indices)}/{len(pseudo_records)}"
                )

            if (
                strict_geometry is not None
                and iteration % opt.strict_track_log_interval == 0
            ):
                if strict_geometry_result is not None:
                    mean_error = float(
                        strict_geometry_result["mean_error"].detach().item()
                    )
                    print(
                        f"[Strict Geometry] iteration={iteration} "
                        f"mean_reprojection_error={mean_error:.6f}px"
                    )
                    if tb_writer:
                        tb_writer.add_scalar(
                            "tracks/reprojection_error", mean_error, iteration
                        )
                        tb_writer.add_scalar(
                            "tracks/geometric_loss",
                            geometric_constraint_loss.detach().item(),
                            iteration,
                        )
                if gaussians.get_xyz.grad is not None:
                    active = torch.as_tensor(
                        strict_geometry.active_track_indices,
                        device=gaussians.get_xyz.device,
                        dtype=torch.long,
                    )
                    associated = strict_geometry.associated_gaussian_ids.index_select(
                        0, active
                    )
                    associated = torch.unique(associated[associated >= 0])
                    if len(associated):
                        norms = gaussians.get_xyz.grad.index_select(
                            0, associated
                        ).norm(dim=-1)
                        coverage = float((norms > 1e-12).float().mean().item())
                        if tb_writer:
                            tb_writer.add_scalar(
                                "tracks/gradient_coverage", coverage, iteration
                            )
                            tb_writer.add_scalar(
                                "tracks/gradient_mean_norm",
                                float(norms.mean().item()),
                                iteration,
                            )
                            tb_writer.add_scalar(
                                "tracks/gradient_zero_ratio", 1.0 - coverage, iteration
                            )

            # Legacy validation is also required when the legacy constraint path is enabled.
            if (constraint_system is not None and reprojection_validator is not None and
                iteration % 100 == 0):  # 每100次迭代验证一次
                def _validate_legacy_constraints():
                    # 验证几何约束
                    if constraint_result is not None:
                        # 已修复: 传入了正确的参数并解包了返回值
                        is_valid, validation_metrics = reprojection_validator.validate_constraints(
                            constraint_result,
                            active_trajectories,
                            iteration
                        )

                        # 记录验证指标到tensorboard
                        if tb_writer:
                            tb_writer.add_scalar('geometric_constraints/constraint_satisfaction',
                                                 validation_metrics.constraint_satisfaction, iteration)
                            tb_writer.add_scalar('geometric_constraints/geometric_consistency',
                                                 validation_metrics.geometric_consistency, iteration)
                            # 已修复: 从 constraint_result 获取平均误差
                            tb_writer.add_scalar('geometric_constraints/reprojection_error_mean',
                                                 constraint_result.mean_error, iteration)
                            tb_writer.add_scalar('geometric_constraints/constraint_loss',
                                                 geometric_constraint_loss.item(), iteration)

                        # A low satisfaction score is a diagnostic result, not
                        # a hidden fallback; the computed loss remains active.
                        if not is_valid:
                            print(f"[EvidenceTrack-GS] Warning: Constraint validation failed at iteration {iteration}. "
                                  f"Satisfaction: {validation_metrics.constraint_satisfaction:.3f}")
                fail_closed(
                    args.model_path,
                    stage="legacy_constraint_validation",
                    iteration=iteration,
                    operation=_validate_legacy_constraints,
                )

            # Log and save
            # Task: 确保与现有SH系数的兼容性 - Add GT-DCA performance logging
            gt_dca_info = None
            if hasattr(gaussians, 'get_gt_dca_info'):
                gt_dca_info = gaussians.get_gt_dca_info()




            # Densification
            if  iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.prune_threshold, scene.cameras_extent, size_threshold, iteration)
                    if strict_geometry is not None:
                        strict_geometry.associate(
                            gaussians.get_xyz, scene.cameras_extent
                        )


            # Optimizer step with AMP support
            if iteration <= opt.iterations:
                if scaler is not None:
                    # 检查是否有参数有梯度，避免unscale空优化器
                    has_grads = any(p.grad is not None for group in gaussians.optimizer.param_groups for p in group['params'])
                    if has_grads:
                        scaler.unscale_(gaussians.optimizer)
                        scaler.step(gaussians.optimizer)
                        scaler.update()
                    else:
                        # 没有梯度时使用普通step
                        gaussians.optimizer.step()
                else:
                    gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            gaussians.update_learning_rate(iteration)
            if (iteration - args.start_sample_pseudo - 1) % opt.opacity_reset_interval == 0 and \
                    iteration > args.start_sample_pseudo:
                gaussians.reset_opacity()

            training_report(tb_writer, iteration, Ll1, loss, l1_loss,
                            testing_iterations, scene, render, (pipe, background),
                            geometric_constraint_loss if 'geometric_constraint_loss' in locals() else None,
                            gt_dca_info,
                            geometry_reg_loss if 'geometry_reg_loss' in locals() else None)

            if iteration > first_iter and (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            if iteration > first_iter and (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration),
                           scene.model_path + "/chkpnt" + str(iteration) + ".pth")


    if controlled_role in {"B", "SELFRENDER"}:
        _write_pseudo_usage(
            pseudo_usage_path,
            calls=pseudo_supervision_calls,
            unique_indices=pseudo_unique_view_indices,
            pool_size=len(pseudo_records),
            camera_keys=pseudo_camera_keys,
            call_trace=pseudo_call_trace,
            interval=opt.pseudo_rgb_interval,
            start_iteration=opt.pseudo_rgb_start,
            end_iteration=opt.pseudo_rgb_end,
            last_iteration=pseudo_last_supervision_iteration,
        )
        if (
            pseudo_supervision_calls < len(pseudo_records)
            or len(pseudo_unique_view_indices) != len(pseudo_records)
        ):
            raise RuntimeError(
                "Controlled Difix run did not cover the complete pseudo-view pool: "
                f"calls={pseudo_supervision_calls}, unique="
                f"{len(pseudo_unique_view_indices)}/{len(pseudo_records)}"
            )
        print(
            "[Frozen pseudo supervision] controlled coverage gate: PASS; calls="
            f"{pseudo_supervision_calls}, unique="
            f"{len(pseudo_unique_view_indices)}/{len(pseudo_records)}"
        )

    with open(os.path.join(scene.model_path, "real_view_sequence.json"), "w", encoding="utf-8") as handle:
        json.dump(real_view_audit, handle)
    write_training_status(
        scene.model_path,
        status="COMPLETED",
        stage="training_complete",
        iteration=int(opt.iterations),
    )


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer



def training_report(tb_writer, iteration, Ll1, loss, l1_loss, testing_iterations, scene : Scene, renderFunc, renderArgs, geometric_constraint_loss=None, gt_dca_info=None, geometry_reg_loss=None):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        # Record the geometry objective.
        if geometric_constraint_loss is not None:
            tb_writer.add_scalar('train_loss_patches/geometric_constraint_loss', geometric_constraint_loss.item(), iteration)

        # 记录几何正则化损失
        if geometry_reg_loss is not None:
            tb_writer.add_scalar('train_loss_patches/geometry_regularization_loss', geometry_reg_loss.item(), iteration)

        # Task: 确保与现有SH系数的兼容性 - GT-DCA performance logging
        if gt_dca_info is not None and gt_dca_info.get('status') == 'initialized':
            tb_writer.add_scalar('gt_dca/enabled', 1 if gt_dca_info.get('enabled', False) else 0, iteration)

            # Log GT-DCA performance statistics if available
            if 'performance_stats' in gt_dca_info:
                perf_stats = gt_dca_info['performance_stats']
                if 'forward_calls' in perf_stats:
                    tb_writer.add_scalar('gt_dca/forward_calls', perf_stats['forward_calls'], iteration)
                if 'average_processing_time' in perf_stats:
                    tb_writer.add_scalar('gt_dca/avg_processing_time', perf_stats['average_processing_time'], iteration)

            # Log GT-DCA configuration
            if 'config' in gt_dca_info:
                config = gt_dca_info['config']
                if hasattr(config, 'feature_dim'):
                    tb_writer.add_scalar('gt_dca/feature_dim', config.feature_dim, iteration)
                if hasattr(config, 'num_sample_points'):
                    tb_writer.add_scalar('gt_dca/num_sample_points', config.num_sample_points, iteration)

        # tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()},
                              {'name': 'train', 'cameras' : scene.getTrainCameras()})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test, psnr_test, ssim_test, lpips_test = 0.0, 0.0, 0.0, 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 8):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()

                    _mask = None
                    _psnr = psnr(image, gt_image, _mask).mean().double()
                    _ssim = ssim(image, gt_image, _mask).mean().double()
                    _lpips = lpips(image, gt_image, _mask, net_type='vgg')
                    psnr_test += _psnr
                    ssim_test += _ssim
                    lpips_test += _lpips
                psnr_test /= len(config['cameras'])
                ssim_test /= len(config['cameras'])
                lpips_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} SSIM {} LPIPS {} ".format(
                    iteration, config['name'], l1_test, psnr_test, ssim_test, lpips_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

    if tb_writer:
        tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
    torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--llff_holdout", type=int, default=0, help="Holdout factor for LLFF data. 1/N of images are used for testing. Default=0 means all for training.")

    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1000, 2000, 3000, 5000, 10000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[5000, 10000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--data_type", type=str, default="colmap", help="Type of dataset, e.g., 'colmap', 'blender', '360'")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[5000, 10000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--train_bg", action="store_true")

    # Note: Geometric constraint parameters are already defined in ModelParams and OptimizationParams classes

    # Task: 添加GT-DCA启用/禁用的配置选项
    # GT-DCA: Add arguments for GT-DCA appearance modeling
    parser.add_argument("--use_gt_dca", action="store_true", help="Enable GT-DCA enhanced appearance modeling.")
    parser.add_argument("--gt_dca_feature_dim", type=int, default=256, help="GT-DCA feature dimension.")
    parser.add_argument("--gt_dca_num_sample_points", type=int, default=8, help="Number of sampling points for GT-DCA deformable sampling.")
    parser.add_argument("--gt_dca_hidden_dim", type=int, default=128, help="Hidden dimension for GT-DCA MLPs.")
    parser.add_argument("--gt_dca_confidence_threshold", type=float, default=0.5, help="Confidence threshold for GT-DCA track points.")
    parser.add_argument("--gt_dca_min_track_points", type=int, default=4, help="Minimum number of track points required for GT-DCA.")
    parser.add_argument("--gt_dca_enable_caching", action="store_true", help="Enable GT-DCA feature caching for performance.")
    parser.add_argument("--gt_dca_dropout_rate", type=float, default=0.1, help="Dropout rate for GT-DCA modules.")
    parser.add_argument("--gt_dca_attention_heads", type=int, default=8, help="Number of attention heads for GT-DCA cross-attention.")
    # 混合精度选项 - 注意：混合精度现在通过 --mixed_precision 全局控制
    # amp_dtype 参数已移至 OptimizationParams 中统一管理

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print(args.test_iterations)

    # Validate explicit legacy geometry configuration before starting training.
    if hasattr(args, 'enable_geometric_constraints') and args.enable_geometric_constraints:
        if not setup_geometric_constraints_config(args):
            error = RuntimeError("Failed to set up enabled geometric constraints")
            mark_training_invalid(
                args.model_path,
                stage="constraint_configuration",
                error=error,
            )
            raise error

        # Print the explicit legacy configuration for the run record.
        print_geometric_constraints_summary(args)

    # Task: 添加GT-DCA启用/禁用的配置选项
    # GT-DCA: Setup GT-DCA configuration from command line arguments
    if hasattr(args, 'use_gt_dca') and args.use_gt_dca:
        from gt_dca.core.data_structures import GTDCAConfig

        # Create GT-DCA configuration optimized for Tesla T4 16GB
        gt_dca_config = GTDCAConfig(
            feature_dim=getattr(args, 'gt_dca_feature_dim', 64),   # Further reduced to 64
            hidden_dim=getattr(args, 'gt_dca_hidden_dim', 32),    # Further reduced to 32
            num_sample_points=getattr(args, 'gt_dca_num_sample_points', 2),  # Minimal sampling points
            confidence_threshold=getattr(args, 'gt_dca_confidence_threshold', 0.8),  # Higher threshold
            min_track_points=getattr(args, 'gt_dca_min_track_points', 4),
            enable_caching=True,  # Enable caching for performance
            dropout_rate=getattr(args, 'gt_dca_dropout_rate', 0.0),  # Disable dropout for speed
            attention_heads=getattr(args, 'gt_dca_attention_heads', 2),  # Minimal attention heads
            use_mixed_precision=getattr(args, 'mixed_precision', False),
            amp_dtype=getattr(args, 'amp_dtype', 'fp16')
        )

        # Attach GT-DCA configuration to args
        args.gt_dca_config = gt_dca_config

        # GT-DCA: 独立的轨迹文件处理（不依赖几何约束系统）
        if hasattr(args, 'track_path') and args.track_path:
            if os.path.exists(args.track_path):
                print(f"✅ GT-DCA轨迹文件已找到: {args.track_path}")
            else:
                print(f"⚠️ GT-DCA轨迹文件不存在: {args.track_path}")
                print("请确保轨迹文件路径正确，或生成轨迹文件后重新运行")
        else:
            print("⚠️ 未指定GT-DCA轨迹文件路径，请使用 --track_path 参数指定")

        print(f"✅ GT-DCA配置已设置: {gt_dca_config}")
    else:
        args.use_gt_dca = False
        args.gt_dca_config = None

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet, getattr(args, "experiment_seed", 1))

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    try:
        training(lp.extract(args), op.extract(args), pp.extract(args), args)
    except Exception as exc:
        mark_training_invalid(args.model_path, stage="training", error=exc)
        raise

    # All done
    print("\nTraining complete.")
