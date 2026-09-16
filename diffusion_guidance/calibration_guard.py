"""Pixel-center calibrated projection for the inherited CUDA rasterizer.

The CUDA kernel uses ndc2Pix(v,S)=((v+1)*S-1)/2. This module makes
its projected centers agree with K[R|t] and the resized source observations.
It changes camera calibration only; it is not a learned model component.
"""
from __future__ import annotations
import math
import numpy as np
import torch


INTRINSIC_FIELDS = ("fx", "fy", "cx", "cy", "width", "height")
PSEUDO_CAMERA_SOURCE = "source_camera_interpolation_only"
PSEUDO_POSE_GENERATOR = "pinned_upstream_generate_random_poses_from_train_cameras"


def _validated_intrinsics(calibration: dict) -> dict:
    missing = [name for name in INTRINSIC_FIELDS if name not in calibration]
    if missing:
        raise ValueError(f"Missing intrinsic calibration fields: {missing}")
    result = {
        name: int(calibration[name]) if name in {"width", "height"}
        else float(calibration[name])
        for name in INTRINSIC_FIELDS
    }
    values = np.asarray([result[name] for name in INTRINSIC_FIELDS], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Camera intrinsics must be finite")
    if result["width"] <= 0 or result["height"] <= 0:
        raise ValueError("Camera dimensions must be positive")
    if result["fx"] <= 0 or result["fy"] <= 0:
        raise ValueError("Camera focal lengths must be positive")
    return result


def resized_intrinsics(calibration: dict, width: int, height: int) -> dict:
    calibration = _validated_intrinsics(calibration)
    width, height = int(width), int(height)
    if width <= 0 or height <= 0:
        raise ValueError("Resized camera dimensions must be positive")
    sx = width / float(calibration["width"])
    sy = height / float(calibration["height"])
    return dict(
        fx=calibration["fx"] * sx,
        fy=calibration["fy"] * sy,
        cx=(calibration["cx"] + 0.5) * sx - 0.5,
        cy=(calibration["cy"] + 0.5) * sy - 0.5,
        width=width,
        height=height,
    )


def intrinsic_projection_matrix(k: dict, near: float, far: float, *, device='cpu', dtype=torch.float32):
    k = _validated_intrinsics(k)
    if near <= 0 or far <= near or k['fx'] <= 0 or k['fy'] <= 0:
        raise ValueError('Invalid intrinsics or clipping planes')
    p = torch.zeros((4, 4), device=device, dtype=dtype)
    p[0, 0] = 2*k['fx']/k['width']; p[1, 1] = 2*k['fy']/k['height']
    p[0, 2] = (2*k['cx']+1)/k['width']-1
    p[1, 2] = (2*k['cy']+1)/k['height']-1
    p[2, 2] = far/(far-near); p[2, 3] = -far*near/(far-near)
    p[3, 2] = 1
    return p


def apply_intrinsics(camera, k: dict):
    width, height = int(camera.image_width), int(camera.image_height)
    k = _validated_intrinsics(k)
    if (k['width'], k['height']) != (width, height):
        k = resized_intrinsics(k, width, height)
    view = camera.world_view_transform
    camera.projection_matrix = intrinsic_projection_matrix(
        k, camera.znear, camera.zfar, device=view.device, dtype=view.dtype).T.contiguous()
    camera.full_proj_transform = view @ camera.projection_matrix
    camera.FoVx = 2*math.atan(width/(2*k['fx']))
    camera.FoVy = 2*math.atan(height/(2*k['fy']))
    camera.research_intrinsics = dict(k)
    return camera


def calibrate_pseudo_cameras(scene, train_cameras, test_cameras=()):
    """Attach audited calibration and source-only pose provenance to live pseudo cameras."""
    pseudo_cameras = list(scene.getPseudoCameras())
    if not pseudo_cameras:
        scene.pseudo_camera_provenance = PSEUDO_CAMERA_SOURCE
        scene.pseudo_camera_pose_source_names = []
        scene.pseudo_camera_heldout_pose_source_names = []
        scene.pseudo_camera_uses_heldout_pose_information = False
        scene.pseudo_camera_pose_generator = PSEUDO_POSE_GENERATOR
        return scene
    if not train_cameras:
        raise RuntimeError("Pseudo cameras exist without a source camera")

    from geometric_constraints.strict_track_store import normalize_image_name

    source_names = [normalize_image_name(camera.image_name) for camera in train_cameras]
    heldout_names = [normalize_image_name(camera.image_name) for camera in test_cameras]
    if len(source_names) != len(set(source_names)):
        raise RuntimeError("Source camera names are not unique")
    if set(source_names) & set(heldout_names):
        raise RuntimeError("Source and held-out camera sets overlap")
    reference = train_cameras[0]
    if not hasattr(reference, "research_intrinsics"):
        raise RuntimeError("Source calibration was not applied before pseudo cameras")

    for index, pseudo in enumerate(pseudo_cameras):
        apply_intrinsics(pseudo, dict(reference.research_intrinsics))
        pseudo.uid = f"pseudo_{index:05d}"
        pseudo.source_camera_name = str(reference.image_name)
        pseudo.calibration_source_camera = normalize_image_name(reference.image_name)
        pseudo.camera_source = PSEUDO_CAMERA_SOURCE
        pseudo.pose_source_camera_names = list(source_names)
        pseudo.pose_uses_heldout_camera_information = False
        pseudo.pose_generator = PSEUDO_POSE_GENERATOR

    scene.pseudo_camera_provenance = PSEUDO_CAMERA_SOURCE
    scene.pseudo_camera_reference_source = normalize_image_name(reference.image_name)
    scene.pseudo_camera_pose_source_names = list(source_names)
    scene.pseudo_camera_heldout_pose_source_names = []
    scene.pseudo_camera_uses_heldout_pose_information = False
    scene.pseudo_camera_pose_generator = PSEUDO_POSE_GENERATOR
    return scene


def calibrate_scene(scene, track_path: str):
    from geometric_constraints.strict_track_store import StrictTrackStore, normalize_image_name
    store = StrictTrackStore.load(track_path)
    # Only the given calibration is consulted. No image pixels or full COLMAP tracks.
    train_cameras = list(scene.getTrainCameras())
    test_cameras = list(scene.getTestCameras())
    for camera in train_cameras + test_cameras:
        key = normalize_image_name(camera.image_name)
        try:
            k = store.camera_calibration(key)
        except KeyError as exc:
            raise RuntimeError(f'Missing calibration for evaluation/training camera {key}') from exc
        apply_intrinsics(camera, resized_intrinsics(k, camera.image_width, camera.image_height))
    # The pinned upstream constructs LLFF/360 pseudo poses from the live train
    # camera list only and copies calibration from its first element.
    return calibrate_pseudo_cameras(scene, train_cameras, test_cameras)
