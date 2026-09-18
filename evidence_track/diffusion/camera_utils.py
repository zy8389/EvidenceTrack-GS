from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Sequence

import numpy as np
import torch


def _numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def camera_payload(camera: Any, decimals: int = 7) -> dict:
    missing = [
        name for name in ("R", "T", "FoVx", "FoVy", "image_width", "image_height")
        if not hasattr(camera, name)
    ]
    if missing:
        raise AttributeError(f"Camera is missing required fields: {missing}")
    payload = {
        "R": np.round(_numpy(camera.R).reshape(3, 3), decimals).tolist(),
        "T": np.round(_numpy(camera.T).reshape(3), decimals).tolist(),
        "FoVx": round(float(camera.FoVx), decimals),
        "FoVy": round(float(camera.FoVy), decimals),
        "width": int(camera.image_width),
        "height": int(camera.image_height),
    }
    if hasattr(camera, "research_intrinsics"):
        payload["intrinsics"] = dict(camera.research_intrinsics)
    return canonical_camera_payload(payload, decimals)


def canonical_camera_payload(payload: dict, decimals: int = 7) -> dict:
    # Accept both the Phase-2 flattened/fov_x schema and the canonical Phase-2.1 schema.
    fov_x = payload.get("FoVx", payload.get("fov_x"))
    fov_y = payload.get("FoVy", payload.get("fov_y"))
    if fov_x is None or fov_y is None:
        raise ValueError("Camera payload must declare FoVx/FoVy")
    rotation = np.asarray(payload.get("R"), dtype=np.float64)
    translation = np.asarray(payload.get("T"), dtype=np.float64)
    if rotation.shape != (3, 3) or translation.shape not in {(3,), (3, 1)}:
        raise ValueError("Camera payload requires R=[3,3] and T=[3]")
    if not np.isfinite(rotation).all() or not np.isfinite(translation).all():
        raise ValueError("Camera extrinsics contain non-finite values")
    width, height = int(payload["width"]), int(payload["height"])
    fov_x, fov_y = float(fov_x), float(fov_y)
    if width <= 0 or height <= 0:
        raise ValueError("Camera width and height must be positive")
    if not all(math.isfinite(value) and 0.0 < value < math.pi for value in (fov_x, fov_y)):
        raise ValueError("Camera FoV values must be finite and lie in (0, pi)")
    result = {
        "R": np.round(rotation.reshape(3, 3), decimals).tolist(),
        "T": np.round(translation.reshape(3), decimals).tolist(),
        "FoVx": round(fov_x, decimals),
        "FoVy": round(fov_y, decimals),
        "width": width,
        "height": height,
    }
    if "intrinsics" in payload:
        k = payload["intrinsics"]
        missing = [name for name in ("fx", "fy", "cx", "cy") if name not in k]
        if missing:
            raise ValueError(f"Camera intrinsics are incomplete: {missing}")
        values = {name: float(k[name]) for name in ("fx", "fy", "cx", "cy")}
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError("Camera intrinsics contain non-finite values")
        if values["fx"] <= 0.0 or values["fy"] <= 0.0:
            raise ValueError("Camera focal lengths must be positive")
        if not (-0.5 <= values["cx"] <= width - 0.5 and -0.5 <= values["cy"] <= height - 0.5):
            raise ValueError("Camera principal point lies outside the pixel domain")
        if "width" in k and int(k["width"]) != width:
            raise ValueError("Camera/intrinsics width mismatch")
        if "height" in k and int(k["height"]) != height:
            raise ValueError("Camera/intrinsics height mismatch")
        result["intrinsics"] = {
            name: round(value, decimals) for name, value in values.items()
        }
        result["intrinsics"].update(width=result["width"], height=result["height"])
    return result


def camera_fingerprint_from_payload(payload: dict, prefix: str = "pv") -> str:
    canonical = canonical_camera_payload(payload)
    serialized = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return f"{prefix}_{hashlib.sha1(serialized.encode('utf-8')).hexdigest()[:20]}"


def camera_fingerprint(camera: Any, prefix: str = "pv") -> str:
    return camera_fingerprint_from_payload(camera_payload(camera), prefix=prefix)


def pseudo_camera_manifest_fields(camera: Any) -> dict:
    """Return explicit, redundant fields so pseudo-camera provenance is auditable."""
    payload = camera_payload(camera)
    if "intrinsics" not in payload:
        raise ValueError("Pseudo camera lacks calibrated fx/fy/cx/cy")
    required_attributes = (
        "uid",
        "camera_source",
        "calibration_source_camera",
        "pose_source_camera_names",
        "pose_uses_heldout_camera_information",
        "pose_generator",
    )
    missing = [name for name in required_attributes if not hasattr(camera, name)]
    if missing:
        raise ValueError(f"Pseudo camera lacks provenance attributes: {missing}")
    intrinsics = payload["intrinsics"]
    pose_sources = [str(name) for name in camera.pose_source_camera_names]
    if not pose_sources or len(pose_sources) != len(set(pose_sources)):
        raise ValueError("Pseudo camera pose sources must be non-empty and unique")
    fingerprint = camera_fingerprint_from_payload(payload)
    return {
        "camera_id": str(camera.uid),
        "camera_source": str(camera.camera_source),
        "calibration_source_camera": str(camera.calibration_source_camera),
        "pose_source_camera_names": pose_sources,
        "uses_heldout_pose_information": bool(
            camera.pose_uses_heldout_camera_information
        ),
        "pose_generator": str(camera.pose_generator),
        "pose_convention": (
            "upstream Camera.R storage; world-to-camera rotation is R.T, "
            "translation is T in X_cam=R.T@X_world+T"
        ),
        "camera_fingerprint": fingerprint,
        "key": fingerprint,
        "fx": intrinsics["fx"],
        "fy": intrinsics["fy"],
        "cx": intrinsics["cx"],
        "cy": intrinsics["cy"],
        "width": payload["width"],
        "height": payload["height"],
        "R": payload["R"],
        "t": payload["T"],
        "camera": payload,
    }


def _camera_center(camera: Any) -> np.ndarray:
    if hasattr(camera, "camera_center"):
        return np.asarray(_numpy(camera.camera_center), dtype=np.float64).reshape(3)
    rotation = np.asarray(_numpy(camera.R), dtype=np.float64).reshape(3, 3)
    translation = np.asarray(_numpy(camera.T), dtype=np.float64).reshape(3)
    return -(rotation @ translation)


def _view_direction(camera: Any) -> np.ndarray:
    rotation = np.asarray(_numpy(camera.R), dtype=np.float64).reshape(3, 3)
    direction = rotation[:, 2]
    return direction / max(float(np.linalg.norm(direction)), 1e-12)


def nearest_reference_camera(
    target_camera: Any,
    source_cameras: Sequence[Any],
    center_weight: float = 1.0,
    direction_weight: float = 1.0,
) -> Any:
    if not source_cameras:
        raise ValueError("source_cameras is empty")
    target_center = _camera_center(target_camera)
    target_direction = _view_direction(target_camera)
    distances = np.asarray(
        [np.linalg.norm(_camera_center(camera) - target_center) for camera in source_cameras]
    )
    positive = distances[distances > 0]
    normalizer = float(np.median(positive)) if len(positive) else 1.0
    scores = []
    for camera, distance in zip(source_cameras, distances):
        direction_distance = 1.0 - float(np.dot(_view_direction(camera), target_direction))
        scores.append(
            center_weight * float(distance) / max(normalizer, 1e-12)
            + direction_weight * direction_distance
        )
    return source_cameras[int(np.argmin(scores))]
