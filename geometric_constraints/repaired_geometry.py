"""Differentiable source-only Track-to-Gaussian reprojection constraints."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Dict, Iterable, Sequence

import numpy as np
import torch

from .strict_track_store import StrictTrackStore, normalize_image_name


@dataclass(frozen=True)
class AssociationStatistics:
    active_tracks: int
    associated_tracks: int
    unique_associated_gaussians: int
    association_collision_rate: float
    mean_association_distance: float
    p95_association_distance: float
    rejected_associations: int

    def emit(self, prefix: str = "[Strict Track Association]") -> None:
        labels = {
            "active_tracks": "Active Tracks",
            "associated_tracks": "Associated Tracks",
            "unique_associated_gaussians": "Unique Associated Gaussians",
            "association_collision_rate": "Association Collision Rate",
            "mean_association_distance": "Mean Association Distance",
            "p95_association_distance": "P95 Association Distance",
            "rejected_associations": "Rejected Associations",
        }
        for key, value in asdict(self).items():
            print(f"{prefix} {labels[key]}: {value}")


def project_points_pixel(
    points_world: torch.Tensor,
    camera,
    calibration: Dict[str, np.ndarray | float | int | str],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project world points into resized training-image pixel coordinates."""
    if (
        points_world.ndim != 2
        or points_world.shape[1] != 3
        or points_world.shape[0] == 0
        or not points_world.is_floating_point()
    ):
        raise ValueError("Projection points must be a non-empty floating [N,3] tensor")
    if not bool(torch.isfinite(points_world).all()):
        raise ValueError("Projection points contain NaN or Inf")
    if hasattr(camera, "world_view_transform"):
        world_to_camera = camera.world_view_transform.transpose(0, 1).to(
            device=points_world.device, dtype=points_world.dtype
        )
        rotation = world_to_camera[:3, :3]
        translation = world_to_camera[:3, 3]
    else:
        rotation = torch.as_tensor(
            calibration["rotation_w2c"],
            device=points_world.device,
            dtype=points_world.dtype,
        )
        translation = torch.as_tensor(
            calibration["translation_w2c"],
            device=points_world.device,
            dtype=points_world.dtype,
        )
    point_camera = points_world @ rotation.transpose(0, 1) + translation
    depth = point_camera[:, 2]

    target_width = int(getattr(camera, "image_width", calibration["width"]))
    target_height = int(getattr(camera, "image_height", calibration["height"]))
    if target_width <= 0 or target_height <= 0:
        raise ValueError("Projection camera dimensions must be positive")
    intrinsics = getattr(camera, "research_intrinsics", calibration)
    from diffusion_guidance.calibration_guard import resized_intrinsics

    intrinsics = resized_intrinsics(intrinsics, target_width, target_height)
    fx, fy = float(intrinsics["fx"]), float(intrinsics["fy"])
    cx, cy = float(intrinsics["cx"]), float(intrinsics["cy"])
    if not all(math.isfinite(value) for value in (fx, fy, cx, cy)):
        raise ValueError("Projection intrinsics contain NaN or Inf")
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError("Projection focal lengths must be positive")
    if not bool(torch.isfinite(rotation).all()) or not bool(
        torch.isfinite(translation).all()
    ):
        raise ValueError("Projection extrinsics contain NaN or Inf")
    safe_depth = torch.clamp(depth, min=torch.finfo(points_world.dtype).eps)
    u = fx * point_camera[:, 0] / safe_depth + cx
    v = fy * point_camera[:, 1] / safe_depth + cy
    return torch.stack([u, v], dim=-1), depth


class StrictGeometryManager:
    def __init__(
        self,
        store: StrictTrackStore,
        source_cameras: Sequence,
        *,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
        min_length: int = 2,
        min_quality: float = 0.05,
        max_tracks: int = 0,
        huber_delta: float = 2.0,
        association_chunk_size: int = 1024,
        max_association_distance_ratio: float = 0.05,
        collision_warning_rate: float = 0.25,
    ):
        self.store = store
        self.device = torch.device(device)
        self.dtype = dtype
        if not dtype.is_floating_point:
            raise ValueError("Strict geometry dtype must be floating point")
        if isinstance(min_length, bool) or int(min_length) < 2:
            raise ValueError("min_length must be an integer >= 2")
        if not math.isfinite(float(min_quality)) or not 0.0 <= float(min_quality) <= 1.0:
            raise ValueError("min_quality must be finite and lie in [0, 1]")
        if isinstance(max_tracks, bool) or int(max_tracks) < 0:
            raise ValueError("max_tracks must be a non-negative integer")
        self.huber_delta = float(huber_delta)
        self.association_chunk_size = int(association_chunk_size)
        self.max_association_distance_ratio = float(max_association_distance_ratio)
        self.collision_warning_rate = float(collision_warning_rate)
        if not math.isfinite(self.huber_delta) or self.huber_delta <= 0.0:
            raise ValueError("huber_delta must be finite and positive")
        if isinstance(association_chunk_size, bool) or self.association_chunk_size <= 0:
            raise ValueError("association_chunk_size must be a positive integer")
        if (
            not math.isfinite(self.max_association_distance_ratio)
            or self.max_association_distance_ratio < 0.0
        ):
            raise ValueError(
                "max_association_distance_ratio must be finite and non-negative"
            )
        if (
            not math.isfinite(self.collision_warning_rate)
            or not 0.0 <= self.collision_warning_rate <= 1.0
        ):
            raise ValueError("collision_warning_rate must lie in [0, 1]")
        source_cameras = list(source_cameras)
        camera_names = [
            normalize_image_name(camera.image_name) for camera in source_cameras
        ]
        if len(camera_names) != len(set(camera_names)):
            raise ValueError("Source Scene camera names must be unique")
        self.camera_by_name = dict(zip(camera_names, source_cameras))
        expected = set(store.source_images)
        if expected != set(self.camera_by_name):
            raise RuntimeError(
                f"Strict source cameras mismatch: H5={sorted(expected)}, "
                f"scene={sorted(self.camera_by_name)}"
            )

        lengths = np.zeros(len(store), dtype=np.int32)
        flat_track_indices = np.repeat(
            np.arange(len(store), dtype=np.int64),
            np.diff(store.observation_offsets).astype(np.int64),
        )
        for track_index in range(len(store)):
            start = int(store.observation_offsets[track_index])
            end = int(store.observation_offsets[track_index + 1])
            lengths[track_index] = int(store.use_for_anchor[start:end].sum())
        active = np.flatnonzero(
            np.logical_and(lengths >= int(min_length), store.quality >= float(min_quality))
        )
        if max_tracks > 0 and len(active) > max_tracks:
            order = np.argsort(store.quality[active])[::-1][:max_tracks]
            active = active[order]
        if len(active) == 0:
            raise RuntimeError("No strict tracks pass the active-track filters")
        self.active_track_indices = np.asarray(active, dtype=np.int64)
        self._active_lookup = np.zeros(len(store), dtype=bool)
        self._active_lookup[self.active_track_indices] = True
        source_mask = store.use_for_anchor & self._active_lookup[flat_track_indices]
        self.observation_track_indices = torch.as_tensor(
            flat_track_indices[source_mask], device=self.device, dtype=torch.long
        )
        self.observation_xy = torch.as_tensor(
            store.observation_xy[source_mask], device=self.device, dtype=self.dtype
        )
        self.observation_confidence = torch.as_tensor(
            store.observation_confidence[source_mask], device=self.device, dtype=self.dtype
        )
        self.observation_image_names = np.asarray(
            [normalize_image_name(v) for v in store.observation_image_names[source_mask]],
            dtype=object,
        )
        self.anchors = torch.as_tensor(store.xyz, device=self.device, dtype=self.dtype)
        self.quality = torch.as_tensor(store.quality, device=self.device, dtype=self.dtype)
        self.associated_gaussian_ids = torch.full(
            (len(store),), -1, device=self.device, dtype=torch.long
        )
        self.association_distances = torch.full(
            (len(store),), float("inf"), device=self.device, dtype=self.dtype
        )
        self.last_association_statistics = None

    @classmethod
    def from_path(cls, path: str, source_cameras: Sequence, **kwargs) -> "StrictGeometryManager":
        return cls(StrictTrackStore.load(path), source_cameras, **kwargs)

    def associate(self, gaussian_xyz: torch.Tensor, scene_radius: float) -> AssociationStatistics:
        if gaussian_xyz.ndim != 2 or gaussian_xyz.shape[1] != 3 or len(gaussian_xyz) == 0:
            raise ValueError("Gaussian XYZ must be a non-empty [N,3] tensor")
        if not gaussian_xyz.is_floating_point():
            raise ValueError("Gaussian XYZ must use a floating dtype")
        if gaussian_xyz.device != self.device:
            raise ValueError(
                f"Gaussian XYZ device {gaussian_xyz.device} differs from geometry device {self.device}"
            )
        if not bool(torch.isfinite(gaussian_xyz).all()):
            raise ValueError("Gaussian XYZ contains NaN or Inf")
        if not math.isfinite(float(scene_radius)) or float(scene_radius) <= 0.0:
            raise ValueError("scene_radius must be finite and positive")
        active_tensor = torch.as_tensor(
            self.active_track_indices, device=self.device, dtype=torch.long
        )
        anchors = self.anchors.index_select(0, active_tensor)
        ids, distances = [], []
        with torch.no_grad():
            for start in range(0, len(anchors), self.association_chunk_size):
                chunk = anchors[start : start + self.association_chunk_size]
                pairwise = torch.cdist(chunk, gaussian_xyz.detach())
                values, indices = pairwise.min(dim=1)
                ids.append(indices)
                distances.append(values)
            nearest_ids = torch.cat(ids)
            nearest_distances = torch.cat(distances)
            threshold = self.max_association_distance_ratio * float(scene_radius)
            accepted = (
                nearest_distances <= threshold
                if self.max_association_distance_ratio > 0
                else torch.ones_like(nearest_distances, dtype=torch.bool)
            )
            self.associated_gaussian_ids.fill_(-1)
            self.association_distances.fill_(float("inf"))
            self.associated_gaussian_ids[active_tensor[accepted]] = nearest_ids[accepted]
            self.association_distances[active_tensor[accepted]] = nearest_distances[accepted]

        accepted_ids = nearest_ids[accepted]
        accepted_distances = nearest_distances[accepted]
        associated = int(accepted.sum().item())
        unique = int(torch.unique(accepted_ids).numel()) if associated else 0
        collision_rate = (associated - unique) / associated if associated else 0.0
        statistics = AssociationStatistics(
            active_tracks=len(self.active_track_indices),
            associated_tracks=associated,
            unique_associated_gaussians=unique,
            association_collision_rate=float(collision_rate),
            mean_association_distance=(
                float(accepted_distances.mean().item()) if associated else float("nan")
            ),
            p95_association_distance=(
                float(torch.quantile(accepted_distances, 0.95).item())
                if associated else float("nan")
            ),
            rejected_associations=len(self.active_track_indices) - associated,
        )
        self.last_association_statistics = statistics
        statistics.emit()
        if statistics.association_collision_rate > self.collision_warning_rate:
            print(
                "[Strict Track Association] WARNING: collision rate is high. "
                "Report it before considering unique assignment, a stricter distance "
                "threshold, or best-quality track filtering. Hungarian matching is not "
                "enabled in Phase 2.1."
            )
        return statistics

    def _observation_selection(self, track_indices: Iterable[int] | None) -> torch.Tensor:
        associated = self.associated_gaussian_ids.index_select(
            0, self.observation_track_indices
        ) >= 0
        if track_indices is None:
            return associated
        selected = torch.zeros(len(self.store), device=self.device, dtype=torch.bool)
        values = list(track_indices)
        if any(isinstance(value, bool) or not 0 <= int(value) < len(self.store) for value in values):
            raise ValueError("track_indices contains an invalid Track index")
        indices = torch.as_tensor(values, device=self.device, dtype=torch.long)
        if len(indices):
            selected[indices] = True
        return associated & selected.index_select(0, self.observation_track_indices)

    def compute_loss(
        self,
        gaussian_xyz: torch.Tensor,
        *,
        track_indices: Iterable[int] | None = None,
    ) -> Dict[str, torch.Tensor | int]:
        if (
            gaussian_xyz.ndim != 2
            or gaussian_xyz.shape[1] != 3
            or gaussian_xyz.shape[0] == 0
            or gaussian_xyz.device != self.device
        ):
            raise ValueError("Gaussian XYZ must be a non-empty [N,3] tensor on the geometry device")
        if not bool(torch.isfinite(gaussian_xyz).all()):
            raise RuntimeError("Gaussian XYZ contains NaN or Inf")
        selection = self._observation_selection(track_indices)
        if not bool(selection.any()):
            raise RuntimeError("No associated source observations are available")
        losses, errors, weights = [], [], []
        selected_names = self.observation_image_names[
            selection.detach().cpu().numpy()
        ]
        selected_track_indices = self.observation_track_indices[selection]
        selected_xy = self.observation_xy[selection]
        selected_confidence = self.observation_confidence[selection]
        for image_name in sorted(set(selected_names.tolist())):
            name_mask_np = selected_names == image_name
            name_mask = torch.as_tensor(name_mask_np, device=self.device, dtype=torch.bool)
            track_ids = selected_track_indices[name_mask]
            gaussian_ids = self.associated_gaussian_ids.index_select(0, track_ids)
            points = gaussian_xyz.index_select(0, gaussian_ids)
            camera = self.camera_by_name[image_name]
            calibration = self.store.camera_calibration(image_name)
            prediction, depth = project_points_pixel(points, camera, calibration)
            target = selected_xy[name_mask]
            scale_x = int(getattr(camera, "image_width", calibration["width"])) / float(calibration["width"])
            scale_y = int(getattr(camera, "image_height", calibration["height"])) / float(calibration["height"])
            target = (target + 0.5) * torch.tensor(
                [scale_x, scale_y], device=self.device, dtype=self.dtype
            ) - 0.5
            error = torch.linalg.norm(prediction - target, dim=-1)
            valid_depth = depth > torch.finfo(self.dtype).eps
            weight = self.quality.index_select(0, track_ids) * selected_confidence[name_mask]
            weight = weight * valid_depth.to(self.dtype)
            delta = self.huber_delta
            robust = torch.where(
                error <= delta,
                0.5 * error.square(),
                delta * (error - 0.5 * delta),
            )
            losses.append(robust)
            errors.append(error)
            weights.append(weight)
        all_losses = torch.cat(losses)
        all_errors = torch.cat(errors)
        all_weights = torch.cat(weights)
        if not bool(torch.isfinite(all_losses).all()) or not bool(
            torch.isfinite(all_errors).all()
        ):
            raise RuntimeError("Non-finite source reprojection loss or error")
        if not bool(torch.isfinite(all_weights).all()) or bool((all_weights < 0).any()):
            raise RuntimeError("Source reprojection weights must be finite and non-negative")
        if float(all_weights.sum().detach()) <= 0.0:
            raise RuntimeError("All source observations invalid; refusing silent zero geometry loss")
        denominator = all_weights.sum()
        return {
            "loss": (all_losses * all_weights).sum() / denominator,
            "mean_error": (all_errors * all_weights).sum() / denominator,
            "errors": all_errors,
            "weights": all_weights,
            "observation_count": int(all_errors.numel()),
        }

    def gradient_probe(
        self,
        gaussian_xyz: torch.Tensor,
        *,
        count: int = 8,
        offset: float = 1e-4,
    ) -> Dict[str, float]:
        if isinstance(count, bool) or int(count) <= 0:
            raise ValueError("Gradient probe count must be positive")
        if not math.isfinite(float(offset)) or float(offset) <= 0.0:
            raise ValueError("Gradient probe offset must be finite and positive")
        active = torch.as_tensor(
            self.active_track_indices, device=self.device, dtype=torch.long
        )
        associated_mask = self.associated_gaussian_ids.index_select(0, active) >= 0
        tracks = active[associated_mask][:count]
        if len(tracks) == 0:
            raise RuntimeError("Geometry gradient probe has no associated tracks")
        gaussian_ids = torch.unique(self.associated_gaussian_ids.index_select(0, tracks))
        saved = gaussian_xyz.index_select(0, gaussian_ids).detach().clone()
        if gaussian_xyz.grad is not None:
            gaussian_xyz.grad.zero_()
        try:
            with torch.no_grad():
                gaussian_xyz[gaussian_ids, 0] += float(offset)
            result = self.compute_loss(gaussian_xyz, track_indices=tracks.tolist())
            result["loss"].backward()
            norms = gaussian_xyz.grad.index_select(0, gaussian_ids).norm(dim=-1)
            coverage = float((norms > 1e-12).float().mean().item())
            return {
                "loss": float(result["loss"].detach().item()),
                "observation_count": int(result["observation_count"]),
                "coverage": coverage,
                "zero_gradient_ratio": 1.0 - coverage,
                "mean_gradient_norm": float(norms.mean().item()),
            }
        finally:
            with torch.no_grad():
                gaussian_xyz[gaussian_ids] = saved
            if gaussian_xyz.grad is not None:
                gaussian_xyz.grad.zero_()
