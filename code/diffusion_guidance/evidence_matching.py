"""Training-free Track evidence baselines and resolution-aware metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .evidence_features import SpatialFeatureMap, feature_grid_pixel_coordinates


PCK_THRESHOLDS = (3, 5, 8, 16)


def project_anchor_numpy(xyz: np.ndarray, calibration: Dict) -> np.ndarray:
    rotation = np.asarray(calibration["rotation_w2c"], dtype=np.float64)
    translation = np.asarray(calibration["translation_w2c"], dtype=np.float64)
    point_camera = rotation @ np.asarray(xyz, dtype=np.float64) + translation
    if point_camera[2] <= 1e-10:
        return np.asarray([np.nan, np.nan], dtype=np.float64)
    return np.asarray(
        [
            calibration["fx"] * point_camera[0] / point_camera[2] + calibration["cx"],
            calibration["fy"] * point_camera[1] / point_camera[2] + calibration["cy"],
        ],
        dtype=np.float64,
    )


def error_metrics(
    predicted: np.ndarray, ground_truth: np.ndarray, *, feature_stride: float,
) -> Dict[str, float | int]:
    """PCK counts prediction failures in its denominator, not as missing data.

    Mean/median are conditional on valid predictions; pairwise reports must
    also show failure rate and common support. Do not compare conditional means
    alone when coverage differs.
    """
    predicted = np.asarray(predicted, dtype=np.float64)
    ground_truth = np.asarray(ground_truth, dtype=np.float64)
    if predicted.shape != ground_truth.shape or predicted.ndim != 2 or predicted.shape[1] != 2:
        raise ValueError("predicted and ground_truth must share shape [N,2]")
    eligible = np.isfinite(ground_truth).all(axis=1)
    success = eligible & np.isfinite(predicted).all(axis=1)
    errors = np.linalg.norm(predicted[success] - ground_truth[success], axis=1)
    total = int(eligible.sum())
    result = {
        "count": total, "successful_count": int(success.sum()),
        "failure_rate": float(1 - success.sum() / total) if total else float("nan"),
        "mean_error": float(errors.mean()) if len(errors) else float("nan"),
        "median_error": float(np.median(errors)) if len(errors) else float("nan"),
        "normalized_error_feature_stride": float(errors.mean() / feature_stride)
            if len(errors) and feature_stride > 0 else float("nan"),
    }
    result.update({f"pck{t}": float((errors <= t).sum() / total)
                   if total else float("nan") for t in PCK_THRESHOLDS})
    return result


def local_feature_match(
    queries: torch.Tensor,
    target: SpatialFeatureMap,
    centers_xy: np.ndarray,
    *,
    window_radius: float,
    temperature: float,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    queries = F.normalize(queries.to(target.features.device), dim=-1)
    coordinates = feature_grid_pixel_coordinates(target)
    flattened_coordinates = coordinates.reshape(-1, 2)
    flattened_features = target.features.permute(1, 2, 0).reshape(-1, target.features.shape[0])
    predictions, entropy, peak_probability, peak_similarity = [], [], [], []
    for query, center in zip(queries, np.asarray(centers_xy, dtype=np.float64)):
        center_tensor = torch.as_tensor(
            center, device=flattened_coordinates.device, dtype=flattened_coordinates.dtype
        )
        delta = torch.abs(flattened_coordinates - center_tensor)
        mask = (delta[:, 0] <= window_radius) & (delta[:, 1] <= window_radius)
        if not bool(mask.any()):
            predictions.append(np.asarray([np.nan, np.nan]))
            entropy.append(float("nan"))
            peak_probability.append(float("nan"))
            peak_similarity.append(float("nan"))
            continue
        candidate_coordinates = flattened_coordinates[mask]
        similarities = flattened_features[mask] @ query
        probability = torch.softmax(similarities / temperature, dim=0)
        prediction = (probability[:, None] * candidate_coordinates).sum(dim=0)
        predictions.append(prediction.detach().cpu().numpy())
        entropy.append(
            float((-(probability * torch.log(probability + 1e-12)).sum()).item())
        )
        peak_probability.append(float(probability.max().item()))
        peak_similarity.append(float(similarities.max().item()))
    return np.stack(predictions), {
        "entropy": np.asarray(entropy),
        "peak_probability": np.asarray(peak_probability),
        "peak_similarity": np.asarray(peak_similarity),
    }


@dataclass(frozen=True)
class HardNegativeSelection:
    """Auditable hard-negative assignments for one target image.

    ``indices`` is only a tensor-indexing convenience.  A selection is a
    valid hard negative *only* when ``within_radius`` is true and its mode is
    ``local_hard_negative``.  ``global_fallback`` is retained for secondary
    analysis, while ``missing`` never substitutes the query itself as a
    negative.
    """

    indices: np.ndarray
    projection_distance_px: np.ndarray
    within_radius: np.ndarray
    selection_mode: np.ndarray


def select_hard_negatives(
    source_queries: torch.Tensor,
    projected_centers: np.ndarray,
    *,
    nearby_radius: float,
) -> HardNegativeSelection:
    """Select local hard negatives and explicitly label nonlocal fallbacks.

    Matching with a global fallback is useful as a secondary counterfactual,
    but it does not share the local projection prior and therefore cannot
    enter the primary identity margin.
    """
    if not np.isfinite(nearby_radius) or nearby_radius < 0.0:
        raise ValueError("nearby_radius must be a finite non-negative value")
    normalized = F.normalize(source_queries.detach().cpu(), dim=-1)
    similarity = normalized @ normalized.transpose(0, 1)
    centers = np.asarray(projected_centers, dtype=np.float64)
    if centers.ndim != 2 or centers.shape[1] != 2:
        raise ValueError("projected_centers must have shape [N,2]")
    if len(normalized) != len(centers):
        raise ValueError("source_queries and projected_centers must have equal length")
    distances = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=-1)
    count = len(centers)
    indices = np.empty(count, dtype=np.int64)
    selected_distances = np.full(count, np.nan, dtype=np.float64)
    within_radius = np.zeros(count, dtype=bool)
    modes = np.full(count, "missing", dtype=object)
    other_indices = np.arange(count)
    for index in range(count):
        nearby = np.flatnonzero(
            (distances[index] <= nearby_radius)
            & (other_indices != index)
            & np.isfinite(distances[index])
        )
        if len(nearby):
            candidate_similarity = similarity[index, torch.as_tensor(nearby)].numpy()
            chosen = int(nearby[int(np.argmax(candidate_similarity))])
            indices[index] = chosen
            selected_distances[index] = distances[index, chosen]
            within_radius[index] = True
            modes[index] = "local_hard_negative"
            continue

        # Keep a nonlocal candidate only as an explicitly invalid, secondary
        # diagnostic.  With a single track there is no negative at all.
        global_candidates = np.flatnonzero(
            (other_indices != index) & np.isfinite(distances[index])
        )
        if len(global_candidates) == 0:
            indices[index] = index
            continue
        candidate_similarity = similarity[
            index, torch.as_tensor(global_candidates)
        ].numpy()
        chosen = int(global_candidates[int(np.argmax(candidate_similarity))])
        indices[index] = chosen
        selected_distances[index] = distances[index, chosen]
        modes[index] = "global_fallback"
    return HardNegativeSelection(
        indices=indices,
        projection_distance_px=selected_distances,
        within_radius=within_radius,
        selection_mode=modes,
    )


def choose_hard_shuffled_indices(
    source_queries: torch.Tensor,
    projected_centers: np.ndarray,
    *,
    nearby_radius: float,
    return_metadata: bool = False,
):
    """Compatibility wrapper around the auditable hard-negative selection."""
    selection = select_hard_negatives(
        source_queries, projected_centers, nearby_radius=nearby_radius
    )
    if return_metadata:
        return selection.indices, {
            "within_radius": selection.within_radius,
            "projection_distance_px": selection.projection_distance_px,
            "selection_mode": selection.selection_mode,
        }
    return selection.indices


def shifted_centers(
    projected_centers: np.ndarray,
    shift_pixels: float,
) -> np.ndarray:
    shifted = np.asarray(projected_centers, dtype=np.float64).copy()
    shifted[:, 0] += float(shift_pixels)
    return shifted


def valid_shift_recovery_mask(
    shifted: np.ndarray,
    ground_truth: np.ndarray,
    window_radius: float,
) -> np.ndarray:
    delta = np.abs(
        np.asarray(shifted, dtype=np.float64)
        - np.asarray(ground_truth, dtype=np.float64)
    )
    return (
        np.isfinite(delta).all(axis=1)
        & (delta[:, 0] <= window_radius)
        & (delta[:, 1] <= window_radius)
    )
