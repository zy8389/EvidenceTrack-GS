#!/usr/bin/env python3
"""Report strict-track coverage and current Track-to-Gaussian association health."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from geometric_constraints.strict_track_store import StrictTrackStore, normalize_image_name


def _summary(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "count": int(len(values)), "mean": float(values.mean()),
        "median": float(np.median(values)), "min": float(values.min()), "max": float(values.max()),
    }


def _grid_coverage(store: StrictTrackStore, grid_size: int) -> dict:
    source_keys = set(store.source_images)
    coverage = {}
    for image_name in sorted(source_keys):
        calibration = store.camera_calibration(image_name)
        width, height = int(calibration["width"]), int(calibration["height"])
        grid = np.zeros((grid_size, grid_size), dtype=np.int64)
        for index, observed_name in enumerate(store.observation_image_names):
            if normalize_image_name(observed_name) != image_name or not store.use_for_anchor[index]:
                continue
            x, y = store.observation_xy[index]
            col = min(grid_size - 1, max(0, int(np.floor(x / max(width, 1) * grid_size))))
            row = min(grid_size - 1, max(0, int(np.floor(y / max(height, 1) * grid_size))))
            grid[row, col] += 1
        coverage[image_name] = {
            "grid_size": grid_size,
            "occupied_cells": int((grid > 0).sum()),
            "total_cells": grid_size * grid_size,
            "occupancy_fraction": float((grid > 0).mean()),
            "counts": grid.tolist(),
        }
    return coverage


def report(path: Path, grid_size: int) -> dict:
    store = StrictTrackStore.load(path)
    lengths = np.diff(store.observation_offsets)
    source_lengths = np.asarray([
        store.observations_for(index)["use_for_anchor"].sum() for index in range(len(store))
    ], dtype=np.int64)
    covariance_eigvals = np.linalg.eigvalsh(store.covariance.astype(np.float64))
    condition = covariance_eigvals[:, -1] / covariance_eigvals[:, 0]
    xyz = store.xyz.astype(np.float64)
    return {
        "track_path": str(path.resolve()),
        "valid_track_count": int(len(store)),
        "all_observation_count": int(len(store.observation_xy)),
        "track_length_distribution": _summary(lengths),
        "source_track_length_distribution": _summary(source_lengths),
        "source_reprojection_error_px": _summary(store.source_reprojection_error),
        "triangulation_covariance_condition_number": _summary(condition),
        "two_dimensional_source_grid_coverage": _grid_coverage(store, grid_size),
        "three_dimensional_spatial_spread": {
            "min": xyz.min(axis=0).tolist(), "max": xyz.max(axis=0).tolist(),
            "mean": xyz.mean(axis=0).tolist(), "std": xyz.std(axis=0).tolist(),
            "axis_range": (xyz.max(axis=0) - xyz.min(axis=0)).tolist(),
        },
        "association": {
            "status": "NOT_AVAILABLE_FROM_STATIC_H5",
            "required_server_fields": [
                "associated_track_count", "unique_gaussian_count",
                "association_collision_rate", "rejected_association_count",
            ],
            "reason": "Association depends on the live Gaussian state and must be emitted by the GPU geometry gate/training log.",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("track_h5", type=Path)
    parser.add_argument("--grid-size", type=int, choices=[8, 16], default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = report(args.track_h5, args.grid_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
