#!/usr/bin/env python3
"""Failure-aware paired analysis for Correct-vs-Wrong identity evidence."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from diffusion_guidance.evidence_protocol import identity_protocol_from_metadata

VARIANTS = (
    "GS Render + feature matching",
    "Difix Output + feature matching",
    "Real Target + feature matching [reference diagnostic]",
)


def capped_error(value: float, penalty: float) -> tuple[float, bool]:
    valid = math.isfinite(value)
    if valid and value < 0.0:
        raise ValueError(f"Pixel errors must be non-negative, got {value}")
    return (min(value, penalty) if valid else penalty), valid


def _parse_bool(value, label: str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized not in {"true", "false"}:
        raise ValueError(f"{label} must be true or false, got {value!r}")
    return normalized == "true"


def _canonical_float(value, label: str) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if math.isnan(number):
        return "nan"
    if math.isinf(number):
        return "inf" if number > 0 else "-inf"
    return number.hex()


def _strict_support_entry(row: dict) -> dict:
    required = (
        "image_name",
        "track_id",
        "image_width",
        "image_height",
        "feature_width",
        "feature_height",
        "feature_stride",
        "projection_x",
        "projection_y",
        "ground_truth_x",
        "ground_truth_y",
        "hard_negative_track_id",
        "hard_negative_projection_distance_px",
        "hard_negative_within_radius",
        "hard_negative_selection_mode",
        "hard_negative_valid",
        "hard_negative_radius",
        "feature_backend",
    )
    missing = [field for field in required if field not in row]
    if missing:
        raise ValueError(f"Cannot construct exact paired support: missing {missing}")
    raw_negative_id = str(row["hard_negative_track_id"]).strip()
    negative_id = (
        ""
        if raw_negative_id.lower() in {"", "none", "null", "nan"}
        else raw_negative_id
    )
    negative_distance = float(row["hard_negative_projection_distance_px"])
    if not math.isfinite(negative_distance) and _parse_bool(
        row["hard_negative_valid"], "hard_negative_valid"
    ):
        raise ValueError("A valid hard negative must have a finite projection distance")
    if math.isfinite(negative_distance) and negative_distance < 0.0:
        raise ValueError("Hard-negative projection distance must be non-negative")
    return {
        "image_name": str(row["image_name"]),
        "track_id": str(row["track_id"]),
        "image_width": int(row["image_width"]),
        "image_height": int(row["image_height"]),
        "feature_width": int(row["feature_width"]),
        "feature_height": int(row["feature_height"]),
        "effective_feature_stride_float64": _canonical_float(
            row["feature_stride"], "feature_stride"
        ),
        "projection_xy_float64": [
            _canonical_float(row["projection_x"], "projection_x"),
            _canonical_float(row["projection_y"], "projection_y"),
        ],
        "ground_truth_xy_float64": [
            _canonical_float(row["ground_truth_x"], "ground_truth_x"),
            _canonical_float(row["ground_truth_y"], "ground_truth_y"),
        ],
        "hard_negative_track_id": negative_id or None,
        "hard_negative_projection_distance_float64": _canonical_float(
            negative_distance,
            "hard_negative_projection_distance_px",
        ),
        "hard_negative_within_radius": _parse_bool(
            row["hard_negative_within_radius"], "hard_negative_within_radius"
        ),
        "hard_negative_selection_mode": str(row["hard_negative_selection_mode"]),
        "hard_negative_valid": _parse_bool(
            row["hard_negative_valid"], "hard_negative_valid"
        ),
        "hard_negative_radius": _canonical_float(
            row["hard_negative_radius"], "hard_negative_radius"
        ),
        "feature_backend": str(row["feature_backend"]),
    }


def _support_digest(entries: list[dict], schema: str) -> str:
    canonical = json.dumps(
        {"schema": schema, "entries": entries},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(canonical).hexdigest()


def analyze(
    rows: list[dict],
    radius: float,
    *,
    metadata: dict | None = None,
    require_exact_support: bool = False,
) -> dict:
    if not math.isfinite(float(radius)) or float(radius) <= 0.0:
        raise ValueError("Report radius must be a finite positive value")
    keyed = defaultdict(dict)
    for row in rows:
        if float(row["window_radius"]) != radius:
            continue
        key = (row["image_name"], row["track_id"])
        variant = row["target_variant"]
        if variant in keyed[key]:
            raise ValueError(f"Duplicate observation {key} / {variant}")
        keyed[key][variant] = row
    if not keyed:
        raise ValueError("No observations at the preregistered radius")

    per_image = defaultdict(list)
    support_entries = []
    for key, group in keyed.items():
        if set(group) != set(VARIANTS):
            raise ValueError(f"Unpaired methods for {key}; do not silently drop failures")
        dimensions = {(int(row["image_width"]), int(row["image_height"])) for row in group.values()}
        if len(dimensions) != 1:
            raise ValueError("Different target resolution across methods")
        width, height = next(iter(dimensions))
        if width <= 0 or height <= 0:
            raise ValueError("Image width and height must be positive")
        penalty = math.hypot(width, height)
        projection = [float(group[variant]["projection_error"]) for variant in VARIANTS]
        if not all(math.isfinite(value) and value >= 0.0 for value in projection):
            raise ValueError("Projection errors must be finite and non-negative")
        if not np.allclose(projection, projection[0], atol=1e-4):
            raise ValueError("Projection baseline differs between paired methods")
        if require_exact_support:
            variant_support = [_strict_support_entry(group[variant]) for variant in VARIANTS]
            if any(entry != variant_support[0] for entry in variant_support[1:]):
                raise ValueError(f"Support/proxy-label mismatch across target variants for {key}")
            for variant in VARIANTS:
                if float(group[variant]["hard_negative_radius"]) != float(radius):
                    raise ValueError(
                        f"Hard-negative radius does not match row window radius for {key} / {variant}"
                    )
            support_entries.append(variant_support[0])
        else:
            support_entries.append(
                {"image_name": str(key[0]), "track_id": str(key[1])}
            )
        record = {"projection_error": min(projection[0], penalty)}
        for label, name in zip(VARIANTS, ("gs", "difix", "real")):
            row = group[label]
            correct_raw = float(row["matching_error"])
            correct, correct_success = capped_error(correct_raw, penalty)
            feature_stride = float(row["feature_stride"])
            if not math.isfinite(feature_stride) or feature_stride <= 0.0:
                raise ValueError("feature_stride must be finite and positive")
            record[f"{name}_failure"] = float(not correct_success)
            record[f"{name}_penalized_error"] = correct
            record[f"{name}_normalized_error"] = correct / feature_stride
            for threshold in (3, 5, 8, 16):
                record[f"{name}_pck{threshold}"] = float(correct_success and correct_raw <= threshold)

            local_hard_valid = _parse_bool(
                row.get("hard_negative_valid", False), "hard_negative_valid"
            )
            if "hard_negative_within_radius" in row or "hard_negative_selection_mode" in row:
                within_radius = _parse_bool(
                    row.get("hard_negative_within_radius", False),
                    "hard_negative_within_radius",
                )
                expected_valid = (
                    within_radius
                    and row.get("hard_negative_selection_mode") == "local_hard_negative"
                )
                if local_hard_valid != expected_valid:
                    raise ValueError(
                        f"Inconsistent hard-negative validity for {key} / {label}"
                    )
            wrong_raw = float(row.get("hard_shuffled_error", float("nan")))
            wrong, wrong_success = capped_error(wrong_raw, penalty)
            record[f"{name}_hard_negative_valid"] = float(local_hard_valid)
            record[f"{name}_correct_failure"] = float(not correct_success) if local_hard_valid else float("nan")
            record[f"{name}_wrong_failure"] = float(not wrong_success) if local_hard_valid else float("nan")
            record[f"{name}_both_success"] = float(correct_success and wrong_success) if local_hard_valid else float("nan")
            record[f"{name}_identity_margin_failure_aware"] = (
                wrong - correct if local_hard_valid else float("nan")
            )
            record[f"{name}_identity_margin_conditional"] = (
                wrong_raw - correct_raw if local_hard_valid and correct_success and wrong_success else float("nan")
            )
            # Compatibility alias: the primary margin is now failure-aware.
            record[f"{name}_identity_margin"] = record[f"{name}_identity_margin_failure_aware"]
            for control in ("global_shuffle", "random_query", "uniform_query"):
                value = row.get(control + "_error")
                control_valid = control != "global_shuffle" or str(row.get("global_shuffle_valid", "False")).lower() == "true"
                if value is None or not control_valid:
                    record[name + "_" + control + "_margin"] = float("nan")
                else:
                    control_error, _ = capped_error(float(value), penalty)
                    record[name + "_" + control + "_margin"] = control_error - correct
        record["difix_gain_vs_projection"] = record["projection_error"] - record["difix_penalized_error"]
        record["difix_gain_vs_gs"] = record["gs_penalized_error"] - record["difix_penalized_error"]
        per_image[key[0]].append(record)

    fields = list(next(iter(per_image.values()))[0])
    image_summaries = []
    for image, records in sorted(per_image.items()):
        row = {"image_name": image, "track_count": len(records)}
        for field in fields:
            values = np.asarray([record[field] for record in records], dtype=np.float64)
            values = values[np.isfinite(values)]
            row[field] = float(values.mean()) if len(values) else None
        for name in ("gs", "difix", "real"):
            valid = np.asarray([record[f"{name}_hard_negative_valid"] for record in records], dtype=np.float64)
            valid_mask = valid == 1.0
            row[f"{name}_eligible_count"] = len(records)
            row[f"{name}_hard_negative_valid_count"] = int(valid_mask.sum())
            for suffix in ("correct_failure", "wrong_failure", "both_success"):
                values = np.asarray([record[f"{name}_{suffix}"] for record in records], dtype=np.float64)
                row[f"{name}_{suffix}_count"] = int(np.nansum(values))
        image_summaries.append(row)

    aggregate = {}
    for field in image_summaries[0]:
        if field == "image_name":
            continue
        values = [row[field] for row in image_summaries if row[field] is not None]
        if field == "track_count" or field.endswith("_count"):
            aggregate[field] = int(sum(values)) if values else 0
        else:
            aggregate[field] = float(np.mean(values)) if values else None
    support_entries.sort(key=lambda row: (row["image_name"], row["track_id"]))
    paired_feature_geometry_by_image = {}
    if require_exact_support:
        for entry in support_entries:
            geometry = {
                "image_name": entry["image_name"],
                "input_resolution_hw": [entry["image_height"], entry["image_width"]],
                "feature_resolution_hw": [
                    entry["feature_height"],
                    entry["feature_width"],
                ],
                "effective_feature_stride_float64": entry[
                    "effective_feature_stride_float64"
                ],
            }
            previous = paired_feature_geometry_by_image.setdefault(
                entry["image_name"], geometry
            )
            if previous != geometry:
                raise ValueError(
                    f"Inconsistent feature geometry within image {entry['image_name']}"
                )
    paired_feature_geometry = [
        paired_feature_geometry_by_image[name]
        for name in sorted(paired_feature_geometry_by_image)
    ]
    support_schema = (
        "identity_support_with_proxy_and_negative_v2"
        if require_exact_support
        else "image_track_support_v1"
    )
    result = {
        "status": "MEASURED_DIAGNOSTIC",
        "radius": radius,
        "aggregation": "tracks->image_mean->scene_mean; no track-level significance test",
        "failure_penalty": "image diagonal, identical across Correct/Wrong/methods",
        "primary_identity_margin": "wrong_penalized_error - correct_penalized_error on within-radius hard negatives only",
        "paired_track_count": len(keyed),
        "paired_support_schema": support_schema,
        "paired_support_sha256": _support_digest(support_entries, support_schema),
        "image_names": sorted(per_image),
        "paired_feature_geometry": paired_feature_geometry,
        "images": image_summaries,
        "scene_metrics": aggregate,
    }
    if metadata is not None:
        protocol = identity_protocol_from_metadata(metadata, radius)
        if require_exact_support:
            row_backends = {entry["feature_backend"] for entry in support_entries}
            if row_backends != {protocol["feature_backend"]}:
                raise ValueError(
                    "Per-track feature backend does not match evaluator metadata"
                )
        result.update(protocol)
        result["paired_identity_arm"] = protocol["paired_identity_arm"]
        if len(result["image_names"]) != protocol["paired_manifest_record_count"]:
            raise ValueError(
                "Per-track report does not cover every paired-manifest held-out image"
            )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--radius", type=float, default=32.0)
    parser.add_argument(
        "--metadata",
        type=Path,
        help="Evaluator metadata.json (defaults to the CSV sibling metadata.json)",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.csv.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    metadata_path = args.metadata or args.csv.parent / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Identity report requires evaluator metadata: {metadata_path}"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    result = analyze(
        rows,
        args.radius,
        metadata=metadata,
        require_exact_support=True,
    )
    result.update(
        {
            "input_csv": str(args.csv.expanduser().resolve()),
            "input_csv_sha256": hashlib.sha256(
                args.csv.expanduser().resolve().read_bytes()
            ).hexdigest(),
            "evaluator_metadata": str(metadata_path.expanduser().resolve()),
            "evaluator_metadata_sha256": hashlib.sha256(
                metadata_path.expanduser().resolve().read_bytes()
            ).hexdigest(),
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result["scene_metrics"], indent=2))


if __name__ == "__main__":
    main()
