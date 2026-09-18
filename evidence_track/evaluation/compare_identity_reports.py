#!/usr/bin/env python3
"""Compare paired A1/B identity reports without promoting them to causality."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path

from diffusion_guidance.evidence_protocol import PAIRED_RECONSTRUCTION_FIELDS
from diffusion_guidance.result_binding import (
    audit_supports_contrast,
    read_pair_audit,
    require_audit_binding,
)
from evidence_track.evaluation.paired_evidence_report import analyze


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path: Path) -> dict:
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("status") != "MEASURED_DIAGNOSTIC":
        raise ValueError(f"Not a measured diagnostic: {path}")
    return result


def extract(report: dict, prefix: str) -> dict:
    source = report.get("scene_metrics", {})
    # paired_evidence_report names variants gs/difix/real.  A1/B diagnostics
    # each use GS-render correspondence as the reconstruction-facing measure.
    return {
        f"{prefix}_correct_error": source["gs_penalized_error"],
        f"{prefix}_identity_margin": source["gs_identity_margin_failure_aware"],
        f"{prefix}_identity_margin_conditional": source[
            "gs_identity_margin_conditional"
        ],
        f"{prefix}_projection_gain": source["projection_error"]
        - source["gs_penalized_error"],
        f"{prefix}_failure_rate": source["gs_failure"],
        f"{prefix}_eligible_count": source["gs_eligible_count"],
        f"{prefix}_hard_negative_valid_count": source[
            "gs_hard_negative_valid_count"
        ],
        f"{prefix}_correct_failure_count": source[
            "gs_correct_failure_count"
        ],
        f"{prefix}_wrong_failure_count": source["gs_wrong_failure_count"],
        f"{prefix}_both_success_count": source["gs_both_success_count"],
    }


def _require_fields(report: dict, label: str, fields: tuple[str, ...]) -> None:
    missing = [field for field in fields if field not in report]
    if missing:
        raise ValueError(f"{label} identity report is incomplete: missing {missing}")


def _hard_negative_outcome_partition(
    *,
    valid_count: int,
    correct_failed: int,
    wrong_failed: int,
    both_success: int,
) -> dict[str, int]:
    both_failed = correct_failed + wrong_failed + both_success - valid_count
    partition = {
        "both_failed": both_failed,
        "correct_only_failed": correct_failed - both_failed,
        "wrong_only_failed": wrong_failed - both_failed,
        "both_success": both_success,
    }
    if any(value < 0 for value in partition.values()) or sum(
        partition.values()
    ) != valid_count:
        raise ValueError("hard-negative outcome counts do not form a disjoint partition")
    return partition


def _recompute_file_backed_report(report: dict, label: str) -> None:
    csv_path = Path(str(report["input_csv"])).expanduser().resolve()
    metadata_path = Path(str(report["evaluator_metadata"])).expanduser().resolve()
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError(f"{label} evaluator metadata must be a JSON object")
    recomputed = analyze(
        rows,
        float(report["radius"]),
        metadata=metadata,
        require_exact_support=True,
    )
    recomputed.update(
        {
            "input_csv": str(csv_path),
            "input_csv_sha256": sha256_file(csv_path),
            "evaluator_metadata": str(metadata_path),
            "evaluator_metadata_sha256": sha256_file(metadata_path),
        }
    )
    if report != recomputed:
        differing = sorted(
            key
            for key in set(report) | set(recomputed)
            if report.get(key) != recomputed.get(key)
        )
        raise ValueError(
            f"{label} identity report is stale or edited; differing fields: {differing}"
        )


def compare_reports(
    a1: dict,
    b: dict,
    *,
    a1_path: Path | None = None,
    b_path: Path | None = None,
) -> dict:
    if a1.get("status") != "MEASURED_DIAGNOSTIC" or b.get("status") != "MEASURED_DIAGNOSTIC":
        raise ValueError("A1 and B inputs must both be measured diagnostic reports")
    protocol_fields = (
        "radius",
        "paired_track_count",
        "paired_support_schema",
        "paired_support_sha256",
        "image_names",
        "paired_feature_geometry",
        "feature_map_geometry",
        "feature_backend",
        "feature_backend_identity",
        "hard_negative_radius",
        "temperature",
        "track_h5_sha256",
        "target_manifest",
        "target_manifest_sha256",
        "dataset",
        "scene",
        "seed",
        "experiment_role",
        "scene_source_path",
        "controlled_pair_id",
        "pair_audit",
        "pair_audit_sha256",
        "track_h5",
        "source_camera_names",
        "source_camera_set_sha256",
        "source_images_dir",
        "source_image_inventory",
        "source_image_inventory_sha256",
        "paired_manifest_metadata",
        "paired_manifest_metadata_sha256",
        "paired_manifest_record_count",
        "a1_evidence_manifest",
        "a1_evidence_manifest_sha256",
        "b_evidence_manifest",
        "b_evidence_manifest_sha256",
        "a1_difix_manifest",
        "a1_difix_manifest_sha256",
        "b_difix_manifest",
        "b_difix_manifest_sha256",
        "a1_difix_run_metadata",
        "a1_difix_run_metadata_sha256",
        "b_difix_run_metadata",
        "b_difix_run_metadata_sha256",
        "a1_difix_cache_run_fingerprint",
        "b_difix_cache_run_fingerprint",
        "dinov2_model",
        "dinov2_local_repository_path",
        "dinov2_repository_commit",
        "dinov2_pretrained_weight_path",
        "dinov2_pretrained_weight_sha256",
    ) + PAIRED_RECONSTRUCTION_FIELDS
    metric_fields = (
        "gs_penalized_error",
        "gs_identity_margin_failure_aware",
        "gs_identity_margin_conditional",
        "projection_error",
        "gs_failure",
        "gs_eligible_count",
        "gs_hard_negative_valid_count",
        "gs_correct_failure_count",
        "gs_wrong_failure_count",
        "gs_both_success_count",
    )
    for label, report in (("A1", a1), ("B", b)):
        _require_fields(report, label, protocol_fields)
        _require_fields(
            report,
            label,
            (
                "input_csv",
                "input_csv_sha256",
                "evaluator_metadata",
                "evaluator_metadata_sha256",
                "per_track_csv",
                "per_track_csv_sha256",
            ),
        )
        if (
            Path(report["input_csv"]).expanduser().resolve()
            != Path(report["per_track_csv"]).expanduser().resolve()
            or report["input_csv_sha256"] != report["per_track_csv_sha256"]
        ):
            raise ValueError(f"{label} report input CSV binding is inconsistent")
        for path_field, hash_field in (
            ("input_csv", "input_csv_sha256"),
            ("evaluator_metadata", "evaluator_metadata_sha256"),
        ):
            source = Path(report[path_field]).expanduser().resolve()
            if not source.is_file() or sha256_file(source) != report[hash_field]:
                raise ValueError(f"{label} report source is stale: {path_field}")
        _recompute_file_backed_report(report, label)
        if report["paired_support_schema"] != "identity_support_with_proxy_and_negative_v2":
            raise ValueError(f"{label} report does not use the exact paired-support schema")
        if re.fullmatch(r"[0-9a-f]{64}", str(report["paired_support_sha256"])) is None:
            raise ValueError(f"{label} paired_support_sha256 is invalid")
        images = report["image_names"]
        if not isinstance(images, list) or images != sorted(set(images)) or not images:
            raise ValueError(f"{label} image_names must be a non-empty sorted unique list")
        scene_metrics = report.get("scene_metrics")
        if not isinstance(scene_metrics, dict):
            raise ValueError(f"{label} scene_metrics is missing")
        _require_fields(scene_metrics, f"{label} scene_metrics", metric_fields)
        for field in (
            "gs_penalized_error",
            "gs_identity_margin_failure_aware",
            "projection_error",
            "gs_failure",
        ):
            value = scene_metrics[field]
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"{label} scene metric {field} is not finite")
        conditional = scene_metrics["gs_identity_margin_conditional"]
        if conditional is not None and (
            not isinstance(conditional, (int, float))
            or not math.isfinite(float(conditional))
        ):
            raise ValueError(f"{label} conditional identity margin is invalid")
        count_fields = (
            "gs_eligible_count",
            "gs_hard_negative_valid_count",
            "gs_correct_failure_count",
            "gs_wrong_failure_count",
            "gs_both_success_count",
        )
        counts = {}
        for field in count_fields:
            value = scene_metrics[field]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{label} scene count {field} is invalid")
            counts[field] = value
        if counts["gs_eligible_count"] != int(report["paired_track_count"]):
            raise ValueError(f"{label} eligible count differs from paired support")
        valid_count = counts["gs_hard_negative_valid_count"]
        if valid_count > counts["gs_eligible_count"] or any(
            counts[field] > valid_count
            for field in (
                "gs_correct_failure_count",
                "gs_wrong_failure_count",
                "gs_both_success_count",
            )
        ):
            raise ValueError(f"{label} hard-negative outcome counts are inconsistent")
        try:
            _hard_negative_outcome_partition(
                valid_count=valid_count,
                correct_failed=counts["gs_correct_failure_count"],
                wrong_failed=counts["gs_wrong_failure_count"],
                both_success=counts["gs_both_success_count"],
            )
        except ValueError as exc:
            raise ValueError(
                f"{label} hard-negative outcome partition is inconsistent: {exc}"
            ) from exc
        if not 0.0 <= float(scene_metrics["gs_failure"]) <= 1.0:
            raise ValueError(f"{label} failure rate must lie in [0, 1]")

    for field in protocol_fields:
        if a1[field] != b[field]:
            raise ValueError(f"A1/B identity protocol mismatch for {field}")
    for field in ("gs_eligible_count", "gs_hard_negative_valid_count"):
        if a1["scene_metrics"][field] != b["scene_metrics"][field]:
            raise ValueError(f"A1/B identity support count mismatch for {field}")
    if a1.get("paired_identity_arm") != "A1" or b.get("paired_identity_arm") != "B":
        raise ValueError("A1/B identity reports do not identify their immutable paired-manifest arms")
    if a1["feature_backend"] != "dinov2":
        raise ValueError("Final A1/B identity comparison requires the preregistered DINOv2 backend")
    for field in (
        "dinov2_model",
        "dinov2_repository_commit",
        "dinov2_pretrained_weight_sha256",
    ):
        if not a1[field]:
            raise ValueError(f"A1/B identity comparison lacks {field}")
    for path_field, hash_field in (
        ("target_manifest", "target_manifest_sha256"),
        ("paired_manifest_metadata", "paired_manifest_metadata_sha256"),
        ("a1_evidence_manifest", "a1_evidence_manifest_sha256"),
        ("b_evidence_manifest", "b_evidence_manifest_sha256"),
        ("a1_difix_manifest", "a1_difix_manifest_sha256"),
        ("b_difix_manifest", "b_difix_manifest_sha256"),
        ("a1_difix_run_metadata", "a1_difix_run_metadata_sha256"),
        ("b_difix_run_metadata", "b_difix_run_metadata_sha256"),
        ("a1_checkpoint", "a1_checkpoint_sha256"),
        ("b_checkpoint", "b_checkpoint_sha256"),
    ):
        path = Path(a1[path_field]).expanduser().resolve()
        if not path.is_file() or sha256_file(path) != a1[hash_field]:
            raise ValueError(f"A1/B identity binding is stale for {path_field}")
    pair_audit_path = Path(a1["pair_audit"]).expanduser().resolve()
    if sha256_file(pair_audit_path) != a1["pair_audit_sha256"]:
        raise ValueError("A1/B identity pair-audit hash is stale")
    _, pair_audit = read_pair_audit(pair_audit_path)
    for arm in ("A1", "B"):
        require_audit_binding(
            pair_audit,
            dataset=a1["dataset"],
            scene=a1["scene"],
            seed=int(a1["seed"]),
            method=arm,
            pair_id=a1["controlled_pair_id"],
            role=a1["experiment_role"],
        )
    if not audit_supports_contrast(pair_audit, "A1", "B"):
        raise ValueError("Identity reports are not authorized by an A1/B audit")

    result = {
        "schema": "paired_identity_association_v4",
        "passed": True,
        "status": "PAIRED_IDENTITY_ASSOCIATION",
        "dataset": a1["dataset"],
        "scene": a1["scene"],
        "seed": int(a1["seed"]),
        "role": a1["experiment_role"],
        "controlled_pair_id": a1["controlled_pair_id"],
        "scene_source_path": a1["scene_source_path"],
        "pair_audit": str(pair_audit_path),
        "pair_audit_sha256": a1["pair_audit_sha256"],
        "pair_audit_input_file_count": pair_audit["audited_input_file_count"],
        "pair_audit_input_fingerprint": pair_audit[
            "audited_input_fingerprint"
        ],
        "track_h5": a1["track_h5"],
        "track_h5_sha256": a1["track_h5_sha256"],
        "source_camera_names": a1["source_camera_names"],
        "source_camera_set_sha256": a1["source_camera_set_sha256"],
        "source_images_dir": a1["source_images_dir"],
        "source_image_inventory": a1["source_image_inventory"],
        "source_image_inventory_sha256": a1["source_image_inventory_sha256"],
        "radius": a1["radius"],
        "paired_manifest_record_count": a1["paired_manifest_record_count"],
        "paired_track_count": a1["paired_track_count"],
        "paired_support_sha256": a1["paired_support_sha256"],
        "target_manifest_sha256": a1["target_manifest_sha256"],
        "image_names": a1["image_names"],
        "feature_backend": a1["feature_backend_identity"],
        "dinov2_model": a1["dinov2_model"],
        "dinov2_local_repository_path": a1["dinov2_local_repository_path"],
        "dinov2_repository_commit": a1["dinov2_repository_commit"],
        "dinov2_pretrained_weight_path": a1["dinov2_pretrained_weight_path"],
        "dinov2_pretrained_weight_sha256": a1[
            "dinov2_pretrained_weight_sha256"
        ],
        **extract(a1, "a1"),
        **extract(b, "b"),
        **{field: a1[field] for field in PAIRED_RECONSTRUCTION_FIELDS},
        "interpretation": "Associates B reconstruction with identity-consistent evidence under a shared protocol; does not establish that identity caused any reconstruction change or that global geometry is correct.",
    }
    result.update(
        {
            "b_minus_a1_correct_error": result["b_correct_error"]
            - result["a1_correct_error"],
            "b_minus_a1_identity_margin": result["b_identity_margin"]
            - result["a1_identity_margin"],
            "b_minus_a1_identity_margin_conditional": (
                result["b_identity_margin_conditional"]
                - result["a1_identity_margin_conditional"]
                if result["a1_identity_margin_conditional"] is not None
                and result["b_identity_margin_conditional"] is not None
                else None
            ),
            "b_minus_a1_projection_gain": result["b_projection_gain"]
            - result["a1_projection_gain"],
            "b_minus_a1_failure_rate": result["b_failure_rate"]
            - result["a1_failure_rate"],
        }
    )
    if a1_path is not None and b_path is not None:
        result.update(
            {
                "a1_report": str(a1_path.resolve()),
                "a1_report_sha256": sha256_file(a1_path),
                "b_report": str(b_path.resolve()),
                "b_report_sha256": sha256_file(b_path),
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a1", type=Path, required=True)
    parser.add_argument("--b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    a1_path = args.a1.expanduser().resolve()
    b_path = args.b.expanduser().resolve()
    a1, b = read(a1_path), read(b_path)
    result = compare_reports(a1, b, a1_path=a1_path, b_path=b_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
