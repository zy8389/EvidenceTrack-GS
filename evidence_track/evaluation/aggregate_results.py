#!/usr/bin/env python3
"""Scene-clustered paired bootstrap with explicit development/confirmatory roles."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from diffusion_guidance.result_binding import (
    audit_supports_contrast,
    read_pair_audit,
    require_audit_binding,
)
from diffusion_guidance.experiment_registry import (
    VALID_ROLES,
    require_complete_confirmatory_rows,
    require_scene_role,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def filter_by_role(rows, role: str):
    if role not in {"development", "confirmatory", "all"}:
        raise ValueError(f"Unknown role: {role}")
    filtered = []
    for row in rows:
        value = str(row.get("role", "")).strip().lower()
        if value not in VALID_ROLES:
            raise ValueError(
                "Every result row must declare role=development or role=confirmatory; "
                f"missing/invalid role for {row.get('dataset')}/{row.get('scene')}"
            )
        require_scene_role(
            str(row.get("dataset", "")), str(row.get("scene", "")), value
        )
        if role == "all" or value == role:
            filtered.append(row)
    if not filtered:
        raise ValueError(f"No {role} result rows are available")
    return filtered


def paired_bootstrap(rows, metric, baseline="A1", treatment="B", seed=20260906, repeats=10000):
    values = {}
    for row in rows:
        if row["method"] not in (baseline, treatment):
            continue
        key = (row["dataset"], row["scene"], int(row["seed"]), row["method"])
        if key in values:
            raise ValueError(f"Duplicate result {key}")
        try:
            value = float(row[metric])
        except (ValueError, TypeError, KeyError) as exc:
            raise ValueError("Missing/unmeasured value: no conclusion can be calculated") from exc
        if not math.isfinite(value):
            raise ValueError("Non-finite value")
        values[key] = value
    pairs = sorted({key[:3] for key in values})
    if not pairs:
        raise ValueError("No paired results; experiments have not been run")
    scenes = {}
    for key in pairs:
        if key + (baseline,) not in values or key + (treatment,) not in values:
            raise ValueError(f"Incomplete {baseline}/{treatment} pair: {key}")
        delta = values[key + (treatment,)] - values[key + (baseline,)]
        scenes.setdefault(key[:2], []).append(delta)
    differences = np.asarray([np.mean(value) for value in scenes.values()])
    rng = np.random.default_rng(seed)
    interval = None if len(differences) < 2 else np.quantile(
        rng.choice(differences, (repeats, len(differences)), replace=True).mean(axis=1),
        [0.025, 0.975],
    ).tolist()
    return {
        "metric": metric,
        "contrast": f"{treatment}-{baseline}",
        "mean_delta": float(differences.mean()),
        "scene_count": len(scenes),
        "paired_run_count": len(pairs),
        "scene_bootstrap_ci95": interval,
        "bootstrap_seed": seed,
        "bootstrap_repeats": repeats,
        "warning": "Interpret only the selected preregistered role; do not pool development tuning scenes into a confirmatory claim.",
    }


def _require_row_bound_pair_audits(
    rows: list[dict], *, baseline: str, treatment: str, csv_path: Path
) -> None:
    """Require every measured arm to cite its own passed group-specific audit."""
    if not rows:
        raise ValueError("No result rows supplied for pair-audit validation")
    unsupported_methods = sorted(
        {str(row.get("method", "")) for row in rows} - {baseline, treatment}
    )
    if unsupported_methods:
        raise ValueError(
            "Aggregate one controlled contrast per CSV; unrelated methods would "
            "require a second audit binding and are not accepted here: "
            f"{unsupported_methods}"
        )
    cache: dict[Path, dict] = {}
    audit_groups: dict[Path, tuple[str, str, int, str]] = {}
    for index, row in enumerate(rows, start=2):
        required = (
            "dataset",
            "scene",
            "role",
            "seed",
            "method",
            "pair_id",
            "pair_audit_path",
            "pair_audit_sha256",
            "metric_log",
            "metric_log_sha256",
        )
        missing = [field for field in required if not str(row.get(field, "")).strip()]
        if missing:
            raise ValueError(
                f"Result CSV row {index} lacks group-specific pair binding fields: {missing}"
            )
        method = str(row["method"])
        if method not in {baseline, treatment}:
            raise ValueError(
                "Aggregate exactly one declared contrast per CSV; unrelated "
                f"method {method!r} appears at result row {index}"
            )
        audit_path, payload = read_pair_audit(
            row["pair_audit_path"], base_dir=csv_path.parent
        )
        if _sha256_file(audit_path) != str(row["pair_audit_sha256"]):
            raise ValueError(f"Result CSV row {index} pair audit hash is stale")
        log_path = Path(str(row["metric_log"])).expanduser()
        if not log_path.is_absolute():
            log_path = csv_path.parent / log_path
        log_path = log_path.resolve()
        if not log_path.is_file() or _sha256_file(log_path) != str(
            row["metric_log_sha256"]
        ):
            raise ValueError(
                f"Result CSV row {index} training log is missing or changed"
            )
        cache[audit_path] = payload
        require_audit_binding(
            payload,
            dataset=str(row["dataset"]),
            scene=str(row["scene"]),
            seed=int(row["seed"]),
            method=method,
            pair_id=str(row["pair_id"]),
            role=str(row["role"]),
        )
        group = (
            str(row["dataset"]),
            str(row["scene"]),
            int(row["seed"]),
            str(row["pair_id"]),
        )
        previous = audit_groups.setdefault(audit_path, group)
        if previous != group:
            raise ValueError(
                "One pair audit cannot authorize unrelated dataset/scene/seed/pair groups: "
                f"{audit_path}"
            )
    required_groups = {
        (str(row["dataset"]), str(row["scene"]), int(row["seed"]))
        for row in rows
        if str(row["method"]) in {baseline, treatment}
    }
    for dataset, scene, seed in sorted(required_groups):
        arms = [
            row
            for row in rows
            if (str(row["dataset"]), str(row["scene"]), int(row["seed"]))
            == (dataset, scene, seed)
            and str(row["method"]) in {baseline, treatment}
        ]
        methods = {str(row["method"]) for row in arms}
        if len(arms) != 2 or methods != {baseline, treatment}:
            raise ValueError(
                f"Incomplete {baseline}/{treatment} result group for audit binding: "
                f"{dataset}/{scene}/seed{seed}"
            )
        pair_ids = {str(row["pair_id"]) for row in arms}
        audit_paths = {
            read_pair_audit(row["pair_audit_path"], base_dir=csv_path.parent)[0]
            for row in arms
        }
        if len(pair_ids) != 1 or len(audit_paths) != 1:
            raise ValueError(
                f"{baseline}/{treatment} rows must share exactly one pair_id and audit: "
                f"{dataset}/{scene}/seed{seed}"
            )
        audit = cache[next(iter(audit_paths))]
        if not audit_supports_contrast(audit, baseline, treatment):
            raise ValueError(
                f"Pair audit does not cover required contrast {treatment}-{baseline}: "
                f"{dataset}/{scene}/seed{seed}"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--metric", choices=["psnr", "ssim", "lpips", "geometry_error"], required=True)
    parser.add_argument("--role", choices=["development", "confirmatory", "all"], default="confirmatory")
    parser.add_argument("--baseline", default="A1")
    parser.add_argument("--treatment", default="B")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.csv.open(encoding="utf-8", newline="") as handle:
        rows = filter_by_role(list(csv.DictReader(handle)), args.role)
    _require_row_bound_pair_audits(
        rows, baseline=args.baseline, treatment=args.treatment, csv_path=args.csv
    )
    if args.role == "confirmatory":
        require_complete_confirmatory_rows(
            rows, methods=(args.baseline, args.treatment)
        )
    datasets = sorted({row["dataset"] for row in rows})
    result = {
        "passed": True,
        "schema": "controlled_paired_aggregate_v2",
        "role": args.role,
        "datasets": {
            dataset: paired_bootstrap(
                [row for row in rows if row["dataset"] == dataset],
                args.metric,
                baseline=args.baseline,
                treatment=args.treatment,
            )
            for dataset in datasets
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
