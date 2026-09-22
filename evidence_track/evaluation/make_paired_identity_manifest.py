#!/usr/bin/env python3
"""Bind A1/B held-out renders and Difix targets into one immutable selector."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from diffusion_guidance.control_identity import source_camera_set_sha256
from diffusion_guidance.checkpoint_state import RENDER_STATE_SCHEMA, load_checkpoint_summary
from diffusion_guidance.difix_provenance import (
    HELDOUT_DIAGNOSTIC_MANIFEST_SCHEMA,
    difix_run_metadata_path,
    load_difix_run_metadata,
    validate_difix_target,
)
from diffusion_guidance.evidence_protocol import (
    canonical_source_image_inventory,
    source_image_inventory_sha256,
)
from diffusion_guidance.experiment_registry import require_scene_role
from diffusion_guidance.result_binding import (
    audit_supports_contrast,
    read_pair_audit,
    require_audit_binding,
    require_final_checkpoint_binding,
)


PAIRED_MANIFEST_SCHEMA = "paired_identity_manifest_v4"
PAIRED_MANIFEST_METADATA_SCHEMA = "paired_identity_manifest_metadata_v4"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(base: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Manifest path value must be a non-empty string")
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _source_image_inventory(
    scene_source_path: Path, source_camera_names: list[str]
) -> tuple[Path, list[dict[str, str]], str]:
    """Record the exact source-image bytes used by the identity evaluator."""
    images_dir = (scene_source_path / "images").resolve()
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Scene source image directory is missing: {images_dir}")
    records: list[dict[str, str]] = []
    for camera_name in source_camera_names:
        name = Path(str(camera_name).replace("\\", "/")).stem
        direct = images_dir / str(camera_name)
        if not direct.is_file():
            matches = sorted(images_dir.glob(name + ".*"))
            if len(matches) != 1 or not matches[0].is_file():
                raise FileNotFoundError(
                    f"Cannot resolve source camera image {camera_name!r} in {images_dir}"
                )
            direct = matches[0]
        image_path = direct.resolve()
        records.append(
            {
                "camera_name": name,
                "path": str(image_path),
                "sha256": sha256_file(image_path),
            }
        )
    canonical = canonical_source_image_inventory(records)
    return images_dir, canonical, source_image_inventory_sha256(canonical)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_number}")
            rows.append(row)
    if not rows:
        raise ValueError(f"Manifest is empty: {path}")
    return rows


def _one_value(records: list[dict[str, Any]], field: str, label: str) -> Any:
    values = [record.get(field) for record in records]
    if any(value in (None, "") for value in values):
        raise ValueError(f"{label} evidence manifest lacks {field}")
    canonical = {
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        for value in values
    }
    if len(canonical) != 1:
        raise ValueError(f"{label} evidence manifest changes {field} across records")
    return values[0]


def read_records(path: Path, *, arm: str) -> tuple[dict[str, dict], dict[str, Any]]:
    path = path.expanduser().resolve()
    rows = _read_jsonl(path)
    required = (
        "manifest_schema",
        "image_name",
        "camera",
        "camera_fingerprint",
        "camera_source",
        "gs_render",
        "difix_output",
        "real_target",
        "reference_image",
        "gs_render_sha256",
        "reference_image_sha256",
        "real_target_sha256",
        "dataset",
        "scene",
        "seed",
        "experiment_role",
        "scene_source_path",
        "paired_identity_arm",
        "controlled_pair_id",
        "pair_audit",
        "pair_audit_sha256",
        "track_h5",
        "track_h5_sha256",
        "source_camera_names",
        "source_camera_set_sha256",
        "checkpoint",
        "checkpoint_sha256",
        "checkpoint_iteration",
        "checkpoint_state_format",
        "checkpoint_gaussian_count",
        "checkpoint_render_state_schema",
        "checkpoint_render_state_sha256",
        "checkpoint_controlled_provenance_sha256",
        "scene_ply_gaussian_count",
        "scene_ply_checkpoint_count_match",
        "scene_ply_render_state_schema",
        "scene_ply_render_state_sha256",
        "scene_ply_checkpoint_render_state_match",
    )
    records: dict[str, dict] = {}
    for line_number, record in enumerate(rows, start=1):
        missing = [field for field in required if record.get(field) in (None, "")]
        if missing:
            raise ValueError(
                f"Incomplete {arm} evidence record at {path}:{line_number}: {missing}"
            )
        if record["manifest_schema"] != HELDOUT_DIAGNOSTIC_MANIFEST_SCHEMA:
            raise ValueError(f"{arm} evidence uses the wrong manifest schema")
        if record["paired_identity_arm"] != arm:
            raise ValueError(f"{arm} evidence record declares another arm")
        if record["camera_source"] != "heldout_evaluation_camera":
            raise ValueError(f"{arm} evidence record is not held-out")
        image_name = str(record["image_name"])
        if image_name in records:
            raise ValueError(f"Duplicate image_name at {path}:{line_number}")
        for path_field, hash_field in (
            ("gs_render", "gs_render_sha256"),
            ("reference_image", "reference_image_sha256"),
            ("real_target", "real_target_sha256"),
            ("checkpoint", "checkpoint_sha256"),
        ):
            asset = _resolve(path.parent, record[path_field])
            if not asset.is_file() or record[hash_field] != sha256_file(asset):
                raise ValueError(
                    f"{arm} evidence asset is missing or changed: {image_name}/{path_field}"
                )
            record[path_field] = str(asset)
        record["difix_output"] = str(_resolve(path.parent, record["difix_output"]))
        checkpoint_summary = load_checkpoint_summary(
            Path(record["checkpoint"]),
            expected_iteration=12000,
            require_cuda_rng=True,
            require_controlled_provenance=True,
        )
        if (
            record["checkpoint_state_format"] != checkpoint_summary["format"]
            or int(record["checkpoint_iteration"]) != 12000
            or int(record["checkpoint_gaussian_count"])
            != checkpoint_summary["gaussian_count"]
            or record["checkpoint_render_state_schema"]
            != checkpoint_summary["render_state_schema"]
            or record["checkpoint_render_state_sha256"]
            != checkpoint_summary["render_state_sha256"]
            or record["checkpoint_controlled_provenance_sha256"]
            != checkpoint_summary["controlled_provenance_sha256"]
        ):
            raise ValueError(
                f"{arm} evidence checkpoint summary differs from serialized state: {image_name}"
            )
        records[image_name] = record

    context_fields = (
        "dataset",
        "scene",
        "seed",
        "experiment_role",
        "scene_source_path",
        "controlled_pair_id",
        "pair_audit",
        "pair_audit_sha256",
        "track_h5",
        "track_h5_sha256",
        "source_camera_names",
        "source_camera_set_sha256",
    )
    context = {
        field: _one_value(rows, field, arm) for field in context_fields
    }
    role = require_scene_role(
        context["dataset"], context["scene"], context["experiment_role"]
    )
    pair_audit_path = _resolve(path.parent, context["pair_audit"])
    if context["pair_audit_sha256"] != sha256_file(pair_audit_path):
        raise ValueError(f"{arm} evidence pair-audit hash is stale")
    _, pair_audit = read_pair_audit(pair_audit_path)
    require_audit_binding(
        pair_audit,
        dataset=context["dataset"],
        scene=context["scene"],
        seed=int(context["seed"]),
        method=arm,
        pair_id=context["controlled_pair_id"],
        role=role,
    )
    if not audit_supports_contrast(pair_audit, "A1", "B"):
        raise ValueError("Identity evidence requires an A1/B controlled-pair audit")
    scene_source_path = _resolve(path.parent, context["scene_source_path"])
    if (
        not scene_source_path.is_dir()
        or scene_source_path.name != str(context["scene"])
        or scene_source_path
        != Path(pair_audit["scene_source_path"]).expanduser().resolve()
    ):
        raise ValueError(f"{arm} evidence scene-source binding is invalid")
    track_path = _resolve(path.parent, context["track_h5"])
    if context["track_h5_sha256"] != sha256_file(track_path):
        raise ValueError(f"{arm} evidence Track H5 hash is stale")
    source_names = context["source_camera_names"]
    if not isinstance(source_names, list) or source_camera_set_sha256(source_names) != context[
        "source_camera_set_sha256"
    ]:
        raise ValueError(f"{arm} evidence source-camera binding is invalid")

    # Every held-out row in one arm must be produced by the same final
    # checkpoint.  This is stricter than checking only a per-row hash later.
    checkpoint_fields = (
        "checkpoint",
        "checkpoint_sha256",
        "checkpoint_iteration",
        "checkpoint_state_format",
        "checkpoint_gaussian_count",
        "checkpoint_render_state_schema",
        "checkpoint_render_state_sha256",
        "checkpoint_controlled_provenance_sha256",
        "scene_ply_gaussian_count",
        "scene_ply_checkpoint_count_match",
        "scene_ply_render_state_schema",
        "scene_ply_render_state_sha256",
        "scene_ply_checkpoint_render_state_match",
    )
    context.update({
        field: _one_value(rows, field, arm) for field in checkpoint_fields
    })
    if context["checkpoint_render_state_schema"] != RENDER_STATE_SCHEMA:
        raise ValueError(f"{arm} evidence uses an unsupported render-state schema")
    if context["scene_ply_render_state_schema"] != context[
        "checkpoint_render_state_schema"
    ]:
        raise ValueError(f"{arm} Scene PLY render-state schema differs from checkpoint")
    context["checkpoint"] = str(_resolve(path.parent, context["checkpoint"]))
    final_checkpoint_summary = load_checkpoint_summary(
        Path(context["checkpoint"]),
        expected_iteration=12000,
        require_cuda_rng=True,
        require_controlled_provenance=True,
    )
    require_final_checkpoint_binding(
        final_checkpoint_summary,
        pair_audit,
        method=arm,
        checkpoint_path=Path(context["checkpoint"]),
    )
    if context["checkpoint_controlled_provenance_sha256"] != final_checkpoint_summary[
        "controlled_provenance_sha256"
    ]:
        raise ValueError(f"{arm} final checkpoint provenance hash is stale")
    context["pair_audit"] = str(pair_audit_path)
    context["track_h5"] = str(track_path)
    context["scene_source_path"] = str(scene_source_path)

    difix_manifest = path.parent / "difix_manifest.jsonl"
    difix_rows = _read_jsonl(difix_manifest)
    difix_by_key = {}
    for record in difix_rows:
        key = str(record.get("key", ""))
        if not key or key in difix_by_key:
            raise ValueError(f"{arm} Difix manifest has an invalid/duplicate key")
        difix_by_key[key] = record
    run_metadata = load_difix_run_metadata(
        difix_manifest,
        expected_record_count=len(records),
        require_reproducibility=True,
    )
    if run_metadata["manifest_schema"] != HELDOUT_DIAGNOSTIC_MANIFEST_SCHEMA:
        raise ValueError(f"{arm} Difix run metadata uses the wrong schema")
    if set(difix_by_key) != {
        record["camera_fingerprint"] for record in records.values()
    }:
        raise ValueError(f"{arm} evidence and Difix camera sets differ")
    context["difix_protocol"] = {
        field: run_metadata[field]
        for field in (
            "model_id",
            "model_revision",
            "difix_code_commit",
            "coordinate_policy",
            "dtype",
            "timesteps",
            "guidance_scale",
            "prompt",
        )
    }
    context["difix_manifest"] = str(difix_manifest.resolve())
    context["difix_manifest_sha256"] = sha256_file(difix_manifest)
    context["difix_run_metadata"] = str(
        difix_run_metadata_path(difix_manifest).resolve()
    )
    context["difix_run_metadata_sha256"] = sha256_file(
        difix_run_metadata_path(difix_manifest)
    )
    context["difix_cache_run_fingerprint"] = run_metadata[
        "cache_run_fingerprint"
    ]
    for record in records.values():
        cache_record = difix_by_key[record["camera_fingerprint"]]
        if cache_record.get("manifest_schema") != HELDOUT_DIAGNOSTIC_MANIFEST_SCHEMA:
            raise ValueError(f"{arm} Difix record uses the wrong manifest schema")
        context_compare_fields = (
            "camera",
            "camera_fingerprint",
            "camera_source",
            "dataset",
            "scene",
            "seed",
            "experiment_role",
            "paired_identity_arm",
            "controlled_pair_id",
            "pair_audit",
            "pair_audit_sha256",
            "track_h5",
            "track_h5_sha256",
            "source_camera_names",
            "source_camera_set_sha256",
            "checkpoint",
            "checkpoint_sha256",
            "checkpoint_iteration",
            "checkpoint_state_format",
            "checkpoint_gaussian_count",
            "checkpoint_render_state_schema",
            "checkpoint_render_state_sha256",
            "checkpoint_controlled_provenance_sha256",
            "scene_ply_gaussian_count",
            "scene_ply_checkpoint_count_match",
            "scene_ply_render_state_schema",
            "scene_ply_render_state_sha256",
            "scene_ply_checkpoint_render_state_match",
        )
        for field in context_compare_fields:
            expected = record.get(field)
            observed = cache_record.get(
                "key" if field == "camera_fingerprint" else field
            )
            if field in {"checkpoint", "pair_audit", "track_h5"}:
                observed = str(_resolve(difix_manifest.parent, observed))
                expected = str(_resolve(path.parent, expected))
            if json.dumps(observed, sort_keys=True, separators=(",", ":")) != json.dumps(
                expected, sort_keys=True, separators=(",", ":")
            ):
                raise ValueError(
                    f"{arm} evidence/Difix scientific context mismatch for {field}: "
                    f"{record['image_name']}"
                )
        input_path = _resolve(difix_manifest.parent, cache_record["input"])
        reference_path = _resolve(
            difix_manifest.parent, cache_record["reference_image"]
        )
        target_path = _resolve(difix_manifest.parent, cache_record["target"])
        if (
            input_path != Path(record["gs_render"])
            or reference_path != Path(record["reference_image"])
            or target_path != Path(record["difix_output"])
        ):
            raise ValueError(f"{arm} evidence/Difix asset paths differ")
        sidecar = validate_difix_target(
            target_path=target_path,
            input_path=input_path,
            reference_path=reference_path,
            camera_fingerprint=record["camera_fingerprint"],
            camera_resolution=(
                int(record["camera"]["width"]),
                int(record["camera"]["height"]),
            ),
            run_metadata=run_metadata,
            require_reproducibility=True,
        )
        record["difix_output_sha256"] = sidecar["output_sha256"]

    context.update(
        {
            "experiment_role": role,
            "pair_audit": str(pair_audit_path),
            "track_h5": str(track_path),
            "evidence_manifest": str(path),
            "evidence_manifest_sha256": sha256_file(path),
            "difix_manifest": str(difix_manifest.resolve()),
            "difix_manifest_sha256": sha256_file(difix_manifest),
            "difix_run_metadata": str(difix_run_metadata_path(difix_manifest)),
            "difix_run_metadata_sha256": sha256_file(
                difix_run_metadata_path(difix_manifest)
            ),
            "difix_cache_run_fingerprint": run_metadata[
                "cache_run_fingerprint"
            ],
        }
    )
    return records, context


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a1-manifest", type=Path, required=True)
    parser.add_argument("--b-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    a1, a1_context = read_records(args.a1_manifest, arm="A1")
    b, b_context = read_records(args.b_manifest, arm="B")
    if set(a1) != set(b):
        raise ValueError("A1 and B held-out manifests have different camera sets")
    shared_fields = (
        "dataset",
        "scene",
        "seed",
        "experiment_role",
        "scene_source_path",
        "controlled_pair_id",
        "pair_audit",
        "pair_audit_sha256",
        "track_h5",
        "track_h5_sha256",
        "source_camera_names",
        "source_camera_set_sha256",
        "source_images_dir",
        "source_image_inventory",
        "source_image_inventory_sha256",
    )
    source_images_dir, source_image_inventory, source_image_inventory_digest = (
        _source_image_inventory(
            Path(a1_context["scene_source_path"]).expanduser().resolve(),
            a1_context["source_camera_names"],
        )
    )
    a1_context.update(
        {
            "source_images_dir": str(source_images_dir),
            "source_image_inventory": source_image_inventory,
            "source_image_inventory_sha256": source_image_inventory_digest,
        }
    )
    b_images_dir, b_source_image_inventory, b_source_image_inventory_digest = (
        _source_image_inventory(
            Path(b_context["scene_source_path"]).expanduser().resolve(),
            b_context["source_camera_names"],
        )
    )
    b_context.update(
        {
            "source_images_dir": str(b_images_dir),
            "source_image_inventory": b_source_image_inventory,
            "source_image_inventory_sha256": b_source_image_inventory_digest,
        }
    )
    for field in shared_fields:
        if a1_context[field] != b_context[field]:
            raise ValueError(f"A1/B identity context mismatch for {field}")
    if a1_context["difix_protocol"] != b_context["difix_protocol"]:
        raise ValueError(
            "A1/B Difix model, code, dtype, timestep, guidance, prompt, or "
            "coordinate policy differs"
        )

    rows = []
    for image_name in sorted(a1):
        left, right = a1[image_name], b[image_name]
        for field in (
            "camera",
            "camera_fingerprint",
            "real_target",
            "real_target_sha256",
            "reference_image",
            "reference_image_sha256",
        ):
            if left[field] != right[field]:
                raise ValueError(f"A1/B {field} mismatch for {image_name}")
        for label, record in (("A1", left), ("B", right)):
            if (
                int(record["checkpoint_iteration"]) != 12000
                or record["checkpoint_state_format"] != "geotrack-research-v2"
                or int(record["checkpoint_gaussian_count"]) <= 0
                or record["checkpoint_render_state_schema"] != RENDER_STATE_SCHEMA
                or record["scene_ply_render_state_schema"]
                != record["checkpoint_render_state_schema"]
                or int(record["scene_ply_gaussian_count"])
                != int(record["checkpoint_gaussian_count"])
                or record["scene_ply_checkpoint_count_match"] is not True
                or record["scene_ply_checkpoint_render_state_match"] is not True
                or record["scene_ply_render_state_sha256"]
                != record["checkpoint_render_state_sha256"]
            ):
                raise ValueError(
                    f"{label} final checkpoint/Scene PLY binding is invalid"
                )
        rows.append(
            {
                "manifest_schema": PAIRED_MANIFEST_SCHEMA,
                **{field: a1_context[field] for field in shared_fields},
                "a1_evidence_manifest": a1_context["evidence_manifest"],
                "a1_evidence_manifest_sha256": a1_context[
                    "evidence_manifest_sha256"
                ],
                "b_evidence_manifest": b_context["evidence_manifest"],
                "b_evidence_manifest_sha256": b_context[
                    "evidence_manifest_sha256"
                ],
                "a1_difix_manifest": a1_context["difix_manifest"],
                "a1_difix_manifest_sha256": a1_context[
                    "difix_manifest_sha256"
                ],
                "b_difix_manifest": b_context["difix_manifest"],
                "b_difix_manifest_sha256": b_context["difix_manifest_sha256"],
                "a1_difix_run_metadata": a1_context["difix_run_metadata"],
                "a1_difix_run_metadata_sha256": a1_context[
                    "difix_run_metadata_sha256"
                ],
                "b_difix_run_metadata": b_context["difix_run_metadata"],
                "b_difix_run_metadata_sha256": b_context[
                    "difix_run_metadata_sha256"
                ],
                "a1_difix_cache_run_fingerprint": a1_context[
                    "difix_cache_run_fingerprint"
                ],
                "b_difix_cache_run_fingerprint": b_context[
                    "difix_cache_run_fingerprint"
                ],
                "a1_difix_protocol": a1_context["difix_protocol"],
                "b_difix_protocol": b_context["difix_protocol"],
                "image_name": image_name,
                "camera": left["camera"],
                "camera_fingerprint": left["camera_fingerprint"],
                "a1_render": left["gs_render"],
                "a1_render_sha256": left["gs_render_sha256"],
                "a1_difix_output": left["difix_output"],
                "a1_difix_output_sha256": left["difix_output_sha256"],
                "b_render": right["gs_render"],
                "b_render_sha256": right["gs_render_sha256"],
                "b_difix_output": right["difix_output"],
                "b_difix_output_sha256": right["difix_output_sha256"],
                "real_target": left["real_target"],
                "real_target_sha256": left["real_target_sha256"],
                "reference_image": left["reference_image"],
                "reference_image_sha256": left["reference_image_sha256"],
                "a1_checkpoint": left["checkpoint"],
                "a1_checkpoint_sha256": left["checkpoint_sha256"],
                "a1_checkpoint_iteration": left["checkpoint_iteration"],
                "a1_checkpoint_state_format": left["checkpoint_state_format"],
                "a1_checkpoint_gaussian_count": left["checkpoint_gaussian_count"],
                "a1_checkpoint_render_state_schema": left[
                    "checkpoint_render_state_schema"
                ],
                "a1_checkpoint_render_state_sha256": left[
                    "checkpoint_render_state_sha256"
                ],
                "a1_checkpoint_controlled_provenance_sha256": left[
                    "checkpoint_controlled_provenance_sha256"
                ],
                "a1_scene_ply_gaussian_count": left["scene_ply_gaussian_count"],
                "a1_scene_ply_checkpoint_count_match": left[
                    "scene_ply_checkpoint_count_match"
                ],
                "a1_scene_ply_render_state_schema": left[
                    "scene_ply_render_state_schema"
                ],
                "a1_scene_ply_render_state_sha256": left[
                    "scene_ply_render_state_sha256"
                ],
                "a1_scene_ply_checkpoint_render_state_match": left[
                    "scene_ply_checkpoint_render_state_match"
                ],
                "b_checkpoint": right["checkpoint"],
                "b_checkpoint_sha256": right["checkpoint_sha256"],
                "b_checkpoint_iteration": right["checkpoint_iteration"],
                "b_checkpoint_state_format": right["checkpoint_state_format"],
                "b_checkpoint_gaussian_count": right["checkpoint_gaussian_count"],
                "b_checkpoint_render_state_schema": right[
                    "checkpoint_render_state_schema"
                ],
                "b_checkpoint_render_state_sha256": right[
                    "checkpoint_render_state_sha256"
                ],
                "b_checkpoint_controlled_provenance_sha256": right[
                    "checkpoint_controlled_provenance_sha256"
                ],
                "b_scene_ply_gaussian_count": right["scene_ply_gaussian_count"],
                "b_scene_ply_checkpoint_count_match": right[
                    "scene_ply_checkpoint_count_match"
                ],
                "b_scene_ply_render_state_schema": right[
                    "scene_ply_render_state_schema"
                ],
                "b_scene_ply_render_state_sha256": right[
                    "scene_ply_render_state_sha256"
                ],
                "b_scene_ply_checkpoint_render_state_match": right[
                    "scene_ply_checkpoint_render_state_match"
                ],
            }
        )

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    metadata = {
        "schema": PAIRED_MANIFEST_METADATA_SCHEMA,
        "passed": True,
        "paired_manifest": str(output),
        "paired_manifest_sha256": sha256_file(output),
        "record_count": len(rows),
        **{field: a1_context[field] for field in shared_fields},
        "a1_evidence_manifest": a1_context["evidence_manifest"],
        "a1_evidence_manifest_sha256": a1_context[
            "evidence_manifest_sha256"
        ],
        "b_evidence_manifest": b_context["evidence_manifest"],
        "b_evidence_manifest_sha256": b_context["evidence_manifest_sha256"],
        "a1_difix_manifest": a1_context["difix_manifest"],
        "a1_difix_manifest_sha256": a1_context["difix_manifest_sha256"],
        "a1_difix_run_metadata": a1_context["difix_run_metadata"],
        "a1_difix_run_metadata_sha256": a1_context[
            "difix_run_metadata_sha256"
        ],
        "a1_difix_cache_run_fingerprint": a1_context[
            "difix_cache_run_fingerprint"
        ],
        "b_difix_manifest": b_context["difix_manifest"],
        "b_difix_manifest_sha256": b_context["difix_manifest_sha256"],
        "b_difix_run_metadata": b_context["difix_run_metadata"],
        "b_difix_run_metadata_sha256": b_context[
            "difix_run_metadata_sha256"
        ],
        "b_difix_cache_run_fingerprint": b_context[
            "difix_cache_run_fingerprint"
        ],
        "a1_difix_protocol": a1_context["difix_protocol"],
        "b_difix_protocol": b_context["difix_protocol"],
        "a1_checkpoint": a1_context["checkpoint"],
        "a1_checkpoint_sha256": a1_context["checkpoint_sha256"],
        "a1_checkpoint_iteration": a1_context["checkpoint_iteration"],
        "a1_checkpoint_state_format": a1_context["checkpoint_state_format"],
        "a1_checkpoint_gaussian_count": a1_context["checkpoint_gaussian_count"],
        "a1_checkpoint_render_state_schema": a1_context[
            "checkpoint_render_state_schema"
        ],
        "a1_checkpoint_render_state_sha256": a1_context[
            "checkpoint_render_state_sha256"
        ],
        "a1_checkpoint_controlled_provenance_sha256": a1_context[
            "checkpoint_controlled_provenance_sha256"
        ],
        "a1_scene_ply_gaussian_count": a1_context[
            "scene_ply_gaussian_count"
        ],
        "a1_scene_ply_checkpoint_count_match": a1_context[
            "scene_ply_checkpoint_count_match"
        ],
        "a1_scene_ply_render_state_schema": a1_context[
            "scene_ply_render_state_schema"
        ],
        "a1_scene_ply_render_state_sha256": a1_context[
            "scene_ply_render_state_sha256"
        ],
        "a1_scene_ply_checkpoint_render_state_match": a1_context[
            "scene_ply_checkpoint_render_state_match"
        ],
        "b_checkpoint": b_context["checkpoint"],
        "b_checkpoint_sha256": b_context["checkpoint_sha256"],
        "b_checkpoint_iteration": b_context["checkpoint_iteration"],
        "b_checkpoint_state_format": b_context["checkpoint_state_format"],
        "b_checkpoint_gaussian_count": b_context["checkpoint_gaussian_count"],
        "b_checkpoint_render_state_schema": b_context[
            "checkpoint_render_state_schema"
        ],
        "b_checkpoint_render_state_sha256": b_context[
            "checkpoint_render_state_sha256"
        ],
        "b_checkpoint_controlled_provenance_sha256": b_context[
            "checkpoint_controlled_provenance_sha256"
        ],
        "b_scene_ply_gaussian_count": b_context["scene_ply_gaussian_count"],
        "b_scene_ply_checkpoint_count_match": b_context[
            "scene_ply_checkpoint_count_match"
        ],
        "b_scene_ply_render_state_schema": b_context[
            "scene_ply_render_state_schema"
        ],
        "b_scene_ply_render_state_sha256": b_context[
            "scene_ply_render_state_sha256"
        ],
        "b_scene_ply_checkpoint_render_state_match": b_context[
            "scene_ply_checkpoint_render_state_match"
        ],
    }
    metadata_path = output.with_suffix(output.suffix + ".metadata.json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"Wrote immutable paired identity manifest for {len(rows)} held-out "
        f"views: {output}"
    )


if __name__ == "__main__":
    main()
