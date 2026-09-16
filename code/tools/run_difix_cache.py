#!/usr/bin/env python3
"""Generate a reproducible frozen Difix cache in a separate environment."""

from __future__ import annotations

import argparse
from io import BytesIO
import hashlib
import json
from pathlib import Path
import re
import sys
import subprocess

import numpy as np
from PIL import Image
import torch

from diffusion_guidance.difix_provenance import (
    A0_PSEUDO_MANIFEST_SCHEMA,
    CACHE_IDENTITY_FIELDS,
    HELDOUT_DIAGNOSTIC_MANIFEST_SCHEMA,
    cache_run_fingerprint,
    load_difix_run_metadata,
    reproducibility_check_sha256,
    manifest_record_sha256,
    require_manifest_schema,
    validate_difix_target,
)
from diffusion_guidance.camera_utils import camera_fingerprint_from_payload
from diffusion_guidance.checkpoint_state import RENDER_STATE_SCHEMA, load_checkpoint_summary
from diffusion_guidance.control_identity import (
    source_camera_set_sha256,
    validate_controlled_checkpoint_provenance_files,
)
from diffusion_guidance.experiment_registry import require_scene_role
from diffusion_guidance.result_binding import (
    read_pair_audit,
    require_audit_binding,
    require_final_checkpoint_binding,
)


def load_manifest(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc


def resolve_manifest_path(manifest: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (manifest.parent / path).resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value, label: str) -> str:
    digest = str(value).lower()
    if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError(f"{label} must be a SHA256 digest")
    return digest


def _validate_common_frozen_assets(
    manifest: Path, record: dict, index: int
) -> tuple[Path, Path]:
    target_kind = str(record.get("supervision_target_kind", "difix")).lower()
    if target_kind != "difix":
        raise ValueError(
            f"Difix cache expects supervision_target_kind=difix at record {index}"
        )
    required = (
        "manifest_schema",
        "input_sha256",
        "reference_image_sha256",
    )
    missing = [field for field in required if record.get(field) in (None, "")]
    if missing:
        raise ValueError(
            f"Difix manifest record {index} lacks frozen input provenance: {missing}"
        )
    input_path = resolve_manifest_path(manifest, record["input"])
    reference_value = record.get("reference_image", record.get("ref"))
    reference_path = resolve_manifest_path(manifest, reference_value)
    for label, path in (
        ("GS input", input_path),
        ("reference image", reference_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {label} at manifest record {index}: {path}"
            )
    if _require_sha256(record["input_sha256"], "input_sha256") != sha256_file(
        input_path
    ):
        raise ValueError(f"GS input hash mismatch at manifest record {index}")
    if _require_sha256(
        record["reference_image_sha256"], "reference_image_sha256"
    ) != sha256_file(reference_path):
        raise ValueError(f"Reference image hash mismatch at manifest record {index}")
    try:
        camera_resolution = (
            int(record["camera"]["width"]),
            int(record["camera"]["height"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid camera resolution at manifest record {index}"
        ) from exc
    if any(value <= 0 for value in camera_resolution):
        raise ValueError(f"Non-positive camera resolution at manifest record {index}")
    with Image.open(input_path) as image:
        input_resolution = tuple(map(int, image.size))
    if input_resolution != camera_resolution:
        raise ValueError(
            f"GS input resolution differs from camera domain at manifest record {index}: "
            f"input={input_resolution}, camera={camera_resolution}"
        )
    with Image.open(reference_path) as image:
        reference_resolution = tuple(map(int, image.size))
    if any(value <= 0 for value in reference_resolution):
        raise ValueError(
            f"Reference image has a non-positive resolution at manifest record {index}"
        )
    target_path = resolve_manifest_path(manifest, record["target"])
    if target_path in {input_path, reference_path}:
        raise ValueError(f"Difix output aliases an input at manifest record {index}")
    return input_path, reference_path


def _validate_a0_pseudo_assets(manifest: Path, record: dict, index: int) -> None:
    _validate_common_frozen_assets(manifest, record, index)
    required = (
        "a0_checkpoint",
        "a0_checkpoint_sha256",
        "a0_checkpoint_iteration",
        "a0_checkpoint_state_format",
        "a0_checkpoint_gaussian_count",
        "a0_checkpoint_render_state_schema",
        "a0_checkpoint_render_state_sha256",
        "a0_checkpoint_controlled_provenance_sha256",
        "a0_render_state_source",
        "live_pseudo_audit",
        "live_pseudo_audit_sha256",
        "projection_context_fingerprint",
        "track_h5",
        "track_h5_sha256",
        "source_camera_names",
        "source_camera_set_sha256",
        "export_seed",
    )
    missing = [field for field in required if not record.get(field)]
    if missing:
        raise ValueError(
            f"Difix manifest record {index} lacks immutable A0 provenance: {missing}"
        )
    checkpoint_path = resolve_manifest_path(manifest, record["a0_checkpoint"])
    live_audit_path = resolve_manifest_path(manifest, record["live_pseudo_audit"])
    track_path = resolve_manifest_path(manifest, record["track_h5"])
    for label, path in (
        ("A0 checkpoint", checkpoint_path),
        ("live pseudo-camera audit", live_audit_path),
        ("strict Track H5", track_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label} at manifest record {index}: {path}")
    if _require_sha256(
        record["a0_checkpoint_sha256"], "a0_checkpoint_sha256"
    ) != sha256_file(checkpoint_path):
        raise ValueError(f"A0 checkpoint hash mismatch at manifest record {index}")
    if (
        int(record["a0_checkpoint_iteration"]) != 10000
        or record["a0_checkpoint_state_format"] != "geotrack-research-v2"
    ):
        raise ValueError(f"Invalid A0 checkpoint iteration at manifest record {index}")
    if (
        int(record["a0_checkpoint_gaussian_count"]) <= 0
        or record["a0_render_state_source"]
        != "complete_checkpoint_restore_after_scene_ply_load"
    ):
        raise ValueError(f"Invalid A0 render-state provenance at manifest record {index}")
    _require_sha256(
        record["a0_checkpoint_render_state_sha256"],
        "a0_checkpoint_render_state_sha256",
    )
    source_names = record["source_camera_names"]
    if not isinstance(source_names, list) or source_camera_set_sha256(source_names) != record[
        "source_camera_set_sha256"
    ]:
        raise ValueError(f"Source-camera set mismatch at manifest record {index}")
    checkpoint_summary = load_checkpoint_summary(
        checkpoint_path,
        expected_iteration=10000,
        require_cuda_rng=True,
        require_controlled_provenance=True,
    )
    checkpoint_provenance = validate_controlled_checkpoint_provenance_files(
        checkpoint_summary["controlled_provenance"]
    )
    if (
        checkpoint_provenance["role"] != "A0"
        or checkpoint_provenance["track_h5_path"] != str(track_path)
        or checkpoint_provenance["track_h5_sha256"] != record["track_h5_sha256"]
        or checkpoint_provenance["source_camera_names"] != source_names
        or checkpoint_provenance["source_camera_set_sha256"]
        != record["source_camera_set_sha256"]
    ):
        raise ValueError(
            f"A0 checkpoint provenance differs from manifest inputs at record {index}"
        )
    if checkpoint_summary["gaussian_count"] != int(
        record["a0_checkpoint_gaussian_count"]
    ):
        raise ValueError(f"A0 checkpoint Gaussian count mismatch at record {index}")
    if record["a0_checkpoint_render_state_schema"] != RENDER_STATE_SCHEMA:
        raise ValueError(f"A0 render-state schema mismatch at record {index}")
    if record["a0_checkpoint_render_state_sha256"] != checkpoint_summary[
        "render_state_sha256"
    ]:
        raise ValueError(f"A0 render-state tensor hash mismatch at record {index}")
    if _require_sha256(
        record["a0_checkpoint_controlled_provenance_sha256"],
        "a0_checkpoint_controlled_provenance_sha256",
    ) != checkpoint_summary["controlled_provenance_sha256"]:
        raise ValueError(f"A0 controlled-provenance hash mismatch at record {index}")
    if _require_sha256(record["track_h5_sha256"], "track_h5_sha256") != sha256_file(
        track_path
    ):
        raise ValueError(f"Track H5 hash mismatch at manifest record {index}")
    if record["live_pseudo_audit_sha256"] != sha256_file(live_audit_path):
        raise ValueError(f"Live pseudo-camera audit hash mismatch at manifest record {index}")
    live_audit = json.loads(live_audit_path.read_text(encoding="utf-8"))
    if live_audit.get("passed") is not True:
        raise ValueError(f"Live pseudo-camera audit is not passed at manifest record {index}")
    if live_audit.get("projection_context_fingerprint") != record[
        "projection_context_fingerprint"
    ]:
        raise ValueError(
            f"Live pseudo-camera audit projection context mismatch at manifest record {index}"
        )
    live_keys = {
        item.get("camera_fingerprint")
        for item in live_audit.get("records", [])
        if isinstance(item, dict)
    }
    if record["key"] not in live_keys:
        raise ValueError(f"Pseudo camera absent from live audit at manifest record {index}")


def _validate_heldout_diagnostic_assets(
    manifest: Path, record: dict, index: int
) -> None:
    _validate_common_frozen_assets(manifest, record, index)
    required = (
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
    missing = [field for field in required if record.get(field) in (None, "")]
    if missing:
        raise ValueError(
            f"Held-out Difix record {index} lacks immutable provenance: {missing}"
        )
    arm = str(record["paired_identity_arm"])
    if arm not in {"A1", "B"}:
        raise ValueError(f"Held-out diagnostic arm is invalid at record {index}")
    role = require_scene_role(
        record["dataset"], record["scene"], record["experiment_role"]
    )
    pair_audit_path = resolve_manifest_path(manifest, record["pair_audit"])
    if _require_sha256(
        record["pair_audit_sha256"], "pair_audit_sha256"
    ) != sha256_file(pair_audit_path):
        raise ValueError(f"Pair audit hash mismatch at manifest record {index}")
    _, pair_audit = read_pair_audit(pair_audit_path)
    require_audit_binding(
        pair_audit,
        dataset=str(record["dataset"]),
        scene=str(record["scene"]),
        seed=int(record["seed"]),
        method=arm,
        pair_id=str(record["controlled_pair_id"]),
        role=role,
    )
    scene_source_path = resolve_manifest_path(manifest, record["scene_source_path"])
    if (
        not scene_source_path.is_dir()
        or scene_source_path.name != str(record["scene"])
        or scene_source_path
        != Path(pair_audit["scene_source_path"]).expanduser().resolve()
    ):
        raise ValueError(f"Scene-source binding mismatch at manifest record {index}")
    checkpoint_path = resolve_manifest_path(manifest, record["checkpoint"])
    if _require_sha256(
        record["checkpoint_sha256"], "checkpoint_sha256"
    ) != sha256_file(checkpoint_path):
        raise ValueError(f"Final checkpoint hash mismatch at manifest record {index}")
    checkpoint_summary = load_checkpoint_summary(
        checkpoint_path,
        expected_iteration=12000,
        require_cuda_rng=True,
        require_controlled_provenance=True,
    )
    require_final_checkpoint_binding(
        checkpoint_summary,
        pair_audit,
        method=arm,
        checkpoint_path=checkpoint_path,
    )
    if (
        int(record["checkpoint_iteration"]) != checkpoint_summary["iteration"]
        or record["checkpoint_state_format"] != checkpoint_summary["format"]
        or record["checkpoint_render_state_schema"] != checkpoint_summary[
            "render_state_schema"
        ]
        or int(record["checkpoint_gaussian_count"])
        != checkpoint_summary["gaussian_count"]
        or int(record["scene_ply_gaussian_count"])
        != checkpoint_summary["gaussian_count"]
        or record["scene_ply_checkpoint_count_match"] is not True
        or record["scene_ply_render_state_schema"] != checkpoint_summary[
            "render_state_schema"
        ]
        or record["scene_ply_checkpoint_render_state_match"] is not True
        or record["scene_ply_render_state_sha256"]
        != record["checkpoint_render_state_sha256"]
        or record["checkpoint_render_state_sha256"]
        != checkpoint_summary["render_state_sha256"]
        or record["checkpoint_controlled_provenance_sha256"]
        != checkpoint_summary["controlled_provenance_sha256"]
    ):
        raise ValueError(f"Final checkpoint/Scene PLY mismatch at manifest record {index}")
    _require_sha256(
        record["checkpoint_render_state_sha256"],
        "checkpoint_render_state_sha256",
    )
    track_path = resolve_manifest_path(manifest, record["track_h5"])
    if _require_sha256(record["track_h5_sha256"], "track_h5_sha256") != sha256_file(
        track_path
    ):
        raise ValueError(f"Track H5 hash mismatch at manifest record {index}")
    source_names = record["source_camera_names"]
    if not isinstance(source_names, list) or source_camera_set_sha256(source_names) != record[
        "source_camera_set_sha256"
    ]:
        raise ValueError(f"Source-camera set mismatch at manifest record {index}")
    if record.get("camera_source") != "heldout_evaluation_camera":
        raise ValueError(f"Held-out diagnostic camera source is invalid at record {index}")


def validate_frozen_manifest_assets(manifest: Path, record: dict, index: int) -> str:
    """Dispatch strict validation by the manifest's explicit scientific role."""
    schema = require_manifest_schema(record.get("manifest_schema"))
    if schema == A0_PSEUDO_MANIFEST_SCHEMA:
        _validate_a0_pseudo_assets(manifest, record, index)
    elif schema == HELDOUT_DIAGNOSTIC_MANIFEST_SCHEMA:
        _validate_heldout_diagnostic_assets(manifest, record, index)
    else:  # pragma: no cover - require_manifest_schema already rejects this.
        raise ValueError(f"Unsupported manifest schema: {schema}")
    return schema


def validate_manifest_camera_key(record: dict, index: int) -> None:
    key = str(record["key"])
    match = re.fullmatch(r"(.+)_([0-9a-f]{20})", key)
    if match is None:
        raise ValueError(f"Invalid camera fingerprint at manifest record {index}: {key}")
    expected = camera_fingerprint_from_payload(record["camera"], prefix=match.group(1))
    if key != expected:
        raise ValueError(f"Camera fingerprint mismatch at manifest record {index}")


def png_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="PNG", compress_level=6)
    return buffer.getvalue()


def run_pipeline_once(
    pipe,
    *,
    prompt: str,
    input_image: Image.Image,
    reference_image: Image.Image,
    timestep: int,
    guidance_scale: float,
    seed: int,
    device: str,
) -> Image.Image:
    generator_device = device if str(device).startswith("cuda") else "cpu"
    generator = torch.Generator(device=generator_device)
    generator.manual_seed(int(seed))
    output = pipe(
        prompt,
        image=input_image,
        ref_image=reference_image,
        num_inference_steps=1,
        timesteps=[int(timestep)],
        guidance_scale=float(guidance_scale),
        generator=generator,
        height=max(8, (input_image.height//8)*8),
        width=max(8, (input_image.width//8)*8),
    ).images[0]
    # Explicit full-frame resampling, not cropping: retains the input camera domain.
    return output.resize(input_image.size, Image.Resampling.BICUBIC)


def verify_pair(first: Image.Image, second: Image.Image) -> dict:
    first_array = np.asarray(first.convert("RGB"), dtype=np.int16)
    second_array = np.asarray(second.convert("RGB"), dtype=np.int16)
    if first_array.shape != second_array.shape:
        raise RuntimeError("Difix reproducibility check changed output shape")
    absolute = np.abs(first_array - second_array)
    result = {
        "byte_equal": png_bytes(first) == png_bytes(second),
        "max_abs_uint8": int(absolute.max(initial=0)),
        "mean_abs_uint8": float(absolute.mean()),
    }
    pixel_exact = result["max_abs_uint8"] == 0
    metric_tolerance = (
        result["max_abs_uint8"] <= 1
        and result["mean_abs_uint8"] <= 0.01
    )
    result["pixel_exact"] = pixel_exact
    result["passed"] = bool(result["byte_equal"] or pixel_exact or metric_tolerance)
    if result["byte_equal"]:
        result["mode"] = "byte_exact"
    elif pixel_exact:
        result["mode"] = "pixel_exact"
    elif metric_tolerance:
        result["mode"] = "metric_tolerance"
    else:
        result["mode"] = "failed"
    result["tolerance"] = {
        "max_abs_uint8": 1,
        "mean_abs_uint8": 0.01,
    }
    if not result["passed"]:
        raise RuntimeError(f"Difix reproducibility check failed: {result}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--difix-repo", required=True, type=Path)
    parser.add_argument("--model-id", default="nvidia/difix_ref")
    parser.add_argument("--model-revision", required=True, help="Immutable Hugging Face model commit SHA")
    parser.add_argument("--prompt", default="remove degradation")
    parser.add_argument("--timestep", type=int, default=199)
    parser.add_argument("--guidance-scale", type=float, default=0.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--reproducibility-check", action="store_true")
    args = parser.parse_args()

    if args.skip_existing:
        parser.error("--skip-existing is disabled: rebuild the cache to avoid stale-target reuse")
    if not args.reproducibility_check:
        parser.error("--reproducibility-check is mandatory for a consumable Difix cache")
    if len(args.model_revision) != 40 or any(c not in "0123456789abcdef" for c in args.model_revision.lower()):
        parser.error("--model-revision must be a 40-character commit SHA")
    args.manifest = args.manifest.expanduser().resolve()
    args.difix_repo = args.difix_repo.expanduser().resolve()
    difix_commit = subprocess.check_output(["git", "-C", str(args.difix_repo), "rev-parse", "HEAD"], text=True).strip()
    difix_dirty = subprocess.check_output(
        ["git", "-C", str(args.difix_repo), "status", "--porcelain"], text=True
    ).strip()
    if difix_dirty:
        raise RuntimeError("Difix repository must be clean so its commit fully identifies the cache code")
    if len(difix_commit) not in (40, 64) or any(
        character not in "0123456789abcdef" for character in difix_commit.lower()
    ):
        raise RuntimeError("Difix repository HEAD is not a full immutable commit")
    source_dir = args.difix_repo.expanduser().resolve() / "src"
    if not (source_dir / "pipeline_difix.py").exists():
        raise FileNotFoundError(f"Official Difix pipeline not found in {source_dir}")
    sys.path.insert(0, str(source_dir))
    from pipeline_difix import DifixPipeline  # type: ignore

    records = list(load_manifest(args.manifest))
    if not records:
        raise ValueError("Difix manifest is empty")
    seen_keys, seen_targets, manifest_schemas = set(), set(), set()
    for index, record in enumerate(records):
        missing = [
            field
            for field in ("key", "camera", "input", "target")
            if field not in record
        ]
        if record.get("reference_image", record.get("ref")) is None:
            missing.append("reference_image")
        if missing:
            raise ValueError(f"Difix manifest record {index} is missing {missing}")
        validate_manifest_camera_key(record, index)
        schema = validate_frozen_manifest_assets(args.manifest, record, index)
        manifest_schemas.add(schema)
        seed_field = "export_seed" if schema == A0_PSEUDO_MANIFEST_SCHEMA else "seed"
        if isinstance(record.get(seed_field), bool) or int(record.get(seed_field, -1)) != int(
            args.seed
        ):
            raise ValueError(
                f"Difix --seed does not match {seed_field} at manifest record {index}"
            )
        key = str(record["key"])
        target = resolve_manifest_path(args.manifest, record["target"])
        if key in seen_keys or target in seen_targets:
            raise ValueError(f"Duplicate Difix key or target at manifest record {index}")
        seen_keys.add(key)
        seen_targets.add(target)
        if target.exists() or target.with_suffix(target.suffix + ".metadata.json").exists():
            raise FileExistsError(
                f"Refuse to overwrite immutable Difix cache target/sidecar: {target}"
            )
    if len(manifest_schemas) != 1:
        raise ValueError(
            f"One Difix cache cannot mix manifest schemas: {sorted(manifest_schemas)}"
        )
    manifest_schema = next(iter(manifest_schemas))

    metadata_path = args.manifest.with_suffix(
        args.manifest.suffix + ".difix_metadata.json"
    )
    if metadata_path.exists():
        raise FileExistsError(f"Refuse to overwrite Difix run metadata: {metadata_path}")

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    pipe = DifixPipeline.from_pretrained(
        args.model_id, revision=args.model_revision, trust_remote_code=True, torch_dtype=dtype
    )
    if args.cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(args.device)

    run_metadata = {
        "manifest_schema": manifest_schema,
        "seed": args.seed,
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "difix_code_commit": difix_commit,
        "difix_code_clean": True,
        "coordinate_policy": "full_frame_to_multiple8_then_bicubic_to_original",
        "dtype": args.dtype,
        "timesteps": [args.timestep],
        "guidance_scale": args.guidance_scale,
        "prompt": args.prompt,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "record_count": len(records),
        "reproducibility_check": None,
        "reproducibility_check_sha256": None,
    }
    run_metadata["cache_run_fingerprint"] = None
    for index, record in enumerate(records):
        input_path = resolve_manifest_path(args.manifest, record["input"])
        reference_value = record.get("reference_image", record.get("ref"))
        if reference_value is None:
            raise ValueError(f"Manifest record {index} has no reference image")
        reference_path = resolve_manifest_path(args.manifest, reference_value)
        target_path = resolve_manifest_path(args.manifest, record["target"])
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if args.skip_existing and target_path.exists():
            continue
        with Image.open(input_path) as image:
            input_image = image.convert("RGB")
        with Image.open(reference_path) as image:
            reference_image = image.convert("RGB")

        output = run_pipeline_once(
            pipe,
            prompt=args.prompt,
            input_image=input_image,
            reference_image=reference_image,
            timestep=args.timestep,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
            device=args.device,
        )
        if args.reproducibility_check and run_metadata["reproducibility_check"] is None:
            repeated = run_pipeline_once(
                pipe,
                prompt=args.prompt,
                input_image=input_image,
                reference_image=reference_image,
                timestep=args.timestep,
                guidance_scale=args.guidance_scale,
                seed=args.seed,
                device=args.device,
            )
            run_metadata["reproducibility_check"] = verify_pair(output, repeated)
            run_metadata["reproducibility_check"].update(
                {
                    "camera_fingerprint": str(record["key"]),
                    "input_sha256": sha256_file(input_path),
                    "reference_sha256": sha256_file(reference_path),
                    "output_sha256": hashlib.sha256(png_bytes(output)).hexdigest(),
                    "repeated_output_sha256": hashlib.sha256(
                        png_bytes(repeated)
                    ).hexdigest(),
                }
            )
            run_metadata["reproducibility_check_sha256"] = (
                reproducibility_check_sha256(
                    run_metadata["reproducibility_check"]
                )
            )
            run_metadata["cache_run_fingerprint"] = cache_run_fingerprint(
                run_metadata
            )
        if run_metadata["cache_run_fingerprint"] is None:
            raise RuntimeError("Difix cache lacks a completed reproducibility gate")
        target_path.write_bytes(png_bytes(output))
        metadata = {
            **{field: run_metadata[field] for field in CACHE_IDENTITY_FIELDS},
            "cache_run_fingerprint": run_metadata["cache_run_fingerprint"],
            "reference_image": str(reference_path),
            "input_image": str(input_path),
            "target_image": str(target_path),
            "input_sha256": sha256_file(input_path),
            "reference_sha256": sha256_file(reference_path),
            "output_sha256": sha256_file(target_path),
            "camera_fingerprint": str(record["key"]),
            "resolution": [int(output.width), int(output.height)],
            "source_manifest_record": record,
            "source_manifest_record_sha256": manifest_record_sha256(record),
        }
        target_path.with_suffix(target_path.suffix + ".metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        validate_difix_target(
            target_path=target_path,
            input_path=input_path,
            reference_path=reference_path,
            camera_fingerprint=str(record["key"]),
            camera_resolution=(
                int(record["camera"]["width"]),
                int(record["camera"]["height"]),
            ),
            run_metadata=run_metadata,
            require_reproducibility=True,
        )
        print(f"[{index + 1}/{len(records)}] {target_path}")

    metadata_path.write_text(
        json.dumps(run_metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    load_difix_run_metadata(
        args.manifest,
        expected_record_count=len(records),
        require_reproducibility=True,
    )
    print(f"Difix cache metadata: {metadata_path}")


if __name__ == "__main__":
    main()
