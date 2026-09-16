from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from diffusion_guidance.difix_provenance import (
    A0_PSEUDO_MANIFEST_SCHEMA,
    CACHE_IDENTITY_FIELDS,
    cache_run_fingerprint,
    difix_run_metadata_path,
    load_difix_run_metadata,
    manifest_record_sha256,
    reproducibility_check_sha256,
    sha256_file,
    validate_difix_target,
    validate_self_render_target,
)
from diffusion_guidance.checkpoint_state import renderer_state_sha256


def make_cache(tmp_path: Path):
    manifest = tmp_path / "manifest.jsonl"
    records = []
    paths = []
    for index in range(2):
        input_path = tmp_path / f"input_{index}.png"
        reference_path = tmp_path / f"reference_{index}.png"
        target_path = tmp_path / f"target_{index}.png"
        Image.new("RGB", (8, 6), color=(index, 1, 2)).save(input_path)
        Image.new("RGB", (8, 6), color=(3, index, 4)).save(reference_path)
        Image.new("RGB", (8, 6), color=(5, 6, index)).save(target_path)
        records.append(
            {
                "key": f"camera-{index}",
                "input": str(input_path),
                "reference_image": str(reference_path),
                "target": str(target_path),
            }
        )
        paths.append((input_path, reference_path, target_path))
    manifest.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    run_metadata = {
        "manifest_schema": A0_PSEUDO_MANIFEST_SCHEMA,
        "manifest": str(manifest.resolve()),
        "manifest_sha256": sha256_file(manifest),
        "record_count": 2,
        "seed": 1,
        "model_id": "nvidia/difix_ref",
        "model_revision": "a" * 40,
        "difix_code_commit": "b" * 40,
        "difix_code_clean": True,
        "coordinate_policy": "full_frame_to_multiple8_then_bicubic_to_original",
        "dtype": "fp16",
        "timesteps": [199],
        "guidance_scale": 0.0,
        "prompt": "remove degradation",
    }
    run_metadata["reproducibility_check"] = {
        "passed": True,
        "mode": "byte_exact",
        "camera_fingerprint": records[0]["key"],
        "input_sha256": sha256_file(paths[0][0]),
        "reference_sha256": sha256_file(paths[0][1]),
        "output_sha256": sha256_file(paths[0][2]),
        "repeated_output_sha256": sha256_file(paths[0][2]),
    }
    run_metadata["reproducibility_check_sha256"] = reproducibility_check_sha256(
        run_metadata["reproducibility_check"]
    )
    run_metadata["cache_run_fingerprint"] = cache_run_fingerprint(run_metadata)
    difix_run_metadata_path(manifest).write_text(
        json.dumps(run_metadata), encoding="utf-8"
    )
    for record, (input_path, reference_path, target_path) in zip(records, paths):
        sidecar = {
            **{field: run_metadata[field] for field in CACHE_IDENTITY_FIELDS},
            "cache_run_fingerprint": run_metadata["cache_run_fingerprint"],
            "input_image": str(input_path.resolve()),
            "reference_image": str(reference_path.resolve()),
            "target_image": str(target_path.resolve()),
            "input_sha256": sha256_file(input_path),
            "reference_sha256": sha256_file(reference_path),
            "output_sha256": sha256_file(target_path),
            "camera_fingerprint": record["key"],
            "resolution": [8, 6],
            "source_manifest_record": record,
            "source_manifest_record_sha256": manifest_record_sha256(record),
        }
        target_path.with_suffix(".png.metadata.json").write_text(
            json.dumps(sidecar), encoding="utf-8"
        )
    return manifest, records, paths


def test_renderer_state_hash_is_shared_by_checkpoint_and_live_model():
    core = {
        "active_sh_degree": 1,
        "_xyz": torch.tensor([[1.0, 2.0, 3.0]]),
        "_features_dc": torch.tensor([[[0.1, 0.2, 0.3]]]),
        "_features_rest": torch.zeros((1, 3, 3)),
        "_scaling": torch.zeros((1, 3)),
        "_rotation": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        "_opacity": torch.tensor([[0.5]]),
    }
    model = SimpleNamespace(**core)
    assert renderer_state_sha256({"core": core}) == renderer_state_sha256(model)
    changed = dict(core)
    changed["_opacity"] = torch.tensor([[0.6]])
    assert renderer_state_sha256(changed) != renderer_state_sha256(core)


def validate_record(record, paths, run_metadata, *, require_reproducibility=False):
    input_path, reference_path, target_path = paths
    return validate_difix_target(
        target_path=target_path,
        input_path=input_path,
        reference_path=reference_path,
        camera_fingerprint=record["key"],
        camera_resolution=(8, 6),
        run_metadata=run_metadata,
        require_reproducibility=require_reproducibility,
    )


def test_every_sidecar_is_bound_to_one_run_metadata_record(tmp_path):
    manifest, records, paths = make_cache(tmp_path)
    run_metadata = load_difix_run_metadata(manifest, expected_record_count=2)
    for record, record_paths in zip(records, paths):
        validate_record(record, record_paths, run_metadata)


def test_reproducibility_gate_is_bound_to_exact_manifest_assets(tmp_path):
    manifest, records, paths = make_cache(tmp_path)
    run_metadata = load_difix_run_metadata(
        manifest, expected_record_count=2, require_reproducibility=True
    )
    validate_record(
        records[0], paths[0], run_metadata, require_reproducibility=True
    )
    metadata_path = difix_run_metadata_path(manifest)
    tampered = json.loads(metadata_path.read_text(encoding="utf-8"))
    tampered["reproducibility_check"]["camera_fingerprint"] = "camera-1"
    metadata_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="reproducibility"):
        load_difix_run_metadata(
            manifest, expected_record_count=2, require_reproducibility=True
        )


def test_sidecar_model_revision_mismatch_fails_closed(tmp_path):
    manifest, records, paths = make_cache(tmp_path)
    run_metadata = load_difix_run_metadata(manifest, expected_record_count=2)
    sidecar_path = paths[1][2].with_suffix(".png.metadata.json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["model_revision"] = "c" * 40
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    with pytest.raises(ValueError, match="model_revision"):
        validate_record(records[1], paths[1], run_metadata)


def test_target_and_manifest_hash_mismatches_fail_closed(tmp_path):
    manifest, records, paths = make_cache(tmp_path)
    run_metadata = load_difix_run_metadata(manifest, expected_record_count=2)
    paths[0][2].write_bytes(b"tampered")
    with pytest.raises((ValueError, OSError)):
        validate_record(records[0], paths[0], run_metadata)
    manifest.write_text(manifest.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="manifest hash mismatch"):
        load_difix_run_metadata(manifest, expected_record_count=2)


def test_self_render_target_is_bound_to_the_a0_input_hash(tmp_path):
    input_path = tmp_path / "a0.png"
    Image.new("RGB", (8, 6), color=(1, 2, 3)).save(input_path)
    digest = sha256_file(input_path)
    validate_self_render_target(
        input_path=input_path,
        target_path=input_path,
        camera_resolution=(8, 6),
        input_sha256=digest,
        target_sha256=digest,
    )
    input_path.write_bytes(b"overwritten")
    with pytest.raises(ValueError, match="hash"):
        validate_self_render_target(
            input_path=input_path,
            target_path=input_path,
            camera_resolution=(8, 6),
            input_sha256=digest,
            target_sha256=digest,
        )
