from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from diffusion_guidance.control_identity import source_camera_set_sha256
from diffusion_guidance.evidence_protocol import source_image_inventory_sha256
from tools import evaluate_track_evidence as evidence


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_evidence_difix_comparison_covers_complete_scientific_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    camera = {"width": 8, "height": 6}
    paths = {
        "gs_render": str((tmp_path / "gs.png").resolve()),
        "reference_image": str((tmp_path / "reference.png").resolve()),
        "difix_output": str((tmp_path / "difix.png").resolve()),
        "checkpoint": str((tmp_path / "checkpoint.pth").resolve()),
        "pair_audit": str((tmp_path / "pair.json").resolve()),
        "track_h5": str((tmp_path / "tracks.h5").resolve()),
        "scene_source_path": str((tmp_path / "fern").resolve()),
    }
    context = {
        "manifest_schema": "heldout_identity_diagnostic_v2",
        "camera": camera,
        "camera_source": "heldout_evaluation_camera",
        "dataset": "LLFF",
        "scene": "fern",
        "seed": 1,
        "experiment_role": "development",
        "scene_source_path": paths["scene_source_path"],
        "paired_identity_arm": "A1",
        "controlled_pair_id": "controlled_pair_fixture",
        "pair_audit": paths["pair_audit"],
        "pair_audit_sha256": "1" * 64,
        "track_h5": paths["track_h5"],
        "track_h5_sha256": "2" * 64,
        "source_camera_names": ["image_1", "image_2", "image_3"],
        "source_camera_set_sha256": "3" * 64,
        "checkpoint": paths["checkpoint"],
        "checkpoint_sha256": "4" * 64,
        "checkpoint_iteration": 12000,
        "checkpoint_state_format": "geotrack-research-v2",
        "checkpoint_gaussian_count": 17,
        "checkpoint_render_state_schema": "renderer-visible-state-v1",
        "checkpoint_render_state_sha256": "5" * 64,
        "checkpoint_controlled_provenance_sha256": "9" * 64,
        "scene_ply_gaussian_count": 17,
        "scene_ply_checkpoint_count_match": True,
        "scene_ply_render_state_schema": "renderer-visible-state-v1",
        "scene_ply_render_state_sha256": "5" * 64,
        "scene_ply_checkpoint_render_state_match": True,
    }
    record = {
        **context,
        "image_name": "heldout_1",
        "camera_fingerprint": "heldout-camera-fixture",
        **paths,
        "gs_render_sha256": "6" * 64,
        "reference_image_sha256": "7" * 64,
    }
    cache_record = {
        **context,
        "key": record["camera_fingerprint"],
        "input": paths["gs_render"],
        "reference_image": paths["reference_image"],
        "target": paths["difix_output"],
        "input_sha256": record["gs_render_sha256"],
        "reference_image_sha256": record["reference_image_sha256"],
    }
    monkeypatch.setattr(
        evidence,
        "validate_difix_target",
        lambda **kwargs: {"output_sha256": "8" * 64},
    )

    assert evidence._validate_evidence_difix(record, cache_record, {}) == {
        "output_sha256": "8" * 64
    }
    cache_record["scene_ply_gaussian_count"] = 18
    with pytest.raises(ValueError, match="scientific context mismatch"):
        evidence._validate_evidence_difix(record, cache_record, {})


def _write(path: Path, content: str = "fixture\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path.resolve()


def _paired_metadata_fixture(tmp_path: Path) -> tuple[Path, dict, dict[str, dict]]:
    paired = _write(tmp_path / "paired.jsonl", "{}\n")
    pair_audit = _write(tmp_path / "pair_audit.json", "{}\n")
    track = _write(tmp_path / "tracks.h5")
    scene = tmp_path / "fern"
    images = scene / "images"
    source_camera_names = ["image_1", "image_2", "image_3"]
    source_inventory = []
    for name in source_camera_names:
        image = _write(images / f"{name}.png", f"{name}\n")
        source_inventory.append(
            {"camera_name": name, "path": str(image), "sha256": sha256_file(image)}
        )
    protocol = {
        "model_id": "nvidia/difix_ref",
        "model_revision": "a" * 40,
        "difix_code_commit": "b" * 40,
        "coordinate_policy": "full_frame_to_multiple8_then_bicubic_to_original",
        "dtype": "fp16",
        "timesteps": [199],
        "guidance_scale": 0.0,
        "prompt": "remove degradation",
    }
    metadata = {
        "schema": "paired_identity_manifest_metadata_v4",
        "passed": True,
        "paired_manifest": str(paired),
        "paired_manifest_sha256": sha256_file(paired),
        "record_count": 1,
        "dataset": "LLFF",
        "scene": "fern",
        "seed": 1,
        "experiment_role": "development",
        "scene_source_path": str(scene.resolve()),
        "controlled_pair_id": "controlled_pair_fixture",
        "pair_audit": str(pair_audit),
        "pair_audit_sha256": sha256_file(pair_audit),
        "track_h5": str(track),
        "track_h5_sha256": sha256_file(track),
        "source_camera_names": source_camera_names,
        "source_camera_set_sha256": source_camera_set_sha256(source_camera_names),
        "source_images_dir": str(images.resolve()),
        "source_image_inventory": source_inventory,
        "source_image_inventory_sha256": source_image_inventory_sha256(
            source_inventory
        ),
    }
    run_metadata: dict[str, dict] = {}
    for prefix in ("a1", "b"):
        evidence_manifest = _write(tmp_path / prefix / "evidence_manifest.jsonl")
        difix_manifest = _write(tmp_path / prefix / "difix_manifest.jsonl")
        difix_metadata = _write(
            difix_manifest.with_suffix(
                difix_manifest.suffix + ".difix_metadata.json"
            )
        )
        checkpoint = _write(tmp_path / prefix / "chkpnt12000.pth")
        fingerprint = ("a" if prefix == "a1" else "b") * 64
        run_metadata[str(difix_manifest)] = {
            **protocol,
            "cache_run_fingerprint": fingerprint,
        }
        metadata.update(
            {
                f"{prefix}_evidence_manifest": str(evidence_manifest),
                f"{prefix}_evidence_manifest_sha256": sha256_file(evidence_manifest),
                f"{prefix}_difix_manifest": str(difix_manifest),
                f"{prefix}_difix_manifest_sha256": sha256_file(difix_manifest),
                f"{prefix}_difix_run_metadata": str(difix_metadata),
                f"{prefix}_difix_run_metadata_sha256": sha256_file(difix_metadata),
                f"{prefix}_difix_cache_run_fingerprint": fingerprint,
                f"{prefix}_difix_protocol": dict(protocol),
                f"{prefix}_checkpoint": str(checkpoint),
                f"{prefix}_checkpoint_sha256": sha256_file(checkpoint),
                f"{prefix}_checkpoint_iteration": 12000,
                f"{prefix}_checkpoint_state_format": "geotrack-research-v2",
                f"{prefix}_checkpoint_gaussian_count": 17,
                f"{prefix}_checkpoint_render_state_schema": "renderer-visible-state-v1",
                f"{prefix}_checkpoint_render_state_sha256": "c" * 64,
                f"{prefix}_checkpoint_controlled_provenance_sha256": "d" * 64,
                f"{prefix}_scene_ply_gaussian_count": 17,
                f"{prefix}_scene_ply_checkpoint_count_match": True,
                f"{prefix}_scene_ply_render_state_schema": "renderer-visible-state-v1",
                f"{prefix}_scene_ply_render_state_sha256": "c" * 64,
                f"{prefix}_scene_ply_checkpoint_render_state_match": True,
            }
        )
    return paired, metadata, run_metadata


def test_paired_v3_protocol_is_recomputed_from_difix_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paired, metadata, run_metadata = _paired_metadata_fixture(tmp_path)
    metadata_path = paired.with_suffix(paired.suffix + ".metadata.json")
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    monkeypatch.setattr(evidence, "require_scene_role", lambda *args: "development")
    monkeypatch.setattr(
        evidence,
        "read_pair_audit",
        lambda path: (
            Path(path).resolve(),
            {"scene_source_path": metadata["scene_source_path"]},
        ),
    )
    monkeypatch.setattr(evidence, "require_audit_binding", lambda *args, **kwargs: None)
    monkeypatch.setattr(evidence, "audit_supports_contrast", lambda *args: True)
    monkeypatch.setattr(
        evidence,
        "load_difix_run_metadata",
        lambda manifest, **kwargs: run_metadata[str(Path(manifest).resolve())],
    )
    monkeypatch.setattr(
        evidence,
        "load_checkpoint_summary",
        lambda *args, **kwargs: {
            "iteration": 12000,
            "format": "geotrack-research-v2",
            "gaussian_count": 17,
            "render_state_schema": "renderer-visible-state-v1",
            "render_state_sha256": "c" * 64,
            "controlled_provenance": {"fixture": True},
            "controlled_provenance_sha256": "d" * 64,
        },
    )
    final_bindings = []
    monkeypatch.setattr(
        evidence,
        "require_final_checkpoint_binding",
        lambda summary, audit, *, method, checkpoint_path: final_bindings.append(
            (
                summary["controlled_provenance_sha256"],
                method,
                Path(checkpoint_path).name,
            )
        ),
    )

    assert evidence.validate_paired_manifest_metadata(paired, "A1")["passed"] is True
    assert final_bindings == [
        ("d" * 64, "A1", "chkpnt12000.pth"),
        ("d" * 64, "B", "chkpnt12000.pth"),
    ]
    metadata["a1_difix_protocol"]["dtype"] = "fp32"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="stale or fabricated"):
        evidence.validate_paired_manifest_metadata(paired, "A1")
