from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from diffusion_guidance.control_identity import source_camera_set_sha256
from tools import make_paired_identity_manifest as paired


def _write(path: Path, content: bytes = b"fixture\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path.resolve()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_read_records_binds_the_resolved_final_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scene = tmp_path / "fern"
    scene.mkdir()
    render = _write(tmp_path / "arm" / "render.png")
    reference = _write(tmp_path / "arm" / "reference.png")
    real_target = _write(tmp_path / "arm" / "real.png")
    difix_output = _write(tmp_path / "arm" / "difix.png")
    checkpoint = _write(tmp_path / "arm" / "chkpnt12000.pth")
    pair_audit = _write(tmp_path / "pair_audit.json", b"{}\n")
    track = _write(tmp_path / "tracks.h5")
    source_names = ["source_1"]
    record = {
        "manifest_schema": paired.HELDOUT_DIAGNOSTIC_MANIFEST_SCHEMA,
        "image_name": "heldout_1",
        "camera": {"width": 8, "height": 6},
        "camera_fingerprint": "heldout-fixture",
        "camera_source": "heldout_evaluation_camera",
        "gs_render": str(render),
        "difix_output": str(difix_output),
        "real_target": str(real_target),
        "reference_image": str(reference),
        "gs_render_sha256": _sha256(render),
        "reference_image_sha256": _sha256(reference),
        "real_target_sha256": _sha256(real_target),
        "dataset": "LLFF",
        "scene": "fern",
        "seed": 1,
        "experiment_role": "development",
        "scene_source_path": str(scene.resolve()),
        "paired_identity_arm": "A1",
        "controlled_pair_id": "controlled-pair-fixture",
        "pair_audit": str(pair_audit),
        "pair_audit_sha256": _sha256(pair_audit),
        "track_h5": str(track),
        "track_h5_sha256": _sha256(track),
        "source_camera_names": source_names,
        "source_camera_set_sha256": source_camera_set_sha256(source_names),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_iteration": 12000,
        "checkpoint_state_format": "geotrack-research-v2",
        "checkpoint_gaussian_count": 17,
        "checkpoint_render_state_schema": "renderer-visible-state-v1",
        "checkpoint_render_state_sha256": "c" * 64,
        "checkpoint_controlled_provenance_sha256": "d" * 64,
        "scene_ply_gaussian_count": 17,
        "scene_ply_checkpoint_count_match": True,
        "scene_ply_render_state_schema": "renderer-visible-state-v1",
        "scene_ply_render_state_sha256": "c" * 64,
        "scene_ply_checkpoint_render_state_match": True,
    }
    evidence_manifest = tmp_path / "arm" / "evidence_manifest.jsonl"
    evidence_manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")

    cache_record = dict(record)
    cache_record.update(
        {
            "key": record["camera_fingerprint"],
            "input": str(render),
            "reference_image": str(reference),
            "target": str(difix_output),
        }
    )
    difix_manifest = tmp_path / "arm" / "difix_manifest.jsonl"
    difix_manifest.write_text(json.dumps(cache_record) + "\n", encoding="utf-8")
    _write(paired.difix_run_metadata_path(difix_manifest), b"{}\n")

    checkpoint_summary = {
        "iteration": 12000,
        "format": "geotrack-research-v2",
        "gaussian_count": 17,
        "render_state_schema": "renderer-visible-state-v1",
        "render_state_sha256": "c" * 64,
        "controlled_provenance": {"fixture": True},
        "controlled_provenance_sha256": "d" * 64,
    }
    run_metadata = {
        "manifest_schema": paired.HELDOUT_DIAGNOSTIC_MANIFEST_SCHEMA,
        "model_id": "nvidia/difix_ref",
        "model_revision": "a" * 40,
        "difix_code_commit": "b" * 40,
        "coordinate_policy": "full_frame_to_multiple8_then_bicubic_to_original",
        "dtype": "fp16",
        "timesteps": [199],
        "guidance_scale": 0.0,
        "prompt": "remove degradation",
        "cache_run_fingerprint": "e" * 64,
    }
    monkeypatch.setattr(paired, "require_scene_role", lambda *args: "development")
    monkeypatch.setattr(
        paired,
        "read_pair_audit",
        lambda path: (Path(path).resolve(), {"scene_source_path": str(scene.resolve())}),
    )
    monkeypatch.setattr(paired, "require_audit_binding", lambda *args, **kwargs: None)
    monkeypatch.setattr(paired, "audit_supports_contrast", lambda *args: True)
    monkeypatch.setattr(paired, "load_checkpoint_summary", lambda *args, **kwargs: checkpoint_summary)
    monkeypatch.setattr(paired, "load_difix_run_metadata", lambda *args, **kwargs: run_metadata)
    monkeypatch.setattr(
        paired,
        "validate_difix_target",
        lambda **kwargs: {"output_sha256": _sha256(difix_output)},
    )
    final_bindings: list[tuple[str, Path]] = []
    monkeypatch.setattr(
        paired,
        "require_final_checkpoint_binding",
        lambda summary, audit, *, method, checkpoint_path: final_bindings.append(
            (method, Path(checkpoint_path).resolve())
        ),
    )

    records, context = paired.read_records(evidence_manifest, arm="A1")

    assert list(records) == ["heldout_1"]
    assert Path(context["checkpoint"]) == checkpoint
    assert final_bindings == [("A1", checkpoint)]


def test_paired_support_accepts_distinct_paths_only_for_identical_bytes() -> None:
    left = {
        "camera": {"width": 8, "height": 6},
        "camera_fingerprint": "heldout-fixture",
        "real_target": "/evidence_a1/real.png",
        "real_target_sha256": "a" * 64,
        "reference_image": "/evidence_a1/reference.png",
        "reference_image_sha256": "b" * 64,
    }
    right = {
        **left,
        "real_target": "/evidence_b/real.png",
        "reference_image": "/evidence_b/reference.png",
    }

    paired._require_shared_heldout_support(left, right, "heldout_1")

    right["real_target_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="real_target_sha256"):
        paired._require_shared_heldout_support(left, right, "heldout_1")

    right["real_target_sha256"] = left["real_target_sha256"]
    right["reference_image_sha256"] = "d" * 64
    with pytest.raises(ValueError, match="reference_image_sha256"):
        paired._require_shared_heldout_support(left, right, "heldout_1")

    right["reference_image_sha256"] = left["reference_image_sha256"]
    right["camera_fingerprint"] = "different-camera"
    with pytest.raises(ValueError, match="camera_fingerprint"):
        paired._require_shared_heldout_support(left, right, "heldout_1")
