from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from diffusion_guidance.camera_utils import camera_fingerprint_from_payload
from diffusion_guidance.difix_provenance import (
    A0_PSEUDO_MANIFEST_SCHEMA,
    CACHE_IDENTITY_FIELDS,
    cache_run_fingerprint,
    reproducibility_check_sha256,
    validate_difix_target,
)
from diffusion_guidance.evidence_matching import choose_hard_shuffled_indices
from tools.aggregate_results import filter_by_role
from tools.audit_controlled_pair import audit
from tools.paired_evidence_report import VARIANTS, analyze


def png(path: Path, value: int = 0):
    Image.new("RGB", (8, 6), color=(value, value, value)).save(path)


def sha(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def camera():
    return dict(R=np.eye(3).tolist(), T=[0, 0, 0], FoVx=1.0, FoVy=1.0, width=8, height=6,
                intrinsics=dict(fx=4.0, fy=4.0, cx=3.5, cy=2.5, width=8, height=6))


def test_cache_hash_validation_fail_closed(tmp_path):
    inp, ref, target = tmp_path / "input.png", tmp_path / "ref.png", tmp_path / "target.png"
    png(inp, 1); png(ref, 2); png(target, 3)
    payload = camera(); key = camera_fingerprint_from_payload(payload)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({"key": key, "camera": payload, "input": str(inp), "reference_image": str(ref), "target": str(target)}) + "\n", encoding="utf-8")
    run_metadata = {
        "manifest_schema": A0_PSEUDO_MANIFEST_SCHEMA,
        "manifest": str(manifest.resolve()),
        "manifest_sha256": sha(manifest), "record_count": 1, "seed": 1,
        "model_id": "fixture/difix", "model_revision": "a" * 40,
        "difix_code_commit": "b" * 40, "difix_code_clean": True,
        "coordinate_policy": "full_frame_to_multiple8_then_bicubic_to_original",
        "dtype": "fp32", "timesteps": [199], "guidance_scale": 0.0,
        "prompt": "remove degradation",
    }
    run_metadata["reproducibility_check"] = {
        "passed": True,
        "mode": "byte_exact",
        "camera_fingerprint": key,
        "input_sha256": sha(inp),
        "reference_sha256": sha(ref),
        "output_sha256": sha(target),
        "repeated_output_sha256": sha(target),
    }
    run_metadata["reproducibility_check_sha256"] = reproducibility_check_sha256(
        run_metadata["reproducibility_check"]
    )
    run_metadata["cache_run_fingerprint"] = cache_run_fingerprint(run_metadata)
    manifest.with_suffix(".jsonl.difix_metadata.json").write_text(json.dumps(run_metadata), encoding="utf-8")
    sidecar = {
        **{field: run_metadata[field] for field in CACHE_IDENTITY_FIELDS},
        "cache_run_fingerprint": run_metadata["cache_run_fingerprint"],
        "input_image": str(inp.resolve()), "reference_image": str(ref.resolve()),
        "target_image": str(target.resolve()),
        "input_sha256": sha(inp), "reference_sha256": sha(ref),
        "output_sha256": sha(target), "camera_fingerprint": key,
        "resolution": [8, 6],
    }
    target.with_suffix(".png.metadata.json").write_text(json.dumps(sidecar), encoding="utf-8")
    validate_difix_target(
        target_path=target,
        input_path=inp,
        reference_path=ref,
        camera_fingerprint=key,
        camera_resolution=(8, 6),
        run_metadata=run_metadata,
    )
    target.write_bytes(b"tampered")
    with pytest.raises((ValueError, OSError)):
        validate_difix_target(
            target_path=target,
            input_path=inp,
            reference_path=ref,
            camera_fingerprint=key,
            camera_resolution=(8, 6),
            run_metadata=run_metadata,
        )


def test_hard_negative_locality_metadata():
    queries = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])
    indices, meta = choose_hard_shuffled_indices(queries, np.array([[0., 0.], [4., 0.], [100., 0.]]), nearby_radius=10, return_metadata=True)
    assert meta["selection_mode"].tolist()[:2] == ["local_hard_negative", "local_hard_negative"]
    assert meta["within_radius"].tolist()[:2] == [True, True]
    assert meta["selection_mode"][2] == "global_fallback"
    assert not meta["within_radius"][2]
    assert indices[2] in (0, 1)


def identity_rows(correct=np.nan, wrong=np.nan, valid=True):
    return [dict(image_name="x", track_id="1", window_radius=32, target_variant=variant,
                 image_width=3, image_height=4, projection_error=2, matching_error=correct,
                 feature_stride=1, hard_negative_valid=valid, hard_shuffled_error=wrong,
                 global_shuffle_valid=False) for variant in VARIANTS]


def test_failure_aware_identity_margin_counts_wrong_failure():
    report = analyze(identity_rows(correct=1, wrong=np.nan), 32)["scene_metrics"]
    assert report["gs_identity_margin_failure_aware"] == 4
    assert report["gs_identity_margin_conditional"] is None
    assert report["gs_wrong_failure_count"] == 1
    assert report["gs_both_success_count"] == 0


def test_invalid_hard_negative_excluded_from_primary_margin():
    report = analyze(identity_rows(correct=1, wrong=3, valid=False), 32)["scene_metrics"]
    assert report["gs_identity_margin_failure_aware"] is None
    assert report["gs_hard_negative_valid_count"] == 0


def write_pair_run(path: Path, role: str, start_hash: str, sequence, pseudo=False):
    path.mkdir()
    (path / "controlled_ab_metadata.json").write_text(json.dumps({
        "role": role, "seed": 1, "start_checkpoint_sha256": start_hash,
        "checkpoint_iteration": 10000, "final_iteration": 12000,
        "real_view_sampler": "independent_seeded_cycle_shuffle_v1",
        "optimizer_state_restored": True,
        "checkpoint_state_format": "geotrack-research-v2",
        "difix_enabled": pseudo,
    }), encoding="utf-8")
    (path / "real_view_sequence.json").write_text(json.dumps(sequence), encoding="utf-8")
    if pseudo:
        (path / "pseudo_supervision_usage.json").write_text(json.dumps({"pseudo_supervision_calls": 39, "unique_pseudo_views_used": 32}), encoding="utf-8")


def test_pair_audit_emits_false_instead_of_throwing(tmp_path):
    seq = [[i, "image"] for i in range(2000)]
    write_pair_run(tmp_path / "a1", "A1", "same", seq)
    write_pair_run(tmp_path / "b", "B", "different", seq, pseudo=True)
    report = audit(tmp_path / "a1", tmp_path / "b", require_pseudo=True)
    assert report["passed"] is False
    assert "start_checkpoint_hash_mismatch" in report["failures"]


def test_confirmatory_filter_requires_role_and_excludes_development():
    rows = [
        dict(dataset="LLFF", role="development", scene="fern"),
        dict(dataset="LLFF", role="confirmatory", scene="flower"),
    ]
    assert [row["scene"] for row in filter_by_role(rows, "confirmatory")] == ["flower"]
    with pytest.raises(ValueError):
        filter_by_role([dict(dataset="LLFF", scene="fern")], "confirmatory")
