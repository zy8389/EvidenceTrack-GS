from __future__ import annotations

import copy
import csv
import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from diffusion_guidance.control_identity import source_camera_set_sha256
from diffusion_guidance.evidence_protocol import source_image_inventory_sha256
from tools import compare_identity_reports as comparison
from tools.paired_evidence_report import VARIANTS, analyze


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_fixture(path: Path, content: str = "fixture\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path.resolve()


def rows(*, ground_truth_x: float = 11.0):
    result = []
    for variant in VARIANTS:
        result.append(
            {
                "image_name": "heldout.png",
                "track_id": "7",
                "target_variant": variant,
                "window_radius": 32,
                "feature_backend": "dinov2",
                "image_width": 80,
                "image_height": 60,
                "feature_stride": 14,
                "feature_width": 5,
                "feature_height": 4,
                "projection_x": 10.0,
                "projection_y": 20.0,
                "ground_truth_x": ground_truth_x,
                "ground_truth_y": 21.0,
                "projection_error": 2 ** 0.5,
                "matching_error": 1.0,
                "hard_shuffled_error": 3.0,
                "hard_negative_track_id": "8",
                "hard_negative_projection_distance_px": 4.0,
                "hard_negative_within_radius": True,
                "hard_negative_selection_mode": "local_hard_negative",
                "hard_negative_valid": True,
                "hard_negative_radius": 32,
                "global_shuffle_valid": False,
            }
        )
    return result


def protocol_fixture(tmp_path: Path) -> dict:
    scene = tmp_path / "fern"
    source_image = write_fixture(scene / "images" / "source_1.png", "source\n")
    source_inventory = [
        {
            "camera_name": "source_1",
            "path": str(source_image),
            "sha256": sha256_file(source_image),
        }
    ]
    pair_audit = write_fixture(tmp_path / "pair_audit.json", "{}\n")
    track = write_fixture(tmp_path / "tracks.h5")
    paired = write_fixture(tmp_path / "paired.jsonl", "{}\n")
    paired_metadata = write_fixture(tmp_path / "paired.jsonl.metadata.json", "{}\n")
    a1_evidence = write_fixture(tmp_path / "A1" / "evidence_manifest.jsonl")
    b_evidence = write_fixture(tmp_path / "B" / "evidence_manifest.jsonl")
    a1_difix = write_fixture(tmp_path / "A1" / "difix_manifest.jsonl")
    b_difix = write_fixture(tmp_path / "B" / "difix_manifest.jsonl")
    a1_difix_metadata = write_fixture(
        tmp_path / "A1" / "difix_manifest.jsonl.difix_metadata.json"
    )
    b_difix_metadata = write_fixture(
        tmp_path / "B" / "difix_manifest.jsonl.difix_metadata.json"
    )
    a1_checkpoint = write_fixture(tmp_path / "A1" / "chkpnt12000.pth")
    b_checkpoint = write_fixture(tmp_path / "B" / "chkpnt12000.pth")
    dino_repository = tmp_path / "dinov2"
    write_fixture(dino_repository / "hubconf.py")
    dino_weights = write_fixture(tmp_path / "weights" / "dinov2_vits14.pth")
    difix_protocol = {
        "model_id": "nvidia/difix_ref",
        "model_revision": "d" * 40,
        "difix_code_commit": "e" * 40,
        "coordinate_policy": "full_frame_to_multiple8_then_bicubic_to_original",
        "dtype": "fp16",
        "timesteps": [199],
        "guidance_scale": 0.0,
        "prompt": "remove degradation",
    }
    protocol = {
        "feature_map_geometry": [{"backend": "dinov2:dinov2_vits14"}],
        "feature_backend": "dinov2",
        "feature_backend_identity": "dinov2:dinov2_vits14",
        "hard_negative_radius": 32.0,
        "temperature": 0.07,
        "track_h5": str(track),
        "track_h5_sha256": sha256_file(track),
        "target_manifest": str(paired),
        "target_manifest_sha256": sha256_file(paired),
        "dataset": "LLFF",
        "scene": "fern",
        "seed": 1,
        "experiment_role": "development",
        "scene_source_path": str(scene.resolve()),
        "controlled_pair_id": "controlled_pair_fixture",
        "pair_audit": str(pair_audit),
        "pair_audit_sha256": sha256_file(pair_audit),
        "source_camera_names": ["source_1"],
        "source_camera_set_sha256": source_camera_set_sha256(["source_1"]),
        "source_images_dir": str((scene / "images").resolve()),
        "source_image_inventory": source_inventory,
        "source_image_inventory_sha256": source_image_inventory_sha256(
            source_inventory
        ),
        "paired_manifest_metadata": str(paired_metadata),
        "paired_manifest_metadata_sha256": sha256_file(paired_metadata),
        "paired_manifest_record_count": 1,
        "a1_evidence_manifest": str(a1_evidence),
        "a1_evidence_manifest_sha256": sha256_file(a1_evidence),
        "b_evidence_manifest": str(b_evidence),
        "b_evidence_manifest_sha256": sha256_file(b_evidence),
        "a1_difix_manifest": str(a1_difix),
        "a1_difix_manifest_sha256": sha256_file(a1_difix),
        "b_difix_manifest": str(b_difix),
        "b_difix_manifest_sha256": sha256_file(b_difix),
        "a1_difix_run_metadata": str(a1_difix_metadata),
        "a1_difix_run_metadata_sha256": sha256_file(a1_difix_metadata),
        "b_difix_run_metadata": str(b_difix_metadata),
        "b_difix_run_metadata_sha256": sha256_file(b_difix_metadata),
        "a1_difix_cache_run_fingerprint": "2" * 64,
        "b_difix_cache_run_fingerprint": "3" * 64,
        "dinov2_model": "dinov2_vits14",
        "dinov2_local_repository_path": str(dino_repository.resolve()),
        "dinov2_repository_commit": "a" * 40,
        "dinov2_pretrained_weight_path": str(dino_weights),
        "dinov2_pretrained_weight_sha256": sha256_file(dino_weights),
        "a1_difix_protocol": dict(difix_protocol),
        "b_difix_protocol": dict(difix_protocol),
    }
    for prefix, checkpoint in (("a1", a1_checkpoint), ("b", b_checkpoint)):
        protocol.update(
            {
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
    return protocol


def report(payload: list[dict], tmp_path: Path, arm: str) -> dict:
    result = analyze(payload, 32, require_exact_support=True)
    result.update(protocol_fixture(tmp_path))
    csv_path = tmp_path / arm.lower() / "per_track.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(payload[0]))
        writer.writeheader()
        writer.writerows(payload)
    evaluator_metadata = write_fixture(
        tmp_path / arm.lower() / "metadata.json", json.dumps({"arm": arm}) + "\n"
    )
    result.update(
        {
            "paired_identity_arm": arm,
            "input_csv": str(csv_path.resolve()),
            "input_csv_sha256": sha256_file(csv_path),
            "per_track_csv": str(csv_path.resolve()),
            "per_track_csv_sha256": sha256_file(csv_path),
            "evaluator_metadata": str(evaluator_metadata),
            "evaluator_metadata_sha256": sha256_file(evaluator_metadata),
        }
    )
    return result


def patch_external_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(comparison, "_recompute_file_backed_report", lambda *args: None)
    monkeypatch.setattr(
        comparison,
        "read_pair_audit",
        lambda path, **kwargs: (
            Path(path).resolve(),
            {
                "pair_id": "controlled_pair_fixture",
                "audited_contrasts": [["A1", "B"]],
                "audited_input_file_count": 1,
                "audited_input_fingerprint": "f" * 64,
            },
        ),
    )
    monkeypatch.setattr(comparison, "require_audit_binding", lambda *args, **kwargs: None)
    monkeypatch.setattr(comparison, "audit_supports_contrast", lambda *args: True)


def test_hard_negative_counts_must_form_a_disjoint_partition():
    assert comparison._hard_negative_outcome_partition(
        valid_count=10,
        correct_failed=2,
        wrong_failed=3,
        both_success=6,
    ) == {
        "both_failed": 1,
        "correct_only_failed": 1,
        "wrong_only_failed": 2,
        "both_success": 6,
    }
    with pytest.raises(ValueError, match="disjoint partition"):
        comparison._hard_negative_outcome_partition(
            valid_count=10,
            correct_failed=2,
            wrong_failed=9,
            both_success=3,
        )


def test_exact_support_hash_is_order_independent_and_comparable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    patch_external_validation(monkeypatch)
    a1 = report(rows(), tmp_path, "A1")
    b = report(list(reversed(rows())), tmp_path, "B")
    result = comparison.compare_reports(a1, b)
    assert result["paired_support_sha256"] == a1["paired_support_sha256"]
    assert result["image_names"] == ["heldout.png"]


def test_proxy_label_change_changes_support_hash_and_blocks_comparison(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    patch_external_validation(monkeypatch)
    a1 = report(rows(), tmp_path, "A1")
    b = report(rows(ground_truth_x=12.0), tmp_path, "B")
    assert a1["paired_support_sha256"] != b["paired_support_sha256"]
    with pytest.raises(ValueError, match="paired_support_sha256"):
        comparison.compare_reports(a1, b)


def test_dinov2_commit_or_weight_mismatch_blocks_comparison(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    patch_external_validation(monkeypatch)
    a1 = report(rows(), tmp_path, "A1")
    b = copy.deepcopy(a1)
    b["paired_identity_arm"] = "B"
    b["dinov2_repository_commit"] = "d" * 40
    with pytest.raises(ValueError, match="dinov2_repository_commit"):
        comparison.compare_reports(a1, b)
