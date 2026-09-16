from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from diffusion_guidance.evidence_protocol import dinov2_protocol_from_metadata


def dino_metadata(tmp_path: Path):
    repository = tmp_path / "dinov2"
    repository.mkdir()
    (repository / "hubconf.py").write_text("# fixture\n", encoding="utf-8")
    weights = tmp_path / "weights" / "dinov2_vits14.pth"
    weights.parent.mkdir()
    weights.write_bytes(b"fixed-dinov2-weight-fixture")
    return {
        "feature_backend": "dinov2",
        "dinov2_model": "dinov2_vits14",
        "window_radii": [16, 32, 48],
        "primary_radius": 32,
        "hard_negative_radius": 32,
        "temperature": 0.07,
        "track_h5_sha256": "d" * 64,
        "feature_maps": [
            {
                "backend": "dinov2:dinov2_vits14",
                "input_image_resolution": [60, 80],
                "model_input_resolution": [56, 70],
                "feature_resolution": [4, 5],
                "feature_stride": [15.0, 16.0],
                "effective_feature_stride": (240.0) ** 0.5,
                "interpolation_method": "bicubic_input_bilinear_coordinate_sampling",
                "target_variant": "GS Render + feature matching",
            }
        ],
        "extractor_provenance": {
            "backend": "dinov2:dinov2_vits14",
            "local_repository_path": str(repository),
            "repository_commit": "a" * 40,
            "pretrained_weight_path": str(weights),
            "pretrained_weight_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
            "weight_loading": "explicit_local_state_dict",
        },
    }


def test_report_protocol_contains_immutable_dinov2_identity(tmp_path: Path):
    protocol = dinov2_protocol_from_metadata(dino_metadata(tmp_path))
    assert protocol["feature_backend_identity"] == "dinov2:dinov2_vits14"
    assert protocol["dinov2_repository_commit"] == "a" * 40
    assert len(protocol["dinov2_pretrained_weight_sha256"]) == 64


def test_missing_or_malformed_dinov2_weight_hash_fails_closed(tmp_path: Path):
    metadata = dino_metadata(tmp_path)
    metadata["extractor_provenance"]["pretrained_weight_sha256"] = "unknown"
    with pytest.raises(ValueError, match="pretrained_weight_sha256"):
        dinov2_protocol_from_metadata(metadata)


def test_feature_extractor_hashes_explicit_local_weights(monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    from diffusion_guidance import evidence_features

    repository = tmp_path / "dinov2"
    repository.mkdir()
    (repository / "hubconf.py").write_text("# fixture\n", encoding="utf-8")
    weights = tmp_path / "weights.pth"
    weights.write_bytes(b"fixed-weight-fixture")
    monkeypatch.setenv("DINOV2_REPO", str(repository))
    monkeypatch.setenv("DINOV2_WEIGHT_PATH", str(weights))

    class Model:
        def load_state_dict(self, state, strict=False):
            return [], []

        def to(self, device):
            return self

        def eval(self):
            return self

    def git_output(command, text=True):
        return "a" * 40 + "\n" if "rev-parse" in command else ""

    monkeypatch.setattr(evidence_features.subprocess, "check_output", git_output)
    monkeypatch.setattr(torch.hub, "load", lambda *args, **kwargs: Model())
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: {})
    extractor = evidence_features.FeatureExtractor("dinov2", device="cpu")
    provenance = extractor.reproducibility_metadata()
    assert provenance["repository_commit"] == "a" * 40
    assert provenance["pretrained_weight_sha256"] == hashlib.sha256(
        weights.read_bytes()
    ).hexdigest()
