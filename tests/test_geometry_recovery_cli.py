from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import test_geometry_recovery as recovery_gate


def test_recovery_cli_supplies_production_gaussian_pipeline_parameters(
    tmp_path, monkeypatch
):
    output_path = tmp_path / "recovery.json"
    captured = {}

    def fake_run_real(args):
        # Consume the values at the same boundary as Gaussian/Scene setup,
        # rather than merely checking that argparse happened to create names.
        captured.update(
            sh_degree=args.sh_degree,
            source_path=args.source_path,
            model_path=args.model_path,
            use_color=args.use_color,
            convert_SHs_python=args.convert_SHs_python,
            compute_cov3D_python=args.compute_cov3D_python,
            train_bg=args.train_bg,
            strict_track_min_length=args.strict_track_min_length,
            experiment_seed=args.experiment_seed,
        )
        return {"passed": True}

    monkeypatch.setattr(recovery_gate, "run_real", fake_run_real)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "test_geometry_recovery.py",
            "-s",
            str(tmp_path / "scene"),
            "-m",
            str(tmp_path / "recovery-scratch"),
            "--track_path",
            str(tmp_path / "tracks.h5"),
            "--checkpoint",
            str(tmp_path / "chkpnt10000.pth"),
            "--strict_source_only_geometry",
            "--experiment_seed",
            "1",
            "--output",
            str(output_path),
        ],
    )

    recovery_gate.main()

    assert captured == {
        "sh_degree": 3,
        "source_path": str(tmp_path / "scene"),
        "model_path": str(tmp_path / "recovery-scratch"),
        "use_color": True,
        "convert_SHs_python": False,
        "compute_cov3D_python": False,
        "train_bg": False,
        "strict_track_min_length": 2,
        "experiment_seed": 1,
    }
    assert json.loads(output_path.read_text(encoding="utf-8"))["passed"] is True


def test_recovery_cli_reaches_gaussian_scene_setup_boundary(tmp_path, monkeypatch):
    class SceneBoundaryReached(RuntimeError):
        pass

    captured = {}
    fake_scene_module = ModuleType("scene")
    fake_general_utils = ModuleType("utils.general_utils")

    class BoundaryGaussianModel:
        def __init__(self, args):
            captured.update(
                sh_degree=args.sh_degree,
                use_color=args.use_color,
                train_bg=args.train_bg,
                strict_track_min_length=args.strict_track_min_length,
                experiment_seed=args.experiment_seed,
            )

    def boundary_scene(args, gaussians, *, shuffle):
        assert isinstance(gaussians, BoundaryGaussianModel)
        assert shuffle is False
        raise SceneBoundaryReached

    fake_scene_module.GaussianModel = BoundaryGaussianModel
    fake_scene_module.Scene = boundary_scene
    fake_general_utils.safe_state = lambda quiet, seed: None
    monkeypatch.setitem(sys.modules, "scene", fake_scene_module)
    monkeypatch.setitem(sys.modules, "utils.general_utils", fake_general_utils)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "test_geometry_recovery.py",
            "-s",
            str(tmp_path / "scene"),
            "-m",
            str(tmp_path / "recovery-scratch"),
            "--track_path",
            str(tmp_path / "tracks.h5"),
            "--checkpoint",
            str(tmp_path / "chkpnt10000.pth"),
            "--strict_source_only_geometry",
            "--experiment_seed",
            "1",
        ],
    )

    with pytest.raises(SceneBoundaryReached):
        recovery_gate.main()

    assert captured == {
        "sh_degree": 3,
        "use_color": True,
        "train_bg": False,
        "strict_track_min_length": 2,
        "experiment_seed": 1,
    }
