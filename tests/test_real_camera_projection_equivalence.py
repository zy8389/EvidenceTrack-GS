from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from diffusion_guidance.calibration_guard import resized_intrinsics
from geometric_constraints.repaired_geometry import project_points_pixel
from tools import check_camera_projection_equivalence as projection_gate
from tools.check_camera_projection_equivalence import compare_camera, direct_projection


def live_camera(calibration: dict, *, width: int, height: int):
    rotation = torch.as_tensor(calibration["rotation_w2c"], dtype=torch.float64)
    translation = torch.as_tensor(calibration["translation_w2c"], dtype=torch.float64)
    world_view = torch.eye(4, dtype=torch.float64)
    world_view[:3, :3] = rotation
    world_view[:3, 3] = translation
    return SimpleNamespace(
        image_name="source_000.png",
        image_width=width,
        image_height=height,
        world_view_transform=world_view.transpose(0, 1),
        research_intrinsics=resized_intrinsics(
            {
                "fx": calibration["fx"],
                "fy": calibration["fy"],
                "cx": calibration["cx"],
                "cy": calibration["cy"],
                "width": calibration["width"],
                "height": calibration["height"],
            },
            width,
            height,
        ),
    )


def calibration():
    return {
        "width": 160,
        "height": 120,
        "fx": 100.0,
        "fy": 97.0,
        "cx": 79.5,
        "cy": 59.5,
        "rotation_w2c": np.eye(3),
        "translation_w2c": np.asarray([0.1, -0.2, 0.3]),
    }


def test_h5_and_live_projection_match_at_native_and_resized_resolution():
    k = calibration()
    points = torch.tensor(
        [[0.2, -0.1, 2.0], [-0.4, 0.3, 3.0], [0.1, 0.2, 4.5]],
        dtype=torch.float64,
    )
    camera = live_camera(k, width=80, height=60)
    records = compare_camera(points, camera, k, tolerance=1e-8, matrix_tolerance=1e-8)
    assert [record["resolution_kind"] for record in records] == ["original", "training"]
    assert all(record["passed"] for record in records)
    assert all(record["max_pixel_difference"] <= 1e-8 for record in records)
    assert all(record["rotation_max_abs_difference"] <= 1e-8 for record in records)


def test_projection_path_uses_calibrated_live_intrinsics():
    k = calibration()
    points = torch.tensor([[0.2, -0.1, 2.0]], dtype=torch.float64)
    camera = live_camera(k, width=80, height=60)
    expected, _ = direct_projection(points, k, 80, 60)
    actual, _ = project_points_pixel(points, camera, k)
    assert torch.allclose(actual, expected, atol=1e-8, rtol=0.0)


def test_matrix_mismatch_fails_equivalence_record():
    k = calibration()
    camera = live_camera(k, width=160, height=120)
    camera.world_view_transform[0, 3] += 0.01
    points = torch.tensor([[0.2, -0.1, 2.0]], dtype=torch.float64)
    records = compare_camera(points, camera, k, tolerance=1e-8, matrix_tolerance=1e-8)
    assert not records[0]["passed"]
    assert records[0]["homogeneous_row_max_abs_difference"] > 1e-8


def test_projection_cli_accepts_run_stage_dataset_selectors(tmp_path, monkeypatch):
    track_path = tmp_path / "tracks.h5"
    output_path = tmp_path / "projection.json"
    captured = {}

    def fake_run(args):
        captured["data_type"] = args.data_type
        captured["llff_holdout"] = args.llff_holdout
        captured["use_color"] = args.use_color
        captured["train_bg"] = args.train_bg
        return {"passed": True}

    monkeypatch.setattr(projection_gate, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check_camera_projection_equivalence.py",
            "--track-h5",
            str(track_path),
            "--track_path",
            str(track_path),
            "--output",
            str(output_path),
            "--data_type",
            "colmap",
            "--llff_holdout",
            "8",
            "--strict_source_only_geometry",
        ],
    )

    projection_gate.main()

    assert captured == {
        "data_type": "colmap",
        "llff_holdout": 8,
        "use_color": True,
        "train_bg": False,
    }
    assert json.loads(output_path.read_text(encoding="utf-8"))["passed"] is True
