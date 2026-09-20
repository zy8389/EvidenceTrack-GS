#!/usr/bin/env python3
"""Pre-A0 server gate for live Scene pseudo-camera provenance and calibration."""
from __future__ import annotations

from argparse import ArgumentParser
import hashlib
import json
from pathlib import Path
import subprocess

from arguments import ModelParams, PipelineParams
from diffusion_guidance.calibration_guard import calibrate_scene
from scene import GaussianModel, Scene
from evidence_track.evaluation.audit_pseudo_camera_manifest import audit_live_scene
from utils.general_utils import safe_state


PINNED_UPSTREAM_COMMIT = "81ada6a32c918591ae7c7a0279dc6ca7a8018e2f"


def main() -> None:
    parser = ArgumentParser()
    ModelParams(parser)
    PipelineParams(parser)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data_type", default="colmap", choices=["colmap", "360"])
    parser.add_argument("--llff_holdout", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--use_gt_dca", action="store_true")
    parser.add_argument("--train_bg", action="store_true")
    # This gate always creates a fresh scratch model directory, so there is no
    # prior cfg_args file to merge. Parse the explicit protocol arguments only.
    args = parser.parse_args()
    if not args.strict_source_only_geometry:
        parser.error("Live pseudo-camera gate requires --strict_source_only_geometry")
    if not args.track_path:
        parser.error("Live pseudo-camera gate requires --track_path")
    if not str(getattr(args, "model_path", "")).strip():
        parser.error("Live pseudo-camera gate requires -m/--model_path")
    if not hasattr(args, "gt_dca_config"):
        args.gt_dca_config = None

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    repository_root = Path(__file__).resolve().parents[2]
    marker_path = repository_root / ".research_revision_v2.json"
    track_path = Path(args.track_path).expanduser().resolve()
    current_commit = None
    upstream_commit = None
    track_sha256 = None
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        upstream_commit = marker.get("upstream_commit")
        if upstream_commit != PINNED_UPSTREAM_COMMIT:
            raise RuntimeError(
                "Live pseudo-camera gate has the wrong pinned upstream provenance: "
                f"current={upstream_commit}, expected={PINNED_UPSTREAM_COMMIT}"
            )
        current_commit = subprocess.check_output(
            ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
        track_sha256 = hashlib.sha256(track_path.read_bytes()).hexdigest()
        safe_state(args.quiet, args.seed)
        gaussians = GaussianModel(args)
        scene = Scene(args, gaussians, shuffle=False)
        calibrate_scene(scene, args.track_path)
        result = audit_live_scene(scene)
        result.update(
            {
                "upstream_commit_expected": PINNED_UPSTREAM_COMMIT,
                "upstream_commit_current": upstream_commit,
                "repository_commit": current_commit,
                "track_h5": str(track_path),
                "track_h5_sha256": track_sha256,
                "scene_model_path": str(Path(args.model_path).expanduser().resolve()),
                "seed": int(args.seed),
                "passed": bool(result["passed"]),
            }
        )
    except Exception as exc:
        result = {
            "gate": "P0_live_pseudo_camera_pose_intrinsics_provenance",
            "upstream_commit_expected": PINNED_UPSTREAM_COMMIT,
            "upstream_commit_current": upstream_commit,
            "repository_commit": current_commit,
            "track_h5": str(track_path),
            "track_h5_sha256": track_sha256,
            "scene_model_path": str(Path(args.model_path).expanduser().resolve()),
            "seed": int(args.seed),
            "passed": False,
            "failure": f"{type(exc).__name__}: {exc}",
        }
    output.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
