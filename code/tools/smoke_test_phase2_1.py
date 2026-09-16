#!/usr/bin/env python3
"""CPU smoke tests for every Phase 2.1 integrity patch."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
import random
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import h5py
import numpy as np
from PIL import Image
import torch

from build_anchor_tracks_from_colmap import (
    build_source_tracks,
    build_record,
    intrinsics_matrix,
    project,
    source_covariance,
    source_quality,
    source_scene_scale,
)
from diffusion_guidance.pseudo_manifest import load_pseudo_manifest, pseudo_view_index
from diffusion_guidance.camera_utils import camera_fingerprint_from_payload
from diffusion_guidance.checkpoint_state import (
    FORMAT as CHECKPOINT_FORMAT,
    RENDER_STATE_SCHEMA,
    load_checkpoint_summary,
)
from diffusion_guidance.control_identity import (
    PREREGISTERED_CONTROLLED_OPTIMIZATION,
    controlled_checkpoint_provenance_from_metadata,
    controlled_checkpoint_provenance_sha256,
    controlled_pair_id,
    controlled_training_protocol_sha256,
    source_camera_set_sha256,
    source_image_inventory_sha256,
    training_camera_inventory_sha256,
)
from diffusion_guidance.difix_provenance import (
    A0_PSEUDO_MANIFEST_SCHEMA,
    CACHE_IDENTITY_FIELDS,
    HELDOUT_DIAGNOSTIC_MANIFEST_SCHEMA,
    cache_run_fingerprint,
    manifest_record_sha256,
    reproducibility_check_sha256,
)
from diffusion_guidance.result_binding import RESULT_BINDING_SCHEMA, read_pair_audit
from diffusion_guidance.evidence_features import SpatialFeatureMap
from diffusion_guidance.evidence_matching import (
    choose_hard_shuffled_indices,
    error_metrics,
    local_feature_match,
    select_hard_negatives,
    valid_shift_recovery_mask,
)
from geometric_constraints.repaired_geometry import StrictGeometryManager
from geometric_constraints.strict_track_store import (
    ANCHOR_PROVENANCE,
    FORMAT_VERSION,
    POINT_CLOUD_PROVENANCE,
    STRICT_PROTOCOL,
    TRACK_MEMBERSHIP_PROVENANCE,
    StrictTrackStore,
)
from run_difix_cache import run_pipeline_once, verify_pair
from test_geometry_recovery import run_synthetic


def camera(center_x: float, camera_id: int) -> tuple[dict, dict]:
    camera_record = {
        "id": camera_id,
        "model": "PINHOLE",
        "width": 160,
        "height": 120,
        "params": np.asarray([100.0, 100.0, 80.0, 60.0]),
    }
    image_record = {
        "id": camera_id,
        "camera_id": camera_id,
        "name": f"image_{camera_id}.png",
        "rotation_w2c": np.eye(3),
        "tvec": np.asarray([-center_x, 0.0, 0.0]),
    }
    return camera_record, image_record


def observation_for(xyz, image_record, camera_record):
    temporary_cameras = {camera_record["id"]: camera_record}
    temporary_images = {image_record["id"]: image_record}
    observation = {
        "image_id": image_record["id"],
        "image_name": image_record["name"],
        "xy": np.zeros(2),
        "confidence": 1.0,
    }
    matrix = intrinsics_matrix(camera_record)
    point_camera = image_record["rotation_w2c"] @ xyz + image_record["tvec"]
    homogeneous = matrix @ point_camera
    observation["xy"] = homogeneous[:2] / homogeneous[2]
    assert np.allclose(
        project(xyz, observation, temporary_cameras, temporary_images)[0],
        observation["xy"],
    )
    return observation


def write_strict_h5(path: Path, track_xyz: np.ndarray) -> None:
    cameras, images = {}, {}
    centers = (-0.4, 0.4, 0.0)
    for camera_id, center in enumerate(centers, start=1):
        camera_record, image_record = camera(center, camera_id)
        cameras[camera_id] = camera_record
        images[camera_id] = image_record
    all_observations = []
    offsets = [0]
    for xyz in track_xyz:
        all_observations.extend(
            observation_for(xyz, images[index], cameras[index])
            for index in (1, 2, 3)
        )
        offsets.append(len(all_observations))
    source_mask = np.tile(np.asarray([True, True, False]), len(track_xyz))
    string_dtype = h5py.string_dtype("utf-8")
    with h5py.File(path, "w") as handle:
        handle.attrs["format_version"] = FORMAT_VERSION
        handle.attrs["geometry_protocol"] = STRICT_PROTOCOL
        handle.attrs["anchor_provenance"] = ANCHOR_PROVENANCE
        handle.attrs["track_membership_provenance"] = (
            TRACK_MEMBERSHIP_PROVENANCE
        )
        handle.attrs["point_cloud_provenance"] = POINT_CLOUD_PROVENANCE
        handle.attrs["source_scene_scale"] = 0.8
        handle.attrs["source_images_json"] = json.dumps(["image_1.png", "image_2.png"])
        handle.attrs["heldout_images_json"] = json.dumps(["image_3.png"])
        tracks = handle.create_group("tracks")
        tracks.create_dataset("id", data=np.arange(len(track_xyz), dtype=np.int64))
        tracks.create_dataset("xyz", data=track_xyz.astype(np.float32))
        tracks.create_dataset("rgb_source", data=np.full((len(track_xyz), 3), 0.5, dtype=np.float32))
        tracks.create_dataset("covariance", data=np.tile(np.eye(3, dtype=np.float32)[None] * 1e-4, (len(track_xyz), 1, 1)))
        tracks.create_dataset("quality", data=np.ones(len(track_xyz), dtype=np.float32))
        tracks.create_dataset("source_reprojection_error", data=np.zeros(len(track_xyz), dtype=np.float32))
        observations = handle.create_group("observations")
        observations.create_dataset("offsets", data=np.asarray(offsets, dtype=np.int64))
        observations.create_dataset("xy", data=np.stack([item["xy"] for item in all_observations]).astype(np.float32))
        observations.create_dataset("image_id", data=np.asarray([item["image_id"] for item in all_observations], dtype=np.int32))
        observations.create_dataset("image_name", data=np.asarray([item["image_name"] for item in all_observations], dtype=object), dtype=string_dtype)
        observations.create_dataset("confidence", data=np.ones(len(all_observations), dtype=np.float32))
        observations.create_dataset("use_for_anchor", data=source_mask)
        observations.create_dataset("use_for_quality", data=source_mask)
        observations.create_dataset("use_for_source_features", data=source_mask)
        cameras_group = handle.create_group("cameras")
        records = [(images[index], cameras[index]) for index in (1, 2, 3)]
        cameras_group.create_dataset("image_id", data=np.asarray([item[0]["id"] for item in records], dtype=np.int32))
        cameras_group.create_dataset("image_name", data=np.asarray([item[0]["name"] for item in records], dtype=object), dtype=string_dtype)
        cameras_group.create_dataset("model", data=np.asarray([item[1]["model"] for item in records], dtype=object), dtype=string_dtype)
        cameras_group.create_dataset("width", data=np.asarray([item[1]["width"] for item in records], dtype=np.int32))
        cameras_group.create_dataset("height", data=np.asarray([item[1]["height"] for item in records], dtype=np.int32))
        cameras_group.create_dataset("fx", data=np.asarray([100.0] * 3))
        cameras_group.create_dataset("fy", data=np.asarray([100.0] * 3))
        cameras_group.create_dataset("cx", data=np.asarray([80.0] * 3))
        cameras_group.create_dataset("cy", data=np.asarray([60.0] * 3))
        cameras_group.create_dataset("rotation_w2c", data=np.stack([item[0]["rotation_w2c"] for item in records]))
        cameras_group.create_dataset("translation_w2c", data=np.stack([item[0]["tvec"] for item in records]))


def fake_scene_cameras():
    values = []
    for camera_id, center in ((1, -0.4), (2, 0.4)):
        _, image = camera(center, camera_id)
        transform = np.eye(4, dtype=np.float32)
        transform[:3, :3] = image["rotation_w2c"]
        transform[:3, 3] = image["tvec"]
        values.append(
            SimpleNamespace(
                image_name=f"image_{camera_id}",
                image_width=160,
                image_height=120,
                world_view_transform=torch.from_numpy(transform).transpose(0, 1),
            )
        )
    return values


def test_heldout_invariance(root: Path) -> None:
    cameras, images = {}, {}
    for camera_id, center in enumerate((-0.4, 0.4, 0.0), start=1):
        cameras[camera_id], images[camera_id] = camera(center, camera_id)
        Image.new("RGB", (160, 120), color=(64 * camera_id, 20, 10)).save(
            root / images[camera_id]["name"]
        )
    xyz = np.asarray([0.1, -0.05, 2.5])
    observations = [observation_for(xyz, images[index], cameras[index]) for index in (1, 2, 3)]
    point_track = {"id": 7}
    scale = source_scene_scale(images, {"image_1", "image_2"})
    first = build_record(
        point_track,
        observations,
        {"image_1", "image_2"},
        {"image_3"},
        cameras,
        images,
        root,
        1.0,
        4.0,
        scale,
    )
    changed = [dict(item) for item in observations]
    changed[2]["xy"] = changed[2]["xy"] + np.asarray([900.0, -700.0])
    second = build_record(
        point_track,
        changed,
        {"image_1", "image_2"},
        {"image_3"},
        cameras,
        images,
        root,
        1.0,
        4.0,
        scale,
    )
    deleted = build_record(
        point_track,
        observations[:2],
        {"image_1", "image_2"},
        {"image_3"},
        cameras,
        images,
        root,
        1.0,
        4.0,
        scale,
    )
    assert np.array_equal(first["xyz"], second["xyz"])
    assert np.array_equal(first["xyz"], deleted["xyz"])
    assert np.array_equal(first["covariance"], second["covariance"])
    assert first["quality"] == second["quality"]


def test_source_rgb_membership_only() -> None:
    feature_sets = {
        image_id: {
            "image_id": image_id,
            "image_name": f"source_{image_id}.png",
            "xy": np.asarray([[10.0 + image_id, 20.0], [80.0, 40.0]]),
            "descriptors": np.zeros((2, 128), dtype=np.float32),
        }
        for image_id in (1, 2, 3)
    }
    source_edges = [
        {
            "left_image_id": 1,
            "left_feature_index": 0,
            "right_image_id": 2,
            "right_feature_index": 0,
            "descriptor_distance": 1.0,
            "epipolar_distance": 0.1,
        },
        {
            "left_image_id": 2,
            "left_feature_index": 0,
            "right_image_id": 3,
            "right_feature_index": 0,
            "descriptor_distance": 1.1,
            "epipolar_distance": 0.1,
        },
    ]
    tracks = build_source_tracks(feature_sets, source_edges)
    assert len(tracks) == 1
    assert tracks[0]["source_nodes"] == [(1, 0), (2, 0), (3, 0)]
    snapshot = tuple(tracks[0]["source_nodes"])
    # Held-out observations and arbitrary legacy point ids are not inputs to
    # source membership construction and therefore cannot change the identity.
    legacy_or_heldout_noise = {"point3d_id": 999, "xy": [900.0, -700.0]}
    assert legacy_or_heldout_noise
    rebuilt = build_source_tracks(feature_sets, source_edges)
    assert tuple(rebuilt[0]["source_nodes"]) == snapshot


def test_quality_scale_invariance() -> None:
    cameras, images = {}, {}
    for camera_id, center in enumerate((-0.4, 0.0, 0.4), start=1):
        cameras[camera_id], images[camera_id] = camera(center, camera_id)
    source_keys = {"image_1", "image_2", "image_3"}
    xyz = np.asarray([0.1, -0.05, 2.5])
    observations = [
        observation_for(xyz, images[index], cameras[index])
        for index in (1, 2, 3)
    ]
    scale = source_scene_scale(images, source_keys)
    covariance = source_covariance(xyz, observations, cameras, images, 1.0)
    quality = source_quality(0.5, 3, covariance, scale)

    scaled_images = {
        image_id: {
            **image,
            "tvec": np.asarray(image["tvec"]) * 10.0,
        }
        for image_id, image in images.items()
    }
    scaled_xyz = xyz * 10.0
    scaled_observations = [
        observation_for(scaled_xyz, scaled_images[index], cameras[index])
        for index in (1, 2, 3)
    ]
    scaled_scene = source_scene_scale(scaled_images, source_keys)
    scaled_covariance = source_covariance(
        scaled_xyz,
        scaled_observations,
        cameras,
        scaled_images,
        1.0,
    )
    scaled_quality = source_quality(
        0.5, 3, scaled_covariance, scaled_scene
    )
    assert np.allclose(scaled_covariance, covariance * 100.0, rtol=1e-5)
    assert np.isclose(scaled_scene, scale * 10.0, rtol=1e-12)
    assert np.isclose(scaled_quality, quality, rtol=1e-8, atol=1e-10)


def test_store_association_and_gradient(root: Path) -> None:
    track_xyz = np.asarray(
        [[0.0, 0.0, 2.5], [0.001, 0.0, 2.5], [0.5, 0.0, 2.5]],
        dtype=np.float32,
    )
    path = root / "strict.h5"
    write_strict_h5(path, track_xyz)
    store = StrictTrackStore.load(path)
    audit = store.leakage_audit()
    assert audit.passed
    gaussians = torch.nn.Parameter(
        torch.tensor([[0.0, 0.0, 2.5], [0.5, 0.0, 2.5]], dtype=torch.float32)
    )
    manager = StrictGeometryManager(
        store,
        fake_scene_cameras(),
        device="cpu",
        min_quality=0.0,
        max_association_distance_ratio=1.0,
    )
    statistics = manager.associate(gaussians, scene_radius=1.0)
    assert statistics.associated_tracks == 3
    assert statistics.unique_associated_gaussians == 2
    assert statistics.association_collision_rate > 0.0
    probe = manager.gradient_probe(gaussians, count=3, offset=1e-3)
    assert probe["coverage"] == 1.0
    assert probe["mean_gradient_norm"] > 0.0


def test_evidence_metrics_and_window() -> None:
    predicted = np.asarray([[0.0, 0.0], [6.0, 0.0], [10.0, 0.0]])
    ground_truth = np.zeros_like(predicted)
    metrics = error_metrics(predicted, ground_truth, feature_stride=5.0)
    assert metrics["pck3"] == 1.0 / 3.0
    assert metrics["pck8"] == 2.0 / 3.0
    assert np.isclose(metrics["normalized_error_feature_stride"], predicted[:, 0].mean() / 5.0)
    shifted = np.asarray([[8.0, 0.0], [16.0, 0.0]])
    gt = np.zeros_like(shifted)
    assert valid_shift_recovery_mask(shifted, gt, 16.0).all()
    assert not valid_shift_recovery_mask(shifted, gt, 7.0).any()

    features = torch.zeros((2, 5, 5), dtype=torch.float32)
    features[0] = 1.0
    fmap = SpatialFeatureMap(
        features=torch.nn.functional.normalize(features, dim=0),
        input_resolution=(33, 33),
        model_input_resolution=(33, 33),
        feature_resolution=(5, 5),
        feature_stride=(8.25, 8.25),
        interpolation_method="test",
        backend="test",
    )
    match, _ = local_feature_match(
        torch.tensor([[1.0, 0.0]]),
        fmap,
        np.asarray([[16.0, 16.0]]),
        window_radius=16.0,
        temperature=0.1,
    )
    assert np.isfinite(match).all()


def test_hard_negatives() -> None:
    queries = torch.tensor(
        [[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]], dtype=torch.float32
    )
    centers = np.asarray([[0.0, 0.0], [5.0, 0.0], [100.0, 0.0]])
    negatives = choose_hard_shuffled_indices(queries, centers, nearby_radius=10.0)
    assert negatives[0] == 1 and negatives[1] == 0
    selection = select_hard_negatives(queries, centers, nearby_radius=10.0)
    assert selection.selection_mode[2] == "global_fallback"
    assert not selection.within_radius[2]


class FakeDifix:
    def __call__(self, prompt, image, ref_image, num_inference_steps, timesteps, guidance_scale, generator, height=None, width=None):
        values = torch.rand((8, 8, 3), generator=generator)
        array = (values.numpy() * 255.0).astype(np.uint8)
        return SimpleNamespace(images=[Image.fromarray(array)])


def test_difix_reproducibility() -> None:
    image = Image.new("RGB", (8, 8), color=(1, 2, 3))
    kwargs = dict(
        pipe=FakeDifix(),
        prompt="remove degradation",
        input_image=image,
        reference_image=image,
        timestep=199,
        guidance_scale=0.0,
        seed=17,
        device="cpu",
    )
    first = run_pipeline_once(**kwargs)
    second = run_pipeline_once(**kwargs)
    result = verify_pair(first, second)
    assert result["byte_equal"] and result["mode"] == "byte_exact"

    reference = np.zeros((20, 20, 3), dtype=np.uint8)
    nearly_equal = reference.copy()
    nearly_equal[0, 0, 0] = 1
    tolerant = verify_pair(Image.fromarray(reference), Image.fromarray(nearly_equal))
    assert tolerant["passed"]
    assert tolerant["mode"] == "metric_tolerance"
    assert tolerant["max_abs_uint8"] == 1
    assert tolerant["mean_abs_uint8"] <= 0.01


def test_pseudo_pool_coverage() -> None:
    call_iterations = [
        iteration
        for iteration in range(10001, 12001)
        if iteration > 10000
        and iteration < 12000
        and (iteration - 10000) % 50 == 0
    ]
    indices = [
        pseudo_view_index(call_count, 32)
        for call_count, _ in enumerate(call_iterations)
    ]
    assert len(call_iterations) == 39
    assert len(set(indices)) == 32
    assert set(indices) == set(range(32))


def test_controlled_real_schedule() -> None:
    cameras = list(range(3))
    def schedule(seed, pseudo_calls):
        real_rng = random.Random(seed)
        pseudo_rng = random.Random(seed + 1000003)
        output, stack = [], []
        for iteration in range(2000):
            if not stack:
                stack = cameras.copy()
                real_rng.shuffle(stack)
            output.append(stack.pop())
            for _ in range(pseudo_calls(iteration)):
                pseudo_rng.random()
        return output
    a1 = schedule(1, lambda _: 0)
    b = schedule(1, lambda iteration: int(iteration % 50 == 0))
    assert a1 == b


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path.resolve()


def _controlled_protocol(role: str, scene: Path, track: Path, start: Path | None) -> dict:
    iterations = 10000 if role == "A0" else 12000
    common = {
        **PREREGISTERED_CONTROLLED_OPTIMIZATION,
        "iterations": iterations,
        "experiment_seed": 1,
        "position_lr_init": 0.00016,
    }
    return {
        "schema": "controlled_training_protocol_v1",
        "model": {
            "source_path": str(scene),
            "track_path": str(track),
            "images": "images",
            "strict_tracks": True,
            "strict_source_only_geometry": True,
        },
        "optimization": dict(common),
        "pipeline": {"debug": False},
        "runtime": {
            **common,
            "source_path": str(scene),
            "track_path": str(track),
            "start_checkpoint": None if start is None else str(start),
            "test_iterations": [iterations],
            "save_iterations": [iterations],
            "checkpoint_iterations": [iterations],
            "quiet": False,
        },
    }


def _checkpoint_state(provenance: dict) -> dict:
    count = 3
    core = {
        "active_sh_degree": 0,
        "max_sh_degree": 0,
        "_xyz": torch.tensor(
            [[0.0, 0.0, 2.5], [0.2, 0.0, 2.5], [-0.2, 0.1, 2.5]],
            dtype=torch.float32,
        ),
        "_features_dc": torch.zeros((count, 1, 3), dtype=torch.float32),
        "_features_rest": torch.zeros((count, 0, 3), dtype=torch.float32),
        "_scaling": torch.zeros((count, 3), dtype=torch.float32),
        "_rotation": torch.zeros((count, 4), dtype=torch.float32),
        "_opacity": torch.zeros((count, 1), dtype=torch.float32),
        "max_radii2D": torch.zeros(count, dtype=torch.float32),
        "xyz_gradient_accum": torch.zeros((count, 1), dtype=torch.float32),
        "denom": torch.ones((count, 1), dtype=torch.float32),
    }
    groups = {
        "xyz": core["_xyz"],
        "f_dc": core["_features_dc"],
        "f_rest": core["_features_rest"],
        "opacity": core["_opacity"],
        "scaling": core["_scaling"],
        "rotation": core["_rotation"],
    }
    optimizer_state, optimizer_groups = {}, []
    for identifier, (name, tensor) in enumerate(groups.items()):
        optimizer_state[identifier] = {
            "step": 1,
            "exp_avg": torch.zeros_like(tensor),
            "exp_avg_sq": torch.zeros_like(tensor),
        }
        optimizer_groups.append({"name": name, "params": [identifier]})
    return {
        "format": CHECKPOINT_FORMAT,
        "core": core,
        "spatial_lr_scale": 1.0,
        "optimizer": {"state": optimizer_state, "param_groups": optimizer_groups},
        "confidence": torch.ones((count, 1), dtype=torch.float32),
        "init_point": core["_xyz"].clone(),
        "bg_color": torch.zeros(3, dtype=torch.float32),
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": [torch.get_rng_state().clone()],
        "controlled_provenance": provenance,
    }


def _write_checkpoint(path: Path, iteration: int, provenance: dict) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save((_checkpoint_state(provenance), iteration), path)
    return load_checkpoint_summary(
        path,
        expected_iteration=iteration,
        require_cuda_rng=True,
        require_controlled_provenance=True,
    )


def _cache_protocol() -> dict:
    return {
        "model_id": "fixture/difix",
        "model_revision": "a" * 40,
        "difix_code_commit": "b" * 40,
        "difix_code_clean": True,
        "coordinate_policy": "full_frame_to_multiple8_then_bicubic_to_original",
        "dtype": "fp32",
        "timesteps": [199],
        "guidance_scale": 0.0,
        "prompt": "remove degradation",
    }


def _write_strict_difix_cache(
    manifest: Path,
    record: dict,
    *,
    input_path: Path,
    reference_path: Path,
    target_path: Path,
) -> dict:
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
    reproducibility = {
        "passed": True,
        "camera_fingerprint": record["key"],
        "input_sha256": _digest(input_path),
        "reference_sha256": _digest(reference_path),
        "output_sha256": _digest(target_path),
        "repeated_output_sha256": _digest(target_path),
        "mode": "byte_exact",
    }
    metadata = {
        "manifest_schema": record["manifest_schema"],
        "manifest": str(manifest.resolve()),
        "manifest_sha256": _digest(manifest),
        "record_count": 1,
        "seed": 1,
        **_cache_protocol(),
        "reproducibility_check": reproducibility,
        "reproducibility_check_sha256": reproducibility_check_sha256(reproducibility),
    }
    metadata["cache_run_fingerprint"] = cache_run_fingerprint(metadata)
    _write_json(manifest.with_suffix(manifest.suffix + ".difix_metadata.json"), metadata)
    sidecar = {
        **{field: metadata[field] for field in CACHE_IDENTITY_FIELDS},
        "cache_run_fingerprint": metadata["cache_run_fingerprint"],
        "input_image": str(input_path.resolve()),
        "reference_image": str(reference_path.resolve()),
        "target_image": str(target_path.resolve()),
        "input_sha256": _digest(input_path),
        "reference_sha256": _digest(reference_path),
        "output_sha256": _digest(target_path),
        "camera_fingerprint": record["key"],
        "resolution": [160, 120],
        "source_manifest_record": record,
        "source_manifest_record_sha256": manifest_record_sha256(record),
    }
    _write_json(target_path.with_suffix(target_path.suffix + ".metadata.json"), sidecar)
    return metadata


def _final_checkpoint_record(path: Path, summary: dict, provenance: dict) -> dict:
    return {
        "path": str(path.resolve()),
        "sha256": _digest(path),
        "iteration": summary["iteration"],
        "state_format": summary["format"],
        "gaussian_count": summary["gaussian_count"],
        "render_state_schema": summary["render_state_schema"],
        "render_state_sha256": summary["render_state_sha256"],
        "controlled_provenance": provenance,
        "controlled_provenance_sha256": summary["controlled_provenance_sha256"],
    }


def test_evidence_end_to_end(root: Path) -> None:
    """Exercise v2 Difix, v4 paired identity and v2 checkpoint binding together."""
    scene = root / "fern"
    images_dir = scene / "images"
    images_dir.mkdir(parents=True)
    h5_path = scene / "tracks.h5"
    track_xyz = np.asarray(
        [[0.0, 0.0, 2.5], [0.2, 0.0, 2.5], [-0.2, 0.1, 2.5]],
        dtype=np.float32,
    )
    write_strict_h5(h5_path, track_xyz)

    image_y, image_x = np.mgrid[0:120, 0:160]
    array = np.stack(
        [image_x / 159.0, image_y / 119.0, (image_x + image_y) / 278.0],
        axis=-1,
    )
    array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    for name in ("image_1.png", "image_2.png", "image_3.png"):
        Image.fromarray(array).save(images_dir / name)

    source_names = ["image_1", "image_2"]
    source_inventory = [
        {
            "camera_name": name,
            "path": str((images_dir / f"{name}.png").resolve()),
            "sha256": _digest(images_dir / f"{name}.png"),
        }
        for name in source_names
    ]
    source_inventory_sha256 = source_image_inventory_sha256(source_inventory)
    source_camera_sha256 = source_camera_set_sha256(source_names)
    source_cameras = []
    for name, translation in (("image_1", [0.4, 0.0, 0.0]), ("image_2", [-0.4, 0.0, 0.0])):
        source_cameras.append(
            {
                "camera_name": name,
                "camera": {
                    "R": np.eye(3).tolist(),
                    "T": translation,
                    "FoVx": 1.0,
                    "FoVy": 1.0,
                    "width": 160,
                    "height": 120,
                    "intrinsics": {
                        "fx": 100.0,
                        "fy": 100.0,
                        "cx": 80.0,
                        "cy": 60.0,
                        "width": 160,
                        "height": 120,
                    },
                },
            }
        )
    camera_inventory_sha256 = training_camera_inventory_sha256(source_cameras)
    strict_geometry = {
        "strict_tracks": True,
        "strict_source_only_geometry": True,
        "strict_track_weight": 0.1,
        "densify_until_iter": 10000,
        "mixed_precision": False,
        "disable_legacy_pseudo_depth": True,
        "disable_depth_loss": True,
        "geometry_reg_enabled": False,
        "use_gt_dca": False,
    }

    def metadata_for(role: str, start: Path | None, a0_digest: str | None, pseudo: Path | None = None) -> dict:
        protocol = _controlled_protocol(role, scene.resolve(), h5_path.resolve(), start)
        return {
            "role": role,
            "seed": 1,
            "checkpoint_iteration": 10000,
            "final_iteration": 10000 if role == "A0" else 12000,
            "start_checkpoint": None if start is None else str(start.resolve()),
            "start_checkpoint_sha256": None if start is None else _digest(start),
            "start_checkpoint_controlled_provenance_sha256": a0_digest,
            "scene_source_path": str(scene.resolve()),
            "track_h5_path": str(h5_path.resolve()),
            "track_h5_sha256": _digest(h5_path),
            "source_camera_names": source_names,
            "source_camera_set_sha256": source_camera_sha256,
            "source_images_dir": str(images_dir.resolve()),
            "source_image_inventory": source_inventory,
            "source_image_inventory_sha256": source_inventory_sha256,
            "source_training_camera_inventory": source_cameras,
            "source_training_camera_inventory_sha256": camera_inventory_sha256,
            "controlled_training_protocol": protocol,
            "controlled_training_protocol_sha256": controlled_training_protocol_sha256(protocol),
            "strict_geometry_protocol": strict_geometry,
            "pseudo_manifest_path": None if pseudo is None else str(pseudo.resolve()),
            "pseudo_manifest_sha256": None if pseudo is None else _digest(pseudo),
            "pseudo_camera_pool_sha256": None if pseudo is None else "e" * 64,
            "pseudo_target_kind": None if pseudo is None else "difix",
            "pseudo_rgb_strict_cache": pseudo is not None,
        }

    a0_metadata = metadata_for("A0", None, None)
    a0_provenance = controlled_checkpoint_provenance_from_metadata(a0_metadata)
    a0_path = root / "A0" / "chkpnt10000.pth"
    a0_summary = _write_checkpoint(a0_path, 10000, a0_provenance)
    a0_digest = a0_summary["controlled_provenance_sha256"]

    heldout_camera = {
        "R": np.eye(3).tolist(),
        "T": [0.0, 0.0, 0.0],
        "FoVx": 1.0,
        "FoVy": 1.0,
        "width": 160,
        "height": 120,
        "intrinsics": {
            "fx": 100.0,
            "fy": 100.0,
            "cx": 80.0,
            "cy": 60.0,
            "width": 160,
            "height": 120,
        },
    }
    pseudo_key = camera_fingerprint_from_payload(heldout_camera)
    pseudo_input = root / "A0" / "pseudo_input.png"
    pseudo_target = root / "B" / "pseudo_target.png"
    pseudo_target.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(pseudo_input)
    Image.fromarray(array).save(pseudo_target)
    live_pseudo_audit = _write_json(
        root / "A0" / "live_pseudo_audit.json",
        {
            "passed": True,
            "projection_context_fingerprint": "f" * 64,
            "track_h5_sha256": _digest(h5_path),
            "records": [{"camera_fingerprint": pseudo_key}],
        },
    )
    pseudo_manifest = root / "B" / "pseudo_supervision.jsonl"
    pseudo_record = {
        "manifest_schema": A0_PSEUDO_MANIFEST_SCHEMA,
        "key": pseudo_key,
        "camera": heldout_camera,
        "input": str(pseudo_input.resolve()),
        "reference_image": str((images_dir / "image_1.png").resolve()),
        "target": str(pseudo_target.resolve()),
        "supervision_target_kind": "difix",
        "input_sha256": _digest(pseudo_input),
        "reference_image_sha256": _digest(images_dir / "image_1.png"),
        "a0_checkpoint": str(a0_path.resolve()),
        "a0_checkpoint_sha256": _digest(a0_path),
        "a0_checkpoint_iteration": 10000,
        "a0_checkpoint_state_format": a0_summary["format"],
        "a0_checkpoint_gaussian_count": a0_summary["gaussian_count"],
        "a0_checkpoint_render_state_schema": a0_summary["render_state_schema"],
        "a0_checkpoint_render_state_sha256": a0_summary["render_state_sha256"],
        "a0_checkpoint_controlled_provenance_sha256": a0_digest,
        "a0_render_state_source": "complete_checkpoint_restore_after_scene_ply_load",
        "live_pseudo_audit": str(live_pseudo_audit),
        "live_pseudo_audit_sha256": _digest(live_pseudo_audit),
        "projection_context_fingerprint": "f" * 64,
        "track_h5": str(h5_path.resolve()),
        "track_h5_sha256": _digest(h5_path),
        "source_camera_names": source_names,
        "source_camera_set_sha256": source_camera_sha256,
    }
    _write_strict_difix_cache(
        pseudo_manifest,
        pseudo_record,
        input_path=pseudo_input,
        reference_path=images_dir / "image_1.png",
        target_path=pseudo_target,
    )
    assert len(load_pseudo_manifest(pseudo_manifest, strict_targets=True)) == 1

    a1_metadata = metadata_for("A1", a0_path, a0_digest)
    b_metadata = metadata_for("B", a0_path, a0_digest, pseudo_manifest)
    pair_id = controlled_pair_id(a1_metadata)
    a1_metadata["controlled_pair_id"] = pair_id
    b_metadata["controlled_pair_id"] = pair_id
    a1_provenance = controlled_checkpoint_provenance_from_metadata(a1_metadata)
    b_provenance = controlled_checkpoint_provenance_from_metadata(b_metadata)
    a1_path = root / "A1" / "chkpnt12000.pth"
    b_path = root / "B" / "chkpnt12000.pth"
    a1_summary = _write_checkpoint(a1_path, 12000, a1_provenance)
    b_summary = _write_checkpoint(b_path, 12000, b_provenance)

    audit = {
        "audit_schema": RESULT_BINDING_SCHEMA,
        "passed": True,
        "failures": [],
        "pair_id": pair_id,
        "controlled_pair_id": pair_id,
        "dataset": "LLFF",
        "scene": "fern",
        "role": "development",
        "seed": 1,
        "scene_source_path": str(scene.resolve()),
        "start_checkpoint_path": str(a0_path.resolve()),
        "start_checkpoint_sha256": _digest(a0_path),
        "checkpoint_iteration": 10000,
        "final_iteration": 12000,
        "track_h5_path": str(h5_path.resolve()),
        "track_h5_sha256": _digest(h5_path),
        "source_camera_names": source_names,
        "source_camera_set_sha256": source_camera_sha256,
        "source_images_dir": str(images_dir.resolve()),
        "source_image_inventory": source_inventory,
        "source_image_inventory_sha256": source_inventory_sha256,
        "source_training_camera_inventory": source_cameras,
        "source_training_camera_inventory_sha256": camera_inventory_sha256,
        "controlled_training_protocol": a1_metadata["controlled_training_protocol"],
        "controlled_training_protocol_sha256": a1_metadata["controlled_training_protocol_sha256"],
        "start_checkpoint_controlled_provenance": a0_provenance,
        "start_checkpoint_controlled_provenance_sha256": a0_digest,
        "strict_geometry_protocol": strict_geometry,
        "audited_methods": ["A1", "B"],
        "audited_contrasts": [["A1", "B"]],
        "pseudo_validated_methods": ["B"],
        "final_checkpoints": {
            "A1": _final_checkpoint_record(a1_path, a1_summary, a1_provenance),
            "B": _final_checkpoint_record(b_path, b_summary, b_provenance),
        },
    }
    audited_paths = [a0_path, h5_path, images_dir / "image_1.png", images_dir / "image_2.png", a1_path, b_path]
    audit["audited_input_files"] = [
        {"path": str(path.resolve()), "sha256": _digest(path)}
        for path in sorted(audited_paths, key=lambda item: str(item.resolve()))
    ]
    audit["audited_input_file_count"] = len(audit["audited_input_files"])
    audit["audited_input_fingerprint"] = hashlib.sha256(
        json.dumps(
            audit["audited_input_files"],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()
    audit_path = _write_json(root / "pair_audit.json", audit)
    _, checked_audit = read_pair_audit(audit_path)
    assert checked_audit["final_checkpoints"]["B"]["controlled_provenance"] == b_provenance

    common_context = {
        "dataset": "LLFF",
        "scene": "fern",
        "seed": 1,
        "experiment_role": "development",
        "scene_source_path": str(scene.resolve()),
        "controlled_pair_id": pair_id,
        "pair_audit": str(audit_path),
        "pair_audit_sha256": _digest(audit_path),
        "track_h5": str(h5_path.resolve()),
        "track_h5_sha256": _digest(h5_path),
        "source_camera_names": source_names,
        "source_camera_set_sha256": source_camera_sha256,
    }
    reference_path = images_dir / "image_1.png"
    real_target = images_dir / "image_3.png"
    arm_contexts = {}
    for arm, checkpoint, summary in (("A1", a1_path, a1_summary), ("B", b_path, b_summary)):
        arm_dir = root / arm
        render = arm_dir / "heldout_gs.png"
        target = arm_dir / "heldout_difix.png"
        Image.fromarray(array).save(render)
        Image.fromarray(array).save(target)
        key = camera_fingerprint_from_payload(heldout_camera, prefix="heldout")
        checkpoint_fields = {
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": _digest(checkpoint),
            "checkpoint_iteration": 12000,
            "checkpoint_state_format": summary["format"],
            "checkpoint_gaussian_count": summary["gaussian_count"],
            "checkpoint_render_state_schema": summary["render_state_schema"],
            "checkpoint_render_state_sha256": summary["render_state_sha256"],
            "checkpoint_controlled_provenance_sha256": summary["controlled_provenance_sha256"],
            "scene_ply_gaussian_count": summary["gaussian_count"],
            "scene_ply_checkpoint_count_match": True,
            "scene_ply_render_state_schema": summary["render_state_schema"],
            "scene_ply_render_state_sha256": summary["render_state_sha256"],
            "scene_ply_checkpoint_render_state_match": True,
        }
        cache_record = {
            "manifest_schema": HELDOUT_DIAGNOSTIC_MANIFEST_SCHEMA,
            "key": key,
            "image_name": "image_3.png",
            "camera": heldout_camera,
            "camera_source": "heldout_evaluation_camera",
            "input": str(render.resolve()),
            "reference_image": str(reference_path.resolve()),
            "target": str(target.resolve()),
            "input_sha256": _digest(render),
            "reference_image_sha256": _digest(reference_path),
            **common_context,
            "paired_identity_arm": arm,
            **checkpoint_fields,
        }
        difix_manifest = arm_dir / "difix_manifest.jsonl"
        cache_metadata = _write_strict_difix_cache(
            difix_manifest,
            cache_record,
            input_path=render,
            reference_path=reference_path,
            target_path=target,
        )
        evidence_record = {
            "manifest_schema": HELDOUT_DIAGNOSTIC_MANIFEST_SCHEMA,
            "image_name": "image_3.png",
            "camera_fingerprint": key,
            "camera": heldout_camera,
            "camera_source": "heldout_evaluation_camera",
            "gs_render": str(render.resolve()),
            "difix_output": str(target.resolve()),
            "real_target": str(real_target.resolve()),
            "reference_image": str(reference_path.resolve()),
            "gs_render_sha256": _digest(render),
            "reference_image_sha256": _digest(reference_path),
            "real_target_sha256": _digest(real_target),
            **common_context,
            "paired_identity_arm": arm,
            **checkpoint_fields,
        }
        evidence_manifest = arm_dir / "evidence_manifest.jsonl"
        evidence_manifest.write_text(
            json.dumps(evidence_record, sort_keys=True) + "\n", encoding="utf-8"
        )
        arm_contexts[arm.lower()] = {
            "evidence_manifest": evidence_manifest.resolve(),
            "difix_manifest": difix_manifest.resolve(),
            "difix_metadata": difix_manifest.with_suffix(difix_manifest.suffix + ".difix_metadata.json").resolve(),
            "cache_metadata": cache_metadata,
            "checkpoint_fields": checkpoint_fields,
            "render": render.resolve(),
            "target": target.resolve(),
            "key": key,
        }

    paired_manifest = root / "paired_identity_manifest.jsonl"
    difix_protocol = {
        field: arm_contexts["a1"]["cache_metadata"][field]
        for field in (
            "model_id", "model_revision", "difix_code_commit", "coordinate_policy",
            "dtype", "timesteps", "guidance_scale", "prompt",
        )
    }
    paired_row = {
        "manifest_schema": "paired_identity_manifest_v4",
        **common_context,
        "source_images_dir": str(images_dir.resolve()),
        "source_image_inventory": source_inventory,
        "source_image_inventory_sha256": source_inventory_sha256,
        "a1_difix_protocol": difix_protocol,
        "b_difix_protocol": difix_protocol,
        "image_name": "image_3.png",
        "camera": heldout_camera,
        "camera_fingerprint": arm_contexts["a1"]["key"],
        "a1_render": str(arm_contexts["a1"]["render"]),
        "a1_render_sha256": _digest(arm_contexts["a1"]["render"]),
        "a1_difix_output": str(arm_contexts["a1"]["target"]),
        "a1_difix_output_sha256": _digest(arm_contexts["a1"]["target"]),
        "b_render": str(arm_contexts["b"]["render"]),
        "b_render_sha256": _digest(arm_contexts["b"]["render"]),
        "b_difix_output": str(arm_contexts["b"]["target"]),
        "b_difix_output_sha256": _digest(arm_contexts["b"]["target"]),
        "reference_image": str(reference_path.resolve()),
        "reference_image_sha256": _digest(reference_path),
        "real_target": str(real_target.resolve()),
        "real_target_sha256": _digest(real_target),
    }
    for prefix, arm in (("a1", "a1"), ("b", "b")):
        for field, value in arm_contexts[arm]["checkpoint_fields"].items():
            paired_row[f"{prefix}_{field}"] = value
    paired_manifest.write_text(json.dumps(paired_row, sort_keys=True) + "\n", encoding="utf-8")
    paired_metadata = {
        "schema": "paired_identity_manifest_metadata_v4",
        "passed": True,
        "paired_manifest": str(paired_manifest.resolve()),
        "paired_manifest_sha256": _digest(paired_manifest),
        "record_count": 1,
        **common_context,
        "source_images_dir": str(images_dir.resolve()),
        "source_image_inventory": source_inventory,
        "source_image_inventory_sha256": source_inventory_sha256,
        "a1_difix_protocol": difix_protocol,
        "b_difix_protocol": difix_protocol,
    }
    for prefix, arm in (("a1", "a1"), ("b", "b")):
        context = arm_contexts[arm]
        paired_metadata.update(
            {
                f"{prefix}_evidence_manifest": str(context["evidence_manifest"]),
                f"{prefix}_evidence_manifest_sha256": _digest(context["evidence_manifest"]),
                f"{prefix}_difix_manifest": str(context["difix_manifest"]),
                f"{prefix}_difix_manifest_sha256": _digest(context["difix_manifest"]),
                f"{prefix}_difix_run_metadata": str(context["difix_metadata"]),
                f"{prefix}_difix_run_metadata_sha256": _digest(context["difix_metadata"]),
                f"{prefix}_difix_cache_run_fingerprint": context["cache_metadata"]["cache_run_fingerprint"],
            }
        )
        for field, value in context["checkpoint_fields"].items():
            paired_metadata[f"{prefix}_{field}"] = value
    _write_json(paired_manifest.with_suffix(paired_manifest.suffix + ".metadata.json"), paired_metadata)

    for arm in ("A1", "B"):
        output = root / f"evidence_{arm.lower()}"
        completed = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("evaluate_track_evidence.py")),
                "--track-h5", str(h5_path),
                "--images-dir", str(images_dir),
                "--target-manifest", str(paired_manifest),
                "--paired-arm", arm,
                "--feature-backend", "rgb",
                "--device", "cpu",
                "--window-radii", "32",
                "--output-dir", str(output),
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"{arm} paired evidence evaluator failed:\n{completed.stdout}"
            )
        summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        for key in (
            "projection_mean_error", "projection_median_error", "projection_pck3",
            "projection_pck5", "projection_pck8", "gs_mean_error", "difix_mean_error",
            "real_reference_mean_error", "difix_vs_projection_error_reduction",
            "difix_vs_gs_error_reduction",
        ):
            assert key in summary


def main() -> None:
    tests = [
        ("strict held-out anchor invariance", test_heldout_invariance),
        ("leakage audit + collision + gradient", test_store_association_and_gradient),
    ]
    passed = []
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for name, function in tests:
            function(root)
            passed.append(name)
        test_evidence_end_to_end(root)
        passed.append("Projection/GS/Difix/Real evidence end-to-end")
    for name, function in (
        ("source RGB-only Track membership", test_source_rgb_membership_only),
        ("dimensionless quality scale invariance", test_quality_scale_invariance),
        ("resolution-aware PCK + non-trivial shift window", test_evidence_metrics_and_window),
        ("hard shuffled negatives", test_hard_negatives),
        ("Difix exact-or-tolerance reproducibility", test_difix_reproducibility),
        ("32-view Difix pool coverage at interval 50", test_pseudo_pool_coverage),
        ("A1/B identical real-view schedule", test_controlled_real_schedule),
    ):
        function()
        passed.append(name)
    recovery = run_synthetic(100)
    assert recovery["passed"], recovery
    passed.append("geometry recovery functional synthetic")
    for name in passed:
        print(f"PASS: {name}")
    print(f"PHASE 2.1 CPU SMOKE TESTS: {len(passed)}/{len(passed)} PASS")


if __name__ == "__main__":
    main()
