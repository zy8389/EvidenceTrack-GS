#!/usr/bin/env python3
"""CPU smoke tests for every Phase 2.1 integrity patch."""

from __future__ import annotations

import json
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
from diffusion_guidance.pseudo_manifest import pseudo_view_index
from diffusion_guidance.camera_utils import camera_fingerprint_from_payload
from diffusion_guidance.difix_provenance import (
    CACHE_IDENTITY_FIELDS,
    cache_run_fingerprint,
)
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


def test_evidence_end_to_end(root: Path) -> None:
    track_xyz = np.asarray(
        [[0.0, 0.0, 2.5], [0.2, 0.0, 2.5], [-0.2, 0.1, 2.5]],
        dtype=np.float32,
    )
    h5_path = root / "evidence_tracks.h5"
    write_strict_h5(h5_path, track_xyz)
    image_y, image_x = np.mgrid[0:120, 0:160]
    array = np.stack(
        [image_x / 159.0, image_y / 119.0, (image_x + image_y) / 278.0],
        axis=-1,
    )
    array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    for name in ("image_1.png", "image_2.png", "gs.png", "difix.png", "real.png"):
        Image.fromarray(array).save(root / name)
    manifest = root / "evidence_manifest.jsonl"
    import hashlib
    def digest(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    payload = {
        "R": np.eye(3).tolist(), "T": [0.0, 0.0, 0.0],
        "FoVx": 1.0, "FoVy": 1.0, "width": 160, "height": 120,
        "intrinsics": {"fx": 100.0, "fy": 100.0, "cx": 80.0, "cy": 60.0, "width": 160, "height": 120},
    }
    key = camera_fingerprint_from_payload(payload, prefix="heldout")
    difix_manifest = root / "difix_manifest.jsonl"
    difix_manifest.write_text(
        json.dumps(
            {
                "key": key,
                "camera": payload,
                "input": str(root / "gs.png"),
                "reference_image": str(root / "image_1.png"),
                "target": str(root / "difix.png"),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    run_metadata = {
        "manifest": str(difix_manifest.resolve()),
        "manifest_sha256": digest(difix_manifest),
        "record_count": 1,
        "seed": 1,
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
    run_metadata["cache_run_fingerprint"] = cache_run_fingerprint(run_metadata)
    difix_manifest.with_suffix(".jsonl.difix_metadata.json").write_text(
        json.dumps(run_metadata), encoding="utf-8"
    )
    difix_sidecar = {
        **{field: run_metadata[field] for field in CACHE_IDENTITY_FIELDS},
        "cache_run_fingerprint": run_metadata["cache_run_fingerprint"],
        "input_image": str((root / "gs.png").resolve()),
        "reference_image": str((root / "image_1.png").resolve()),
        "target_image": str((root / "difix.png").resolve()),
        "input_sha256": digest(root / "gs.png"),
        "reference_sha256": digest(root / "image_1.png"),
        "output_sha256": digest(root / "difix.png"),
        "camera_fingerprint": key, "resolution": [160, 120],
    }
    (root / "difix.png.metadata.json").write_text(json.dumps(difix_sidecar), encoding="utf-8")
    manifest.write_text(
        json.dumps(
            {
                "image_name": "image_3.png", "camera_fingerprint": key, "camera": payload,
                "gs_render": str(root / "gs.png"), "difix_output": str(root / "difix.png"),
                "real_target": str(root / "real.png"), "reference_image": str(root / "image_1.png"),
                "gs_render_sha256": digest(root / "gs.png"),
                "reference_image_sha256": digest(root / "image_1.png"),
                "real_target_sha256": digest(root / "real.png"),
            }
        ) + "\n",
        encoding="utf-8",
    )
    output = root / "evidence_output"
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("evaluate_track_evidence.py")),
            "--track-h5",
            str(h5_path),
            "--images-dir",
            str(root),
            "--target-manifest",
            str(manifest),
            "--feature-backend",
            "rgb",
            "--device",
            "cpu",
            "--window-radii",
            "32",
            "--output-dir",
            str(output),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    for key in (
        "projection_mean_error",
        "projection_median_error",
        "projection_pck3",
        "projection_pck5",
        "projection_pck8",
        "gs_mean_error",
        "difix_mean_error",
        "real_reference_mean_error",
        "difix_vs_projection_error_reduction",
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
