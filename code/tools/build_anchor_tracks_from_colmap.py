#!/usr/bin/env python3
"""Build strict source-only Tracks without full-COLMAP point membership.

Strict Track identity is constructed exclusively from pairwise feature matches
among the declared source RGB images.  COLMAP cameras/images provide only the
given intrinsics, extrinsics and image names; ``points3D.bin`` is never opened.
Every exported XYZ is triangulated and refined from source observations only.
Source-to-held-out matching is performed only after the source Track set is
frozen, and those observations are retained solely as evaluation labels.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
import struct
from typing import Dict, Iterable, List, Sequence

import h5py
import numpy as np
from PIL import Image

from geometric_constraints.strict_track_store import (
    ANCHOR_PROVENANCE,
    FORMAT_VERSION,
    POINT_CLOUD_PROVENANCE,
    STRICT_PROTOCOL,
    TRACK_MEMBERSHIP_PROVENANCE,
    normalize_image_name,
)


CAMERA_MODELS = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}
SUPPORTED_MODELS = {"PINHOLE", "SIMPLE_PINHOLE"}


def read_exact(handle, size: int) -> bytes:
    value = handle.read(size)
    if len(value) != size:
        raise EOFError(f"Expected {size} bytes, received {len(value)}")
    return value


def unpack(handle, fmt: str):
    size = struct.calcsize("<" + fmt)
    return struct.unpack("<" + fmt, read_exact(handle, size))


def qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    w, x, y, z = qvec
    return np.asarray(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * z * x + 2 * w * y],
            [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
            [2 * z * x - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
        ],
        dtype=np.float64,
    )


def read_cameras(path: Path) -> Dict[int, dict]:
    cameras: Dict[int, dict] = {}
    with path.open("rb") as handle:
        for _ in range(unpack(handle, "Q")[0]):
            camera_id, model_id, width, height = unpack(handle, "iiQQ")
            if model_id not in CAMERA_MODELS:
                raise ValueError(f"Unknown COLMAP camera model id {model_id}")
            model, count = CAMERA_MODELS[model_id]
            params = np.asarray(unpack(handle, "d" * count), dtype=np.float64)
            if model not in SUPPORTED_MODELS:
                raise ValueError(
                    f"Strict protocol requires undistorted PINHOLE/SIMPLE_PINHOLE; got {model}"
                )
            cameras[int(camera_id)] = {
                "id": int(camera_id),
                "model": model,
                "width": int(width),
                "height": int(height),
                "params": params,
            }
    return cameras


def read_images(path: Path) -> Dict[int, dict]:
    images: Dict[int, dict] = {}
    with path.open("rb") as handle:
        for _ in range(unpack(handle, "Q")[0]):
            values = unpack(handle, "idddddddi")
            image_id = int(values[0])
            qvec = np.asarray(values[1:5], dtype=np.float64)
            tvec = np.asarray(values[5:8], dtype=np.float64)
            camera_id = int(values[8])
            raw_name = bytearray()
            while True:
                char = read_exact(handle, 1)
                if char == b"\x00":
                    break
                raw_name.extend(char)
            name = raw_name.decode("utf-8")
            count = unpack(handle, "Q")[0]
            # Binary image observations may reflect the full COLMAP model.
            # Consume their bytes to advance the stream but never retain or use
            # them in the strict source-only Track builder.
            for _ in range(count):
                unpack(handle, "ddq")
            images[image_id] = {
                "id": image_id,
                "qvec": qvec,
                "rotation_w2c": qvec_to_rotmat(qvec),
                "tvec": tvec,
                "camera_id": camera_id,
                "name": name,
            }
    return images


def intrinsics(camera: dict) -> tuple[float, float, float, float]:
    params = camera["params"]
    if camera["model"] == "SIMPLE_PINHOLE":
        return float(params[0]), float(params[0]), float(params[1]), float(params[2])
    if camera["model"] == "PINHOLE":
        return tuple(map(float, params[:4]))
    raise ValueError(f"Unsupported camera model {camera['model']}")


def intrinsics_matrix(camera: dict) -> np.ndarray:
    fx, fy, cx, cy = intrinsics(camera)
    return np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


def resolve_rgb_path(images_dir: Path, image_name: str) -> Path:
    direct = images_dir / image_name
    if direct.exists():
        return direct
    matches = list(images_dir.glob(normalize_image_name(image_name) + ".*"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Cannot resolve RGB image {image_name!r} in {images_dir}"
        )
    return matches[0]


def extract_sift_features(
    images_dir: Path,
    image: dict,
    *,
    max_features: int,
    contrast_threshold: float,
) -> dict:
    """Extract local features from one RGB image without COLMAP observations."""
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "Strict source RGB membership requires OpenCV with SIFT support"
        ) from exc

    path = resolve_rgb_path(images_dir, image["name"])
    grayscale = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if grayscale is None:
        raise RuntimeError(f"OpenCV failed to read {path}")
    detector = cv2.SIFT_create(
        nfeatures=int(max_features),
        contrastThreshold=float(contrast_threshold),
    )
    keypoints, descriptors = detector.detectAndCompute(grayscale, None)
    if descriptors is None or not keypoints:
        raise RuntimeError(f"No SIFT features detected in source image {path}")
    return {
        "image_id": int(image["id"]),
        "image_name": str(image["name"]),
        "xy": np.asarray([point.pt for point in keypoints], dtype=np.float64),
        "descriptors": np.asarray(descriptors, dtype=np.float32),
    }


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(vector, dtype=np.float64)
    return np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def fundamental_from_known_cameras(
    left_image: dict,
    right_image: dict,
    cameras: Dict[int, dict],
) -> np.ndarray:
    left_rotation = left_image["rotation_w2c"]
    right_rotation = right_image["rotation_w2c"]
    relative_rotation = right_rotation @ left_rotation.T
    relative_translation = (
        right_image["tvec"]
        - relative_rotation @ left_image["tvec"]
    )
    essential = _skew(relative_translation) @ relative_rotation
    left_intrinsics = intrinsics_matrix(cameras[left_image["camera_id"]])
    right_intrinsics = intrinsics_matrix(cameras[right_image["camera_id"]])
    fundamental = (
        np.linalg.inv(right_intrinsics).T
        @ essential
        @ np.linalg.inv(left_intrinsics)
    )
    norm = float(np.linalg.norm(fundamental))
    if norm <= 1e-15:
        raise ValueError("Source camera pair has a degenerate epipolar geometry")
    return fundamental / norm


def sampson_distance(
    left_xy: np.ndarray,
    right_xy: np.ndarray,
    fundamental: np.ndarray,
) -> np.ndarray:
    left = np.column_stack([left_xy, np.ones(len(left_xy))])
    right = np.column_stack([right_xy, np.ones(len(right_xy))])
    f_left = left @ fundamental.T
    ft_right = right @ fundamental
    residual = np.sum(right * f_left, axis=1)
    denominator = (
        f_left[:, 0] ** 2
        + f_left[:, 1] ** 2
        + ft_right[:, 0] ** 2
        + ft_right[:, 1] ** 2
    )
    return np.abs(residual) / np.sqrt(np.maximum(denominator, 1e-20))


def match_feature_pair(
    left: dict,
    right: dict,
    left_image: dict,
    right_image: dict,
    cameras: Dict[int, dict],
    *,
    ratio_threshold: float,
    epipolar_threshold: float,
) -> List[dict]:
    """Mutual SIFT ratio matching followed by known-pose epipolar filtering."""
    import cv2

    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    forward_knn = matcher.knnMatch(left["descriptors"], right["descriptors"], k=2)
    reverse_knn = matcher.knnMatch(right["descriptors"], left["descriptors"], k=2)

    def ratio_map(values):
        accepted = {}
        for candidates in values:
            if len(candidates) < 2:
                continue
            best, second = candidates[:2]
            if float(best.distance) < float(ratio_threshold) * float(second.distance):
                accepted[int(best.queryIdx)] = (
                    int(best.trainIdx),
                    float(best.distance),
                )
        return accepted

    forward = ratio_map(forward_knn)
    reverse = ratio_map(reverse_knn)
    mutual = [
        (left_index, right_index, distance)
        for left_index, (right_index, distance) in forward.items()
        if reverse.get(right_index, (-1, float("inf")))[0] == left_index
    ]
    if not mutual:
        return []
    fundamental = fundamental_from_known_cameras(
        left_image, right_image, cameras
    )
    left_xy = np.asarray([left["xy"][item[0]] for item in mutual])
    right_xy = np.asarray([right["xy"][item[1]] for item in mutual])
    distances = sampson_distance(left_xy, right_xy, fundamental)
    matches = []
    for item, geometric_distance in zip(mutual, distances):
        if float(geometric_distance) > float(epipolar_threshold):
            continue
        left_index, right_index, descriptor_distance = item
        matches.append(
            {
                "left_image_id": int(left["image_id"]),
                "left_feature_index": int(left_index),
                "right_image_id": int(right["image_id"]),
                "right_feature_index": int(right_index),
                "descriptor_distance": float(descriptor_distance),
                "epipolar_distance": float(geometric_distance),
            }
        )
    return matches


def build_source_tracks(
    feature_sets: Dict[int, dict], pair_matches: Sequence[dict]
) -> List[dict]:
    """Build conflict-free components using source-source edges only."""
    parent = {}
    component_images = {}

    def add(node):
        if node not in parent:
            parent[node] = node
            component_images[node] = {node[0]}

    def find(node):
        root = node
        while parent[root] != root:
            root = parent[root]
        while parent[node] != node:
            following = parent[node]
            parent[node] = root
            node = following
        return root

    def merge(left_node, right_node):
        add(left_node)
        add(right_node)
        left_root, right_root = find(left_node), find(right_node)
        if left_root == right_root:
            return
        # A Track may contain at most one feature from any source image.
        if component_images[left_root] & component_images[right_root]:
            return
        if repr(left_root) > repr(right_root):
            left_root, right_root = right_root, left_root
        parent[right_root] = left_root
        component_images[left_root] |= component_images.pop(right_root)

    ordered_edges = sorted(
        pair_matches,
        key=lambda item: (
            item["descriptor_distance"],
            item["epipolar_distance"],
            item["left_image_id"],
            item["left_feature_index"],
            item["right_image_id"],
            item["right_feature_index"],
        ),
    )
    for match in ordered_edges:
        merge(
            (int(match["left_image_id"]), int(match["left_feature_index"])),
            (int(match["right_image_id"]), int(match["right_feature_index"])),
        )

    components = defaultdict(list)
    for node in parent:
        components[find(node)].append(node)
    source_tracks = []
    for nodes in components.values():
        if len(nodes) < 2:
            continue
        nodes = sorted(nodes)
        observations = []
        for image_id, feature_index in nodes:
            feature_set = feature_sets[image_id]
            observations.append(
                {
                    "image_id": image_id,
                    "image_name": feature_set["image_name"],
                    "xy": feature_set["xy"][feature_index].astype(np.float64),
                    "confidence": 1.0,
                }
            )
        source_tracks.append(
            {"source_nodes": nodes, "observations": observations}
        )
    source_tracks.sort(
        key=lambda item: tuple(
            (
                feature_sets[image_id]["image_name"],
                round(float(feature_sets[image_id]["xy"][feature_index, 0]), 4),
                round(float(feature_sets[image_id]["xy"][feature_index, 1]), 4),
            )
            for image_id, feature_index in item["source_nodes"]
        )
    )
    for track_id, track in enumerate(source_tracks):
        track["id"] = track_id
    return source_tracks


def attach_heldout_evaluation_observations(
    records: Sequence[dict],
    feature_sets: Dict[int, dict],
    source_to_heldout_matches: Sequence[dict],
    heldout_image_ids: Sequence[int],
    *,
    minimum_votes: int,
) -> int:
    """Attach held-out labels after source records/quality have been frozen."""
    by_source_node = defaultdict(list)
    for match in source_to_heldout_matches:
        source_node = (
            int(match["left_image_id"]),
            int(match["left_feature_index"]),
        )
        by_source_node[source_node].append(match)

    attached = 0
    for record in records:
        source_nodes = list(record["source_nodes"])
        for heldout_image_id in heldout_image_ids:
            candidates = defaultdict(list)
            for source_node in source_nodes:
                for match in by_source_node.get(source_node, []):
                    if int(match["right_image_id"]) == int(heldout_image_id):
                        candidates[int(match["right_feature_index"])].append(
                            float(match["descriptor_distance"])
                        )
            required = min(max(int(minimum_votes), 1), len(source_nodes))
            eligible = [
                (feature_index, values)
                for feature_index, values in candidates.items()
                if len(values) >= required
            ]
            if not eligible:
                continue
            feature_index, values = min(
                eligible,
                key=lambda item: (-len(item[1]), float(np.mean(item[1])), item[0]),
            )
            heldout = feature_sets[int(heldout_image_id)]
            confidence = (
                len(values)
                / max(len(source_nodes), 1)
                * math.exp(-float(np.mean(values)) / 128.0)
            )
            record["observations"].append(
                {
                    "image_id": int(heldout_image_id),
                    "image_name": heldout["image_name"],
                    "xy": heldout["xy"][feature_index].astype(np.float64),
                    "confidence": float(np.clip(confidence, 0.0, 1.0)),
                }
            )
            attached += 1
    return attached


def source_scene_scale(
    images: Dict[int, dict], source_keys: set[str]
) -> float:
    centers = []
    for image in images.values():
        if normalize_image_name(image["name"]) not in source_keys:
            continue
        centers.append(-image["rotation_w2c"].T @ image["tvec"])
    if len(centers) < 2:
        raise ValueError("At least two source camera centers are required")
    centers = np.asarray(centers, dtype=np.float64)
    baselines = [
        float(np.linalg.norm(centers[left] - centers[right]))
        for left, right in combinations(range(len(centers)), 2)
    ]
    positive = [value for value in baselines if value > 1e-12]
    if not positive:
        raise ValueError("Source camera median baseline is degenerate")
    return float(np.median(positive))


def projection_matrix(observation: dict, cameras: Dict[int, dict], images: Dict[int, dict]) -> np.ndarray:
    image = images[int(observation["image_id"])]
    camera = cameras[image["camera_id"]]
    return intrinsics_matrix(camera) @ np.column_stack(
        [image["rotation_w2c"], image["tvec"]]
    )


def triangulate_source_observations(
    observations: Sequence[dict], cameras: Dict[int, dict], images: Dict[int, dict]
) -> np.ndarray:
    if len(observations) < 2:
        raise ValueError("At least two source observations are required")
    rows = []
    for observation in observations:
        matrix = projection_matrix(observation, cameras, images)
        x, y = np.asarray(observation["xy"], dtype=np.float64)
        rows.extend([x * matrix[2] - matrix[0], y * matrix[2] - matrix[1]])
    _, _, vh = np.linalg.svd(np.asarray(rows, dtype=np.float64), full_matrices=False)
    homogeneous = vh[-1]
    if abs(float(homogeneous[3])) < 1e-12:
        raise ValueError("Degenerate source-only triangulation")
    return homogeneous[:3] / homogeneous[3]


def project(xyz: np.ndarray, observation: dict, cameras: Dict[int, dict], images: Dict[int, dict]) -> tuple[np.ndarray, float]:
    image = images[int(observation["image_id"])]
    camera = cameras[image["camera_id"]]
    fx, fy, cx, cy = intrinsics(camera)
    point_camera = image["rotation_w2c"] @ xyz + image["tvec"]
    depth = float(point_camera[2])
    if depth <= 1e-10:
        raise ValueError("Source-only anchor has non-positive source depth")
    pixel = np.asarray(
        [fx * point_camera[0] / depth + cx, fy * point_camera[1] / depth + cy],
        dtype=np.float64,
    )
    return pixel, depth


def projection_jacobian(xyz: np.ndarray, observation: dict, cameras: Dict[int, dict], images: Dict[int, dict]) -> np.ndarray:
    image = images[int(observation["image_id"])]
    camera = cameras[image["camera_id"]]
    fx, fy, _, _ = intrinsics(camera)
    rotation = image["rotation_w2c"]
    point_camera = rotation @ xyz + image["tvec"]
    x, y, z = point_camera
    if z <= 1e-10:
        raise ValueError("Source-only anchor has non-positive source depth")
    camera_jacobian = np.asarray(
        [[fx / z, 0.0, -fx * x / (z * z)], [0.0, fy / z, -fy * y / (z * z)]],
        dtype=np.float64,
    )
    return camera_jacobian @ rotation


def refine_source_anchor(
    initial_xyz: np.ndarray,
    observations: Sequence[dict],
    cameras: Dict[int, dict],
    images: Dict[int, dict],
    iterations: int = 15,
) -> np.ndarray:
    xyz = np.asarray(initial_xyz, dtype=np.float64).copy()
    for _ in range(iterations):
        residuals, jacobians = [], []
        for observation in observations:
            prediction, _ = project(xyz, observation, cameras, images)
            residuals.append(prediction - observation["xy"])
            jacobians.append(projection_jacobian(xyz, observation, cameras, images))
        residual = np.concatenate(residuals)
        jacobian = np.concatenate(jacobians, axis=0)
        hessian = jacobian.T @ jacobian
        damping = np.eye(3) * max(float(np.trace(hessian)), 1.0) * 1e-10
        delta = np.linalg.solve(hessian + damping, -(jacobian.T @ residual))
        xyz += delta
        if float(np.linalg.norm(delta)) < 1e-10:
            break
    # Positivity is checked only in source cameras; held-out cameras do not filter tracks.
    for observation in observations:
        project(xyz, observation, cameras, images)
    return xyz


def source_reprojection_error(xyz: np.ndarray, observations: Sequence[dict], cameras: Dict[int, dict], images: Dict[int, dict]) -> float:
    errors = [
        float(np.linalg.norm(project(xyz, obs, cameras, images)[0] - obs["xy"]))
        for obs in observations
    ]
    return float(np.mean(errors))


def source_covariance(
    xyz: np.ndarray,
    observations: Sequence[dict],
    cameras: Dict[int, dict],
    images: Dict[int, dict],
    pixel_sigma: float,
) -> np.ndarray:
    information = np.zeros((3, 3), dtype=np.float64)
    precision = 1.0 / max(float(pixel_sigma) ** 2, 1e-12)
    for observation in observations:
        jacobian = projection_jacobian(xyz, observation, cameras, images)
        information += precision * (jacobian.T @ jacobian)
    # Relative damping is equivariant to similarity-scale changes; an absolute
    # world-unit floor would destroy this property for very large scenes.
    spectrum = np.linalg.eigvalsh(information)
    if not np.isfinite(spectrum).all() or spectrum[-1] <= 0:
        raise ValueError("Non-finite or zero triangulation information")
    if spectrum[0] / spectrum[-1] < 1e-10:
        raise ValueError("Degenerate source triangulation: insufficient parallax")
    scale = float(np.trace(information) / 3.0)
    covariance = np.linalg.inv(information / scale + np.eye(3) * 1e-8) / scale
    return 0.5 * (covariance + covariance.T)


def source_quality(
    error: float,
    length: int,
    covariance: np.ndarray,
    scene_scale: float,
) -> float:
    length_term = min(1.0, math.log1p(length) / math.log1p(6))
    reprojection_term = math.exp(-max(error, 0.0) / 2.0)
    if not math.isfinite(scene_scale) or scene_scale <= 0.0:
        raise ValueError("source-only scene scale must be finite and positive")
    uncertainty_world = math.sqrt(max(float(np.trace(covariance)), 0.0))
    uncertainty_dimensionless = uncertainty_world / float(scene_scale)
    uncertainty_term = 1.0 / (1.0 + uncertainty_dimensionless)
    return float(np.clip(length_term * reprojection_term * uncertainty_term, 0.0, 1.0))


def load_name_list(path: Path | None, repeated: Iterable[str]) -> List[str]:
    values: List[str] = []
    if path is not None:
        values.extend(
            line.strip() for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    values.extend(v for v in repeated if v)
    return list(dict.fromkeys(normalize_image_name(value) for value in values))


def sample_source_rgb(
    images_dir: Path,
    observations: Sequence[dict],
    cache: Dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    samples = []
    cache = {} if cache is None else cache
    for observation in observations:
        name = str(observation["image_name"])
        path = images_dir / name
        if not path.exists():
            matches = list(images_dir.glob(normalize_image_name(name) + ".*"))
            if len(matches) != 1:
                raise FileNotFoundError(f"Cannot resolve source RGB image {name!r} in {images_dir}")
            path = matches[0]
        key = str(path.resolve())
        if key not in cache:
            with Image.open(path) as image:
                cache[key] = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        array = cache[key]
        x, y = np.rint(observation["xy"]).astype(int)
        x = int(np.clip(x, 0, array.shape[1] - 1))
        y = int(np.clip(y, 0, array.shape[0] - 1))
        samples.append(array[y, x])
    return np.mean(np.stack(samples), axis=0)


def build_record(
    point_track: dict,
    all_observations: Sequence[dict],
    source_keys: set[str],
    heldout_keys: set[str],
    cameras: Dict[int, dict],
    images: Dict[int, dict],
    images_dir: Path,
    pixel_sigma: float,
    max_source_error: float,
    scene_scale: float,
    rgb_cache: Dict[str, np.ndarray] | None = None,
) -> dict | None:
    source = [obs for obs in all_observations if normalize_image_name(obs["image_name"]) in source_keys]
    heldout = [obs for obs in all_observations if normalize_image_name(obs["image_name"]) in heldout_keys]
    if len(source) < 2:
        return None
    try:
        xyz = triangulate_source_observations(source, cameras, images)
        xyz = refine_source_anchor(xyz, source, cameras, images)
        error = source_reprojection_error(xyz, source, cameras, images)
        if error > max_source_error:
            return None
        covariance = source_covariance(xyz, source, cameras, images, pixel_sigma)
        rgb = sample_source_rgb(images_dir, source, rgb_cache)
    except (ValueError, np.linalg.LinAlgError):
        return None
    return {
        "id": int(point_track["id"]),
        "xyz": xyz,
        "rgb_source": rgb,
        "covariance": covariance,
        "quality": source_quality(
            error, len(source), covariance, scene_scale
        ),
        "source_reprojection_error": error,
        "source_observation_count": len(source),
        "observations": list(source) + list(heldout),
        "source_nodes": list(point_track.get("source_nodes", [])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sparse-dir", required=True, type=Path)
    parser.add_argument("--images-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--anchor-image-list", required=True, type=Path)
    parser.add_argument("--heldout-image-list", type=Path)
    parser.add_argument("--heldout-image", action="append", default=[])
    parser.add_argument("--pixel-sigma", type=float, default=1.0)
    parser.add_argument("--max-source-reprojection-error", type=float, default=4.0)
    parser.add_argument("--max-tracks", type=int, default=0)
    parser.add_argument("--max-features-per-image", type=int, default=8192)
    parser.add_argument("--sift-contrast-threshold", type=float, default=0.02)
    parser.add_argument("--match-ratio-threshold", type=float, default=0.75)
    parser.add_argument("--epipolar-threshold", type=float, default=1.5)
    parser.add_argument("--min-heldout-match-votes", type=int, default=2)
    args = parser.parse_args()

    source_names = load_name_list(args.anchor_image_list, [])
    heldout_names = load_name_list(args.heldout_image_list, args.heldout_image)
    if len(source_names) < 2:
        raise ValueError("Strict source-only geometry needs at least two source images")
    if set(source_names) & set(heldout_names):
        raise ValueError("Source and held-out image lists overlap")

    sparse = args.sparse_dir.expanduser().resolve()
    images_dir = args.images_dir.expanduser().resolve()
    cameras = read_cameras(sparse / "cameras.bin")
    images = read_images(sparse / "images.bin")
    image_key_to_id = {
        normalize_image_name(image["name"]): image_id
        for image_id, image in images.items()
    }
    missing = (set(source_names) | set(heldout_names)) - set(image_key_to_id)
    if missing:
        raise KeyError(f"Images are absent from COLMAP images.bin: {sorted(missing)}")

    source_keys, heldout_keys = set(source_names), set(heldout_names)
    scale = source_scene_scale(images, source_keys)
    source_image_ids = [image_key_to_id[name] for name in source_names]
    heldout_image_ids = [image_key_to_id[name] for name in heldout_names]

    # Strict source Track identity is frozen before any held-out RGB is opened.
    source_features = {
        image_id: extract_sift_features(
            images_dir,
            images[image_id],
            max_features=args.max_features_per_image,
            contrast_threshold=args.sift_contrast_threshold,
        )
        for image_id in source_image_ids
    }
    source_pair_matches = []
    for left_id, right_id in combinations(source_image_ids, 2):
        source_pair_matches.extend(
            match_feature_pair(
                source_features[left_id],
                source_features[right_id],
                images[left_id],
                images[right_id],
                cameras,
                ratio_threshold=args.match_ratio_threshold,
                epipolar_threshold=args.epipolar_threshold,
            )
        )
    source_tracks = build_source_tracks(source_features, source_pair_matches)
    if not source_tracks:
        raise RuntimeError(
            "No source-only Tracks were formed from source-source RGB matches"
        )

    retained = []
    rgb_cache: Dict[str, np.ndarray] = {}
    for source_track in source_tracks:
        record = build_record(
            source_track,
            source_track["observations"],
            source_keys,
            heldout_keys,
            cameras,
            images,
            images_dir,
            args.pixel_sigma,
            args.max_source_reprojection_error,
            scale,
            rgb_cache,
        )
        if record is not None:
            retained.append(record)

    retained.sort(
        key=lambda item: (item["quality"], item["source_observation_count"]),
        reverse=True,
    )
    if args.max_tracks > 0:
        retained = retained[: args.max_tracks]
    if not retained:
        raise RuntimeError("No source-only tracks survived triangulation")

    # Source Tracks, anchors, covariance, quality, RGB and max-track selection
    # are now immutable. Held-out RGB is accessed only to add evaluation labels.
    feature_sets = dict(source_features)
    source_to_heldout_matches = []
    matched_heldout_image_ids = []
    for heldout_id in heldout_image_ids:
        try:
            feature_sets[heldout_id] = extract_sift_features(
                images_dir,
                images[heldout_id],
                max_features=args.max_features_per_image,
                contrast_threshold=args.sift_contrast_threshold,
            )
        except RuntimeError as exc:
            print(
                "[Held-out Evaluation Labels] Warning: skipping "
                f"{images[heldout_id]['name']}: {exc}"
            )
            continue
        matched_heldout_image_ids.append(heldout_id)
        for source_id in source_image_ids:
            source_to_heldout_matches.extend(
                match_feature_pair(
                    source_features[source_id],
                    feature_sets[heldout_id],
                    images[source_id],
                    images[heldout_id],
                    cameras,
                    ratio_threshold=args.match_ratio_threshold,
                    epipolar_threshold=args.epipolar_threshold,
                )
            )
    heldout_label_count = attach_heldout_evaluation_observations(
        retained,
        feature_sets,
        source_to_heldout_matches,
        matched_heldout_image_ids,
        minimum_votes=args.min_heldout_match_votes,
    )

    lengths = np.asarray([len(item["observations"]) for item in retained], dtype=np.int32)
    offsets = np.concatenate([[0], np.cumsum(lengths, dtype=np.int64)])
    observations = [obs for item in retained for obs in item["observations"]]
    use_source = np.asarray(
        [normalize_image_name(obs["image_name"]) in source_keys for obs in observations],
        dtype=np.bool_,
    )
    used_image_ids = sorted(set(source_image_ids + heldout_image_ids))
    string_dtype = h5py.string_dtype(encoding="utf-8")
    source_hash = hashlib.sha256("\n".join(sorted(source_names)).encode("utf-8")).hexdigest()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.output, "w") as handle:
        handle.attrs["format_version"] = FORMAT_VERSION
        handle.attrs["geometry_protocol"] = STRICT_PROTOCOL
        handle.attrs["anchor_provenance"] = ANCHOR_PROVENANCE
        handle.attrs["track_membership_provenance"] = (
            TRACK_MEMBERSHIP_PROVENANCE
        )
        handle.attrs["point_cloud_provenance"] = POINT_CLOUD_PROVENANCE
        handle.attrs["heldout_observation_provenance"] = (
            "evaluation_only_source_to_heldout_sift_after_source_tracks_frozen"
        )
        handle.attrs["source_images_json"] = json.dumps(source_names)
        handle.attrs["heldout_images_json"] = json.dumps(heldout_names)
        handle.attrs["source_image_list_sha256"] = source_hash
        handle.attrs["points3d_usage"] = "none_file_not_opened"
        handle.attrs["source_feature_backend"] = "opencv_sift"
        handle.attrs["source_scene_scale"] = float(scale)
        handle.attrs["source_scene_scale_definition"] = (
            "median_pairwise_source_camera_baseline"
        )
        handle.attrs["quality_uncertainty_definition"] = (
            "sqrt_trace_covariance_divided_by_source_scene_scale"
        )
        handle.attrs["match_ratio_threshold"] = float(args.match_ratio_threshold)
        handle.attrs["epipolar_threshold_pixels"] = float(
            args.epipolar_threshold
        )
        handle.attrs["source_source_match_count"] = len(source_pair_matches)
        handle.attrs["source_track_count_before_geometry_filter"] = len(
            source_tracks
        )
        handle.attrs["heldout_evaluation_label_count"] = heldout_label_count
        handle.attrs["pixel_sigma"] = float(args.pixel_sigma)

        tracks_group = handle.create_group("tracks")
        tracks_group.create_dataset("id", data=np.asarray([v["id"] for v in retained], dtype=np.int64))
        tracks_group.create_dataset("xyz", data=np.stack([v["xyz"] for v in retained]).astype(np.float32))
        tracks_group.create_dataset("rgb_source", data=np.stack([v["rgb_source"] for v in retained]).astype(np.float32))
        tracks_group.create_dataset("covariance", data=np.stack([v["covariance"] for v in retained]).astype(np.float32))
        tracks_group.create_dataset("quality", data=np.asarray([v["quality"] for v in retained], dtype=np.float32))
        tracks_group.create_dataset(
            "source_reprojection_error",
            data=np.asarray([v["source_reprojection_error"] for v in retained], dtype=np.float32),
        )
        tracks_group.create_dataset(
            "source_observation_count",
            data=np.asarray([v["source_observation_count"] for v in retained], dtype=np.int32),
        )

        obs_group = handle.create_group("observations")
        obs_group.create_dataset("offsets", data=offsets)
        obs_group.create_dataset("xy", data=np.stack([v["xy"] for v in observations]).astype(np.float32))
        obs_group.create_dataset("image_id", data=np.asarray([v["image_id"] for v in observations], dtype=np.int32))
        obs_group.create_dataset(
            "image_name",
            data=np.asarray([v["image_name"] for v in observations], dtype=object),
            dtype=string_dtype,
        )
        obs_group.create_dataset("confidence", data=np.asarray([v["confidence"] for v in observations], dtype=np.float32))
        obs_group.create_dataset("use_for_anchor", data=use_source)
        obs_group.create_dataset("use_for_quality", data=use_source)
        obs_group.create_dataset("use_for_source_features", data=use_source)

        camera_group = handle.create_group("cameras")
        camera_records = []
        for image_id in used_image_ids:
            image = images[image_id]
            camera = cameras[image["camera_id"]]
            fx, fy, cx, cy = intrinsics(camera)
            camera_records.append((image, camera, fx, fy, cx, cy))
        camera_group.create_dataset("image_id", data=np.asarray([v[0]["id"] for v in camera_records], dtype=np.int32))
        camera_group.create_dataset(
            "image_name",
            data=np.asarray([v[0]["name"] for v in camera_records], dtype=object),
            dtype=string_dtype,
        )
        camera_group.create_dataset(
            "model",
            data=np.asarray([v[1]["model"] for v in camera_records], dtype=object),
            dtype=string_dtype,
        )
        camera_group.create_dataset("width", data=np.asarray([v[1]["width"] for v in camera_records], dtype=np.int32))
        camera_group.create_dataset("height", data=np.asarray([v[1]["height"] for v in camera_records], dtype=np.int32))
        camera_group.create_dataset("fx", data=np.asarray([v[2] for v in camera_records], dtype=np.float64))
        camera_group.create_dataset("fy", data=np.asarray([v[3] for v in camera_records], dtype=np.float64))
        camera_group.create_dataset("cx", data=np.asarray([v[4] for v in camera_records], dtype=np.float64))
        camera_group.create_dataset("cy", data=np.asarray([v[5] for v in camera_records], dtype=np.float64))
        camera_group.create_dataset("rotation_w2c", data=np.stack([v[0]["rotation_w2c"] for v in camera_records]))
        camera_group.create_dataset("translation_w2c", data=np.stack([v[0]["tvec"] for v in camera_records]))

    print(f"Wrote {len(retained)} strict source-only tracks to {args.output}")
    print(f"Track membership provenance: {TRACK_MEMBERSHIP_PROVENANCE}")
    print("tracks/xyz provenance: source observations DLT + source-only refinement")
    print(f"Source-only scene scale (median baseline): {scale:.9g}")
    print(f"Held-out evaluation observations attached: {heldout_label_count}")
    print("points3D.bin usage: none (file not opened)")


if __name__ == "__main__":
    main()
