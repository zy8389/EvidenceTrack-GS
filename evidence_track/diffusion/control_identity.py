"""Stable identifiers binding one controlled continuation pair to its inputs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


CONTROLLED_TRAINING_PROTOCOL_EXCLUDED_FIELDS = frozenset(
    {
        "model_path",
        "controlled_ab_role",
        "enable_diffusion_pseudo_rgb",
        "pseudo_rgb_manifest",
        "pseudo_rgb_strict_cache",
        "gt_dca_config",
    }
)

CONTROLLED_CHECKPOINT_PROVENANCE_SCHEMA = "controlled_checkpoint_provenance_v2"
PREREGISTERED_CONTROLLED_OPTIMIZATION = {
    "controlled_ab_checkpoint_iteration": 10000,
    "densify_until_iter": 10000,
    "lambda_dssim": 0.2,
    "pseudo_rgb_end": 12000,
    "pseudo_rgb_interval": 50,
    "pseudo_rgb_lambda_dssim": 0.2,
    "pseudo_rgb_start": 10000,
    "pseudo_rgb_weight": 0.1,
    "strict_track_weight": 0.1,
}
CONTROLLED_PHASE_ITERATIONS = {
    "A0": 10000,
    "A1": 12000,
    "SELFRENDER": 12000,
    "B": 12000,
}
CONTROLLED_PHASE_TRANSITION_FIELDS = frozenset(
    {
        "checkpoint_iterations",
        "iterations",
        "save_iterations",
        "start_checkpoint",
        "test_iterations",
    }
)
CONTROLLED_CHECKPOINT_INPUT_FIELDS = (
    "scene_source_path",
    "seed",
    "track_h5_path",
    "track_h5_sha256",
    "source_camera_names",
    "source_camera_set_sha256",
    "source_images_dir",
    "source_image_inventory",
    "source_image_inventory_sha256",
    "source_training_camera_inventory",
    "source_training_camera_inventory_sha256",
    "strict_geometry_protocol",
)
CONTROLLED_CHECKPOINT_LINEAGE_FIELDS = (
    "start_checkpoint",
    "start_checkpoint_sha256",
    "start_checkpoint_controlled_provenance_sha256",
)
CONTROLLED_CHECKPOINT_PSEUDO_FIELDS = (
    "pseudo_manifest_path",
    "pseudo_manifest_sha256",
    "pseudo_camera_pool_sha256",
    "pseudo_target_kind",
    "pseudo_rgb_strict_cache",
)


def normalized_camera_names(names: Sequence[str]) -> list[str]:
    if isinstance(names, (str, bytes)):
        raise ValueError("Source camera names must be a sequence, not one string")
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError("Source camera names must be non-empty strings")
    normalized = sorted(
        Path(name.replace("\\", "/")).stem for name in names
    )
    if (
        not normalized
        or any(not name for name in normalized)
        or len(normalized) != len(set(normalized))
    ):
        raise ValueError("Source camera names must be non-empty and unique")
    return normalized


def source_camera_set_sha256(names: Sequence[str]) -> str:
    serialized = json.dumps(
        normalized_camera_names(names), separators=(",", ":")
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    serialized = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return hashlib.sha256(serialized.encode("ascii")).hexdigest()


def canonical_source_image_inventory(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list) or not value:
        raise ValueError("source_image_inventory must be a non-empty list")
    records: list[dict[str, str]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping) or set(item) != {
            "camera_name",
            "path",
            "sha256",
        }:
            raise ValueError(
                f"source_image_inventory[{index}] has an invalid schema"
            )
        name = normalized_camera_names([item["camera_name"]])[0]
        path_value = item["path"]
        digest = item["sha256"]
        if not isinstance(path_value, str) or not path_value.strip():
            raise ValueError(
                f"source_image_inventory[{index}] has an invalid path"
            )
        path = Path(path_value).expanduser().resolve()
        if path_value != str(path):
            raise ValueError(
                f"source_image_inventory[{index}] path is not canonical"
            )
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(
                f"source_image_inventory[{index}] has an invalid SHA256"
            )
        records.append(
            {"camera_name": name, "path": str(path), "sha256": digest}
        )
    records.sort(key=lambda item: item["camera_name"])
    names = [item["camera_name"] for item in records]
    if len(names) != len(set(names)):
        raise ValueError("source_image_inventory repeats a source camera")
    return records


def source_image_inventory_sha256(value: Any) -> str:
    return _canonical_json_sha256(canonical_source_image_inventory(value))


def build_source_image_inventory(
    images_dir: str | Path, names: Sequence[str]
) -> list[dict[str, str]]:
    directory = Path(images_dir).expanduser().resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"Source image directory is missing: {directory}")
    expected_names = normalized_camera_names(names)
    candidates: dict[str, list[Path]] = {name: [] for name in expected_names}
    for path in directory.iterdir():
        if path.is_file() and path.stem in candidates:
            candidates[path.stem].append(path.resolve())
    records = []
    for name in expected_names:
        matches = candidates[name]
        if len(matches) != 1:
            raise ValueError(
                "Each source camera must resolve to exactly one source image: "
                f"camera={name!r}, matches={[str(path) for path in matches]!r}"
            )
        path = matches[0]
        records.append(
            {"camera_name": name, "path": str(path), "sha256": _sha256_file(path)}
        )
    return canonical_source_image_inventory(records)


def canonical_training_camera_inventory(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError("source_training_camera_inventory must be a non-empty list")
    records: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping) or set(item) != {"camera_name", "camera"}:
            raise ValueError(
                f"source_training_camera_inventory[{index}] has an invalid schema"
            )
        name = normalized_camera_names([item["camera_name"]])[0]
        camera = item["camera"]
        if not isinstance(camera, Mapping) or "intrinsics" not in camera:
            raise ValueError(
                f"source_training_camera_inventory[{index}] lacks calibrated intrinsics"
            )
        from .camera_utils import canonical_camera_payload

        canonical_camera = canonical_camera_payload(dict(camera))
        if dict(camera) != canonical_camera:
            raise ValueError(
                f"source_training_camera_inventory[{index}] camera is not canonical"
            )
        records.append({"camera_name": name, "camera": canonical_camera})
    records.sort(key=lambda item: item["camera_name"])
    names = [item["camera_name"] for item in records]
    if len(names) != len(set(names)):
        raise ValueError("source_training_camera_inventory repeats a camera")
    return records


def training_camera_inventory_sha256(value: Any) -> str:
    return _canonical_json_sha256(canonical_training_camera_inventory(value))


def canonical_controlled_training_protocol(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("controlled_training_protocol must be an object")
    expected = {"schema", "model", "optimization", "pipeline", "runtime"}
    if set(value) != expected:
        raise ValueError(
            "controlled_training_protocol has an invalid schema: "
            f"expected={sorted(expected)}, observed={sorted(map(str, value))}"
        )
    if value.get("schema") != "controlled_training_protocol_v1":
        raise ValueError("controlled_training_protocol has an unsupported schema")
    for namespace in ("model", "optimization", "pipeline", "runtime"):
        section = value.get(namespace)
        if not isinstance(section, Mapping) or not section:
            raise ValueError(
                f"controlled_training_protocol.{namespace} must be a non-empty object"
            )
        leaked = sorted(
            set(section) & CONTROLLED_TRAINING_PROTOCOL_EXCLUDED_FIELDS
        )
        if leaked:
            raise ValueError(
                f"controlled_training_protocol.{namespace} contains "
                f"branch-specific fields: {leaked}"
            )
    return json.loads(
        json.dumps(
            dict(value),
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
    )


def controlled_training_protocol_sha256(value: Any) -> str:
    return _canonical_json_sha256(canonical_controlled_training_protocol(value))


def validate_preregistered_controlled_training_protocol(
    value: Any, *, role: str
) -> dict[str, Any]:
    """Reject controlled runs that are mutually consistent but off protocol."""
    protocol = canonical_controlled_training_protocol(value)
    normalized_role = str(role).upper()
    if normalized_role not in CONTROLLED_PHASE_ITERATIONS:
        raise ValueError(f"Unsupported controlled role: {role!r}")
    model = protocol["model"]
    optimization = protocol["optimization"]
    runtime = protocol["runtime"]
    expected_iterations = CONTROLLED_PHASE_ITERATIONS[normalized_role]
    checks = {
        "model.strict_tracks": model.get("strict_tracks") is True,
        "model.strict_source_only_geometry": model.get(
            "strict_source_only_geometry"
        )
        is True,
        "optimization.iterations": optimization.get("iterations")
        == expected_iterations,
        "runtime.iterations": runtime.get("iterations") == expected_iterations,
    }
    for key, expected in PREREGISTERED_CONTROLLED_OPTIMIZATION.items():
        checks[f"optimization.{key}"] = optimization.get(key) == expected
        checks[f"runtime.{key}"] = runtime.get(key) == expected
    start_checkpoint = runtime.get("start_checkpoint")
    if normalized_role == "A0":
        checks["runtime.start_checkpoint"] = start_checkpoint in (None, "")
    else:
        checks["runtime.start_checkpoint"] = (
            isinstance(start_checkpoint, str)
            and bool(start_checkpoint.strip())
            and start_checkpoint
            == str(Path(start_checkpoint).expanduser().resolve())
        )
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError(
            "Controlled training parameters differ from the preregistered first-run "
            f"protocol: {failed}"
        )
    return protocol


def controlled_phase_common_training_protocol(value: Any) -> dict[str, Any]:
    """Remove only the fields allowed to change at the A0 continuation boundary."""
    protocol = canonical_controlled_training_protocol(value)
    common = json.loads(
        json.dumps(protocol, sort_keys=True, ensure_ascii=True, allow_nan=False)
    )
    for namespace in ("model", "optimization", "pipeline", "runtime"):
        for key in CONTROLLED_PHASE_TRANSITION_FIELDS:
            common[namespace].pop(key, None)
    return common


def canonical_controlled_checkpoint_provenance(value: Any) -> dict[str, Any]:
    """Canonical provenance stored inside every controlled checkpoint."""
    if not isinstance(value, Mapping):
        raise ValueError("controlled checkpoint provenance must be an object")
    expected = {
        "schema",
        "role",
        "final_iteration",
        "controlled_training_protocol",
        "controlled_training_protocol_sha256",
        *CONTROLLED_CHECKPOINT_INPUT_FIELDS,
        *CONTROLLED_CHECKPOINT_LINEAGE_FIELDS,
        *CONTROLLED_CHECKPOINT_PSEUDO_FIELDS,
    }
    if set(value) != expected:
        raise ValueError(
            "controlled checkpoint provenance has an invalid schema: "
            f"expected={sorted(expected)}, observed={sorted(map(str, value))}"
        )
    if value.get("schema") != CONTROLLED_CHECKPOINT_PROVENANCE_SCHEMA:
        raise ValueError("controlled checkpoint provenance has an unsupported schema")
    role = str(value.get("role", "")).upper()
    if role not in CONTROLLED_PHASE_ITERATIONS:
        raise ValueError("controlled checkpoint provenance has an invalid role")
    final_iteration = value.get("final_iteration")
    if (
        isinstance(final_iteration, bool)
        or not isinstance(final_iteration, int)
        or final_iteration != CONTROLLED_PHASE_ITERATIONS[role]
    ):
        raise ValueError("controlled checkpoint provenance has an invalid final iteration")
    seed = value.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("controlled checkpoint provenance seed must be an integer")

    for field in ("scene_source_path", "track_h5_path", "source_images_dir"):
        raw = value.get(field)
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError(f"controlled checkpoint provenance lacks {field}")
        if raw != str(Path(raw).expanduser().resolve()):
            raise ValueError(
                f"controlled checkpoint provenance {field} is not canonical"
            )
    for field in (
        "track_h5_sha256",
        "source_camera_set_sha256",
        "source_image_inventory_sha256",
        "source_training_camera_inventory_sha256",
        "controlled_training_protocol_sha256",
    ):
        digest = value.get(field)
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(
                f"controlled checkpoint provenance {field} is not a lowercase SHA256"
            )

    names = normalized_camera_names(value.get("source_camera_names", []))
    if value.get("source_camera_names") != names:
        raise ValueError("controlled checkpoint source-camera names are not canonical")
    if source_camera_set_sha256(names) != value["source_camera_set_sha256"]:
        raise ValueError("controlled checkpoint source-camera hash is inconsistent")
    images = canonical_source_image_inventory(value.get("source_image_inventory"))
    if value.get("source_image_inventory") != images:
        raise ValueError("controlled checkpoint source-image inventory is not canonical")
    if [item["camera_name"] for item in images] != names:
        raise ValueError("controlled checkpoint source-image cameras differ")
    if source_image_inventory_sha256(images) != value["source_image_inventory_sha256"]:
        raise ValueError("controlled checkpoint source-image hash is inconsistent")
    cameras = canonical_training_camera_inventory(
        value.get("source_training_camera_inventory")
    )
    if value.get("source_training_camera_inventory") != cameras:
        raise ValueError("controlled checkpoint camera inventory is not canonical")
    if [item["camera_name"] for item in cameras] != names:
        raise ValueError("controlled checkpoint calibrated cameras differ")
    if (
        training_camera_inventory_sha256(cameras)
        != value["source_training_camera_inventory_sha256"]
    ):
        raise ValueError("controlled checkpoint camera-inventory hash is inconsistent")
    protocol = validate_preregistered_controlled_training_protocol(
        value.get("controlled_training_protocol"), role=role
    )
    if value.get("controlled_training_protocol") != protocol:
        raise ValueError("controlled checkpoint training protocol is not canonical")
    if (
        controlled_training_protocol_sha256(protocol)
        != value["controlled_training_protocol_sha256"]
    ):
        raise ValueError("controlled checkpoint training-protocol hash is inconsistent")
    model_protocol = protocol["model"]
    optimization_protocol = protocol["optimization"]
    runtime_protocol = protocol["runtime"]
    declared_images = model_protocol.get("images") or "images"
    expected_images_dir = (
        Path(value["scene_source_path"]) / str(declared_images)
    ).expanduser().resolve()
    protocol_binding_checks = {
        "model.source_path": model_protocol.get("source_path")
        == value["scene_source_path"],
        "model.track_path": model_protocol.get("track_path")
        == value["track_h5_path"],
        "model.images": str(expected_images_dir) == value["source_images_dir"],
        "optimization.experiment_seed": optimization_protocol.get(
            "experiment_seed"
        )
        == seed,
        "runtime.source_path": runtime_protocol.get("source_path")
        == value["scene_source_path"],
        "runtime.track_path": runtime_protocol.get("track_path")
        == value["track_h5_path"],
        "runtime.experiment_seed": runtime_protocol.get("experiment_seed") == seed,
    }
    failed_protocol_bindings = sorted(
        name for name, passed in protocol_binding_checks.items() if not passed
    )
    if failed_protocol_bindings:
        raise ValueError(
            "controlled checkpoint training protocol conflicts with its inputs: "
            f"{failed_protocol_bindings}"
        )

    start_checkpoint = value.get("start_checkpoint")
    start_checkpoint_sha256 = value.get("start_checkpoint_sha256")
    start_provenance_sha256 = value.get(
        "start_checkpoint_controlled_provenance_sha256"
    )
    if role == "A0":
        if any(
            item is not None
            for item in (
                start_checkpoint,
                start_checkpoint_sha256,
                start_provenance_sha256,
            )
        ):
            raise ValueError("A0 checkpoint provenance cannot declare a parent checkpoint")
    else:
        if (
            not isinstance(start_checkpoint, str)
            or not start_checkpoint.strip()
            or start_checkpoint != str(Path(start_checkpoint).expanduser().resolve())
        ):
            raise ValueError(
                "continuation checkpoint provenance has a non-canonical parent path"
            )
        for field, digest in (
            ("start_checkpoint_sha256", start_checkpoint_sha256),
            (
                "start_checkpoint_controlled_provenance_sha256",
                start_provenance_sha256,
            ),
        ):
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError(
                    f"continuation checkpoint provenance {field} is not a lowercase SHA256"
                )

    pseudo_manifest_path = value.get("pseudo_manifest_path")
    pseudo_manifest_sha256 = value.get("pseudo_manifest_sha256")
    pseudo_camera_pool_sha256 = value.get("pseudo_camera_pool_sha256")
    pseudo_target_kind = value.get("pseudo_target_kind")
    pseudo_strict = value.get("pseudo_rgb_strict_cache")
    if not isinstance(pseudo_strict, bool):
        raise ValueError("controlled checkpoint pseudo_rgb_strict_cache must be boolean")
    if role in {"A0", "A1"}:
        if (
            any(
                item is not None
                for item in (
                    pseudo_manifest_path,
                    pseudo_manifest_sha256,
                    pseudo_camera_pool_sha256,
                    pseudo_target_kind,
                )
            )
            or pseudo_strict
        ):
            raise ValueError(
                f"{role} checkpoint provenance cannot declare pseudo supervision"
            )
    else:
        if (
            not isinstance(pseudo_manifest_path, str)
            or not pseudo_manifest_path.strip()
            or pseudo_manifest_path
            != str(Path(pseudo_manifest_path).expanduser().resolve())
        ):
            raise ValueError(
                "pseudo-supervised checkpoint provenance has a non-canonical manifest path"
            )
        for field, digest in (
            ("pseudo_manifest_sha256", pseudo_manifest_sha256),
            ("pseudo_camera_pool_sha256", pseudo_camera_pool_sha256),
        ):
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError(
                    f"pseudo-supervised checkpoint provenance {field} is not a lowercase SHA256"
                )
        expected_target_kind = "self_render_a0" if role == "SELFRENDER" else "difix"
        if pseudo_target_kind != expected_target_kind:
            raise ValueError(
                "pseudo-supervised checkpoint provenance has the wrong target kind"
            )
        if pseudo_strict is not True:
            raise ValueError(
                "pseudo-supervised checkpoint provenance requires strict cache validation"
            )
    strict = value.get("strict_geometry_protocol")
    if not isinstance(strict, Mapping) or not strict:
        raise ValueError("controlled checkpoint strict geometry protocol is invalid")
    strict_expected = {
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
    mismatched_strict = sorted(
        key for key, expected_value in strict_expected.items()
        if strict.get(key) != expected_value
    )
    if mismatched_strict:
        raise ValueError(
            "controlled checkpoint strict geometry protocol is off-protocol: "
            f"{mismatched_strict}"
        )
    return json.loads(
        json.dumps(dict(value), sort_keys=True, ensure_ascii=True, allow_nan=False)
    )


def controlled_checkpoint_provenance_sha256(value: Any) -> str:
    return _canonical_json_sha256(canonical_controlled_checkpoint_provenance(value))


def validate_controlled_checkpoint_provenance_files(
    value: Any,
) -> dict[str, Any]:
    provenance = canonical_controlled_checkpoint_provenance(value)
    scene_path = Path(provenance["scene_source_path"])
    track_path = Path(provenance["track_h5_path"])
    images_dir = Path(provenance["source_images_dir"])
    if not scene_path.is_dir():
        raise FileNotFoundError(f"Controlled scene root is missing: {scene_path}")
    if not track_path.is_file() or _sha256_file(track_path) != provenance[
        "track_h5_sha256"
    ]:
        raise ValueError("Controlled checkpoint Track H5 is missing or changed")
    if not images_dir.is_dir():
        raise FileNotFoundError(
            f"Controlled source-image directory is missing: {images_dir}"
        )
    try:
        images_dir.relative_to(scene_path)
    except ValueError as exc:
        raise ValueError("Controlled source-image directory is outside the scene") from exc
    for item in provenance["source_image_inventory"]:
        image_path = Path(item["path"])
        try:
            image_path.relative_to(images_dir)
        except ValueError as exc:
            raise ValueError(
                f"Controlled source image is outside its declared root: {image_path}"
            ) from exc
        if not image_path.is_file() or _sha256_file(image_path) != item["sha256"]:
            raise ValueError(
                f"Controlled checkpoint source image is missing or changed: {image_path}"
            )
    if provenance["role"] != "A0":
        start_checkpoint = Path(provenance["start_checkpoint"])
        if (
            not start_checkpoint.is_file()
            or _sha256_file(start_checkpoint)
            != provenance["start_checkpoint_sha256"]
        ):
            raise ValueError("Controlled parent checkpoint is missing or changed")
    if provenance["role"] in {"SELFRENDER", "B"}:
        pseudo_manifest = Path(provenance["pseudo_manifest_path"])
        if (
            not pseudo_manifest.is_file()
            or _sha256_file(pseudo_manifest)
            != provenance["pseudo_manifest_sha256"]
        ):
            raise ValueError("Controlled pseudo manifest is missing or changed")
    return provenance


def validate_controlled_checkpoint_provenance_inputs(
    value: Any,
    *,
    scene_source_path: str | Path,
    track_h5_path: str | Path,
    source_images_dir: str | Path,
    source_camera_names: Sequence[str],
    source_training_camera_inventory: Any,
    seed: int,
) -> dict[str, Any]:
    """Re-read current inputs and compare them with checkpoint-embedded provenance."""
    provenance = validate_controlled_checkpoint_provenance_files(value)
    scene_path = Path(scene_source_path).expanduser().resolve()
    track_path = Path(track_h5_path).expanduser().resolve()
    images_dir = Path(source_images_dir).expanduser().resolve()
    if not scene_path.is_dir():
        raise FileNotFoundError(f"Controlled scene root is missing: {scene_path}")
    if not track_path.is_file():
        raise FileNotFoundError(f"Controlled Track H5 is missing: {track_path}")
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Controlled source-image directory is missing: {images_dir}")
    try:
        images_dir.relative_to(scene_path)
    except ValueError as exc:
        raise ValueError("Controlled source-image directory is outside the scene") from exc
    names = normalized_camera_names(source_camera_names)
    images = build_source_image_inventory(images_dir, names)
    cameras = canonical_training_camera_inventory(source_training_camera_inventory)
    observed = {
        "scene_source_path": str(scene_path),
        "seed": int(seed),
        "track_h5_path": str(track_path),
        "track_h5_sha256": _sha256_file(track_path),
        "source_camera_names": names,
        "source_camera_set_sha256": source_camera_set_sha256(names),
        "source_images_dir": str(images_dir),
        "source_image_inventory": images,
        "source_image_inventory_sha256": source_image_inventory_sha256(images),
        "source_training_camera_inventory": cameras,
        "source_training_camera_inventory_sha256": training_camera_inventory_sha256(
            cameras
        ),
    }
    mismatched = sorted(
        field for field, actual in observed.items()
        if provenance.get(field) != actual
    )
    if mismatched:
        raise ValueError(
            "Checkpoint-embedded controlled provenance differs from current inputs: "
            f"{mismatched}"
        )
    return provenance


def controlled_checkpoint_provenance_from_metadata(
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    role = str(metadata.get("role", "")).upper()
    payload = {
        "schema": CONTROLLED_CHECKPOINT_PROVENANCE_SCHEMA,
        "role": role,
        "final_iteration": metadata.get("final_iteration"),
        "controlled_training_protocol": metadata.get("controlled_training_protocol"),
        "controlled_training_protocol_sha256": metadata.get(
            "controlled_training_protocol_sha256"
        ),
        **{
            field: metadata.get(field)
            for field in CONTROLLED_CHECKPOINT_INPUT_FIELDS
        },
        **{
            field: metadata.get(field)
            for field in CONTROLLED_CHECKPOINT_LINEAGE_FIELDS
        },
        **{
            field: metadata.get(field)
            for field in CONTROLLED_CHECKPOINT_PSEUDO_FIELDS
        },
    }
    return canonical_controlled_checkpoint_provenance(payload)


def validate_a0_checkpoint_provenance(
    a0_value: Any, continuation_value: Any
) -> dict[str, Any]:
    """Prove that a continuation checkpoint came from the current A0 inputs."""
    a0 = canonical_controlled_checkpoint_provenance(a0_value)
    continuation = canonical_controlled_checkpoint_provenance(continuation_value)
    if a0["role"] != "A0":
        raise ValueError("Continuation checkpoint provenance is not an A0 run")
    if continuation["role"] not in {"A1", "SELFRENDER", "B"}:
        raise ValueError("Current controlled provenance is not a continuation")
    if (
        continuation["start_checkpoint_controlled_provenance_sha256"]
        != controlled_checkpoint_provenance_sha256(a0)
    ):
        raise ValueError(
            "Continuation lineage does not identify the supplied A0 provenance"
        )
    mismatched = [
        field
        for field in CONTROLLED_CHECKPOINT_INPUT_FIELDS
        if a0[field] != continuation[field]
    ]
    if mismatched:
        raise ValueError(
            "A0 checkpoint provenance differs from continuation inputs: "
            f"{mismatched}"
        )
    if controlled_phase_common_training_protocol(
        a0["controlled_training_protocol"]
    ) != controlled_phase_common_training_protocol(
        continuation["controlled_training_protocol"]
    ):
        raise ValueError(
            "A0 checkpoint and continuation differ outside the authorized phase "
            "transition fields"
        )
    return a0


def controlled_pair_payload(metadata: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "scene_source_path",
        "seed",
        "start_checkpoint_sha256",
        "track_h5_sha256",
        "source_camera_set_sha256",
        "checkpoint_iteration",
        "final_iteration",
        "strict_geometry_protocol",
        "controlled_training_protocol_sha256",
        "source_image_inventory_sha256",
        "source_training_camera_inventory_sha256",
        "start_checkpoint_controlled_provenance_sha256",
    )
    missing = [
        field
        for field in fields
        if metadata.get(field) is None or metadata.get(field) == ""
    ]
    if missing:
        raise ValueError(f"Controlled pair identity lacks fields: {missing}")
    return {field: metadata[field] for field in fields}


def controlled_pair_id(metadata: dict[str, Any]) -> str:
    serialized = json.dumps(
        controlled_pair_payload(metadata), sort_keys=True, separators=(",", ":")
    )
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return f"controlled_pair_{digest[:24]}"
