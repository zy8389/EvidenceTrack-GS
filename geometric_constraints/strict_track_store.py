"""Strict source-only track storage and leakage auditing.

The strict format is deliberately incompatible with legacy upstream H5 files.
A strict file proves its anchors, covariance, quality and point-cloud
initialisation were all derived only from the declared source observations.
Held-out observations may be retained only as evaluation labels.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, List

import h5py
import numpy as np


FORMAT_VERSION = "geotrack_anchor_v4_source_rgb_membership"
STRICT_PROTOCOL = "strict_source_only"
ANCHOR_PROVENANCE = "source_observations_dlt_refined"
POINT_CLOUD_PROVENANCE = "source_only_track_anchors_and_source_rgb"
TRACK_MEMBERSHIP_PROVENANCE = (
    "source_rgb_sift_mutual_ratio_epipolar_conflict_free_union"
)


def normalize_image_name(value: str) -> str:
    """Return a stable camera key independent of extension and path style."""
    return Path(str(value).replace("\\", "/")).stem


def _decode_array(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [value.decode("utf-8") if isinstance(value, bytes) else str(value)
         for value in values],
        dtype=object,
    )


def _decode_attr(value) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


@dataclass(frozen=True)
class LeakageAudit:
    source_images: List[str]
    heldout_images: List[str]
    anchor_source_observation_count: int
    heldout_used_for_anchor: int
    heldout_used_for_quality: int
    heldout_used_for_source_features: int
    track_membership_provenance: str
    point_cloud_provenance: str
    source_scene_scale: float

    @property
    def passed(self) -> bool:
        return (
            self.anchor_source_observation_count > 0
            and self.heldout_used_for_anchor == 0
            and self.heldout_used_for_quality == 0
            and self.heldout_used_for_source_features == 0
            and self.track_membership_provenance
            == TRACK_MEMBERSHIP_PROVENANCE
            and self.point_cloud_provenance == POINT_CLOUD_PROVENANCE
            and np.isfinite(self.source_scene_scale)
            and self.source_scene_scale > 0.0
        )

    def emit(self, prefix: str = "[Strict Geometry Leakage Audit]") -> None:
        print(f"{prefix} Source images: {', '.join(self.source_images)}")
        print(f"{prefix} Held-out images: {', '.join(self.heldout_images)}")
        print(
            f"{prefix} Track anchor source observation count: "
            f"{self.anchor_source_observation_count}"
        )
        print(
            f"{prefix} Held-out observations used for anchor = "
            f"{self.heldout_used_for_anchor}"
        )
        print(
            f"{prefix} Held-out observations used for quality = "
            f"{self.heldout_used_for_quality}"
        )
        print(
            f"{prefix} Held-out observations used for source features = "
            f"{self.heldout_used_for_source_features}"
        )
        print(
            f"{prefix} Track membership provenance: "
            f"{self.track_membership_provenance}"
        )
        print(f"{prefix} Point cloud provenance: {self.point_cloud_provenance}")
        print(f"{prefix} Source-only scene scale: {self.source_scene_scale:.9g}")

    def require_pass(self) -> None:
        if not self.passed:
            raise RuntimeError(
                "Strict source-only geometry leakage audit failed; training is blocked"
            )


@dataclass(frozen=True)
class StrictTrackStore:
    path: Path
    track_ids: np.ndarray
    xyz: np.ndarray
    rgb_source: np.ndarray
    covariance: np.ndarray
    quality: np.ndarray
    source_reprojection_error: np.ndarray
    observation_offsets: np.ndarray
    observation_xy: np.ndarray
    observation_image_ids: np.ndarray
    observation_image_names: np.ndarray
    observation_confidence: np.ndarray
    use_for_anchor: np.ndarray
    use_for_quality: np.ndarray
    use_for_source_features: np.ndarray
    camera_image_ids: np.ndarray
    camera_image_names: np.ndarray
    camera_models: np.ndarray
    camera_width: np.ndarray
    camera_height: np.ndarray
    camera_fx: np.ndarray
    camera_fy: np.ndarray
    camera_cx: np.ndarray
    camera_cy: np.ndarray
    camera_rotation_w2c: np.ndarray
    camera_translation_w2c: np.ndarray
    source_images: List[str]
    heldout_images: List[str]
    anchor_provenance: str
    track_membership_provenance: str
    point_cloud_provenance: str
    source_scene_scale: float

    def __len__(self) -> int:
        return int(self.track_ids.shape[0])

    @classmethod
    def load(cls, path: str | Path) -> "StrictTrackStore":
        path = Path(path).expanduser().resolve()
        with h5py.File(path, "r") as handle:
            version = _decode_attr(handle.attrs.get("format_version", ""))
            protocol = _decode_attr(handle.attrs.get("geometry_protocol", ""))
            if version != FORMAT_VERSION or protocol != STRICT_PROTOCOL:
                raise ValueError(
                    f"{path} is not a strict source-only track file: "
                    f"format={version!r}, protocol={protocol!r}; expected "
                    f"{FORMAT_VERSION!r}/{STRICT_PROTOCOL!r}"
                )
            source_images = json.loads(
                _decode_attr(handle.attrs["source_images_json"])
            )
            heldout_images = json.loads(
                _decode_attr(handle.attrs["heldout_images_json"])
            )
            values = dict(
                path=path,
                track_ids=handle["tracks/id"][:],
                xyz=handle["tracks/xyz"][:],
                rgb_source=handle["tracks/rgb_source"][:],
                covariance=handle["tracks/covariance"][:],
                quality=handle["tracks/quality"][:],
                source_reprojection_error=handle[
                    "tracks/source_reprojection_error"
                ][:],
                observation_offsets=handle["observations/offsets"][:],
                observation_xy=handle["observations/xy"][:],
                observation_image_ids=handle["observations/image_id"][:],
                observation_image_names=_decode_array(
                    handle["observations/image_name"][:]
                ),
                observation_confidence=handle["observations/confidence"][:],
                use_for_anchor=handle["observations/use_for_anchor"][:].astype(bool),
                use_for_quality=handle["observations/use_for_quality"][:].astype(bool),
                use_for_source_features=handle[
                    "observations/use_for_source_features"
                ][:].astype(bool),
                camera_image_ids=handle["cameras/image_id"][:],
                camera_image_names=_decode_array(handle["cameras/image_name"][:]),
                camera_models=_decode_array(handle["cameras/model"][:]),
                camera_width=handle["cameras/width"][:],
                camera_height=handle["cameras/height"][:],
                camera_fx=handle["cameras/fx"][:],
                camera_fy=handle["cameras/fy"][:],
                camera_cx=handle["cameras/cx"][:],
                camera_cy=handle["cameras/cy"][:],
                camera_rotation_w2c=handle["cameras/rotation_w2c"][:],
                camera_translation_w2c=handle["cameras/translation_w2c"][:],
                source_images=[normalize_image_name(v) for v in source_images],
                heldout_images=[normalize_image_name(v) for v in heldout_images],
                anchor_provenance=_decode_attr(handle.attrs["anchor_provenance"]),
                track_membership_provenance=_decode_attr(
                    handle.attrs["track_membership_provenance"]
                ),
                point_cloud_provenance=_decode_attr(
                    handle.attrs["point_cloud_provenance"]
                ),
                source_scene_scale=float(handle.attrs["source_scene_scale"]),
            )
        store = cls(**values)
        store.validate()
        return store

    def observations_for(self, track_index: int) -> Dict[str, np.ndarray]:
        if not 0 <= track_index < len(self):
            raise IndexError(track_index)
        start = int(self.observation_offsets[track_index])
        end = int(self.observation_offsets[track_index + 1])
        return {
            "xy": self.observation_xy[start:end],
            "image_id": self.observation_image_ids[start:end],
            "image_name": self.observation_image_names[start:end],
            "confidence": self.observation_confidence[start:end],
            "use_for_anchor": self.use_for_anchor[start:end],
            "use_for_quality": self.use_for_quality[start:end],
            "use_for_source_features": self.use_for_source_features[start:end],
        }

    def camera_calibration(self, image_name: str) -> Dict[str, np.ndarray | float | int | str]:
        key = normalize_image_name(image_name)
        keys = np.asarray([normalize_image_name(v) for v in self.camera_image_names])
        matches = np.flatnonzero(keys == key)
        if len(matches) != 1:
            raise KeyError(f"Expected one calibration for {image_name!r}, found {len(matches)}")
        index = int(matches[0])
        return {
            "image_id": int(self.camera_image_ids[index]),
            "image_name": str(self.camera_image_names[index]),
            "model": str(self.camera_models[index]),
            "width": int(self.camera_width[index]),
            "height": int(self.camera_height[index]),
            "fx": float(self.camera_fx[index]),
            "fy": float(self.camera_fy[index]),
            "cx": float(self.camera_cx[index]),
            "cy": float(self.camera_cy[index]),
            "rotation_w2c": self.camera_rotation_w2c[index],
            "translation_w2c": self.camera_translation_w2c[index],
        }

    def leakage_audit(self) -> LeakageAudit:
        source_keys = set(self.source_images)
        observation_keys = np.asarray(
            [normalize_image_name(v) for v in self.observation_image_names]
        )
        heldout_mask = np.asarray([key not in source_keys for key in observation_keys])
        return LeakageAudit(
            source_images=sorted(source_keys),
            heldout_images=sorted(set(self.heldout_images)),
            anchor_source_observation_count=int(self.use_for_anchor.sum()),
            heldout_used_for_anchor=int(np.logical_and(heldout_mask, self.use_for_anchor).sum()),
            heldout_used_for_quality=int(np.logical_and(heldout_mask, self.use_for_quality).sum()),
            heldout_used_for_source_features=int(
                np.logical_and(heldout_mask, self.use_for_source_features).sum()
            ),
            track_membership_provenance=self.track_membership_provenance,
            point_cloud_provenance=self.point_cloud_provenance,
            source_scene_scale=self.source_scene_scale,
        )

    def validate(self) -> None:
        k = len(self)
        m = int(self.observation_xy.shape[0])
        camera_count = int(self.camera_image_names.shape[0])
        shapes = {
            "track_ids": (self.track_ids.shape, (k,)),
            "xyz": (self.xyz.shape, (k, 3)),
            "rgb_source": (self.rgb_source.shape, (k, 3)),
            "covariance": (self.covariance.shape, (k, 3, 3)),
            "quality": (self.quality.shape, (k,)),
            "source_reprojection_error": (
                self.source_reprojection_error.shape,
                (k,),
            ),
            "observation_offsets": (self.observation_offsets.shape, (k + 1,)),
            "observation_xy": (self.observation_xy.shape, (m, 2)),
            "observation_image_ids": (self.observation_image_ids.shape, (m,)),
            "observation_image_names": (self.observation_image_names.shape, (m,)),
            "observation_confidence": (self.observation_confidence.shape, (m,)),
            "use_for_anchor": (self.use_for_anchor.shape, (m,)),
            "use_for_quality": (self.use_for_quality.shape, (m,)),
            "use_for_source_features": (self.use_for_source_features.shape, (m,)),
            "camera_image_ids": (self.camera_image_ids.shape, (camera_count,)),
            "camera_models": (self.camera_models.shape, (camera_count,)),
            "camera_width": (self.camera_width.shape, (camera_count,)),
            "camera_height": (self.camera_height.shape, (camera_count,)),
            "camera_fx": (self.camera_fx.shape, (camera_count,)),
            "camera_fy": (self.camera_fy.shape, (camera_count,)),
            "camera_cx": (self.camera_cx.shape, (camera_count,)),
            "camera_cy": (self.camera_cy.shape, (camera_count,)),
            "camera_rotation_w2c": (
                self.camera_rotation_w2c.shape,
                (camera_count, 3, 3),
            ),
            "camera_translation_w2c": (
                self.camera_translation_w2c.shape,
                (camera_count, 3),
            ),
        }
        for name, (actual, expected) in shapes.items():
            if actual != expected:
                raise ValueError(f"{name} has shape {actual}, expected {expected}")
        if k == 0:
            raise ValueError("Strict source-only track file is empty")
        if camera_count == 0:
            raise ValueError("Strict source-only track file has no cameras")
        if len(set(map(int, self.track_ids.tolist()))) != k:
            raise ValueError("Track IDs must be unique")
        if int(self.observation_offsets[0]) != 0 or int(self.observation_offsets[-1]) != m:
            raise ValueError("Observation offsets are inconsistent")
        offset_differences = np.diff(self.observation_offsets.astype(np.int64))
        if np.any(offset_differences <= 0):
            raise ValueError("Observation offsets must be strictly increasing")
        if self.anchor_provenance != ANCHOR_PROVENANCE:
            raise ValueError(f"Invalid anchor provenance: {self.anchor_provenance!r}")
        if self.track_membership_provenance != TRACK_MEMBERSHIP_PROVENANCE:
            raise ValueError(
                "Strict source-only Track membership must be built exclusively "
                "from source RGB source-source matches; got "
                f"{self.track_membership_provenance!r}"
            )
        if not np.isfinite(self.source_scene_scale) or self.source_scene_scale <= 0.0:
            raise ValueError("source_scene_scale must be finite and positive")
        for name, array in (
            ("xyz", self.xyz),
            ("rgb_source", self.rgb_source),
            ("covariance", self.covariance),
            ("quality", self.quality),
            ("source_reprojection_error", self.source_reprojection_error),
            ("observation_xy", self.observation_xy),
            ("observation_confidence", self.observation_confidence),
            ("camera_fx", self.camera_fx),
            ("camera_fy", self.camera_fy),
            ("camera_cx", self.camera_cx),
            ("camera_cy", self.camera_cy),
            ("camera_rotation_w2c", self.camera_rotation_w2c),
            ("camera_translation_w2c", self.camera_translation_w2c),
        ):
            if not np.isfinite(array).all():
                raise ValueError(f"{name} contains non-finite values")
        if np.any(self.quality < 0.0) or np.any(self.quality > 1.0):
            raise ValueError("Track quality must lie in [0, 1]")
        if np.any(self.rgb_source < 0.0) or np.any(self.rgb_source > 1.0):
            raise ValueError("Source RGB must lie in [0, 1]")
        if np.any(self.source_reprojection_error < 0.0):
            raise ValueError("Source reprojection errors must be non-negative")
        if np.any(self.observation_confidence < 0.0) or np.any(
            self.observation_confidence > 1.0
        ):
            raise ValueError("Observation confidence must lie in [0, 1]")

        if len(self.source_images) != len(set(self.source_images)):
            raise ValueError("Source image list contains duplicates")
        if len(self.heldout_images) != len(set(self.heldout_images)):
            raise ValueError("Held-out image list contains duplicates")
        source_keys = set(self.source_images)
        heldout_keys = set(self.heldout_images)
        overlap = source_keys & heldout_keys
        if overlap:
            raise ValueError(f"Source and held-out image lists overlap: {sorted(overlap)}")

        camera_keys = [normalize_image_name(value) for value in self.camera_image_names]
        if len(camera_keys) != len(set(camera_keys)):
            raise ValueError("Camera names must be unique after normalization")
        if len(set(map(int, self.camera_image_ids.tolist()))) != camera_count:
            raise ValueError("Camera image IDs must be unique")
        if any(not str(model).strip() for model in self.camera_models):
            raise ValueError("Camera model names must be non-empty")
        if np.any(self.camera_width <= 0) or np.any(self.camera_height <= 0):
            raise ValueError("Camera dimensions must be positive")
        if np.any(self.camera_fx <= 0.0) or np.any(self.camera_fy <= 0.0):
            raise ValueError("Camera focal lengths must be positive")
        if np.any(self.camera_cx < -0.5) or np.any(
            self.camera_cx > self.camera_width - 0.5
        ):
            raise ValueError("Camera cx lies outside the pixel domain")
        if np.any(self.camera_cy < -0.5) or np.any(
            self.camera_cy > self.camera_height - 0.5
        ):
            raise ValueError("Camera cy lies outside the pixel domain")
        identity = np.eye(3, dtype=np.float64)
        rotations = self.camera_rotation_w2c.astype(np.float64)
        if not np.allclose(
            rotations @ np.swapaxes(rotations, 1, 2),
            identity[None],
            atol=1e-5,
            rtol=1e-5,
        ) or np.any(np.linalg.det(rotations) <= 0.0):
            raise ValueError("Camera rotations must be finite proper orthonormal matrices")
        required_camera_keys = source_keys | heldout_keys
        missing_cameras = required_camera_keys - set(camera_keys)
        if missing_cameras:
            raise ValueError(
                f"Source/held-out images lack camera calibration: {sorted(missing_cameras)}"
            )
        camera_index_by_name = {name: index for index, name in enumerate(camera_keys)}

        for track_index in range(k):
            observations = self.observations_for(track_index)
            if int(observations["use_for_anchor"].sum()) < 2:
                raise ValueError(f"Track {track_index} has fewer than two source observations")
            observation_keys = [
                normalize_image_name(name) for name in observations["image_name"]
            ]
            if len(observation_keys) != len(set(observation_keys)):
                raise ValueError(
                    f"Track {track_index} repeats an observation camera"
                )
            for offset, name in enumerate(observations["image_name"]):
                image_key = normalize_image_name(name)
                if image_key not in camera_index_by_name:
                    raise ValueError(
                        f"Track {track_index} observation {name!r} lacks calibration"
                    )
                camera_index = camera_index_by_name[image_key]
                observed_id = int(observations["image_id"][offset])
                if observed_id != int(self.camera_image_ids[camera_index]):
                    raise ValueError(
                        f"Track {track_index} observation {name!r} image ID disagrees with calibration"
                    )
                x, y = map(float, observations["xy"][offset])
                width = int(self.camera_width[camera_index])
                height = int(self.camera_height[camera_index])
                if not (0.0 <= x < width and 0.0 <= y < height):
                    raise ValueError(
                        f"Track {track_index} observation {name!r} lies outside {width}x{height}"
                    )
                is_source = image_key in source_keys
                for field in ("use_for_anchor", "use_for_quality", "use_for_source_features"):
                    if bool(observations[field][offset]) != is_source:
                        raise ValueError(
                            f"Track {track_index} observation {name!r} has invalid {field} flag"
                        )
        covariance = self.covariance.astype(np.float64)
        if not np.allclose(
            covariance, np.swapaxes(covariance, 1, 2), atol=1e-8, rtol=1e-6
        ):
            raise ValueError("Track covariance must be symmetric")
        eigvals = np.linalg.eigvalsh(covariance)
        if np.any(eigvals <= 0.0):
            raise ValueError("Track covariance must be positive definite")
        self.leakage_audit().require_pass()
