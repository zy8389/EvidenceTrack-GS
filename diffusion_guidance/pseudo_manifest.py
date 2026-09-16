from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from .camera_utils import canonical_camera_payload, camera_fingerprint_from_payload
from .checkpoint_state import RENDER_STATE_SCHEMA, load_checkpoint_summary
from .control_identity import (
    normalized_camera_names,
    source_camera_set_sha256,
    validate_controlled_checkpoint_provenance_files,
)
from .difix_provenance import (
    A0_PSEUDO_MANIFEST_SCHEMA,
    load_difix_run_metadata,
    sha256_file,
    validate_difix_target,
    validate_self_render_target,
)

_A0_SOURCE_FIELDS = (
    "manifest_schema",
    "input_sha256",
    "reference_image_sha256",
    "a0_checkpoint",
    "a0_checkpoint_sha256",
    "a0_checkpoint_iteration",
    "a0_checkpoint_state_format",
    "a0_checkpoint_gaussian_count",
    "a0_checkpoint_render_state_schema",
    "a0_checkpoint_render_state_sha256",
    "a0_checkpoint_controlled_provenance_sha256",
    "a0_render_state_source",
    "live_pseudo_audit",
    "live_pseudo_audit_sha256",
    "projection_context_fingerprint",
    "track_h5",
    "track_h5_sha256",
    "source_camera_names",
    "source_camera_set_sha256",
)


@dataclass(frozen=True)
class PseudoRecord:
    key: str
    camera: Dict[str, Any]
    input_path: Path
    target_path: Path
    reference_path: Path
    mask_path: Optional[Path]
    raw: Dict[str, Any]
    run_metadata: Optional[Dict[str, Any]] = None


def _resolve(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return tuple(map(int, image.size))


def _require_sha256(value: Any, label: str) -> str:
    value = str(value)
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value.lower()):
        raise ValueError(f"{label} must be a SHA256 digest")
    return value.lower()


def validate_a0_source_provenance(record: "PseudoRecord") -> None:
    """Verify the immutable A0 assets behind a strict pseudo record.

    Difix sidecars prove the cache output is tied to its declared input.  This
    additional check proves that the declared input was exported from the
    shared complete A0 checkpoint, rather than a same-path PLY replacement.
    """
    raw = record.raw
    missing = [field for field in _A0_SOURCE_FIELDS if raw.get(field) in (None, "")]
    if missing:
        raise ValueError(f"Pseudo manifest lacks immutable A0 provenance: {missing}")
    if raw["manifest_schema"] != A0_PSEUDO_MANIFEST_SCHEMA:
        raise ValueError("Pseudo supervision manifest uses the wrong schema")
    input_size = _image_size(record.input_path)
    expected_size = (int(record.camera["width"]), int(record.camera["height"]))
    if input_size != expected_size:
        raise ValueError(
            "A0 pseudo input resolution differs from its manifest camera domain: "
            f"input={input_size}, camera={expected_size}"
        )
    if _image_size(record.reference_path)[0] <= 0:
        raise ValueError("Pseudo source reference has an invalid resolution")
    if raw["input_sha256"] != sha256_file(record.input_path):
        raise ValueError("A0 pseudo input hash differs from the frozen manifest")
    if raw["reference_image_sha256"] != sha256_file(record.reference_path):
        raise ValueError("Pseudo source reference hash differs from the frozen manifest")
    checkpoint_path = Path(str(raw["a0_checkpoint"])).expanduser().resolve()
    live_audit_path = Path(str(raw["live_pseudo_audit"])).expanduser().resolve()
    track_path = Path(str(raw["track_h5"])).expanduser().resolve()
    for label, path in (
        ("A0 checkpoint", checkpoint_path),
        ("live pseudo-camera audit", live_audit_path),
        ("strict Track H5", track_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label}: {path}")
    if raw["a0_checkpoint_sha256"] != sha256_file(checkpoint_path):
        raise ValueError("A0 checkpoint hash differs from the frozen manifest")
    if raw["live_pseudo_audit_sha256"] != sha256_file(live_audit_path):
        raise ValueError("Live pseudo-camera audit hash differs from the frozen manifest")
    if raw["track_h5_sha256"] != sha256_file(track_path):
        raise ValueError("Track H5 hash differs from the frozen pseudo manifest")
    source_names = normalized_camera_names(raw["source_camera_names"])
    if source_camera_set_sha256(source_names) != raw["source_camera_set_sha256"]:
        raise ValueError("Pseudo manifest source-camera set hash is inconsistent")
    if int(raw["a0_checkpoint_iteration"]) != 10000:
        raise ValueError("First controlled pseudo pool must be bound to A0 iteration 10000")
    if raw["a0_checkpoint_state_format"] != "geotrack-research-v2":
        raise ValueError("Pseudo pool requires a complete geotrack-research-v2 A0 checkpoint")
    if int(raw["a0_checkpoint_gaussian_count"]) <= 0:
        raise ValueError("A0 checkpoint Gaussian count is invalid")
    checkpoint_summary = load_checkpoint_summary(
        checkpoint_path,
        expected_iteration=10000,
        require_cuda_rng=True,
        require_controlled_provenance=True,
    )
    checkpoint_provenance = validate_controlled_checkpoint_provenance_files(
        checkpoint_summary["controlled_provenance"]
    )
    if (
        checkpoint_provenance["role"] != "A0"
        or checkpoint_provenance["track_h5_path"] != str(track_path)
        or checkpoint_provenance["track_h5_sha256"] != raw["track_h5_sha256"]
        or checkpoint_provenance["source_camera_names"] != source_names
        or checkpoint_provenance["source_camera_set_sha256"]
        != raw["source_camera_set_sha256"]
    ):
        raise ValueError(
            "Pseudo manifest inputs differ from checkpoint-embedded A0 provenance"
        )
    if checkpoint_summary["gaussian_count"] != int(
        raw["a0_checkpoint_gaussian_count"]
    ):
        raise ValueError("A0 checkpoint Gaussian count differs from the manifest")
    declared_render_sha256 = _require_sha256(
        raw["a0_checkpoint_render_state_sha256"],
        "a0_checkpoint_render_state_sha256",
    )
    if raw["a0_checkpoint_render_state_schema"] != RENDER_STATE_SCHEMA:
        raise ValueError("A0 pseudo manifest uses an unsupported render-state schema")
    if declared_render_sha256 != checkpoint_summary["render_state_sha256"]:
        raise ValueError("A0 pseudo render-state hash differs from the checkpoint tensors")
    declared_provenance_sha256 = _require_sha256(
        raw["a0_checkpoint_controlled_provenance_sha256"],
        "a0_checkpoint_controlled_provenance_sha256",
    )
    if declared_provenance_sha256 != checkpoint_summary[
        "controlled_provenance_sha256"
    ]:
        raise ValueError(
            "A0 pseudo provenance hash differs from the checkpoint-embedded provenance"
        )
    if (
        raw["a0_render_state_source"]
        != "complete_checkpoint_restore_after_scene_ply_load"
    ):
        raise ValueError("A0 pseudo render-state provenance is incomplete")
    live_audit = json.loads(live_audit_path.read_text(encoding="utf-8"))
    if not isinstance(live_audit, dict) or live_audit.get("passed") is not True:
        raise ValueError("Pseudo manifest is bound to a failed live pseudo-camera audit")
    if live_audit.get("projection_context_fingerprint") != raw[
        "projection_context_fingerprint"
    ]:
        raise ValueError("Pseudo manifest/live audit projection context mismatch")
    if live_audit.get("track_h5_sha256") not in (None, raw["track_h5_sha256"]):
        raise ValueError("Pseudo manifest/live audit Track H5 mismatch")
    live_keys = {
        str(item.get("camera_fingerprint"))
        for item in live_audit.get("records", [])
        if isinstance(item, dict)
    }
    if record.key not in live_keys:
        raise ValueError("Pseudo manifest camera is absent from its live-camera audit")


def validate_frozen_target(
    record: "PseudoRecord", *, run_metadata: Optional[Dict[str, Any]] = None
) -> None:
    """Fail closed unless the declared immutable supervision target is provenance-bound."""
    validate_a0_source_provenance(record)
    target_kind = str(record.raw.get("supervision_target_kind", "difix")).lower()
    for label, path in (("GS input", record.input_path), ("reference", record.reference_path), ("pseudo target", record.target_path)):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label}: {path}")
    if target_kind == "self_render_a0":
        required = ("input_sha256", "target_sha256")
        missing = [field for field in required if not record.raw.get(field)]
        if missing:
            raise ValueError(f"SelfRender manifest lacks frozen A0 hashes: {missing}")
        validate_self_render_target(
            input_path=record.input_path,
            target_path=record.target_path,
            camera_resolution=(int(record.camera["width"]), int(record.camera["height"])),
            input_sha256=str(record.raw["input_sha256"]),
            target_sha256=str(record.raw["target_sha256"]),
        )
        return
    if target_kind != "difix":
        raise ValueError(f"Unknown pseudo supervision target kind: {target_kind}")
    if run_metadata is None:
        raise ValueError("Difix run metadata is required for strict target validation")
    validate_difix_target(
        target_path=record.target_path,
        input_path=record.input_path,
        reference_path=record.reference_path,
        camera_fingerprint=record.key,
        camera_resolution=(int(record.camera["width"]), int(record.camera["height"])),
        run_metadata=run_metadata,
        require_reproducibility=True,
    )


def pseudo_view_index(supervision_call_count: int, pool_size: int) -> int:
    if pool_size <= 0:
        raise ValueError("pseudo-view pool must be non-empty")
    if supervision_call_count < 0:
        raise ValueError("supervision_call_count must be non-negative")
    return int(supervision_call_count) % int(pool_size)


def load_pseudo_manifest(path: str | Path, strict_targets: bool = False) -> List[PseudoRecord]:
    manifest = Path(path).expanduser().resolve()
    records: List[PseudoRecord] = []
    seen_keys: set[str] = set()
    with manifest.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {manifest}:{line_number}") from exc
            camera = canonical_camera_payload(raw["camera"])
            expected_key = camera_fingerprint_from_payload(camera)
            key = str(raw.get("key", raw.get("camera_key", expected_key)))
            if key != expected_key:
                raise ValueError(f"Camera fingerprint mismatch at {manifest}:{line_number}: record={key}, computed={expected_key}")
            if key in seen_keys:
                raise ValueError(f"Duplicate pseudo camera fingerprint: {key}")
            seen_keys.add(key)
            reference = raw.get("reference_image", raw.get("ref"))
            if reference is None:
                raise ValueError(f"Missing reference image at {manifest}:{line_number}")
            record = PseudoRecord(
                key=key, camera=camera,
                input_path=_resolve(manifest.parent, raw["input"]),
                target_path=_resolve(manifest.parent, raw["target"]),
                reference_path=_resolve(manifest.parent, reference),
                mask_path=_resolve(manifest.parent, raw["mask"]) if raw.get("mask") else None,
                raw=raw,
            )
            records.append(record)
    if not records:
        raise RuntimeError(f"Pseudo manifest is empty: {manifest}")
    if strict_targets:
        difix_records = [
            record
            for record in records
            if str(record.raw.get("supervision_target_kind", "difix")).lower() == "difix"
        ]
        run_metadata = None
        if difix_records:
            if len(difix_records) != len(records):
                raise ValueError("A strict pseudo manifest cannot mix Difix and SelfRender targets")
            run_metadata = load_difix_run_metadata(
                manifest,
                expected_record_count=len(records),
                require_reproducibility=True,
            )
        records = [
            PseudoRecord(
                key=record.key,
                camera=record.camera,
                input_path=record.input_path,
                target_path=record.target_path,
                reference_path=record.reference_path,
                mask_path=record.mask_path,
                raw=record.raw,
                run_metadata=run_metadata,
            )
            for record in records
        ]
        for record in records:
            validate_frozen_target(record, run_metadata=record.run_metadata)
    return records


def build_pseudo_camera(record: PseudoRecord):
    from scene.cameras import PseudoCamera
    camera = PseudoCamera(
        R=np.asarray(record.camera["R"], dtype=np.float32), T=np.asarray(record.camera["T"], dtype=np.float32),
        FoVx=float(record.camera["FoVx"]), FoVy=float(record.camera["FoVy"]),
        width=int(record.camera["width"]), height=int(record.camera["height"]),
    )
    if "intrinsics" in record.camera:
        from diffusion_guidance.calibration_guard import apply_intrinsics
        apply_intrinsics(camera, record.camera["intrinsics"])
    return camera


def _image_tensor(path: Path, mode: str) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert(mode), dtype=np.float32).copy() / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous() if mode == "RGB" else torch.from_numpy(array)[None].contiguous()


def _resize(value: torch.Tensor, size: Tuple[int, int], mode: str) -> torch.Tensor:
    if tuple(value.shape[-2:]) == tuple(size):
        return value
    kwargs = {"size": size, "mode": mode}
    if mode in {"bilinear", "bicubic"}:
        kwargs["align_corners"] = False
    return F.interpolate(value[None], **kwargs)[0]


def load_pseudo_target(record: PseudoRecord, *, size: Tuple[int, int], device: torch.device | str, dtype: torch.dtype) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    # Revalidate at every actual cache read.  Start-up validation alone cannot
    # protect a long B/SelfRender continuation from a changed immutable asset.
    validate_frozen_target(record, run_metadata=record.run_metadata)
    target_size = _image_size(record.target_path)
    expected_size = (int(record.camera["width"]), int(record.camera["height"]))
    if target_size != expected_size:
        raise ValueError(
            "Frozen pseudo target dimensions differ from its manifest camera domain: "
            f"target={target_size}, camera={expected_size}"
        )
    image = _resize(_image_tensor(record.target_path, "RGB"), size, "bilinear").to(device=device, dtype=dtype, non_blocking=True)
    mask = None
    if record.mask_path is not None:
        if not record.mask_path.exists():
            raise FileNotFoundError(f"Pseudo supervision mask is declared but missing: {record.mask_path}")
        mask = _resize(_image_tensor(record.mask_path, "L"), size, "nearest").to(device=device, dtype=dtype, non_blocking=True)
    return image, mask
