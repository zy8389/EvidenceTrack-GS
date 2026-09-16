"""Complete continuation checkpoints for controlled A0/A1/SelfRender/B runs."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import random
import math
from typing import Any

import numpy as np
import torch

from .control_identity import (
    canonical_controlled_checkpoint_provenance,
    controlled_checkpoint_provenance_sha256,
)


FORMAT = "geotrack-research-v2"
CORE = (
    "active_sh_degree",
    "max_sh_degree",
    "_xyz",
    "_features_dc",
    "_features_rest",
    "_scaling",
    "_rotation",
    "_opacity",
    "max_radii2D",
    "xyz_gradient_accum",
    "denom",
)
REQUIRED_STATE_FIELDS = (
    "format",
    "core",
    "spatial_lr_scale",
    "optimizer",
    "confidence",
    "init_point",
    "bg_color",
    "python_rng",
    "numpy_rng",
    "torch_rng",
    "cuda_rng",
)

# These are the tensors read by the renderer.  Keep the order and encoding
# stable: every producer and consumer of a render-state digest uses this
# single definition.
RENDER_STATE_SCHEMA = "renderer-visible-state-v1"
RENDER_STATE_FIELDS = (
    "_xyz",
    "_features_dc",
    "_features_rest",
    "_scaling",
    "_rotation",
    "_opacity",
)


def renderer_state_sha256(source: Any) -> str:
    """Return the canonical digest of a renderer-visible Gaussian state.

    ``source`` may be a complete checkpoint mapping (with a ``core`` mapping)
    or a live Gaussian model.  Raw tensor bytes are hashed after a contiguous
    CPU transfer, while dtype and shape are included in the canonical header.
    This prevents a metadata-only render-state claim from being accepted.
    """
    if isinstance(source, Mapping):
        core = source.get("core", source)
        if not isinstance(core, Mapping):
            raise ValueError("Renderer state mapping must contain a core mapping")
        getter = core.get
    else:
        core = None

        def getter(name: str, default=None):
            return getattr(source, name, default)

    active_degree = getter("active_sh_degree")
    if isinstance(active_degree, bool):
        raise ValueError("Renderer active_sh_degree is invalid")
    try:
        active_degree = int(active_degree)
    except (TypeError, ValueError) as exc:
        raise ValueError("Renderer active_sh_degree is invalid") from exc
    if active_degree < 0:
        raise ValueError("Renderer active_sh_degree must be non-negative")

    digest = hashlib.sha256()
    header = {
        "schema": RENDER_STATE_SCHEMA,
        "active_sh_degree": active_degree,
        "fields": list(RENDER_STATE_FIELDS),
    }
    digest.update(
        json.dumps(
            header, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
    )
    for name in RENDER_STATE_FIELDS:
        value = getter(name)
        if not torch.is_tensor(value):
            raise ValueError(f"Renderer state lacks tensor {name}")
        if value.dtype.is_floating_point and not bool(torch.isfinite(value).all()):
            raise ValueError(f"Renderer state tensor {name} is non-finite")
        tensor = value.detach().cpu().contiguous()
        field_header = {
            "name": name,
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "nbytes": int(tensor.numel() * tensor.element_size()),
        }
        digest.update(
            json.dumps(
                field_header, sort_keys=True, separators=(",", ":"), ensure_ascii=True
            ).encode("ascii")
        )
        # Viewing as uint8 avoids NumPy dtype limitations (for example
        # bfloat16) while preserving the exact serialized tensor bytes.
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _require_finite_tensor(value: Any, label: str) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise ValueError(f"Checkpoint {label} must be a tensor")
    if not value.dtype.is_floating_point:
        raise ValueError(f"Checkpoint {label} must use a floating-point dtype")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"Checkpoint {label} contains NaN or Inf")
    return value


def _require_shape(value: torch.Tensor, shape: tuple[int | None, ...], label: str) -> None:
    if value.ndim != len(shape) or any(
        expected is not None and int(observed) != expected
        for observed, expected in zip(value.shape, shape)
    ):
        raise ValueError(
            f"Checkpoint {label} has shape {tuple(value.shape)}, expected {shape}"
        )


def _validate_optimizer_state(
    optimizer: Any, core: Mapping[str, Any]
) -> dict[str, Any]:
    if (
        not isinstance(optimizer, Mapping)
        or "state" not in optimizer
        or "param_groups" not in optimizer
    ):
        raise ValueError("Checkpoint lacks a complete optimizer state_dict")
    states = optimizer["state"]
    groups = optimizer["param_groups"]
    if not isinstance(states, Mapping) or not states:
        raise ValueError("Checkpoint optimizer has no Adam state")
    if not isinstance(groups, list) or not groups:
        raise ValueError("Checkpoint optimizer has no parameter groups")

    tensor_for_group = {
        "xyz": core["_xyz"],
        "f_dc": core["_features_dc"],
        "f_rest": core["_features_rest"],
        "opacity": core["_opacity"],
        "scaling": core["_scaling"],
        "rotation": core["_rotation"],
    }
    required_names = set(tensor_for_group)
    observed_names: set[str] = set()
    observed_ids: set[Any] = set()
    for index, group in enumerate(groups):
        if not isinstance(group, Mapping):
            raise ValueError(f"Checkpoint optimizer param_groups[{index}] is invalid")
        name = group.get("name")
        params = group.get("params")
        if not isinstance(name, str) or not name:
            raise ValueError(f"Checkpoint optimizer param_groups[{index}] lacks a name")
        if name in observed_names:
            raise ValueError(f"Checkpoint optimizer repeats parameter group {name!r}")
        observed_names.add(name)
        if not isinstance(params, list) or len(params) != 1:
            raise ValueError(
                f"Checkpoint optimizer group {name!r} must contain one parameter"
            )
        parameter_id = params[0]
        if parameter_id in observed_ids:
            raise ValueError("Checkpoint optimizer repeats a serialized parameter ID")
        observed_ids.add(parameter_id)
        state = states.get(parameter_id)
        if not isinstance(state, Mapping):
            raise ValueError(f"Checkpoint optimizer lacks Adam state for group {name!r}")
        for field in ("step", "exp_avg", "exp_avg_sq"):
            if field not in state:
                raise ValueError(
                    f"Checkpoint optimizer group {name!r} lacks Adam field {field}"
                )
        exp_avg = _require_finite_tensor(
            state["exp_avg"], f"optimizer.{name}.exp_avg"
        )
        exp_avg_sq = _require_finite_tensor(
            state["exp_avg_sq"], f"optimizer.{name}.exp_avg_sq"
        )
        if exp_avg.shape != exp_avg_sq.shape:
            raise ValueError(f"Checkpoint optimizer moments disagree for group {name!r}")
        if name in tensor_for_group and exp_avg.shape != tensor_for_group[name].shape:
            raise ValueError(
                f"Checkpoint optimizer moment shape disagrees with group {name!r}"
            )
        step = state["step"]
        if torch.is_tensor(step):
            if step.numel() != 1 or not bool(torch.isfinite(step).all()):
                raise ValueError(f"Checkpoint optimizer step is invalid for group {name!r}")
            step_value = float(step.item())
        elif isinstance(step, (int, float)) and not isinstance(step, bool):
            step_value = float(step)
        else:
            raise ValueError(f"Checkpoint optimizer step is invalid for group {name!r}")
        if not math.isfinite(step_value) or step_value <= 0.0:
            raise ValueError(
                f"Checkpoint optimizer step must be finite and positive for group {name!r}"
            )
        for field, value in state.items():
            if torch.is_tensor(value) and (
                value.dtype.is_floating_point
                and not bool(torch.isfinite(value).all())
            ):
                raise ValueError(
                    f"Checkpoint optimizer group {name!r} field {field!r} is non-finite"
                )
    if not required_names.issubset(observed_names):
        raise ValueError(
            "Checkpoint optimizer lacks Gaussian parameter groups: "
            f"{sorted(required_names - observed_names)}"
        )
    if set(states) != observed_ids:
        raise ValueError("Checkpoint optimizer state IDs and parameter groups disagree")
    return {"optimizer_group_names": sorted(observed_names)}


def validate_complete_state(
    state: Any,
    *,
    require_cuda_rng: bool = False,
    require_controlled_provenance: bool = False,
) -> dict[str, Any]:
    """Validate the serialized Gaussian, Adam, densification and RNG state."""
    if not isinstance(state, Mapping) or state.get("format") != FORMAT:
        raise ValueError(f"Checkpoint must use {FORMAT}")
    missing = [field for field in REQUIRED_STATE_FIELDS if field not in state]
    if missing:
        raise ValueError(f"Checkpoint is incomplete: missing {missing}")
    core = state["core"]
    if not isinstance(core, Mapping):
        raise ValueError("Checkpoint core must be a mapping")
    missing_core = [field for field in CORE if field not in core]
    if missing_core:
        raise ValueError(
            f"Checkpoint Gaussian/densification core is incomplete: {missing_core}"
        )

    xyz = _require_finite_tensor(core["_xyz"], "core._xyz")
    _require_shape(xyz, (None, 3), "core._xyz")
    if xyz.shape[0] <= 0:
        raise ValueError("Checkpoint core._xyz must be non-empty [N,3]")
    gaussian_count = int(xyz.shape[0])
    core_shapes = {
        "_features_dc": (gaussian_count, 1, 3),
        "_features_rest": (gaussian_count, None, 3),
        "_scaling": (gaussian_count, 3),
        "_rotation": (gaussian_count, 4),
        "_opacity": (gaussian_count, 1),
        "max_radii2D": (gaussian_count,),
        "xyz_gradient_accum": (gaussian_count, 1),
        "denom": (gaussian_count, 1),
    }
    for name, shape in core_shapes.items():
        value = _require_finite_tensor(core[name], f"core.{name}")
        _require_shape(value, shape, f"core.{name}")
    degree = core["active_sh_degree"]
    if (
        isinstance(degree, bool)
        or not isinstance(degree, (int, np.integer))
        or int(degree) < 0
    ):
        raise ValueError("Checkpoint active_sh_degree must be a non-negative integer")
    max_degree = core["max_sh_degree"]
    if (
        isinstance(max_degree, bool)
        or not isinstance(max_degree, (int, np.integer))
        or int(max_degree) < int(degree)
    ):
        raise ValueError(
            "Checkpoint max_sh_degree must be an integer no smaller than active_sh_degree"
        )
    expected_rest = (int(max_degree) + 1) ** 2 - 1
    if int(core["_features_rest"].shape[1]) != expected_rest:
        raise ValueError(
            "Checkpoint SH feature count disagrees with max_sh_degree"
        )

    spatial_lr_scale = state["spatial_lr_scale"]
    if (
        isinstance(spatial_lr_scale, bool)
        or not isinstance(spatial_lr_scale, (int, float, np.number))
        or not math.isfinite(float(spatial_lr_scale))
        or float(spatial_lr_scale) <= 0.0
    ):
        raise ValueError("Checkpoint spatial_lr_scale must be finite and positive")
    confidence = _require_finite_tensor(state["confidence"], "confidence")
    _require_shape(confidence, (gaussian_count, 1), "confidence")
    init_point = _require_finite_tensor(state["init_point"], "init_point")
    _require_shape(init_point, (None, 3), "init_point")
    if init_point.shape[0] <= 0:
        raise ValueError("Checkpoint init_point must be a non-empty [M,3] tensor")
    bg_color = _require_finite_tensor(state["bg_color"], "bg_color")
    if bg_color.numel() not in (0, 3):
        raise ValueError("Checkpoint bg_color must be empty or contain exactly 3 values")
    optimizer_summary = _validate_optimizer_state(state["optimizer"], core)

    if not isinstance(state["python_rng"], tuple):
        raise ValueError("Checkpoint Python RNG state is invalid")
    try:
        random.Random().setstate(state["python_rng"])
    except Exception as exc:
        raise ValueError("Checkpoint Python RNG state is not restorable") from exc
    numpy_rng = state["numpy_rng"]
    if not isinstance(numpy_rng, tuple) or len(numpy_rng) != 5:
        raise ValueError("Checkpoint NumPy RNG state is invalid")
    try:
        np.random.RandomState().set_state(numpy_rng)
    except Exception as exc:
        raise ValueError("Checkpoint NumPy RNG state is not restorable") from exc
    torch_rng = state["torch_rng"]
    if (
        not torch.is_tensor(torch_rng)
        or torch_rng.dtype != torch.uint8
        or torch_rng.numel() == 0
    ):
        raise ValueError("Checkpoint Torch RNG state is invalid")
    try:
        torch.Generator(device="cpu").set_state(torch_rng.cpu())
    except Exception as exc:
        raise ValueError("Checkpoint Torch RNG state is not restorable") from exc
    cuda_rng = state["cuda_rng"]
    if cuda_rng is not None and (
        not isinstance(cuda_rng, (list, tuple))
        or not cuda_rng
        or any(
            not torch.is_tensor(value)
            or value.dtype != torch.uint8
            or value.numel() == 0
            for value in cuda_rng
        )
    ):
        raise ValueError("Checkpoint CUDA RNG state is invalid")
    if require_cuda_rng and cuda_rng is None:
        raise ValueError("Controlled GPU checkpoint lacks CUDA RNG state")

    controlled_provenance = state.get("controlled_provenance")
    if require_controlled_provenance and controlled_provenance is None:
        raise ValueError("Controlled checkpoint lacks embedded run provenance")
    if controlled_provenance is not None:
        canonical_provenance = canonical_controlled_checkpoint_provenance(
            controlled_provenance
        )
        if controlled_provenance != canonical_provenance:
            raise ValueError("Checkpoint controlled provenance is not canonical")
        controlled_provenance_digest = controlled_checkpoint_provenance_sha256(
            canonical_provenance
        )
    else:
        canonical_provenance = None
        controlled_provenance_digest = None

    render_digest = renderer_state_sha256(core)
    return {
        "format": FORMAT,
        "gaussian_count": gaussian_count,
        "render_state_schema": RENDER_STATE_SCHEMA,
        "render_state_sha256": render_digest,
        "optimizer_state_present": True,
        "densification_state_present": True,
        "python_rng_present": True,
        "numpy_rng_present": True,
        "torch_rng_present": True,
        "cuda_rng_present": cuda_rng is not None,
        "controlled_provenance_present": canonical_provenance is not None,
        "controlled_provenance": canonical_provenance,
        "controlled_provenance_sha256": controlled_provenance_digest,
        **optimizer_summary,
    }


def load_checkpoint_summary(
    path: str | Path,
    *,
    expected_iteration: int | None = None,
    require_cuda_rng: bool = False,
    require_controlled_provenance: bool = False,
) -> dict[str, Any]:
    checkpoint = Path(path).expanduser().resolve()
    try:
        loaded = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise ValueError(f"Cannot load checkpoint {checkpoint}: {exc}") from exc
    if not isinstance(loaded, tuple) or len(loaded) != 2:
        raise ValueError(
            "Checkpoint must contain exactly (complete_state, iteration)"
        )
    state, iteration = loaded
    if isinstance(iteration, bool):
        raise ValueError("Checkpoint iteration must be an integer")
    try:
        iteration = int(iteration)
    except (TypeError, ValueError) as exc:
        raise ValueError("Checkpoint iteration must be an integer") from exc
    if expected_iteration is not None and iteration != int(expected_iteration):
        raise ValueError(
            f"Checkpoint iteration mismatch: observed={iteration}, "
            f"expected={expected_iteration}"
        )
    return {
        "checkpoint": str(checkpoint),
        "iteration": iteration,
        **validate_complete_state(
            state,
            require_cuda_rng=require_cuda_rng,
            require_controlled_provenance=require_controlled_provenance,
        ),
    }


def capture(model) -> dict[str, Any]:
    if model.optimizer is None:
        raise RuntimeError("Cannot checkpoint an uninitialized optimizer")
    state = {
        "format": FORMAT,
        "core": {name: getattr(model, name) for name in CORE},
        "spatial_lr_scale": model.spatial_lr_scale,
        "optimizer": model.optimizer.state_dict(),
        "confidence": model.confidence,
        "init_point": model.init_point,
        "bg_color": model.bg_color,
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
        "controlled_provenance": getattr(model, "controlled_provenance", None),
    }
    validate_complete_state(
        state,
        require_cuda_rng=torch.cuda.is_available(),
        require_controlled_provenance=state["controlled_provenance"] is not None,
    )
    return state


def restore(model, state: Any, opt) -> None:
    role = str(getattr(opt, "controlled_ab_role", "")).upper()
    try:
        summary = validate_complete_state(
            state,
            require_cuda_rng=role in {"A1", "SELFRENDER", "B"},
            require_controlled_provenance=role in {"A1", "SELFRENDER", "B"},
        )
    except ValueError as exc:
        raise RuntimeError(
            "Incomplete checkpoint. Re-run A0 with this revision; controlled "
            "comparisons cannot use a legacy or partial state."
        ) from exc

    for name in CORE:
        setattr(model, name, state["core"][name])
    accum, denom = model.xyz_gradient_accum, model.denom
    model.spatial_lr_scale = state["spatial_lr_scale"]
    for name in ("confidence", "init_point", "bg_color"):
        setattr(model, name, state[name])
    model.training_setup(opt)
    model.xyz_gradient_accum, model.denom = accum, denom
    model.optimizer.load_state_dict(state["optimizer"])
    model.optimizer_state_restored = True
    model.densification_state_restored = True
    model.restored_controlled_provenance = summary["controlled_provenance"]

    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_rng"])
    torch.set_rng_state(state["torch_rng"].cpu())
    if state["cuda_rng"] is not None:
        if not torch.cuda.is_available():
            if role in {"A1", "SELFRENDER", "B"}:
                raise RuntimeError(
                    "Controlled GPU checkpoint cannot restore CUDA RNG without CUDA"
                )
        else:
            if len(state["cuda_rng"]) != torch.cuda.device_count():
                raise RuntimeError(
                    "CUDA RNG device count differs from the A0 checkpoint"
                )
            torch.cuda.set_rng_state_all(
                [value.cpu() for value in state["cuda_rng"]]
            )

    if model.get_xyz.shape[0] != model.get_opacity.shape[0]:
        raise RuntimeError("Corrupt checkpoint tensor lengths")
