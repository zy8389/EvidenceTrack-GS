"""Spatial feature extraction with explicit resolution/stride metadata."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import os
import subprocess
from typing import Dict, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class SpatialFeatureMap:
    features: torch.Tensor  # [C,Hf,Wf]
    input_resolution: Tuple[int, int]  # H,W before backend resize
    model_input_resolution: Tuple[int, int]
    feature_resolution: Tuple[int, int]
    feature_stride: Tuple[float, float]  # y,x in original pixels
    interpolation_method: str
    backend: str

    @property
    def effective_stride(self) -> float:
        return float(np.sqrt(self.feature_stride[0] * self.feature_stride[1]))

    def metadata(self) -> Dict:
        return {
            "backend": self.backend,
            "input_image_resolution": list(self.input_resolution),
            "model_input_resolution": list(self.model_input_resolution),
            "feature_resolution": list(self.feature_resolution),
            "feature_stride": list(self.feature_stride),
            "effective_feature_stride": self.effective_stride,
            "interpolation_method": self.interpolation_method,
        }


def load_rgb(path: str | Path) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32).copy() / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


class FeatureExtractor:
    def __init__(
        self,
        backend: str,
        *,
        device: str = "cuda",
        dinov2_model: str = "dinov2_vits14",
    ):
        self.backend = backend
        self.device = torch.device(device)
        self.dinov2_model_name = dinov2_model
        self.model = None
        self._provenance: Dict[str, object] = {"backend": backend}
        if backend == "dinov2":
            repository_value = os.environ.get("DINOV2_REPO")
            weight_path = os.environ.get("DINOV2_WEIGHT_PATH")
            repository = (
                Path(repository_value).expanduser().resolve()
                if repository_value
                else None
            )
            if repository is None or not (repository / "hubconf.py").is_file():
                raise RuntimeError("Set DINOV2_REPO to a local official clone; mutable remote torch.hub downloads are disabled")
            weights = Path(weight_path).expanduser().resolve() if weight_path else None
            if weights is None or not weights.is_file():
                raise RuntimeError("Set DINOV2_WEIGHT_PATH to the exact local pretrained weight file; cache-only provenance is forbidden")
            try:
                commit = subprocess.check_output(
                    ["git", "-C", str(repository), "rev-parse", "HEAD"],
                    text=True,
                ).strip()
                dirty = subprocess.check_output(
                    ["git", "-C", str(repository), "status", "--porcelain"],
                    text=True,
                ).strip()
            except (OSError, subprocess.CalledProcessError) as exc:
                raise RuntimeError("DINOV2_REPO must be a commit-addressable local git checkout") from exc
            if dirty:
                raise RuntimeError("DINOV2_REPO must be clean so its commit fully identifies the loaded code")
            digest = hashlib.sha256()
            with weights.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            # `pretrained=False` prevents the hub entrypoint from silently selecting a cache/download.
            self.model = torch.hub.load(str(repository), dinov2_model, source="local", pretrained=False)
            state = torch.load(weights, map_location="cpu", weights_only=True)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            if not isinstance(state, dict):
                raise RuntimeError("DINOv2 weight file must contain a state-dict mapping")
            missing, unexpected = self.model.load_state_dict(state, strict=False)
            if missing or unexpected:
                raise RuntimeError(f"DINOv2 weight/model mismatch; missing={len(missing)}, unexpected={len(unexpected)}")
            self.model = self.model.to(self.device).eval()
            self._provenance = {
                "backend": f"dinov2:{dinov2_model}",
                "local_repository_path": str(repository),
                "repository_commit": commit,
                "pretrained_weight_path": str(weights),
                "pretrained_weight_sha256": digest.hexdigest(),
                "weight_loading": "explicit_local_state_dict",
            }
        elif backend == "rgb":
            self._provenance = {"backend": "rgb", "weight_loading": "none"}
        else:
            raise ValueError(f"Unsupported feature backend: {backend}")

    def reproducibility_metadata(self) -> Dict[str, object]:
        return dict(self._provenance)

    @torch.inference_mode()
    def extract(self, path: str | Path) -> SpatialFeatureMap:
        image = load_rgb(path)
        height, width = image.shape[-2:]
        if self.backend == "rgb":
            features = F.normalize(image.to(self.device), dim=0)
            return SpatialFeatureMap(
                features=features,
                input_resolution=(height, width),
                model_input_resolution=(height, width),
                feature_resolution=(height, width),
                feature_stride=(1.0, 1.0),
                interpolation_method="none",
                backend="rgb",
            )

        patch_size = int(getattr(self.model, "patch_size", 14))
        model_height = max(patch_size, (height // patch_size) * patch_size)
        model_width = max(patch_size, (width // patch_size) * patch_size)
        value = F.interpolate(
            image[None].to(self.device),
            size=(model_height, model_width),
            mode="bicubic",
            align_corners=False,
        )
        mean = torch.tensor(
            [0.485, 0.456, 0.406], device=self.device
        ).view(1, 3, 1, 1)
        std = torch.tensor(
            [0.229, 0.224, 0.225], device=self.device
        ).view(1, 3, 1, 1)
        output = self.model.forward_features((value - mean) / std)
        tokens = output["x_norm_patchtokens"]
        feature_height = model_height // patch_size
        feature_width = model_width // patch_size
        features = tokens.reshape(1, feature_height, feature_width, -1)[0]
        features = F.normalize(features.permute(2, 0, 1).contiguous(), dim=0)
        return SpatialFeatureMap(
            features=features,
            input_resolution=(height, width),
            model_input_resolution=(model_height, model_width),
            feature_resolution=(feature_height, feature_width),
            feature_stride=(height / feature_height, width / feature_width),
            interpolation_method="bicubic_input_bilinear_coordinate_sampling",
            backend=f"dinov2:{self.dinov2_model_name}",
        )


def sample_features(
    feature_map: SpatialFeatureMap, xy_pixels: torch.Tensor
) -> torch.Tensor:
    """Bilinearly sample [N,2] original-image pixel coordinates."""
    if xy_pixels.ndim != 2 or xy_pixels.shape[1] != 2:
        raise ValueError("xy_pixels must be [N,2]")
    height, width = feature_map.input_resolution
    # Pixel centers, consistent with align_corners=False resize and patch tokens.
    x = 2.0 * (xy_pixels[:, 0] + 0.5) / width - 1.0
    y = 2.0 * (xy_pixels[:, 1] + 0.5) / height - 1.0
    grid = torch.stack([x, y], dim=-1).view(1, -1, 1, 2)
    sampled = F.grid_sample(
        feature_map.features[None],
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )[0, :, :, 0].transpose(0, 1)
    return F.normalize(sampled, dim=-1)


def feature_grid_pixel_coordinates(feature_map: SpatialFeatureMap) -> torch.Tensor:
    feature_height, feature_width = feature_map.feature_resolution
    input_height, input_width = feature_map.input_resolution
    ys = (torch.arange(feature_height, device=feature_map.features.device,
                       dtype=feature_map.features.dtype) + 0.5) * input_height / feature_height - 0.5
    xs = (torch.arange(feature_width, device=feature_map.features.device,
                       dtype=feature_map.features.dtype) + 0.5) * input_width / feature_width - 0.5
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([grid_x, grid_y], dim=-1)
