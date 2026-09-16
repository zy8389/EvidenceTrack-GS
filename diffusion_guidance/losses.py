from __future__ import annotations

from typing import Callable, Dict, Optional

import torch


def _expanded_mask(mask: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
    if mask.ndim == prediction.ndim - 1:
        mask = mask.unsqueeze(0)
    if mask.ndim != prediction.ndim or mask.shape[-2:] != prediction.shape[-2:]:
        raise ValueError("Mask is incompatible with prediction")
    if mask.shape[-3] == 1 and prediction.shape[-3] != 1:
        mask = mask.expand(*mask.shape[:-3], prediction.shape[-3], *mask.shape[-2:])
    return mask


def masked_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError(f"Shape mismatch: {prediction.shape} vs {target.shape}")
    error = torch.abs(prediction - target)
    if mask is None:
        return error.mean()
    expanded = _expanded_mask(mask, prediction).to(dtype=prediction.dtype)
    return (error * expanded).sum() / (expanded.sum() + eps)


def pseudo_rgb_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    lambda_dssim: float,
    ssim_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    mask: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    if not 0.0 <= lambda_dssim <= 1.0:
        raise ValueError("lambda_dssim must lie in [0,1]")
    l1 = masked_l1(prediction, target, mask)
    # DSSIM remains a full-image term; multiplying images by a mask is not a
    # mathematically valid masked SSIM implementation.
    dssim = 1.0 - ssim_fn(prediction, target)
    total = (1.0 - lambda_dssim) * l1 + lambda_dssim * dssim
    return {"total": total, "l1": l1, "dssim": dssim}
