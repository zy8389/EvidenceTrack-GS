"""Atomic machine-readable status for training runs."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable, Optional, TypeVar


_T = TypeVar("_T")
_STATUSES = {"RUNNING", "COMPLETED", "INVALID"}


def write_training_status(
    model_path: str | os.PathLike[str] | None,
    *,
    status: str,
    stage: str,
    iteration: int | None = None,
    error: BaseException | None = None,
) -> Path | None:
    """Write an atomic status record, or do nothing when no output path exists."""
    if not model_path:
        return None
    if status not in _STATUSES:
        raise ValueError(f"Unsupported training status: {status}")
    if not stage:
        raise ValueError("Training status stage must not be empty")

    destination = Path(model_path)
    destination.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "evidencetrack_training_status_v1",
        "status": status,
        "stage": str(stage),
        "iteration": int(iteration) if iteration is not None else None,
        "error_type": type(error).__name__ if error is not None else None,
        "error": str(error) if error is not None else None,
    }
    target = destination / "run_status.json"
    temporary = target.with_name(f".{target.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, target)
    return target


def mark_training_invalid(
    model_path: str | os.PathLike[str] | None,
    *,
    stage: str,
    error: BaseException,
    iteration: int | None = None,
) -> Path | None:
    """Record an invalid run before the caller re-raises the failure."""
    if model_path:
        target = Path(model_path) / "run_status.json"
        if target.is_file():
            try:
                existing = json.loads(target.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                existing = None
            if isinstance(existing, dict) and existing.get("status") == "INVALID":
                return target
    return write_training_status(
        model_path,
        status="INVALID",
        stage=stage,
        iteration=iteration,
        error=error,
    )


def fail_closed(
    model_path: str | os.PathLike[str] | None,
    *,
    stage: str,
    operation: Callable[[], _T],
    iteration: int | None = None,
) -> _T:
    """Run a required operation and mark the output invalid on any failure."""
    try:
        return operation()
    except Exception as exc:
        mark_training_invalid(
            model_path,
            stage=stage,
            error=exc,
            iteration=iteration,
        )
        raise RuntimeError(
            f"{stage} failed; the training run was marked INVALID: {exc}"
        ) from exc
