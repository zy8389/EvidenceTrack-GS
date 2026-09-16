"""Preregistered dataset partitions used by result-producing tools."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


DEVELOPMENT_SCENES = frozenset({"fern", "room"})
CONFIRMATORY_SCENES = frozenset(
    {"flower", "fortress", "horns", "leaves", "orchids", "trex"}
)
PREREGISTERED_CONFIRMATORY_SEEDS = (1, 2, 3)
VALID_ROLES = frozenset({"development", "confirmatory"})


def normalize_scene_name(scene: str | Path) -> str:
    value = Path(str(scene).replace("\\", "/")).name.strip().casefold()
    if not value:
        raise ValueError("Scene name must be non-empty")
    return value


def expected_scene_role(dataset: str, scene: str | Path) -> str:
    dataset_name = str(dataset).strip().casefold()
    if dataset_name != "llff":
        raise ValueError(
            f"No preregistered development/confirmatory partition for dataset {dataset!r}"
        )
    scene_name = normalize_scene_name(scene)
    if scene_name in DEVELOPMENT_SCENES:
        return "development"
    if scene_name in CONFIRMATORY_SCENES:
        return "confirmatory"
    raise ValueError(f"Unregistered LLFF scene: {scene_name!r}")


def require_scene_role(dataset: str, scene: str | Path, role: str) -> str:
    declared = str(role).strip().casefold()
    if declared not in VALID_ROLES:
        raise ValueError("Role must be development or confirmatory")
    expected = expected_scene_role(dataset, scene)
    if declared != expected:
        raise ValueError(
            f"Scene role mismatch for {dataset}/{normalize_scene_name(scene)}: "
            f"declared={declared}, preregistered={expected}"
        )
    return expected


def require_complete_confirmatory_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    methods: Iterable[str],
) -> None:
    """Require all preregistered LLFF scenes, seeds and requested methods."""
    expected_methods = {str(method) for method in methods}
    if not expected_methods:
        raise ValueError("Confirmatory completeness requires at least one method")
    observed: set[tuple[str, int, str]] = set()
    for row in rows:
        role = require_scene_role(
            str(row.get("dataset", "")),
            str(row.get("scene", "")),
            str(row.get("role", "")),
        )
        if role != "confirmatory":
            raise ValueError("Confirmatory aggregate contains a development row")
        try:
            seed = int(row.get("seed"))
        except (TypeError, ValueError) as exc:
            raise ValueError("Confirmatory result seed must be an integer") from exc
        method = str(row.get("method", ""))
        if method in expected_methods:
            observed.add((normalize_scene_name(str(row.get("scene", ""))), seed, method))

    expected = {
        (scene, seed, method)
        for scene in CONFIRMATORY_SCENES
        for seed in PREREGISTERED_CONFIRMATORY_SEEDS
        for method in expected_methods
    }
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    if missing or extra:
        raise ValueError(
            "Confirmatory aggregate is incomplete or outside the preregistration: "
            f"missing={missing}, extra={extra}"
        )
