from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from diffusion_guidance.evidence_protocol import (
    validate_hard_negative_radius,
    validate_primary_radius,
)


def test_explicit_primary_radius_is_not_inferred_from_maximum():
    assert validate_primary_radius(32, [16, 32, 48]) == 32.0


def test_primary_radius_must_be_one_of_the_evaluated_windows():
    with pytest.raises(ValueError, match="must be one of"):
        validate_primary_radius(32, [16, 48])


def test_primary_hard_negative_radius_cannot_exceed_candidate_window():
    assert validate_hard_negative_radius(32, 32) == 32.0
    with pytest.raises(ValueError, match="must equal"):
        validate_hard_negative_radius(32, 64)


@pytest.mark.parametrize("primary,radii", [(0, [0, 32]), (32, [32, 32]), (32, [32, float("nan")])])
def test_invalid_radius_protocol_fails_closed(primary, radii):
    with pytest.raises(ValueError):
        validate_primary_radius(primary, radii)
