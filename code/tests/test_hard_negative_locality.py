from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from diffusion_guidance.evidence_matching import select_hard_negatives


def test_nonlocal_fallback_is_never_a_valid_hard_negative():
    queries = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])
    selection = select_hard_negatives(
        queries,
        np.asarray([[0.0, 0.0], [4.0, 0.0], [100.0, 0.0]]),
        nearby_radius=8.0,
    )
    assert selection.selection_mode.tolist() == [
        "local_hard_negative",
        "local_hard_negative",
        "global_fallback",
    ]
    assert selection.within_radius.tolist() == [True, True, False]
    assert selection.projection_distance_px[2] > 8.0


def test_single_track_has_missing_negative_instead_of_self_negative():
    selection = select_hard_negatives(
        torch.tensor([[1.0, 0.0]]),
        np.asarray([[0.0, 0.0]]),
        nearby_radius=8.0,
    )
    assert selection.indices.tolist() == [0]
    assert selection.selection_mode.tolist() == ["missing"]
    assert selection.within_radius.tolist() == [False]
    assert np.isnan(selection.projection_distance_px[0])


def test_local_validity_changes_with_the_row_window_radius():
    queries = torch.tensor([[1.0, 0.0], [0.9, 0.1]])
    centers = np.asarray([[0.0, 0.0], [24.0, 0.0]])
    narrow = select_hard_negatives(queries, centers, nearby_radius=16.0)
    primary = select_hard_negatives(queries, centers, nearby_radius=32.0)
    assert narrow.within_radius.tolist() == [False, False]
    assert narrow.selection_mode.tolist() == ["global_fallback", "global_fallback"]
    assert primary.within_radius.tolist() == [True, True]
    assert primary.selection_mode.tolist() == [
        "local_hard_negative",
        "local_hard_negative",
    ]
