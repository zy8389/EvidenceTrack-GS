from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.aggregate_results import filter_by_role


def test_confirmatory_filter_excludes_development_scenes():
    rows = [
        {"dataset": "LLFF", "scene": "fern", "role": "development"},
        {"dataset": "LLFF", "scene": "flower", "role": "confirmatory"},
    ]
    assert filter_by_role(rows, "confirmatory") == [rows[1]]


@pytest.mark.parametrize("role", [None, "", "exploratory"])
def test_missing_or_unknown_role_fails_closed(role):
    with pytest.raises(ValueError, match="declare role"):
        filter_by_role([{"dataset": "LLFF", "scene": "fern", "role": role}], "confirmatory")


def test_empty_confirmatory_partition_is_not_reported_as_a_result():
    with pytest.raises(ValueError, match="No confirmatory"):
        filter_by_role(
            [{"dataset": "LLFF", "scene": "fern", "role": "development"}],
            "confirmatory",
        )
