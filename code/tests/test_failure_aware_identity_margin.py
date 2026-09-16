from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.paired_evidence_report import VARIANTS, analyze


def rows(correct: float, wrong: float, *, valid: bool = True):
    mode = "local_hard_negative" if valid else "global_fallback"
    return [
        {
            "image_name": "heldout.png",
            "track_id": "7",
            "window_radius": 32,
            "target_variant": variant,
            "image_width": 3,
            "image_height": 4,
            "projection_error": 2,
            "matching_error": correct,
            "feature_stride": 1,
            "hard_negative_valid": valid,
            "hard_negative_radius": 32,
            "hard_negative_within_radius": valid,
            "hard_negative_selection_mode": mode,
            "hard_shuffled_error": wrong,
            "global_shuffle_valid": False,
        }
        for variant in VARIANTS
    ]


def test_correct_and_wrong_failures_use_the_same_diagonal_penalty():
    correct_failure = analyze(rows(float("nan"), 2.0), 32)["scene_metrics"]
    wrong_failure = analyze(rows(1.0, float("nan")), 32)["scene_metrics"]
    assert correct_failure["gs_identity_margin_failure_aware"] == -3.0
    assert wrong_failure["gs_identity_margin_failure_aware"] == 4.0
    assert correct_failure["gs_identity_margin_conditional"] is None
    assert wrong_failure["gs_identity_margin_conditional"] is None


def test_nonlocal_negative_is_excluded_from_primary_margin():
    metrics = analyze(rows(1.0, 3.0, valid=False), 32)["scene_metrics"]
    assert metrics["gs_identity_margin_failure_aware"] is None
    assert metrics["gs_hard_negative_valid_count"] == 0


def test_inconsistent_locality_flag_fails_closed():
    payload = rows(1.0, 3.0)
    payload[0]["hard_negative_selection_mode"] = "global_fallback"
    with pytest.raises(ValueError, match="Inconsistent hard-negative validity"):
        analyze(payload, 32)


def test_scene_counts_are_sums_across_images():
    first = rows(1.0, float("nan"))
    second = rows(1.0, float("nan"))
    for row in second:
        row["image_name"] = "heldout_2.png"
    metrics = analyze(first + second, 32)["scene_metrics"]
    assert metrics["track_count"] == 2
    assert metrics["gs_eligible_count"] == 2
    assert metrics["gs_wrong_failure_count"] == 2


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("matching_error", -1.0, "non-negative"),
        ("feature_stride", 0.0, "feature_stride"),
        ("image_width", 0, "width and height"),
    ],
)
def test_invalid_metric_domains_fail_closed(field, value, error):
    payload = rows(1.0, 3.0)
    for row in payload:
        row[field] = value
    with pytest.raises(ValueError, match=error):
        analyze(payload, 32)
