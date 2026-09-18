"""Compatibility import path for evidence_track.evaluation tools."""

from pathlib import Path

__path__ = [
    str(Path(__file__).resolve().parents[1] / "evidence_track" / "evaluation")
]
