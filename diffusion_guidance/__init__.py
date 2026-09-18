"""Compatibility import path for the canonical evidence_track.diffusion package."""

from pathlib import Path

__path__ = [
    str(Path(__file__).resolve().parents[1] / "evidence_track" / "diffusion")
]
