"""Compatibility import path for the canonical evidence_track.geometry package."""

from importlib import import_module
from pathlib import Path

_implementation = import_module("evidence_track.geometry")
__path__ = [
    str(Path(__file__).resolve().parents[1] / "evidence_track" / "geometry")
]
__all__ = list(getattr(_implementation, "__all__", ()))


def __getattr__(name):
    return getattr(_implementation, name)
