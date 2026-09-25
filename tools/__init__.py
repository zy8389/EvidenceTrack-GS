"""Compatibility import path for evidence_track.evaluation tools."""

from pathlib import Path

# Keep the historical ``tools.<audit_module>`` compatibility path while also
# exposing real tool subpackages that live under this directory.
_tools_root = Path(__file__).resolve().parent
__path__ = [
    str(_tools_root),
    str(Path(__file__).resolve().parents[1] / "evidence_track" / "evaluation"),
]
