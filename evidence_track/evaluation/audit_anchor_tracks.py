#!/usr/bin/env python3
"""Fail-closed leakage and coverage audit for strict source-only track H5 files."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from geometric_constraints.strict_track_store import StrictTrackStore
from evidence_track.evaluation.audit_track_coverage import report as coverage_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--min-tracks", type=int, default=32)
    parser.add_argument("--grid-size", type=int, choices=[8, 16], default=8)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    store = StrictTrackStore.load(args.path)
    audit = store.leakage_audit()
    audit.emit()
    if len(store) < args.min_tracks:
        raise RuntimeError(f"Only {len(store)} strict tracks; minimum is {args.min_tracks}")
    audit.require_pass()
    result = coverage_report(args.path, args.grid_size)
    result.update({
        "gate": "P0_strict_source_only_track_build_and_audit",
        "minimum_track_count": int(args.min_tracks),
        "track_sha256": hashlib.sha256(args.path.read_bytes()).hexdigest(),
        "leakage_audit": asdict(audit),
        "passed": True,
    })
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Strict tracks: {len(store)}")
    print("Quality uncertainty: sqrt(trace(covariance)) / source-only scene scale")
    print("STRICT SOURCE-ONLY LEAKAGE + COVERAGE AUDIT: PASS")


if __name__ == "__main__":
    main()
