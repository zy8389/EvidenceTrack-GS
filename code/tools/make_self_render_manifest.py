#!/usr/bin/env python3
"""Build an immutable pseudo-RGB manifest whose target is the shared A0 render.

This control has the same camera pool and schedule as B.  It isolates generic
extra pseudo-view RGB supervision from the effect of the frozen Difix target.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.audit_pseudo_camera_manifest import audit

from diffusion_guidance.pseudo_manifest import (
    load_pseudo_manifest,
    validate_a0_source_provenance,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    args = parser.parse_args()
    source = args.input_manifest.expanduser().resolve()
    output = args.output_manifest.expanduser().resolve()
    if output == source:
        raise ValueError("SelfRender manifest must be a separate immutable file")
    source_audit = audit(source)
    if not source_audit["passed"]:
        raise ValueError("Input pseudo-camera manifest failed provenance audit")
    records = []
    strict_source_records = load_pseudo_manifest(source, strict_targets=False)
    if len(strict_source_records) != 32:
        raise ValueError(
            f"Controlled SelfRender requires exactly 32 source pseudo records, got {len(strict_source_records)}"
        )
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            for field in ("key", "camera", "input", "reference_image"):
                if field not in record:
                    raise ValueError(f"Missing {field} at {source}:{line_number}")
            source_record = strict_source_records[len(records)]
            if record.get("key") != source_record.key:
                raise ValueError(
                    f"Source pseudo record order/fingerprint changed at {source}:{line_number}"
                )
            validate_a0_source_provenance(source_record)
            rewritten = dict(record)
            rewritten["input"] = str(source_record.input_path)
            rewritten["target"] = str(source_record.input_path)
            rewritten["input_sha256"] = record["input_sha256"]
            rewritten["target_sha256"] = record["input_sha256"]
            rewritten["supervision_target_kind"] = "self_render_a0"
            records.append(rewritten)
    if len(records) != 32:
        raise ValueError(f"Controlled SelfRender requires exactly 32 pseudo cameras, got {len(records)}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"Wrote {len(records)} immutable SelfRender records to {output}")


if __name__ == "__main__":
    main()
