#!/usr/bin/env python3
"""Machine-readable SelfRender/B camera-pool and realized-schedule audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from evidence_track.evaluation.audit_controlled_pair import audit_pseudo_control


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-render", type=Path, required=True)
    parser.add_argument("--b", type=Path, required=True)
    parser.add_argument("--self-render-manifest", type=Path, required=True)
    parser.add_argument("--b-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-real-updates", type=int, default=2000)
    parser.add_argument("--expected-pseudo-updates", type=int, default=39)
    parser.add_argument("--expected-unique-pseudo-views", type=int, default=32)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--scene", required=True)
    args = parser.parse_args()
    result = audit_pseudo_control(
        args.self_render.expanduser().resolve(),
        args.b.expanduser().resolve(),
        args.self_render_manifest.expanduser().resolve(),
        args.b_manifest.expanduser().resolve(),
        expected_real_updates=args.expected_real_updates,
        expected_pseudo_updates=args.expected_pseudo_updates,
        expected_unique_pseudo_views=args.expected_unique_pseudo_views,
        dataset=args.dataset,
        scene=args.scene,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
