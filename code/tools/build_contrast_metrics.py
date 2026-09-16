#!/usr/bin/env python3
"""Build one exact two-arm result CSV from immutable final training logs."""
from __future__ import annotations

import argparse
from pathlib import Path

from diffusion_guidance.result_binding import (
    audit_supports_contrast,
    read_pair_audit,
)
from tools.metrics_from_log import extract_metric_row, write_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", choices=["A1", "SelfRender"], required=True)
    parser.add_argument("--baseline-log", type=Path, required=True)
    parser.add_argument("--treatment", choices=["SelfRender", "B"], required=True)
    parser.add_argument("--treatment-log", type=Path, required=True)
    parser.add_argument("--pair-audit", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--iteration", type=int, default=12000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.baseline == args.treatment:
        raise ValueError("A controlled contrast requires two different methods")
    audit_path, audit = read_pair_audit(args.pair_audit)
    if not audit_supports_contrast(audit, args.baseline, args.treatment):
        raise ValueError(
            f"Pair audit does not authorize {args.treatment}-{args.baseline}: "
            f"{audit_path}"
        )
    rows = [
        extract_metric_row(
            log=args.baseline_log,
            dataset=args.dataset,
            scene=args.scene,
            seed=args.seed,
            method=args.baseline,
            pair_id=audit["pair_id"],
            pair_audit=audit_path,
            iteration=args.iteration,
        ),
        extract_metric_row(
            log=args.treatment_log,
            dataset=args.dataset,
            scene=args.scene,
            seed=args.seed,
            method=args.treatment,
            pair_id=audit["pair_id"],
            pair_audit=audit_path,
            iteration=args.iteration,
        ),
    ]
    write_rows(rows, args.output)
    print(
        f"Wrote {args.treatment}-{args.baseline} measured rows to "
        f"{args.output.expanduser().resolve()}"
    )


if __name__ == "__main__":
    main()
