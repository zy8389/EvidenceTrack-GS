#!/usr/bin/env python3
"""Create the exact source/held-out image lists used by the COLMAP reader."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from build_anchor_tracks_from_colmap import read_images


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sparse-dir", required=True, type=Path)
    parser.add_argument("--n-views", required=True, type=int)
    parser.add_argument("--llff-holdout", type=int, default=8)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.n_views < 2:
        raise ValueError("Strict source-only triangulation needs at least two source views")

    images = sorted(read_images(args.sparse_dir / "images.bin").values(), key=lambda item: item["name"])
    if args.llff_holdout > 0:
        train = [item for index, item in enumerate(images) if index % args.llff_holdout != 0]
        heldout = [item for index, item in enumerate(images) if index % args.llff_holdout == 0]
    else:
        train, heldout = images, []
    if len(train) < args.n_views:
        raise ValueError(f"Requested {args.n_views} source views, but only {len(train)} are available")
    if len(train) > args.n_views:
        indices = np.linspace(0, len(train) - 1, args.n_views, dtype=int)
        train = [train[int(index)] for index in indices]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_names = [item["name"] for item in train]
    heldout_names = [item["name"] for item in heldout]
    (args.output_dir / "source_images.txt").write_text(
        "\n".join(source_names) + "\n", encoding="utf-8"
    )
    (args.output_dir / "heldout_images.txt").write_text(
        "\n".join(heldout_names) + ("\n" if heldout_names else ""), encoding="utf-8"
    )
    (args.output_dir / "split.json").write_text(
        json.dumps(
            {
                "n_views": args.n_views,
                "llff_holdout": args.llff_holdout,
                "source_images": source_names,
                "heldout_images": heldout_names,
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    print(f"Source images ({len(source_names)}): {source_names}")
    print(f"Held-out images ({len(heldout_names)}): {heldout_names}")


if __name__ == "__main__":
    main()
