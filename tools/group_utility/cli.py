"""Command line entry point for Group Utility Protocol v1 artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .manifest import load_parent_manifest
from .partition import (
    build_balanced_partition,
    validate_partition,
    write_group_manifests,
    write_group_manifests_reusing_partition,
    write_partition,
)
from .schedule import build_group_schedule, write_group_schedule


def _partition(args: argparse.Namespace) -> int:
    parent = load_parent_manifest(
        args.manifest,
        expected_count=args.expected_count,
        expected_sha256=args.expected_sha256,
        require_files=args.require_files,
    )
    partition = build_balanced_partition(
        parent,
        group_count=args.groups,
        group_size=args.group_size,
        seed=args.seed,
    )
    validate_partition(
        partition,
        parent_manifest_sha256=parent.sha256,
        parent_records=parent,
        expected_group_count=args.groups,
        expected_group_size=args.group_size,
    )
    root = Path(args.output_dir)
    write_partition(root / "partition.json", partition)
    write_group_manifests(parent, partition, root / "groups")
    for group in partition.groups:
        schedule = build_group_schedule(
            group["group_id"],
            group["camera_keys"],
            parent_manifest_sha256=parent.sha256,
            partition_sha256=partition.partition_sha256,
        )
        write_group_schedule(root / "groups" / group["group_id"] / "schedule.json", schedule)
    print(json.dumps({"passed": True, "parent_manifest_sha256": parent.sha256, "partition_sha256": partition.partition_sha256}, sort_keys=True))
    return 0


def _validate(args: argparse.Namespace) -> int:
    parent = load_parent_manifest(
        args.manifest,
        expected_count=args.expected_count,
        expected_sha256=args.expected_sha256,
        require_files=args.require_files,
    )
    print(json.dumps({"passed": True, "records": len(parent), "sha256": parent.sha256}, sort_keys=True))
    return 0


def _materialize(args: argparse.Namespace) -> int:
    parent = load_parent_manifest(
        args.manifest,
        expected_count=args.expected_count,
        expected_sha256=args.expected_sha256,
        require_files=args.require_files,
    )
    partition = json.loads(Path(args.partition).read_text(encoding="utf-8"))
    validate_partition(
        partition,
        parent_records=parent,
        expected_group_count=args.groups,
        expected_group_size=args.group_size,
    )
    paths = write_group_manifests_reusing_partition(
        parent,
        partition,
        args.output_dir,
    )
    print(
        json.dumps(
            {
                "passed": True,
                "parent_manifest_sha256": parent.sha256,
                "partition_sha256": partition["partition_sha256"],
                "groups": {key: str(value) for key, value in paths.items()},
            },
            sort_keys=True,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--manifest", required=True, type=Path)
    common.add_argument("--expected-count", type=int, default=32)
    common.add_argument("--expected-sha256")
    common.add_argument("--require-files", action="store_true")
    validate = subparsers.add_parser("validate-parent", parents=[common])
    validate.set_defaults(func=_validate)
    partition = subparsers.add_parser("partition", parents=[common])
    partition.add_argument("--output-dir", required=True, type=Path)
    partition.add_argument("--groups", type=int, default=4)
    partition.add_argument("--group-size", type=int, default=8)
    partition.add_argument("--seed", type=int, default=0)
    partition.set_defaults(func=_partition)
    materialize = subparsers.add_parser(
        "materialize-reused-partition",
        parents=[common],
        help="materialize a second target-kind parent using an existing camera partition",
    )
    materialize.add_argument("--partition", required=True, type=Path)
    materialize.add_argument("--output-dir", required=True, type=Path)
    materialize.add_argument("--groups", type=int, default=4)
    materialize.add_argument("--group-size", type=int, default=8)
    materialize.set_defaults(func=_materialize)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
