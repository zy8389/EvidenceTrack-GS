#!/usr/bin/env python3
"""Extract one measured final held-out training report; never synthesize values."""
import argparse
import csv
import hashlib
import math
import re
from pathlib import Path

from diffusion_guidance.result_binding import read_pair_audit, require_audit_binding
from diffusion_guidance.experiment_registry import expected_scene_role

NUMBER = r"([-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?)"
VALID_METHODS = frozenset({"A0", "A1", "SelfRender", "B"})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_metric_row(
    *,
    log: Path,
    dataset: str,
    scene: str,
    seed: int,
    method: str,
    pair_id: str,
    pair_audit: Path,
    iteration: int = 12000,
) -> dict[str, object]:
    """Extract one immutable result row under its registered scene role."""
    if method not in VALID_METHODS:
        raise ValueError(f"Unknown controlled method: {method!r}")
    if isinstance(seed, bool):
        raise ValueError("Result seed must be an integer")
    role = expected_scene_role(dataset, scene)
    log = log.expanduser().resolve()
    if not log.is_file():
        raise FileNotFoundError(f"Training log is missing: {log}")
    audit_path, audit = read_pair_audit(pair_audit)
    require_audit_binding(
        audit,
        dataset=dataset,
        scene=scene,
        seed=seed,
        method=method,
        pair_id=pair_id,
        role=role,
    )
    # A0 is the common checkpoint anchor and is audited as a method, while the
    # continuation rows must additionally belong to an explicitly audited
    # contrast.  For A1, either A1/B or A1/SelfRender is valid; a distinct A1
    # CSV row is required for each contrast because each row has one audit.
    if method != "A0" and not any(
        method in contrast for contrast in audit["audited_contrasts"]
    ):
        raise ValueError(
            f"Pair audit does not cover a contrast containing result method "
            f"{method}: {audit['audited_contrasts']!r}"
        )
    text = log.read_text(encoding="utf-8", errors="replace")
    pattern = rf"\[ITER {int(iteration)}\] Evaluating test: L1 .*?PSNR {NUMBER}\s+SSIM {NUMBER}\s+LPIPS {NUMBER}"
    matches = re.findall(pattern, text)
    if len(matches) != 1:
        raise RuntimeError(f"Expected one measured final test record, found {len(matches)}; do not invent a replacement")
    values = list(map(float, matches[0]))
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError("Non-finite metrics")
    return dict(
        dataset=dataset, scene=scene, role=role, seed=int(seed),
        method=method, psnr=values[0], ssim=values[1], lpips=values[2],
        pair_id=audit["pair_id"], pair_audit_path=str(audit_path),
        pair_audit_sha256=sha256_file(audit_path), geometry_error="NR",
        metric_log=str(log), metric_log_sha256=sha256_file(log),
    )


def write_rows(rows: list[dict[str, object]], output: Path) -> None:
    if not rows:
        raise ValueError("Cannot write an empty measured-result CSV")
    fieldnames = list(rows[0])
    if any(list(row) != fieldnames for row in rows):
        raise ValueError("Measured-result rows do not share one exact schema")
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--method", choices=sorted(VALID_METHODS), required=True)
    parser.add_argument(
        "--pair-id",
        required=True,
        help="Exact controlled_pair_id written by this dataset/scene/seed audit",
    )
    parser.add_argument(
        "--pair-audit",
        type=Path,
        required=True,
        help="Passed group-specific pair-audit JSON for this result row",
    )
    parser.add_argument("--iteration", type=int, default=12000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    row = extract_metric_row(
        log=args.log,
        dataset=args.dataset,
        scene=args.scene,
        seed=args.seed,
        method=args.method,
        pair_id=args.pair_id,
        pair_audit=args.pair_audit,
        iteration=args.iteration,
    )
    write_rows([row], args.output)
    print(row)


if __name__ == "__main__":
    main()
