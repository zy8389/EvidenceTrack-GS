#!/usr/bin/env python3
"""Record hardware/software and immutable source hashes without training."""

from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from typing import Optional


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit(root: Path) -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    marker_path = root / ".research_revision_v2.json"
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        marker = {}

    result = {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {},
        "git_commit": _git_commit(root),
        "upstream_commit": marker.get("upstream_commit"),
        "research_revision": marker.get("revision"),
    }
    for name in (
        "torch",
        "torchvision",
        "numpy",
        "opencv-python",
        "opencv-python-headless",
        "h5py",
        "diffusers",
        "transformers",
    ):
        try:
            result["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result["packages"][name] = None

    import torch

    result["cuda_available"] = torch.cuda.is_available()
    result["torch_cuda"] = torch.version.cuda
    result["devices"] = [
        torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
    ]
    result["python_sha256"] = {
        str(path.relative_to(root)).replace("\\", "/"): _sha256(path)
        for path in root.rglob("*.py")
        if ".git" not in path.parts
    }
    protocol_candidates = [
        root / "scripts" / "run_stage.sh",
        root / "configs" / "controlled_protocol.json",
        *sorted(root.glob("env/requirements-*.txt")),
    ]
    result["protocol_sha256"] = {
        str(path.relative_to(root)).replace("\\", "/"): _sha256(path)
        for path in protocol_candidates
        if path.is_file()
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                key: value
                for key, value in result.items()
                if key not in {"python_sha256", "protocol_sha256"}
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
