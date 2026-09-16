# EvidenceTrack-GS

Research package for **EvidenceTrack-GS: Identity-Controlled Geometry Diagnostics for Diffusion-Assisted Sparse-View Gaussian Splatting**.

This repository contains the reproducible research overlay, cumulative patch, experiment protocol, documentation, manuscript source, and integrity manifest. It intentionally excludes datasets, pretrained weights, checkpoints, caches, and generated experiment outputs.

Start with [README_先读.md](README_%E5%85%88%E8%AF%BB.md), then follow [docs/快速开始.md](docs/%E5%BF%AB%E9%80%9F%E5%BC%80%E5%A7%8B.md) and [docs/FERN_PRECHECK_AND_RUN.md](docs/FERN_PRECHECK_AND_RUN.md).

The overlay is tied to upstream GeoTrack-GS commit `81ada6a32c918591ae7c7a0279dc6ca7a8018e2f`. Apply [code/patches/phase2_1_integrity.patch](code/patches/phase2_1_integrity.patch) only once to that pinned revision. Verify the delivered files with `SHA256SUMS.txt` before use.

No training, CPU/CUDA smoke test, or experiment has been run for this final repair snapshot.
