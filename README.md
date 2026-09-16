# EvidenceTrack-GS

Research package for **EvidenceTrack-GS: Identity-Controlled Geometry Diagnostics for Diffusion-Assisted Sparse-View Gaussian Splatting**.

This private repository preserves the reproducible research overlay, cumulative patch, protocols, documentation, manuscript source, experiment matrix, and integrity manifest. It deliberately excludes datasets, pretrained weights, checkpoints, caches, and generated experiment outputs.

Start with [README_先读.md](README_%E5%85%88%E8%AF%BB.md), then follow [docs/快速开始.md](docs/%E5%BF%AB%E9%80%9F%E5%BC%80%E5%A7%8B.md) and [docs/FERN_PRECHECK_AND_RUN.md](docs/FERN_PRECHECK_AND_RUN.md).

## Repository map

- `code/`: source overlay, cumulative patch, integrity gates, scripts, and regression tests.
- `docs/`: Chinese research rationale, repair boundary, and server runbook.
- `experiments/`: preregistered run matrix and measured-result template.
- `paper/`: anonymous pre-experiment manuscript source, figures, and current PDF.
- `validation/`: historical evidence only; it does not validate this source revision.

## Reconstructing the research checkout

The overlay is tied to upstream GeoTrack-GS commit `81ada6a32c918591ae7c7a0279dc6ca7a8018e2f`. Apply [code/patches/phase2_1_integrity.patch](code/patches/phase2_1_integrity.patch) only once to that pinned revision. The documented bootstrap path is in [code/README.md](code/README.md).

`SHA256SUMS.txt` records exact bytes for every tracked delivery artifact except itself. On a POSIX shell, verify it with:

```bash
sha256sum -c SHA256SUMS.txt
```

## Current evidence boundary

No training, current-source CPU/CUDA smoke test, bootstrap assembly, or real experiment has been run for this final repair snapshot. Use the runbook gates before any A0/A1/SelfRender/B conclusion; do not treat `validation/` as current execution evidence.

## Recording a server run

Open the **Controlled experiment record** Issue form after every attempted server run. Attach only small JSON reports, logs, hashes, and failure summaries; keep datasets, model weights, checkpoints, generated images, and Difix caches outside Git.
