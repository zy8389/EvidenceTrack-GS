# EvidenceTrack-GS

[中文](README.md)

**Identity-controlled geometry diagnostics for diffusion-assisted sparse-view Gaussian Splatting.**

This is a materialized full-source repository ready for direct development. The pinned upstream source, cumulative research patch, source overlay, and guarded AST integration are present at the repository root. Future changes, tests, commits, and pushes should be made here; the legacy assembler is archived at [`docs/provenance/archive/bootstrap_legacy.sh`](docs/provenance/archive/bootstrap_legacy.sh) and is not part of the normal workflow.

## Current status

| Component | Status |
| --- | --- |
| Full-source assembly | Passed: two independent assemblies, 0 byte mismatches across 66 target files |
| Patch gates | Passed: apply, cached whitespace, and diff checks |
| Python compileall | Passed |
| Phase-2.1 CPU smoke | 11 / 11 passed |
| Regression tests | 82 / 82 passed |
| Synthetic three-camera recovery | Passed |
| Fern CUDA preflight | Not run |
| A0 / A1 / SelfRender / B | Not run |
| Real Difix / DINOv2 / DTU | Not run |

Validation reports and raw logs are under [`docs/validation/`](docs/validation/). The current evidence verifies full-source reconstruction and CPU integrity gates. It does not establish that the real GPU reconstruction or paper experiments have been completed.

## Development entry point

Clone the full repository:

```bash
git clone --recurse-submodules https://github.com/zy8389/EvidenceTrack-GS.git
cd EvidenceTrack-GS
# CUDA sources are tracked directly; --recurse-submodules is retained for future dependencies.
```

Install the CPU-check dependencies and run the gates:

```bash
pip install -r env/requirements-cpu.txt
export PYTHONPATH="$PWD"
python -m compileall -q evidence_track arguments scene gaussian_renderer train.py utils tests
python -m pytest -q tests
python evidence_track/evaluation/smoke_test_phase2_1.py
python evidence_track/evaluation/test_geometry_recovery.py --synthetic-smoke --output synthetic_recovery.json
```

On Windows PowerShell:

```powershell
$env:PYTHONPATH = (Get-Location).Path
python -m pytest -q tests
python evidence_track/evaluation/smoke_test_phase2_1.py
```

Start real experiments with [`docs/FERN_RUNBOOK.md`](docs/FERN_RUNBOOK.md) and `scripts/run_stage.sh`. Stop downstream GPU work if any P0 gate fails.

Each training output writes `run_status.json`. Metrics and integrity require A0, A1, SelfRender, and B to be `COMPLETED`; missing, `RUNNING`, or `INVALID` runs are rejected.

## Research structure

```text
Source RGB + supplied cameras
          |
          v
Source-only feature tracks
          |
          v
       A0 @ 10k
      /    |     \
 A1 @12k  SelfRender @12k  B @12k
 source     A0 pseudo        Difix pseudo
 only       targets           targets
      \      |      /
       controlled comparison
               |
               v
      identity-controlled diagnostic
               |
               v
 Correct / Hard-Wrong / Random / Uniform / ProjectionOnly
```

The primary comparison is `A1 vs B`. `SelfRender` controls for the effect of additional pseudo-view supervision itself. Matching failures remain in a failure-aware denominator instead of being selectively removed.

## Repository layout

```text
arguments/                 Upstream argument definitions
evidence_track/            Research code (geometry/diffusion/identity/evaluation)
scene/                     Scene and Gaussian state
gaussian_renderer/         CUDA renderer interface
utils/                     Upstream training utilities
tests/                     CPU regression tests
configs/                   Controlled experiment configuration
scripts/                   Training, evaluation, and diagnostics entry points
env/                       Train, Difix, and evidence environment notes
docs/                      Protocol, runbook, reproducibility, and provenance
submodules/                CUDA extension sources
```

`SHA256SUMS.txt` records the bytes of every tracked file except the manifest itself. Verify it in a POSIX shell with `sha256sum -c SHA256SUMS.txt`.

The root-level `diffusion_guidance/`, `geometric_constraints/`, `gt_dca/`, and `tools/` directories are compatibility import shims only. Canonical research code lives under `evidence_track/`; new code should use those canonical paths.

## Provenance and claim boundary

This repository is based on `CPy255/GeoTrack-GS` commit `81ada6a32c918591ae7c7a0279dc6ca7a8018e2f`. The original upstream README is preserved as [`README_UPSTREAM.md`](README_UPSTREAM.md), and the complete provenance record is in [`docs/PROVENANCE.md`](docs/PROVENANCE.md). The former reconstructable-package layout is preserved by Git tag `reproducible-package-2026-09-17`.

The repository does not yet support claims of real Fern reconstruction improvement, diffusion-specific geometry improvement, reproducible real Difix inference, DINOv2 identity attribution, DTU geometry improvement, or state-of-the-art performance. Datasets, model weights, large caches, and checkpoints are not distributed with the repository.
