# EvidenceTrack-GS

[中文](README.md)

**Identity-controlled geometry diagnostics for diffusion-assisted sparse-view Gaussian Splatting.**

This is a materialized full-source repository ready for direct development. The GeoTrack-GS upstream source, cumulative research patch, source overlay, and guarded AST integration are present at the repository root. Future changes, tests, commits, and pushes should be made here; `bootstrap.sh` is no longer part of the normal workflow.

## Current status

| Component | Status |
| --- | --- |
| Full-source assembly | Passed: two independent assemblies, 0 byte mismatches across 66 target files |
| Patch gates | Passed: apply, cached whitespace, and diff checks |
| Python compileall | Passed |
| Phase-2.1 CPU smoke | 11 / 11 passed |
| Regression tests | 79 / 79 passed |
| Synthetic three-camera recovery | Passed |
| Fern CUDA preflight | Not run |
| A0 / A1 / SelfRender / B | Not run |
| Real Difix / DINOv2 / DTU | Not run |

Validation reports and raw logs are under [`validation/`](validation/). The current evidence verifies full-source reconstruction and CPU integrity gates. It does not establish that the real GPU reconstruction or paper experiments have been completed.

## Development entry point

Clone the full repository:

```bash
git clone https://github.com/zy8389/EvidenceTrack-GS.git
cd EvidenceTrack-GS
```

Install the CPU-check dependencies and run the gates:

```bash
pip install -r requirements-cpu.txt
export PYTHONPATH="$PWD"
python -m compileall -q diffusion_guidance geometric_constraints tools tests arguments scene gaussian_renderer train.py utils
python -m pytest -q tests
python tools/smoke_test_phase2_1.py
python tools/test_geometry_recovery.py --synthetic-smoke --output synthetic_recovery.json
```

On Windows PowerShell:

```powershell
$env:PYTHONPATH = (Get-Location).Path
python -m pytest -q tests
python tools/smoke_test_phase2_1.py
```

Start real experiments with [`research_docs/FERN_PRECHECK_AND_RUN.md`](research_docs/FERN_PRECHECK_AND_RUN.md) and `research_scripts/run_stage.sh`. Stop downstream GPU work if any P0 gate fails.

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
arguments/                 Argument definitions
diffusion_guidance/        Diffusion evidence, identity protocol, and checkpoint binding
geometric_constraints/     Strict track-based geometry constraints
scene/                     Scene and Gaussian state
tools/                     Build, audit, evaluation, and recovery tools
tests/                     CPU regression tests
configs/                   Controlled experiment configuration
research_scripts/          CPU and real-experiment scripts
research_docs/             Research notes and Fern runbook
research_provenance/       Cumulative patch and provenance records
validation/                Current full-source validation evidence
```

`SHA256SUMS.txt` records the bytes of every tracked file except the manifest itself. Verify it in a POSIX shell with `sha256sum -c SHA256SUMS.txt`.

## Provenance and claim boundary

This repository is based on `CPy255/GeoTrack-GS` commit `81ada6a32c918591ae7c7a0279dc6ca7a8018e2f`. The original upstream README is preserved as [`README_UPSTREAM.md`](README_UPSTREAM.md), and the complete provenance record is in [`research_provenance/README.md`](research_provenance/README.md). The former reconstructable-package layout is preserved by Git tag `reproducible-package-2026-09-17`.

The repository does not yet support claims of real Fern reconstruction improvement, diffusion-specific geometry improvement, reproducible real Difix inference, DINOv2 identity attribution, DTU geometry improvement, or state-of-the-art performance. Datasets, model weights, large caches, and checkpoints are not distributed with the repository.
