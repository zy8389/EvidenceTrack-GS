# EvidenceTrack-GS

**Identity-controlled geometry diagnostics for diffusion-assisted sparse-view Gaussian Splatting.**

> **Core question:** when diffusion-enhanced views look better, do they also provide **identity-specific geometric utility**, or are the gains explained by appearance improvement, local projection priors, extra optimization, or selective matching success?

[中文说明](README.md)

## Status

This repository is a **pre-experiment research implementation**. The controlled protocol and CPU-side integrity checks are implemented, but current-source CPU validation is not fully passing and the decisive real-scene GPU experiments have not yet been executed.

| Component | Status |
| --- | --- |
| Phase-2.1 CPU smoke checks | Failed: Difix cache metadata guard |
| Regression tests | 76 / 79 passed; 3 failed |
| Synthetic three-camera recovery | ✅ Passed |
| Fern CUDA preflight | ⏳ Not run |
| A0 / A1 / SelfRender / B reconstruction | ⏳ Not run |
| Real Difix inference | ⏳ Not run |
| DINOv2 identity diagnostic | ⏳ Not run |
| DTU independent geometry evaluation | ⏳ Not run |

Current-source CPU validation was rerun on 2026-09-16. Logs, synthetic results, and source hashes are stored under `code/validation/readme_update_*`. The supplied profile bundle's 11/11 smoke and 79/79 regression results belong to a different repaired-source snapshot and do not certify this revision. Older files under the root `validation/` directory are also historical snapshots.

## Research idea

Sparse-view 3D Gaussian Splatting is strongly underconstrained. A frozen diffusion model can improve the visual quality of pseudo or novel views, but visually plausible pixels are not automatically evidence of more accurate 3D geometry.

EvidenceTrack-GS therefore evaluates diffusion assistance under explicit controls rather than assuming that better appearance implies better geometry.

The first controlled experiment follows this structure:

```text
Source RGB + supplied cameras
          |
          v
Source-only feature tracks
          |
          v
      A0 @ 10k
      /   |    \
     /    |     \
 A1 @12k  SelfRender @12k  B @12k
 source     A0 pseudo        Difix pseudo
 only       targets           targets
     \      |      /
      \     |     /
    matched controlled comparison
              |
              v
      A1 / B final renders
              |
              v
 identity-controlled diagnostic
              |
              v
Correct vs Hard-Wrong vs Random
      vs Uniform vs ProjectionOnly
              |
              v
 failure-aware paired geometry evidence
```

The central matched comparison is **A1 vs B**, not A0 vs B. A1 and B must resume from the same complete A0 checkpoint with matched source-view sampling and update count. `SelfRender` is an additional control for generic pseudo-view supervision activity.

## What is being tested

The project separates several questions that are often conflated:

1. **Functional geometry** — does the source-track loss actually move current Gaussian centers in the correct direction?
2. **Diffusion utility** — does the Difix-supervised branch outperform an ordinary matched continuation?
3. **Identity specificity** — does the correct source-track identity outperform wrong, random, uniform, and projection-only controls on the same candidate support?
4. **Independent geometry** — do any apparent image or identity gains survive evaluation against genuine geometric reference data?

A positive image-quality result alone is **not** treated as proof of geometric improvement.

## Controlled branches

- **A0**: repaired source-track baseline trained to 10,000 steps.
- **A1**: ordinary source/track continuation from the exact A0 state to 12,000 steps.
- **SelfRender**: same continuation budget plus pseudo-view supervision using unchanged A0 renders.
- **B**: same continuation budget plus frozen Difix-enhanced pseudo-view RGB supervision.

The protocol audits checkpoint lineage, optimizer/RNG restoration, source-view sequences, pseudo-view schedules, camera fingerprints, cache hashes, and final-checkpoint provenance before results are accepted.

## Identity-controlled diagnostic

The evaluation uses source-track descriptors and compares the same local target support under:

- Correct identity
- Local hard-wrong identity
- Random query
- Uniform query
- Projection-only prediction

Matching failures remain in the denominator. The primary reports therefore include failure-aware error, PCK, hard-negative validity, geometry-relative gain, and identity margin rather than successful-match error alone.

The diagnostic is **evaluation-only**. It does not feed held-out RGB or held-out correspondences back into training and does not establish a causal mediation mechanism by itself.

## Repository structure

```text
code/          Research overlay, controlled protocol, tools, tests, and scripts
experiments/   Preregistered run matrix and measured-result templates
docs/          Protocol notes and Fern server runbook
validation/    Historical CPU validation snapshots
```

The executable source overlay is pinned to upstream GeoTrack-GS commit:

```text
81ada6a32c918591ae7c7a0279dc6ca7a8018e2f
```

## CPU validation

Install the CPU-side dependencies and run:

```bash
cd code
pip install -r requirements-cpu.txt
bash scripts/run_cpu.sh
```

The current repository source was checked on 2026-09-16 with these results:

```text
Phase-2.1 smoke:        FAIL (Difix cache metadata guard)
Pytest regression:      76 passed / 3 failed
Synthetic recovery:     PASS
```

The smoke fixture is missing `manifest_schema` and `reproducibility_check_sha256`. The three regression failures cover DINOv2 metadata (two tests) and camera-matrix mismatch detection (one test). This documentation update does not change research source code or repair those failures.

Because `run_cpu.sh` stops at the first failed gate, the regression and synthetic checks were also run independently. Only the synthetic point-recovery check passed in full; current-source CPU integrity validation remains incomplete. These results are **not** evidence that the real CUDA reconstruction, Difix model, DINOv2 diagnostic, or benchmark geometry pipeline works end-to-end.

## Fern GPU preflight

The first real experiment is intentionally staged. Follow:

```text
docs/FERN_PRECHECK_AND_RUN.md
```

Required order:

```text
Environment
→ Source-track build and audit
→ H5 / live-Scene camera projection equivalence
→ Live pseudo-camera audit
→ CUDA geometry smoke
→ A0 @ 10k
→ A1 / SelfRender / B
→ pair audits
→ reconstruction metrics
→ A1/B identity diagnostic
```

Any P0 integrity gate failure should stop the run before more expensive training.

## Claim boundary

At the current repository state, the following are **not yet established**:

- real-scene reconstruction improvement,
- diffusion-specific geometric improvement,
- real Difix reproducibility on the target GPU stack,
- DINOv2 identity attribution on LLFF,
- DTU geometry improvement,
- runtime or peak-VRAM claims,
- state-of-the-art performance.

The repository is designed so that negative or null results remain valid outcomes of the study.

## Reproducibility

The research package records controlled configuration, hashes, camera provenance, source-image inventories, checkpoint lineage, and pair-audit metadata. Datasets, pretrained weights, generated caches, and large checkpoints are not redistributed here.

`SHA256SUMS.txt` records the exact bytes of every tracked artifact except itself. On a POSIX shell, verify it with `sha256sum -c SHA256SUMS.txt`.

See:

- `code/README.md` for assembly and execution details
- `docs/FERN_PRECHECK_AND_RUN.md` for the first real server run
- `experiments/llff_primary_run_matrix.json` for the preregistered LLFF run matrix
- `code/THIRD_PARTY_NOTICES.md` for third-party boundaries

## Acknowledgement

This project builds on GeoTrack-GS and uses external components such as 3D Gaussian Splatting, Difix, and DINOv2 under their respective licenses and terms. See `code/THIRD_PARTY_NOTICES.md` for the current dependency notice.
