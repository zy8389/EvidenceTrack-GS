# Fern precheck and first controlled run

This is the executable order for a fresh clone of the **materialized full-source repository** and a fresh `$RUN` directory. It preserves the preregistered Fern setting: LLFF, three source views, holdout 8, seed 1, A0 at 10k, and A1/SelfRender/B at 12k. Do not reuse a checkpoint, cache, manifest, or gate report from another run.

The requested checklist placed camera projection before Track construction. The projection gate compares the live Scene path against the same H5 anchors used by the geometry loss, so those anchors must exist first. The strict executable dependency is therefore:

```text
environment -> tracks -> projection -> live pseudo-camera audit -> CUDA geometry smoke -> A0
```

Every machine-readable gate must contain the exact JSON boolean `passed: true`. Missing, malformed, false, or stale reports stop downstream stages. The Track coverage fields are descriptive, but the combined Track report is a gate because it also records the leakage audit, minimum count, and H5 hash.

## 0. Clone a clean checkout

Clone the current full-source repository. The legacy `bootstrap.sh` is not part of this workflow.

```bash
git clone https://github.com/zy8389/EvidenceTrack-GS.git
cd EvidenceTrack-GS
export PYTHONPATH="$PWD"
```

Install the pinned upstream environment and its repository-specific CUDA extensions. Do not substitute similarly named standard 3DGS extensions.

```bash
pip install ./submodules/diff-gaussian-rasterization-confidence
pip install ./submodules/simple-knn
```

Set a new output directory. The role for Fern is always `development`. `DATASET`
is a required protocol label that is written into each pair audit, preventing a
scene/seed audit from being reused for another dataset.

```bash
export SCENE=/datasets/LLFF/fern
export DATASET=LLFF
export RUN=/experiments/fern_v3_seed1_research_v2
export SEED=1
export VIEWS=3
export HOLDOUT=8
test ! -e "$RUN" || { echo "Choose a fresh RUN directory" >&2; exit 2; }
mkdir -p "$RUN"
```

## 1. Environment and dependency gate

```bash
bash research_scripts/run_stage.sh env
```

Required outputs are `$RUN/environment.json` and `$RUN/environment_gate.json`. The gate verifies the pinned upstream commit, research revision marker, CUDA availability, recorded CUDA device/build, and imports of the repository's rasterizer and simple-knn extensions. It also hashes every Python source file plus `research_scripts/run_stage.sh` (or its source-tree `scripts/run_stage.sh` counterpart), `configs/controlled_protocol.json`, and every `requirements-*.txt` file. Downstream stages recompute both inventories and reject additions, removals, or changed bytes until `env` is rerun. Continue only when `environment_gate.json` has `passed: true`.

## 2. Source-track build, leakage audit, and coverage

```bash
bash research_scripts/run_stage.sh tracks
```

This builds the fixed source/held-out split and `$RUN/tracks_source_only.h5`. `$RUN/track_coverage.json` combines the strict source-only leakage audit, minimum 32-track check, H5 SHA256, length/reprojection/conditioning distributions, 8x8 source-image coverage, and 3D spread. The later live Gaussian stage supplies association statistics. A failed Track command removes the previous Track gate, projection gate, and smoke gate for this `$RUN`.

## 3. H5 versus live-Scene projection gate

```bash
bash research_scripts/run_stage.sh projection
```

`$RUN/camera_projection_equivalence.json` compares every source H5 anchor through both projection paths at original and training resolutions. The locked convention is world-to-camera `R @ X + t`; row-vector tensors use `X @ R.T + t`; the live geometry path uses upstream `world_view_transform.T`. The maximum allowed difference is `1e-4 px`.

The stage stamps a context fingerprint over the Track H5, environment reports, COLMAP camera files, source-image bytes, view/holdout settings, revision marker, and relevant projection/training code. Both `smoke` and `A0` recompute it. Any changed input makes the report stale and blocks training until `env`, `tracks`, and `projection` are rerun as needed.

## 4. Live pseudo-camera provenance gate

```bash
bash research_scripts/run_stage.sh pseudo-live
```

This pre-A0 gate audits `scene.getPseudoCameras()` directly, before any expensive training. It must show that the pinned implementation derives pseudo poses from source cameras only and that every live camera has explicit `fx`, `fy`, `cx`, `cy`, width, height, world-to-camera pose, resize convention, source declaration, ID, and fingerprint. Continue only when `$RUN/pseudo_live_camera_audit.json` has `passed: true`.

This establishes camera provenance; it does not turn a pseudo render into an independent physical observation. The claim remains “source-RGB-only geometry conditional on supplied calibration.”

## 5. Real CUDA geometry smoke

```bash
bash research_scripts/run_stage.sh smoke
```

The live training path must emit `$RUN/geometry_smoke/geometry_smoke.json` with finite loss, nonzero gradient, nonempty Track-to-Gaussian association, collision/coverage statistics, and `passed: true`. `run_stage.sh` binds that evidence to the current projection-context fingerprint. A missing live pseudo-camera gate, NaN/Inf, zero gradient, empty supervision, or stale projection report stops here.

## 6. Common A0 at 10k

```bash
bash research_scripts/run_stage.sh A0
```

This stage requires current passed environment, Track, projection, live pseudo-camera, and geometry-smoke gates. It writes the complete log to `$RUN/A0/train.log` and must create `$RUN/A0/chkpnt10000.pth`. That checkpoint embeds `controlled_checkpoint_provenance_v2`, including the canonical scene/Track/source-image inputs, calibrated source cameras, seed, strict geometry settings, and complete common training protocol. Do not use a checkpoint made by an older state format or source revision.

## 7. Read-only real Gaussian recovery gate

```bash
bash research_scripts/run_stage.sh recover
```

`$RUN/recovery.json` tests `0.005`, `0.010`, and `0.020 x camera_extent` perturbations. Every perturbation level must retain finite, nonzero gradients and reduce both reprojection error and anchor distance; the report must have `passed: true`. The A0 checkpoint remains unmodified.

## 8. Export and audit the exact pseudo-camera pool

```bash
bash research_scripts/run_stage.sh export
bash research_scripts/run_stage.sh pseudo-audit
```

The first command exports exactly 32 A0 pseudo renders. The second audits the actual `$RUN/pseudo/manifest.jsonl` records and writes `$RUN/pseudo/provenance_audit.json`. It checks unique camera fingerprints and explicit pose, intrinsics, dimensions, camera source, calibration source, input, target, and reference fields. Each input render, source reference, and shared A0 checkpoint is SHA256-bound in the manifest; every record also stores the checkpoint-embedded A0 provenance digest. The audit rereads those bytes, checks that digest against the checkpoint, and verifies the A0 render dimensions against the camera domain. The passed report is bound to the manifest SHA256; any later manifest edit or same-path asset replacement makes downstream strict validation fail.

The pre-A0 `pseudo-live` and post-A0 `pseudo-audit` stages answer different questions and both are required.

## 9. Build SelfRender and frozen Difix targets

Create the SelfRender manifest first. Its target is the corresponding immutable A0 render.

```bash
bash research_scripts/run_stage.sh self-manifest
```

In the pinned official Difix environment, set immutable provenance and build the B cache:

```bash
export DIFIX_REPO=/workspace/Difix3D
export DIFIX_MODEL_REVISION=<immutable-40-character-model-commit>
bash research_scripts/run_stage.sh cache
```

Continue only when `$RUN/pseudo/manifest.jsonl.difix_metadata.json` contains `reproducibility_check.passed: true`. Every target sidecar must agree with the run metadata and must bind camera fingerprint, full-frame resolution, input/reference/output SHA256, model revision, and Difix code commit. Stale targets are errors; rebuild the entire cache rather than editing metadata.

## 10. Matched A1, SelfRender, and B continuations

```bash
bash research_scripts/run_stage.sh A1
bash research_scripts/run_stage.sh SelfRender
bash research_scripts/run_stage.sh B
```

All three continuations validate the embedded A0 provenance against the current scene, Track, source-image bytes, calibrated cameras, seed, and common protocol before the first training update. They then start from the same complete A0 state and use the same 2,000-source-update sequence. Their logs are `$RUN/A1/train.log`, `$RUN/SelfRender/train.log`, and `$RUN/B/train.log`. SelfRender and B additionally share the exact 32 camera keys, 39-call schedule, pseudo loss, and weight. Their only pseudo-target difference is A0 render versus frozen Difix(A0 render).

The three planned contrasts have distinct meanings:

```text
SelfRender - A1   generic extra pseudo-view supervision
B - SelfRender    frozen Difix-target replacement effect
B - A1            total frozen-Difix branch effect
```

These are controlled contrasts, not assumed positive findings.

## 11. Pair and pseudo-control audits

```bash
bash research_scripts/run_stage.sh pair-self
bash research_scripts/run_stage.sh pair-b
bash research_scripts/run_stage.sh pair-pseudo
```

The first two reports verify the complete A0 SHA and embedded provenance, serialized state format, optimizer/RNG restoration evidence, start/final iterations, exact 2,000-entry source sequence, fixed `10000<iteration<12000` / 50-step pseudo schedule, 39/32 realized usage, and current manifest/target bytes. Pair-audit schema v5 additionally requires the canonical common model/optimization/pipeline/runtime parameter object, every source image's canonical path and SHA256, and every calibrated source training-camera payload. It records the pseudo-supervised methods whose strict usage/manifest validation actually ran; omitting `--require-pseudo` makes an A1/B or A1/SelfRender audit fail. For each audited 12k method it binds the exact checkpoint path and file SHA256, reloads its serialized render state and `controlled_checkpoint_provenance_v2`, verifies the method role, parent A0 path/hash/provenance digest, and pseudo-manifest lineage where applicable, then includes that final checkpoint in the canonical audited-file inventory. These objects and hashes must be identical where the protocol requires them across A1/SelfRender/B. Every downstream audit read rejects non-canonical paths, non-lowercase hashes, incomplete pseudo-validation coverage, or changed bytes. The third independently re-runs strict manifest validation for both pseudo branches: SelfRender must still point to exact A0-render bytes, B must still match its Difix sidecars, cache fingerprint, and output bytes, and both must declare strict frozen-cache validation. It then verifies identical camera keys and realized schedules while preserving the declared target-kind difference. All three JSON reports must have `passed: true`.

Every dataset/scene/seed group needs its own `pair_id` and audit paths. An audit from Fern seed 1 cannot authorize another seed or scene.

## 12. Reconstruction metrics with an explicit role

```bash
bash research_scripts/run_stage.sh metrics
```

The stage reads the three canonical `train.log` files and atomically writes `$RUN/measured_results_a1_b.csv`, `$RUN/measured_results_a1_selfrender.csv`, and `$RUN/measured_results_selfrender_b.csv`. Each measured row carries the exact opaque `pair_id` emitted by its audit. The aggregator rejects a third unrelated arm in a contrast CSV, duplicate arm rows, missing row bindings, cross-group audit reuse, and an audit that does not declare the requested contrast. Do not fill `NR` values from memory or another run. Fern and room always remain `development`; they cannot enter a confirmatory confidence interval.

## 13. A1/B final-render identity diagnostic

Export both final reconstructions on the same held-out camera set, then build the paired-support manifest and frozen diagnostic caches:

```bash
bash research_scripts/run_stage.sh evidence-export-a1
bash research_scripts/run_stage.sh evidence-export-b
bash research_scripts/run_stage.sh evidence-cache-a1
bash research_scripts/run_stage.sh evidence-cache-b
bash research_scripts/run_stage.sh identity-manifest
```

Set an immutable local DINOv2 checkout and exact weight file, then evaluate both arms:

```bash
export DINOV2_REPO=/workspace/dinov2
export DINOV2_WEIGHT_PATH=/workspace/weights/dinov2_vits14.pth
bash research_scripts/run_stage.sh identity-a1
bash research_scripts/run_stage.sh identity-b
bash research_scripts/run_stage.sh identity-report
```

Every evidence export/cache/manifest/identity stage first regenerates the A1/B pair audit from the current continuation metadata, source sequences, A0 checkpoint, pseudo usage, manifest, target bytes, and exact A1/B final checkpoints. The audit is deterministic when those inputs are unchanged. A changed audit hash makes existing held-out manifests or paired selectors stale and forces their regeneration before expensive downstream work. Each held-out export restores the audit-authorized complete 12k checkpoint after loading its matching Scene PLY, rejects a path/hash/role/lineage/count/render-state mismatch, and records the checkpoint provenance digest with every render under `heldout_identity_diagnostic_v2`. `identity-manifest` freezes one `paired_identity_manifest_v4` selector and v4 sidecar containing both exact checkpoint bindings, the shared held-out camera payload, real/reference hashes, and each arm's immutable render hashes. The v4 final association retains both final-checkpoint provenance digests. `identity-a1` and `identity-b` consume that same selector with `--paired-arm A1|B`; they cannot independently choose a different held-out manifest. The paired-support hash, held-out image list, H5 tracks, primary radius, feature backend, DINO repository commit, weight SHA256, image/feature resolutions, and effective stride must match between A1 and B. The primary radius is explicitly 32 px; 16/48 px are robustness analyses.

## 14. Hard-negative and failure-aware review

Inspect both `paired_report.json` files and `$RUN/a1_b_identity_association.json`. The primary identity margin is:

```text
wrong_penalized_error - correct_penalized_error
```

It uses only `local_hard_negative` records with `within_radius=true`. `global_fallback` may be reported separately and must not enter the primary margin. Correct and Wrong failures receive the same image-diagonal penalty. Record eligible count, valid local-hard-negative count, Correct failures, Wrong failures, both-success count, PCK denominator, and failure rate. Successful-only margin is secondary.

The A1/B diagnostic supports an association statement. It does not establish causality or global 3D correctness.

## 15. Final integrity summary

Only after all real logs and group-specific audits exist:

```bash
bash research_scripts/run_stage.sh integrity
```

This command regenerates the three pair audits, the three contrast CSVs, and the A1/B identity association before writing `$RUN/run_integrity.json`. That report is scoped to exactly one dataset/scene/seed and records `aggregate_performed: false`. A passed Fern seed-1 integrity report proves that this run's artifacts are current and mutually bound; it is not a multi-seed or cross-scene efficacy result.

After every required scene/seed has its own passed integrity report, combine each contrast's CSV files into a separate single-header file and aggregate them independently. For example, repeat the command for each metric:

```bash
python tools/aggregate_results.py --csv all_measured_results_a1_b.csv --metric lpips --baseline A1 --treatment B --role confirmatory --output aggregate_a1_b_lpips.json
python tools/aggregate_results.py --csv all_measured_results_a1_selfrender.csv --metric lpips --baseline A1 --treatment SelfRender --role confirmatory --output aggregate_a1_selfrender_lpips.json
python tools/aggregate_results.py --csv all_measured_results_selfrender_b.csv --metric lpips --baseline SelfRender --treatment B --role confirmatory --output aggregate_selfrender_b_lpips.json
```

The aggregator rejects rows whose dataset/scene/seed `pair_id` lacks its own passed audit. For the first Fern run, inspect development results only; do not present a one-scene interval as a confirmatory result.

## Current verification boundary

As of 2026-09-17, the regenerated cumulative patch passed both apply checks (including `--whitespace=error-all`) and `git diff --check`. Two independent fresh assemblies completed successfully with zero byte mismatches across 66 target files. The materialized full-source tree passed compileall, 79/79 pytest checks, 11/11 CPU smoke checks, and synthetic recovery. Evidence is stored under `validation/`.

The following are still **NOT VERIFIED**:

- CUDA extension import and live renderer integration;
- real Fern Track, projection, pseudo-camera, geometry-smoke, A0, and recovery gates;
- real Difix cache/reproducibility and target-sidecar validation;
- real A1, SelfRender, B training and all three pair audits;
- real DINOv2 A1/B identity diagnostics;
- confirmatory six-scene statistics and independent DTU geometry evaluation;
- runtime, peak GPU memory, and final reconstruction metrics.

Files explicitly marked as historical snapshots must not be cited as validation of this revision. A positive B-A1 metric does not isolate the Difix-target replacement, while a positive B-SelfRender metric still does not establish a diffusion-specific mechanism or improved geometry without the planned enhancement control and independent geometry evidence. A passed GPU smoke proves only that the recovery mechanism works under its tested perturbations.
