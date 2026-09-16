#!/usr/bin/env bash
set -euo pipefail

# Historical command reference; use scripts/run_stage.sh for parameterized execution.
GEOTRACK_REPO=/ABS/PATH/GeoTrack-GS
PHASE21_PATCH=/ABS/PATH/phase2_1_integrity.patch
SCENE_PATH=/ABS/DATA/LLFF/fern
WORK_PATH=/ABS/WORK/fern_phase2_1
OUTPUT_PATH=/ABS/OUTPUT/fern_phase2_1
DIFIX_REPO=/ABS/TOOLS/Difix3D

cd "$GEOTRACK_REPO"
git checkout 81ada6a32c918591ae7c7a0279dc6ca7a8018e2f
git apply --check "$PHASE21_PATCH"
git apply "$PHASE21_PATCH"
PYTHONPATH=. python tools/smoke_test_phase2_1.py

mkdir -p "$WORK_PATH" "$OUTPUT_PATH"

# Exact 3-view source / held-out split used by Scene.
python tools/make_sparse_view_split.py \
  --sparse-dir "$SCENE_PATH/sparse/0" \
  --n-views 3 \
  --llff-holdout 8 \
  --output-dir "$WORK_PATH/split"

# Builds Track identity from source-source SIFT matches among the three source
# RGB images. points3D.bin is never opened; poses/intrinsics are given inputs.
python tools/build_anchor_tracks_from_colmap.py \
  --sparse-dir "$SCENE_PATH/sparse/0" \
  --images-dir "$SCENE_PATH/images" \
  --output "$WORK_PATH/tracks_source_only_v4.h5" \
  --anchor-image-list "$WORK_PATH/split/source_images.txt" \
  --heldout-image-list "$WORK_PATH/split/heldout_images.txt" \
  --pixel-sigma 1.0 \
  --max-source-reprojection-error 4.0 \
  --match-ratio-threshold 0.75 \
  --epipolar-threshold 1.5 \
  --min-heldout-match-votes 2

python tools/audit_anchor_tracks.py \
  "$WORK_PATH/tracks_source_only_v4.h5" \
  --min-tracks 32

# 1. Strict Geometry Smoke. This must be the first server gate.
python train.py \
  -s "$SCENE_PATH" \
  -m "$OUTPUT_PATH/strict_geometry_smoke" \
  --data_type colmap \
  --eval \
  --llff_holdout 8 \
  --n_views 3 \
  --track_path "$WORK_PATH/tracks_source_only_v4.h5" \
  --strict_source_only_geometry \
  --strict_tracks \
  --disable_legacy_pseudo_depth \
  --experiment_seed 1 \
  --geometry_smoke_only

# Produce A0, the shared 10k checkpoint reference.
python train.py \
  -s "$SCENE_PATH" \
  -m "$OUTPUT_PATH/A0_repaired_10k" \
  --data_type colmap \
  --eval \
  --llff_holdout 8 \
  --n_views 3 \
  --track_path "$WORK_PATH/tracks_source_only_v4.h5" \
  --strict_source_only_geometry \
  --strict_tracks \
  --strict_track_weight 0.1 \
  --disable_legacy_pseudo_depth \
  --geometry_reg_enabled \
  --experiment_seed 1 \
  --controlled_ab_role A0 \
  --iterations 10000 \
  --test_iterations 3000 5000 10000 \
  --save_iterations 5000 10000 \
  --checkpoint_iterations 5000 10000

A0_CHECKPOINT="$OUTPUT_PATH/A0_repaired_10k/chkpnt10000.pth"

# 2. Geometry Recovery Functional Test. Writes only a separate JSON report;
# the checkpoint is loaded read-only and in-memory XYZ is restored.
python tools/test_geometry_recovery.py \
  -s "$SCENE_PATH" \
  -m "$OUTPUT_PATH/geometry_recovery_scratch" \
  --data_type colmap \
  --eval \
  --llff_holdout 8 \
  --n_views 3 \
  --track_path "$WORK_PATH/tracks_source_only_v4.h5" \
  --strict_source_only_geometry \
  --strict_tracks \
  --checkpoint "$A0_CHECKPOINT" \
  --steps 100 \
  --perturbation-ratios 0.005 0.01 0.02 \
  --experiment_seed 1 \
  --output "$OUTPUT_PATH/geometry_recovery.json"

# Fixed pseudo camera pool and reproducible frozen Difix cache.
python tools/export_pseudo_views.py \
  -s "$SCENE_PATH" \
  -m "$OUTPUT_PATH/A0_repaired_10k" \
  --iteration 10000 \
  --data_type colmap \
  --eval \
  --llff_holdout 8 \
  --n_views 3 \
  --track_path "$WORK_PATH/tracks_source_only_v4.h5" \
  --strict_source_only_geometry \
  --output-dir "$OUTPUT_PATH/pseudo_cache" \
  --max-pseudo-views 32 \
  --seed 1

python tools/run_difix_cache.py \
  --manifest "$OUTPUT_PATH/pseudo_cache/manifest.jsonl" \
  --difix-repo "$DIFIX_REPO" \
  --model-id nvidia/difix_ref \
  --dtype fp16 \
  --timestep 199 \
  --guidance-scale 0.0 \
  --seed 1 \
  --reproducibility-check

# 3A. A1: same 10k checkpoint -> normal continuation to 12k, no Difix.
python train.py \
  -s "$SCENE_PATH" \
  -m "$OUTPUT_PATH/A1_repaired_12k_no_difix" \
  --data_type colmap \
  --eval \
  --llff_holdout 8 \
  --n_views 3 \
  --track_path "$WORK_PATH/tracks_source_only_v4.h5" \
  --strict_source_only_geometry \
  --strict_tracks \
  --strict_track_weight 0.1 \
  --disable_legacy_pseudo_depth \
  --geometry_reg_enabled \
  --experiment_seed 1 \
  --controlled_ab_role A1 \
  --controlled_ab_checkpoint_iteration 10000 \
  --start_checkpoint "$A0_CHECKPOINT" \
  --iterations 12000 \
  --test_iterations 11000 12000 \
  --save_iterations 12000 \
  --checkpoint_iterations 12000

# 3B. B: identical continuation state and real-view schedule; pseudo RGB is the
# only experimental variable.
python train.py \
  -s "$SCENE_PATH" \
  -m "$OUTPUT_PATH/B_repaired_12k_difix" \
  --data_type colmap \
  --eval \
  --llff_holdout 8 \
  --n_views 3 \
  --track_path "$WORK_PATH/tracks_source_only_v4.h5" \
  --strict_source_only_geometry \
  --strict_tracks \
  --strict_track_weight 0.1 \
  --disable_legacy_pseudo_depth \
  --geometry_reg_enabled \
  --experiment_seed 1 \
  --controlled_ab_role B \
  --controlled_ab_checkpoint_iteration 10000 \
  --start_checkpoint "$A0_CHECKPOINT" \
  --iterations 12000 \
  --enable_diffusion_pseudo_rgb \
  --pseudo_rgb_manifest "$OUTPUT_PATH/pseudo_cache/manifest.jsonl" \
  --pseudo_rgb_weight 0.10 \
  --pseudo_rgb_lambda_dssim 0.20 \
  --pseudo_rgb_start 10000 \
  --pseudo_rgb_end 12000 \
  --pseudo_rgb_interval 50 \
  --pseudo_rgb_strict_cache \
  --test_iterations 11000 12000 \
  --save_iterations 12000 \
  --checkpoint_iterations 12000

# Record the realized B coverage in the experiment sheet. The default schedule
# has 39 calls and 32/32 unique views; the training command also enforces full
# pool coverage before returning success.
python - "$OUTPUT_PATH/B_repaired_12k_difix/pseudo_supervision_usage.json" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    usage = json.load(handle)
assert usage["pseudo_supervision_calls"] >= 32, usage
assert usage["unique_pseudo_views_used"] == 32, usage
print(json.dumps(usage, indent=2, sort_keys=True))
PY

# 4. Projection / GS / Difix / Real Track Evidence diagnostic.
python tools/export_heldout_views.py \
  -s "$SCENE_PATH" \
  -m "$OUTPUT_PATH/A1_repaired_12k_no_difix" \
  --iteration 12000 \
  --data_type colmap \
  --eval \
  --llff_holdout 8 \
  --n_views 3 \
  --track_path "$WORK_PATH/tracks_source_only_v4.h5" \
  --strict_source_only_geometry \
  --output-dir "$OUTPUT_PATH/heldout_evidence" \
  --seed 1

python tools/run_difix_cache.py \
  --manifest "$OUTPUT_PATH/heldout_evidence/difix_manifest.jsonl" \
  --difix-repo "$DIFIX_REPO" \
  --model-id nvidia/difix_ref \
  --dtype fp16 \
  --timestep 199 \
  --guidance-scale 0.0 \
  --seed 1 \
  --reproducibility-check

python tools/evaluate_track_evidence.py \
  --track-h5 "$WORK_PATH/tracks_source_only_v4.h5" \
  --images-dir "$SCENE_PATH/images" \
  --target-manifest "$OUTPUT_PATH/heldout_evidence/evidence_manifest.jsonl" \
  --feature-backend dinov2 \
  --dinov2-model dinov2_vits14 \
  --window-radii 32 \
  --shift-pixels 4 8 16 \
  --hard-negative-radius 64 \
  --temperature 0.07 \
  --output-dir "$OUTPUT_PATH/evidence_dinov2"
