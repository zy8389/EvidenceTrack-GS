#!/usr/bin/env bash
# Execute inside the assembled, pinned upstream checkout. Every gate fails closed.
set -euo pipefail

STAGE=${1:?Stages: env tracks projection pseudo-live smoke A0 recover export pseudo-audit self-manifest cache A1 SelfRender B pair-self pair-b pair-pseudo evidence-export-a1 evidence-export-b evidence-cache-a1 evidence-cache-b identity-manifest identity-a1 identity-b identity-report report metrics integrity}
: "${SCENE:?Set SCENE to the calibrated scene root}"
: "${RUN:?Set RUN to an absolute output root for this scene and seed}"
: "${DATASET:?Set DATASET to the protocol dataset label, e.g. LLFF}"
SEED=${SEED:-1}
VIEWS=${VIEWS:-3}
HOLDOUT=${HOLDOUT:-8}
EXPECTED_UPSTREAM_COMMIT=81ada6a32c918591ae7c7a0279dc6ca7a8018e2f

[[ -f .research_revision_v2.json && -f train.py ]] || {
  echo 'Run from the materialized full-source repository' >&2
  exit 2
}
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
mkdir -p "$RUN"

SCENE_NAME=$(basename -- "$SCENE")
EXPERIMENT_ROLE=$(python - "$DATASET" "$SCENE_NAME" <<'PY'
import sys

from diffusion_guidance.experiment_registry import expected_scene_role

print(expected_scene_role(sys.argv[1], sys.argv[2]))
PY
)
readonly SCENE_NAME EXPERIMENT_ROLE

TRACK="$RUN/tracks_source_only.h5"
ENVIRONMENT_REPORT="$RUN/environment.json"
ENVIRONMENT_GATE="$RUN/environment_gate.json"
TRACK_COVERAGE="$RUN/track_coverage.json"
PROJECTION_GATE="$RUN/camera_projection_equivalence.json"
SMOKE_GATE="$RUN/geometry_smoke_gate.json"
SMOKE_REPORT="$RUN/geometry_smoke/geometry_smoke.json"
PSEUDO_MANIFEST="$RUN/pseudo/manifest.jsonl"
PSEUDO_GATE="$RUN/pseudo/provenance_audit.json"
PSEUDO_LIVE_GATE="$RUN/pseudo_live_camera_audit.json"
PROJECTION_MODEL="$RUN/projection_gate_scene"
PSEUDO_LIVE_MODEL="$RUN/pseudo_live_gate_scene"
A0="$RUN/A0"
A1="$RUN/A1"
SELF="$RUN/SelfRender"
B="$RUN/B"
CKPT="$A0/chkpnt10000.pth"
PAIRED_IDENTITY_MANIFEST="$RUN/paired_identity_manifest.jsonl"
PAIRED_IDENTITY_METADATA="$PAIRED_IDENTITY_MANIFEST.metadata.json"
IDENTITY_ASSOCIATION="$RUN/a1_b_identity_association.json"
METRICS_A1_B="$RUN/measured_results_a1_b.csv"
METRICS_A1_SELF="$RUN/measured_results_a1_selfrender.csv"
METRICS_SELF_B="$RUN/measured_results_selfrender_b.csv"
INTEGRITY_REPORT="$RUN/run_integrity.json"
COMMON=(-s "$SCENE" --data_type colmap --eval --llff_holdout "$HOLDOUT" --n_views "$VIEWS" --track_path "$TRACK" --strict_source_only_geometry)
TRAIN=("${COMMON[@]}" --strict_tracks --strict_track_weight 0.1 --disable_legacy_pseudo_depth --disable_depth_loss --experiment_seed "$SEED" --densify_until_iter 10000 --use_color)

die() {
  echo "$*" >&2
  exit 2
}

require_file() {
  local path=$1
  local label=$2
  [[ -s "$path" ]] || die "$label is missing or empty: $path"
}

reset_run_subdir() {
  local path=$1
  local label=$2
  [[ "$path" == "$RUN"/* && "$path" != "$RUN" ]] || {
    die "Refuse to reset $label outside the current RUN: $path"
  }
  rm -rf -- "$path"
  mkdir -p -- "$path"
}

run_logged() {
  local log_path=$1
  shift
  mkdir -p -- "$(dirname -- "$log_path")"
  local -a statuses
  set +e
  "$@" 2>&1 | tee "$log_path"
  statuses=("${PIPESTATUS[@]}")
  set -e
  [[ "${statuses[1]}" -eq 0 ]] || die "tee failed while writing training log: $log_path"
  [[ "${statuses[0]}" -eq 0 ]] || {
    die "Training command failed with status ${statuses[0]}; log retained at $log_path"
  }
  require_file "$log_path" "training log"
}

require_completed_training_runs() {
  local path status
  for path in "$A0" "$A1" "$SELF" "$B"; do
    [[ -f "$path/run_status.json" ]] || die "Training run has no run_status.json; refusing metrics/integrity: $path"
    status=$(python - "$path/run_status.json" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(payload.get("status", "UNKNOWN"))
PY
)
    [[ "$status" == "COMPLETED" ]] || die "Training run is not COMPLETED ($status); refusing metrics/integrity: $path"
  done
}

require_json_object() {
  local path=$1
  local label=$2
  require_file "$path" "$label"
  python - "$path" "$label" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
label = sys.argv[2]
try:
    payload = json.loads(path.read_text(encoding="utf-8"))
except Exception as exc:
    raise SystemExit(f"{label} is not valid JSON: {path}: {exc}")
if not isinstance(payload, dict):
    raise SystemExit(f"{label} must be a JSON object: {path}")
PY
}

require_passed_json() {
  local path=$1
  local label=$2
  require_json_object "$path" "$label"
  python - "$path" "$label" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
label = sys.argv[2]
payload = json.loads(path.read_text(encoding="utf-8"))
if payload.get("passed") is not True:
    reason = payload.get("failures", payload.get("reason", "missing exact boolean passed=true"))
    raise SystemExit(f"{label} did not pass: {path}: {reason}")
print(f"{label}: passed=true")
PY
}

require_nested_passed_json() {
  local path=$1
  local dotted_key=$2
  local label=$3
  require_json_object "$path" "$label"
  python - "$path" "$dotted_key" "$label" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
keys = sys.argv[2].split(".")
label = sys.argv[3]
value = json.loads(path.read_text(encoding="utf-8"))
for key in keys:
    if not isinstance(value, dict) or key not in value:
        raise SystemExit(f"{label} lacks required boolean {'.'.join(keys)}: {path}")
    value = value[key]
if value is not True:
    raise SystemExit(f"{label} requires {'.'.join(keys)}=true: {path}")
print(f"{label}: {'.'.join(keys)}=true")
PY
}

run_json_gate() {
  local report=$1
  local label=$2
  shift 2
  rm -f -- "$report"
  local command_status=0
  "$@" || command_status=$?
  require_passed_json "$report" "$label"
  [[ "$command_status" -eq 0 ]] || die "$label command exited with status $command_status despite its report"
}

write_environment_gate() {
  rm -f -- "$ENVIRONMENT_GATE"
  python - "$ENVIRONMENT_REPORT" "$ENVIRONMENT_GATE" "$EXPECTED_UPSTREAM_COMMIT" <<'PY'
import importlib
import hashlib
import json
import subprocess
import sys
from pathlib import Path

environment_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
expected_commit = sys.argv[3]
environment = json.loads(environment_path.read_text(encoding="utf-8"))

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

checks = {
    "cuda_available": environment.get("cuda_available") is True,
    "cuda_build_recorded": bool(environment.get("torch_cuda")),
    "cuda_device_recorded": bool(environment.get("devices")),
    "torch_version_recorded": bool(environment.get("packages", {}).get("torch")),
}
details = {}
try:
    current_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.STDOUT
    ).strip()
except Exception as exc:
    current_commit = None
    details["git_error"] = str(exc)
checks["pinned_upstream_commit"] = current_commit == expected_commit
checks["environment_report_commit"] = environment.get("git_commit") == expected_commit
details["current_commit"] = current_commit
details["expected_commit"] = expected_commit

marker_path = Path(".research_revision_v2.json").resolve()
try:
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
except Exception as exc:
    marker = {}
    details["revision_marker_error"] = str(exc)
checks["research_revision_marker"] = marker.get("revision") == "research-v2"

root = marker_path.parent
recorded_python = {
    str(relative).replace("\\", "/"): str(digest)
    for relative, digest in environment.get("python_sha256", {}).items()
} if isinstance(environment.get("python_sha256"), dict) else {}
current_python = {
    path.relative_to(root).as_posix(): sha256(path)
    for path in root.rglob("*.py")
    if ".git" not in path.parts
}
checks["python_source_inventory"] = bool(recorded_python) and recorded_python == current_python

protocol_candidates = [
    root / "research_scripts" / "run_stage.sh",
    root / "scripts" / "run_stage.sh",
    root / "configs" / "controlled_protocol.json",
    *sorted(root.glob("requirements-*.txt")),
]
recorded_protocol = {
    str(relative).replace("\\", "/"): str(digest)
    for relative, digest in environment.get("protocol_sha256", {}).items()
} if isinstance(environment.get("protocol_sha256"), dict) else {}
current_protocol = {
    path.relative_to(root).as_posix(): sha256(path)
    for path in protocol_candidates
    if path.is_file()
}
checks["protocol_file_inventory"] = (
    bool(recorded_protocol) and recorded_protocol == current_protocol
)
details["python_source_inventory_count"] = len(current_python)
details["protocol_file_inventory_count"] = len(current_protocol)

for module_name, key in (
    ("diff_gaussian_rasterization", "rasterizer_extension_import"),
    ("simple_knn._C", "simple_knn_extension_import"),
):
    try:
        importlib.import_module(module_name)
        checks[key] = True
    except Exception as exc:
        checks[key] = False
        details[f"{key}_error"] = repr(exc)

result = {
    "schema": "environment_cuda_gate_v1",
    "gate": "P0_environment_and_cuda_dependencies",
    "environment_report": str(environment_path.resolve()),
    "environment_report_sha256": sha256(environment_path),
    "research_revision_marker": str(marker_path),
    "research_revision_marker_sha256": sha256(marker_path),
    "checks": checks,
    "details": details,
    "passed": all(checks.values()),
}
output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(result, indent=2, sort_keys=True))
PY
  require_passed_json "$ENVIRONMENT_GATE" "environment/CUDA gate"
}

require_environment_gate() {
  require_json_object "$ENVIRONMENT_REPORT" "environment report"
  require_passed_json "$ENVIRONMENT_GATE" "environment/CUDA gate"
  python - "$ENVIRONMENT_REPORT" "$ENVIRONMENT_GATE" "$EXPECTED_UPSTREAM_COMMIT" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

environment_path, gate_path = map(Path, sys.argv[1:3])
expected_commit = sys.argv[3]

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

environment = json.loads(environment_path.read_text(encoding="utf-8"))
gate = json.loads(gate_path.read_text(encoding="utf-8"))
marker = Path(str(gate.get("research_revision_marker", ""))).expanduser().resolve()
expected_checks = {
    "cuda_available",
    "cuda_build_recorded",
    "cuda_device_recorded",
    "torch_version_recorded",
    "pinned_upstream_commit",
    "environment_report_commit",
    "research_revision_marker",
    "python_source_inventory",
    "protocol_file_inventory",
    "rasterizer_extension_import",
    "simple_knn_extension_import",
}
if gate.get("schema") != "environment_cuda_gate_v1" or gate.get("gate") != "P0_environment_and_cuda_dependencies":
    raise SystemExit("Environment gate has an unsupported schema or identity; rerun env")
if set(gate.get("checks", {})) != expected_checks or any(value is not True for value in gate["checks"].values()):
    raise SystemExit("Environment gate lacks the complete exact-true check set; rerun env")
if Path(str(gate.get("environment_report", ""))).expanduser().resolve() != environment_path.resolve() or gate.get("environment_report_sha256") != sha256(environment_path):
    raise SystemExit("Environment gate is stale relative to environment.json; rerun env")
if not marker.is_file() or marker != (Path.cwd() / ".research_revision_v2.json").resolve() or gate.get("research_revision_marker_sha256") != sha256(marker):
    raise SystemExit("Environment gate research revision marker is missing or stale; rerun env")
details = gate.get("details", {})
if details.get("current_commit") != expected_commit or details.get("expected_commit") != expected_commit or environment.get("git_commit") != expected_commit:
    raise SystemExit("Environment gate is not bound to the pinned upstream commit; rerun env")
root = marker.parent
recorded_python = {
    str(relative).replace("\\", "/"): str(digest)
    for relative, digest in environment.get("python_sha256", {}).items()
} if isinstance(environment.get("python_sha256"), dict) else {}
current_python = {
    path.relative_to(root).as_posix(): sha256(path)
    for path in root.rglob("*.py")
    if ".git" not in path.parts
}
if not recorded_python or recorded_python != current_python:
    raise SystemExit("Environment Python source inventory is stale; rerun env")
protocol_candidates = [
    root / "research_scripts" / "run_stage.sh",
    root / "scripts" / "run_stage.sh",
    root / "configs" / "controlled_protocol.json",
    *sorted(root.glob("requirements-*.txt")),
]
recorded_protocol = {
    str(relative).replace("\\", "/"): str(digest)
    for relative, digest in environment.get("protocol_sha256", {}).items()
} if isinstance(environment.get("protocol_sha256"), dict) else {}
current_protocol = {
    path.relative_to(root).as_posix(): sha256(path)
    for path in protocol_candidates
    if path.is_file()
}
if not recorded_protocol or recorded_protocol != current_protocol:
    raise SystemExit("Environment protocol-file inventory is stale; rerun env")
PY
}

require_track_gate() {
  require_passed_json "$TRACK_COVERAGE" "strict Track build/leakage/coverage gate"
  python - "$TRACK_COVERAGE" "$TRACK" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

report_path, track_path = map(Path, sys.argv[1:])
if not track_path.is_file():
    raise SystemExit(f"Strict Track H5 is missing: {track_path}")
payload = json.loads(report_path.read_text(encoding="utf-8"))
actual = hashlib.sha256(track_path.read_bytes()).hexdigest()
if payload.get("track_sha256") != actual:
    raise SystemExit("Strict Track gate is stale; rerun tracks")
PY
}

# Stamp or verify the exact inputs that make a projection report reusable. This
# prevents a previously passed report from surviving a changed H5, calibration,
# source image, environment, resolution setting, or relevant code path.
projection_context() {
  local mode=$1
  python - "$mode" "$PROJECTION_GATE" "$TRACK" "$ENVIRONMENT_REPORT" "$ENVIRONMENT_GATE" "$SCENE" "$RUN/split/source_images.txt" "$VIEWS" "$HOLDOUT" "$PWD" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

mode, report_arg, track_arg, environment_arg, environment_gate_arg, scene_arg, source_list_arg, views, holdout, root_arg = sys.argv[1:]
report_path = Path(report_arg)
root = Path(root_arg).resolve()
scene = Path(scene_arg).expanduser().resolve()

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

inputs = {}
def add(label: str, path: Path) -> None:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise SystemExit(f"Projection context input is missing: {label}: {path}")
    inputs[label] = {"path": str(path), "sha256": sha256(path)}

add("strict_track_h5", Path(track_arg))
add("environment_report", Path(environment_arg))
add("environment_gate", Path(environment_gate_arg))
add("research_revision_marker", root / ".research_revision_v2.json")
for relative in (
    "train.py",
    "arguments/__init__.py",
    "tools/check_camera_projection_equivalence.py",
    "geometric_constraints/repaired_geometry.py",
    "geometric_constraints/strict_track_store.py",
    "diffusion_guidance/calibration_guard.py",
    "scene/__init__.py",
    "scene/cameras.py",
    "scene/dataset_readers.py",
    "utils/camera_utils.py",
    "utils/graphics_utils.py",
):
    add(f"code:{relative}", root / relative)

for stem in ("cameras", "images"):
    candidates = [scene / "sparse" / "0" / f"{stem}.bin", scene / "sparse" / "0" / f"{stem}.txt"]
    present = [path for path in candidates if path.is_file()]
    if not present:
        raise SystemExit(f"Projection context lacks COLMAP {stem}.bin/.txt under {scene / 'sparse' / '0'}")
    for path in present:
        add(f"colmap:{path.name}", path)

source_list = Path(source_list_arg)
add("source_image_list", source_list)
source_names = [line.strip() for line in source_list.read_text(encoding="utf-8").splitlines() if line.strip()]
if not source_names:
    raise SystemExit(f"Source image list is empty: {source_list}")
for index, name in enumerate(source_names):
    add(f"source_image:{index}:{name}", scene / "images" / name)

context = {
    "schema": 1,
    "scene": str(scene),
    "n_views": int(views),
    "llff_holdout": int(holdout),
    "strict_source_only_geometry": True,
    "inputs": inputs,
}
serialized = json.dumps(context, sort_keys=True, separators=(",", ":")).encode("utf-8")
fingerprint = hashlib.sha256(serialized).hexdigest()
payload = json.loads(report_path.read_text(encoding="utf-8"))
if payload.get("passed") is not True:
    raise SystemExit(f"Projection gate lacks exact passed=true: {report_path}")

if mode == "stamp":
    payload["run_stage_context"] = {"fingerprint": fingerprint, **context}
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Projection gate context: {fingerprint}")
elif mode == "check":
    stored = payload.get("run_stage_context")
    if not isinstance(stored, dict) or stored.get("schema") != 1 or stored.get("fingerprint") != fingerprint:
        old_inputs = stored.get("inputs", {}) if isinstance(stored, dict) else {}
        changed = sorted(
            key for key in set(old_inputs) | set(inputs)
            if old_inputs.get(key) != inputs.get(key)
        )
        suffix = f"; changed inputs: {changed}" if changed else ""
        raise SystemExit(f"Projection gate is stale; rerun stages env, tracks, projection{suffix}")
    print(f"Projection gate context current: {fingerprint}")
else:
    raise SystemExit(f"Unknown projection context mode: {mode}")
PY
}

require_projection_gate() {
  require_environment_gate
  require_passed_json "$PROJECTION_GATE" "H5/live-Scene projection gate"
  projection_context check
}

write_smoke_gate() {
  python - "$SMOKE_REPORT" "$SMOKE_GATE" "$PROJECTION_GATE" "$TRACK" <<'PY'
import hashlib
import json
import math
import os
import sys
from pathlib import Path

source_path, output_path, projection_path, track_path = map(Path, sys.argv[1:])

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def finite_number(value, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SystemExit(f"Geometry smoke {label} must be numeric")
    value = float(value)
    if not math.isfinite(value):
        raise SystemExit(f"Geometry smoke {label} must be finite")
    return value

if not source_path.is_file():
    raise SystemExit(f"Live train.py smoke report is missing: {source_path}")
if not track_path.is_file():
    raise SystemExit(f"Strict Track H5 is missing: {track_path}")

source = json.loads(source_path.read_text(encoding="utf-8"))
if not isinstance(source, dict):
    raise SystemExit("Live train.py smoke report must be a JSON object")
if source.get("schema") != 1:
    raise SystemExit("Live train.py smoke report has an unsupported schema")
if source.get("gate") != "P0_real_cuda_geometry_smoke":
    raise SystemExit("Live train.py smoke report has the wrong gate identity")
if source.get("passed") is not True:
    raise SystemExit("Live train.py geometry/CUDA smoke did not pass")

actual_track_path = track_path.resolve()
reported_track_path = Path(str(source.get("track_h5", ""))).resolve()
actual_track_sha = sha256(actual_track_path)
if reported_track_path != actual_track_path:
    raise SystemExit("Geometry smoke report refers to a different Track H5")
if source.get("track_h5_sha256") != actual_track_sha:
    raise SystemExit("Geometry smoke Track H5 hash mismatch")

camera_extent = finite_number(source.get("camera_extent"), "camera_extent")
if camera_extent <= 0.0:
    raise SystemExit("Geometry smoke camera_extent must be positive")

gradient = source.get("gradient_probe")
association = source.get("association")
renderer = source.get("renderer_probe")
thresholds = source.get("thresholds")
if not all(isinstance(value, dict) for value in (gradient, association, renderer, thresholds)):
    raise SystemExit("Geometry smoke lacks gradient, association, renderer, or threshold evidence")

coverage = finite_number(gradient.get("coverage"), "gradient coverage")
zero_ratio = finite_number(gradient.get("zero_gradient_ratio"), "zero-gradient ratio")
mean_norm = finite_number(gradient.get("mean_gradient_norm"), "mean gradient norm")
loss_value = finite_number(gradient.get("loss"), "gradient-probe loss")
observation_count = finite_number(
    gradient.get("observation_count"), "gradient-probe observation_count"
)
minimum_coverage = finite_number(
    thresholds.get("minimum_gradient_coverage"), "minimum gradient coverage"
)
if not 0.0 <= minimum_coverage <= 1.0:
    raise SystemExit("Geometry smoke minimum gradient coverage is outside [0, 1]")
if coverage < minimum_coverage or coverage <= 0.0:
    raise SystemExit("Geometry smoke gradient coverage is below its recorded threshold")
if not 0.0 <= zero_ratio <= 1.0 or abs((1.0 - coverage) - zero_ratio) > 1e-6:
    raise SystemExit("Geometry smoke gradient coverage and zero ratio are inconsistent")
if mean_norm <= 0.0:
    raise SystemExit("Geometry smoke mean gradient norm must be positive")
if loss_value < 0.0:
    raise SystemExit("Geometry smoke gradient-probe loss must be non-negative")
if observation_count <= 0.0 or not observation_count.is_integer():
    raise SystemExit("Geometry smoke gradient probe must contain observations")

positive_counts = ("active_tracks", "associated_tracks", "unique_associated_gaussians")
for key in positive_counts:
    value = finite_number(association.get(key), f"association.{key}")
    if value <= 0.0 or not value.is_integer():
        raise SystemExit(f"Geometry smoke association.{key} must be a positive integer")
for key in ("association_collision_rate", "mean_association_distance", "p95_association_distance"):
    value = finite_number(association.get(key), f"association.{key}")
    if value < 0.0:
        raise SystemExit(f"Geometry smoke association.{key} must be non-negative")
collision_rate = float(association["association_collision_rate"])
if collision_rate > 1.0:
    raise SystemExit("Geometry smoke association collision rate is outside [0, 1]")
rejected = finite_number(association.get("rejected_associations"), "association.rejected_associations")
if rejected < 0.0 or not rejected.is_integer():
    raise SystemExit("Geometry smoke rejected_associations must be a non-negative integer")

if renderer.get("passed") is not True:
    raise SystemExit("Live CUDA renderer probe did not pass")
if renderer.get("cuda") is not True or renderer.get("finite") is not True:
    raise SystemExit("Live renderer output must be finite and resident on CUDA")
pixel_count = finite_number(renderer.get("pixel_count"), "renderer pixel_count")
finite_pixel_count = finite_number(
    renderer.get("finite_pixel_count"), "renderer finite_pixel_count"
)
gaussian_count = finite_number(renderer.get("gaussian_count"), "renderer gaussian_count")
source_camera_count = finite_number(
    renderer.get("source_camera_count"), "renderer source_camera_count"
)
if pixel_count <= 0 or gaussian_count <= 0 or source_camera_count <= 0:
    raise SystemExit("Live renderer probe has an empty image, Gaussian set, or source-camera set")
if finite_pixel_count != pixel_count:
    raise SystemExit("Live renderer output contains NaN or Inf values")
if not str(renderer.get("camera_name", "")).strip():
    raise SystemExit("Live renderer probe lacks a source camera name")

for key in (
    "positive_finite_mean_gradient_norm",
    "nonempty_association",
    "live_cuda_renderer",
):
    if thresholds.get(key) is not True:
        raise SystemExit(f"Geometry smoke threshold declaration {key}=true is missing")

projection = json.loads(projection_path.read_text(encoding="utf-8"))
fingerprint = projection.get("run_stage_context", {}).get("fingerprint")
if projection.get("passed") is not True or not fingerprint:
    raise SystemExit("Cannot stamp geometry smoke without a current passed projection gate")

result = dict(source)
result.update({
    "source_report": str(source_path.resolve()),
    "source_report_sha256": sha256(source_path),
    "projection_context_fingerprint": fingerprint,
    "validation_checks": {
        "source_passed_exact_true": True,
        "track_path_and_sha256_current": True,
        "finite_positive_camera_extent": True,
        "finite_nonzero_gradient": True,
        "finite_nonempty_geometry_loss": True,
        "nonempty_association": True,
        "finite_live_cuda_renderer": True,
        "projection_context_bound": True,
    },
    "passed": source.get("passed") is True,
})
temporary = output_path.with_suffix(output_path.suffix + ".tmp")
temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(temporary, output_path)
PY
  require_passed_json "$SMOKE_GATE" "real CUDA geometry smoke gate"
}

require_smoke_gate() {
  require_projection_gate
  require_passed_json "$SMOKE_GATE" "real CUDA geometry smoke gate"
  python - "$SMOKE_GATE" "$SMOKE_REPORT" "$PROJECTION_GATE" "$TRACK" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

smoke = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
source_path = Path(sys.argv[2])
projection = json.loads(Path(sys.argv[3]).read_text(encoding="utf-8"))
track_path = Path(sys.argv[4])

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

current = projection.get("run_stage_context", {}).get("fingerprint")
if not current or smoke.get("projection_context_fingerprint") != current:
    raise SystemExit("Geometry smoke gate is stale relative to the current projection gate; rerun smoke")
if not source_path.is_file() or smoke.get("source_report") != str(source_path.resolve()):
    raise SystemExit("Geometry smoke source report is missing or has moved; rerun smoke")
if smoke.get("source_report_sha256") != sha256(source_path):
    raise SystemExit("Geometry smoke source report changed after validation; rerun smoke")
if not track_path.is_file() or smoke.get("track_h5_sha256") != sha256(track_path):
    raise SystemExit("Geometry smoke Track H5 is missing or stale; rerun tracks and smoke")
if smoke.get("renderer_probe", {}).get("passed") is not True:
    raise SystemExit("Geometry smoke lacks a passed live CUDA renderer probe")
checks = smoke.get("validation_checks")
if not isinstance(checks, dict) or not checks or any(value is not True for value in checks.values()):
    raise SystemExit("Geometry smoke gate lacks complete exact-true validation checks")
PY
}

require_recovery_gate() {
  require_smoke_gate
  require_passed_json "$RUN/recovery.json" "real Gaussian recovery gate"
  python - "$RUN/recovery.json" "$CKPT" "$TRACK" "$SMOKE_REPORT" "$SEED" "$SCENE" "$RUN/split/source_images.txt" <<'PY'
import hashlib
import json
import math
import sys
from pathlib import Path

from diffusion_guidance.checkpoint_state import load_checkpoint_summary
from diffusion_guidance.control_identity import (
    normalized_camera_names,
    source_camera_set_sha256,
)

recovery_path, checkpoint_path, track_path, smoke_path = map(Path, sys.argv[1:5])
seed_value = int(sys.argv[5])
scene_path = Path(sys.argv[6]).expanduser().resolve()
source_list_path = Path(sys.argv[7]).expanduser().resolve()

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def number(value, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SystemExit(f"Recovery {label} must be numeric")
    value = float(value)
    if not math.isfinite(value):
        raise SystemExit(f"Recovery {label} must be finite")
    return value

recovery = json.loads(recovery_path.read_text(encoding="utf-8"))
smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
if recovery.get("schema") != "real_gaussian_geometry_recovery_v1" or recovery.get("gate") != "P0_real_gaussian_geometry_recovery" or recovery.get("mode") != "real" or recovery.get("passed") is not True:
    raise SystemExit("Recovery gate has the wrong schema, mode, or passed state")
if recovery.get("experiment_seed") != seed_value or recovery.get("optimization_steps") != 100 or recovery.get("requested_track_count") != 32 or recovery.get("perturbation_ratios") != [0.005, 0.01, 0.02] or number(recovery.get("minimum_reduction"), "minimum_reduction") != 0.10:
    raise SystemExit("Recovery gate changed the preregistered protocol")
if not scene_path.is_dir() or Path(str(recovery.get("scene_source_path", ""))).expanduser().resolve() != scene_path:
    raise SystemExit("Recovery gate is stale relative to the calibrated scene")
if not source_list_path.is_file():
    raise SystemExit("Recovery gate source-image list is missing")
source_entries = [
    line.strip()
    for line in source_list_path.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
source_names = normalized_camera_names(source_entries)
if recovery.get("source_camera_names") != source_names or recovery.get("source_camera_set_sha256") != source_camera_set_sha256(source_names):
    raise SystemExit("Recovery gate is stale relative to the source-camera set")
if not checkpoint_path.is_file() or recovery.get("checkpoint") != str(checkpoint_path.resolve()) or recovery.get("checkpoint_sha256") != sha256(checkpoint_path) or recovery.get("checkpoint_sha256_after") != recovery.get("checkpoint_sha256") or recovery.get("checkpoint_modified") is not False:
    raise SystemExit("Recovery gate is stale relative to the A0 checkpoint")
checkpoint = load_checkpoint_summary(
    checkpoint_path,
    expected_iteration=10000,
    require_cuda_rng=True,
    require_controlled_provenance=True,
)
if (
    recovery.get("checkpoint_iteration") != checkpoint["iteration"]
    or recovery.get("checkpoint_state_format") != checkpoint["format"]
    or recovery.get("checkpoint_gaussian_count") != checkpoint["gaussian_count"]
    or recovery.get("checkpoint_render_state_schema")
    != checkpoint["render_state_schema"]
    or recovery.get("checkpoint_render_state_sha256")
    != checkpoint["render_state_sha256"]
    or recovery.get("checkpoint_controlled_provenance_sha256")
    != checkpoint["controlled_provenance_sha256"]
):
    raise SystemExit("Recovery gate is stale relative to the complete A0 checkpoint state")
if not track_path.is_file() or recovery.get("track_h5") != str(track_path.resolve()) or recovery.get("track_h5_sha256") != sha256(track_path):
    raise SystemExit("Recovery gate is stale relative to the strict Track H5")
if number(recovery.get("camera_extent"), "camera_extent") != number(smoke.get("camera_extent"), "smoke camera_extent"):
    raise SystemExit("Recovery gate camera extent differs from geometry smoke")
results = recovery.get("results")
if not isinstance(results, list) or len(results) != 3 or number(recovery.get("recovery_success_rate"), "success_rate") != 1.0:
    raise SystemExit("Recovery gate lacks all three passed perturbation results")
for index, (result, ratio) in enumerate(zip(results, [0.005, 0.01, 0.02])):
    if result.get("perturbation_scene_radius_ratio") != ratio or result.get("nan_free") is not True or result.get("passed") is not True or result.get("valid_associated_gaussian_count") != 32 or result.get("unique_gaussian_count") != 32:
        raise SystemExit(f"Recovery result {index} is not an exact passed result")
    initial_pixel = number(result.get("initial_pixel_reprojection_error"), f"result {index} initial pixel")
    final_pixel = number(result.get("final_pixel_reprojection_error"), f"result {index} final pixel")
    initial_3d = number(result.get("initial_3d_anchor_distance"), f"result {index} initial 3D")
    final_3d = number(result.get("final_3d_anchor_distance"), f"result {index} final 3D")
    if initial_pixel <= 0.0 or initial_3d <= 0.0 or final_pixel > initial_pixel * 0.90 or final_3d > initial_3d * 0.90 or number(result.get("convergence_rate"), f"result {index} convergence") < 0.80 or number(result.get("mean_gradient_norm"), f"result {index} gradient") <= 0.0:
        raise SystemExit(f"Recovery result {index} misses the fixed reduction/convergence gate")
PY
}

stamp_pseudo_gate() {
  python - "$PSEUDO_GATE" "$PSEUDO_MANIFEST" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

report_path, manifest_path = map(Path, sys.argv[1:])
payload = json.loads(report_path.read_text(encoding="utf-8"))
if payload.get("passed") is not True:
    raise SystemExit("Cannot stamp a failed pseudo-camera provenance gate")
payload["manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

require_pseudo_gate() {
  require_recovery_gate
  require_pseudo_live_gate
  require_passed_json "$PSEUDO_GATE" "pseudo-camera provenance gate"
  python - "$PSEUDO_GATE" "$PSEUDO_MANIFEST" "$PSEUDO_LIVE_GATE" "$PROJECTION_GATE" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

report_path, manifest_path, live_path, projection_path = map(Path, sys.argv[1:])
if not manifest_path.is_file():
    raise SystemExit(f"Pseudo manifest is missing: {manifest_path}")
payload = json.loads(report_path.read_text(encoding="utf-8"))
actual = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
if payload.get("manifest_sha256") != actual:
    raise SystemExit("Pseudo-camera provenance gate is stale; rerun pseudo-audit")
if not live_path.is_file() or not projection_path.is_file():
    raise SystemExit("Pseudo camera gate requires the current live/projection reports")
live = json.loads(live_path.read_text(encoding="utf-8"))
projection = json.loads(projection_path.read_text(encoding="utf-8"))
current = projection.get("run_stage_context", {}).get("fingerprint")
if live.get("passed") is not True or not current:
    raise SystemExit("Pseudo camera gate lacks a current passed live/projection context")
if payload.get("projection_context_fingerprint") != current:
    raise SystemExit("Pseudo manifest was not generated from the current live/projection context")
if payload.get("live_pseudo_audit_sha256") != hashlib.sha256(live_path.read_bytes()).hexdigest():
    raise SystemExit("Pseudo manifest binds a different live pseudo-camera audit")
records = []
with manifest_path.open(encoding="utf-8") as handle:
    for line in handle:
        if line.strip():
            records.append(json.loads(line))
keys = [str(record.get("key", "")) for record in records]
if len(records) != 32 or len(set(keys)) != 32:
    raise SystemExit("Pseudo manifest must contain exactly 32 unique camera records")
PY
}

require_pseudo_live_gate() {
  require_projection_gate
  require_passed_json "$PSEUDO_LIVE_GATE" "live pseudo-camera provenance gate"
  python - "$PSEUDO_LIVE_GATE" "$PROJECTION_GATE" <<'PY'
import json
import sys
from pathlib import Path

live = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
projection = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
current = projection.get("run_stage_context", {}).get("fingerprint")
stored = live.get("projection_context_fingerprint")
if not current or stored != current:
    raise SystemExit("Live pseudo-camera gate is stale relative to the current projection gate; rerun pseudo-live")
PY
}

require_paired_identity_manifest() {
  audit_pair_b
  require_file "$PAIRED_IDENTITY_MANIFEST" "paired A1/B identity manifest"
  require_passed_json "$PAIRED_IDENTITY_METADATA" "paired identity manifest metadata"
  python - "$PAIRED_IDENTITY_MANIFEST" <<'PY'
import sys
from pathlib import Path

from tools.evaluate_track_evidence import (
    load_paired_target_manifest,
    validate_paired_manifest_metadata,
)

manifest = Path(sys.argv[1]).expanduser().resolve()
validate_paired_manifest_metadata(manifest, "A1")
validate_paired_manifest_metadata(manifest, "B")
load_paired_target_manifest(manifest, "A1")
load_paired_target_manifest(manifest, "B")
print(f"Paired identity manifest and sidecar are current: {manifest}")
PY
}

audit_pair_self() {
  require_recovery_gate
  run_json_gate "$RUN/a1_selfrender_pair_audit.json" "A1/SelfRender controlled-pair gate" \
    python tools/audit_controlled_pair.py --baseline "$A1" --treatment "$SELF" --require-pseudo --dataset "$DATASET" --scene "$SCENE_NAME" --output "$RUN/a1_selfrender_pair_audit.json"
}

audit_pair_b() {
  require_recovery_gate
  run_json_gate "$RUN/a1_b_pair_audit.json" "A1/B controlled-pair gate" \
    python tools/audit_controlled_pair.py --baseline "$A1" --treatment "$B" --require-pseudo --dataset "$DATASET" --scene "$SCENE_NAME" --output "$RUN/a1_b_pair_audit.json"
}

audit_pair_pseudo() {
  require_recovery_gate
  require_passed_json "$RUN/a1_selfrender_pair_audit.json" "A1/SelfRender controlled-pair gate"
  require_passed_json "$RUN/a1_b_pair_audit.json" "A1/B controlled-pair gate"
  run_json_gate "$RUN/selfrender_b_pseudo_control_audit.json" "SelfRender/B pseudo-control gate" \
    python tools/audit_pseudo_control_pair.py --self-render "$SELF" --b "$B" --self-render-manifest "$RUN/pseudo/self_render_manifest.jsonl" --b-manifest "$PSEUDO_MANIFEST" --dataset "$DATASET" --scene "$SCENE_NAME" --output "$RUN/selfrender_b_pseudo_control_audit.json"
}

generate_contrast_metrics() {
  audit_pair_self
  audit_pair_b
  audit_pair_pseudo
  local staging="$RUN/.metrics_staging"
  reset_run_subdir "$staging" "metrics staging directory"
  rm -f -- "$METRICS_A1_B" "$METRICS_A1_SELF" "$METRICS_SELF_B"
  python tools/build_contrast_metrics.py --baseline A1 --baseline-log "$A1/train.log" --treatment B --treatment-log "$B/train.log" --pair-audit "$RUN/a1_b_pair_audit.json" --dataset "$DATASET" --scene "$SCENE_NAME" --seed "$SEED" --output "$staging/a1_b.csv"
  python tools/build_contrast_metrics.py --baseline A1 --baseline-log "$A1/train.log" --treatment SelfRender --treatment-log "$SELF/train.log" --pair-audit "$RUN/a1_selfrender_pair_audit.json" --dataset "$DATASET" --scene "$SCENE_NAME" --seed "$SEED" --output "$staging/a1_selfrender.csv"
  python tools/build_contrast_metrics.py --baseline SelfRender --baseline-log "$SELF/train.log" --treatment B --treatment-log "$B/train.log" --pair-audit "$RUN/selfrender_b_pseudo_control_audit.json" --dataset "$DATASET" --scene "$SCENE_NAME" --seed "$SEED" --output "$staging/selfrender_b.csv"
  mv -- "$staging/a1_b.csv" "$METRICS_A1_B"
  mv -- "$staging/a1_selfrender.csv" "$METRICS_A1_SELF"
  mv -- "$staging/selfrender_b.csv" "$METRICS_SELF_B"
  rm -rf -- "$staging"
}

refresh_identity_association() {
  require_paired_identity_manifest
  run_json_gate "$IDENTITY_ASSOCIATION" "A1/B identity association gate" \
    python tools/compare_identity_reports.py --a1 "$RUN/identity_a1/paired_report.json" --b "$RUN/identity_b/paired_report.json" --output "$IDENTITY_ASSOCIATION"
}

need_difix() {
  : "${DIFIX_REPO:?Set immutable official Difix checkout}"
  : "${DIFIX_MODEL_REVISION:?Set immutable 40-character model revision}"
}

case "$STAGE" in
  env)
    rm -f -- "$ENVIRONMENT_REPORT" "$ENVIRONMENT_GATE"
    python tools/collect_environment.py --output "$ENVIRONMENT_REPORT"
    require_json_object "$ENVIRONMENT_REPORT" "environment report"
    write_environment_gate
    ;;
  tracks)
    require_environment_gate
    rm -f -- "$TRACK" "$TRACK_COVERAGE" "$PROJECTION_GATE" "$SMOKE_GATE"
    python tools/make_sparse_view_split.py --sparse-dir "$SCENE/sparse/0" --n-views "$VIEWS" --llff-holdout "$HOLDOUT" --output-dir "$RUN/split"
    python tools/build_anchor_tracks_from_colmap.py --sparse-dir "$SCENE/sparse/0" --images-dir "$SCENE/images" --output "$TRACK" --anchor-image-list "$RUN/split/source_images.txt" --heldout-image-list "$RUN/split/heldout_images.txt" --pixel-sigma 1 --max-source-reprojection-error 4 --match-ratio-threshold 0.75 --epipolar-threshold 1.5 --min-heldout-match-votes 2
    run_json_gate "$TRACK_COVERAGE" "strict Track build/leakage/coverage gate" \
      python tools/audit_anchor_tracks.py "$TRACK" --min-tracks 32 --grid-size 8 --report "$TRACK_COVERAGE"
    ;;
  projection)
    require_environment_gate
    require_track_gate
    reset_run_subdir "$PROJECTION_MODEL" "projection-gate model directory"
    run_json_gate "$PROJECTION_GATE" "H5/live-Scene projection gate" \
      python tools/check_camera_projection_equivalence.py "${COMMON[@]}" -m "$PROJECTION_MODEL" --track-h5 "$TRACK" --output "$PROJECTION_GATE"
    projection_context stamp
    require_projection_gate
    ;;
  pseudo-live)
    require_environment_gate
    require_track_gate
    require_projection_gate
    reset_run_subdir "$PSEUDO_LIVE_MODEL" "live-pseudo-gate model directory"
    rm -f -- "$PSEUDO_LIVE_GATE"
    run_json_gate "$PSEUDO_LIVE_GATE" "live pseudo-camera provenance gate" \
      python tools/audit_live_pseudo_cameras.py "${COMMON[@]}" -m "$PSEUDO_LIVE_MODEL" --output "$PSEUDO_LIVE_GATE" --seed "$SEED"
    python - "$PSEUDO_LIVE_GATE" "$PROJECTION_GATE" <<'PY'
import json
import sys
from pathlib import Path

live_path, projection_path = map(Path, sys.argv[1:])
live = json.loads(live_path.read_text(encoding="utf-8"))
projection = json.loads(projection_path.read_text(encoding="utf-8"))
fingerprint = projection.get("run_stage_context", {}).get("fingerprint")
if not fingerprint:
    raise SystemExit("Cannot bind live pseudo-camera gate without a current projection fingerprint")
live["projection_context_fingerprint"] = fingerprint
live_path.write_text(json.dumps(live, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
    require_pseudo_live_gate
    ;;
  smoke)
    require_projection_gate
    require_pseudo_live_gate
    reset_run_subdir "$RUN/geometry_smoke" "geometry smoke scratch directory"
    rm -f -- "$SMOKE_GATE" "$SMOKE_REPORT"
    python train.py "${TRAIN[@]}" -m "$RUN/geometry_smoke" --geometry_smoke_only
    write_smoke_gate
    ;;
  A0)
    require_projection_gate
    require_pseudo_live_gate
    require_smoke_gate
    reset_run_subdir "$A0" "A0 run directory"
    run_logged "$A0/train.log" \
      python train.py "${TRAIN[@]}" -m "$A0" --controlled_ab_role A0 --iterations 10000 --test_iterations 10000 --save_iterations 10000 --checkpoint_iterations 10000
    require_file "$CKPT" "A0 final checkpoint"
    ;;
  recover)
    require_projection_gate
    require_pseudo_live_gate
    require_smoke_gate
    require_file "$CKPT" "A0 final checkpoint"
    reset_run_subdir "$RUN/recovery_scratch" "recovery scratch directory"
    run_json_gate "$RUN/recovery.json" "real Gaussian recovery gate" \
      python tools/test_geometry_recovery.py "${COMMON[@]}" --strict_tracks -m "$RUN/recovery_scratch" --checkpoint "$CKPT" --steps 100 --perturbation-ratios 0.005 0.01 0.02 --experiment_seed "$SEED" --output "$RUN/recovery.json"
    require_recovery_gate
    ;;
  export)
    require_projection_gate
    require_pseudo_live_gate
    require_smoke_gate
    require_file "$CKPT" "A0 final checkpoint"
    require_recovery_gate
    reset_run_subdir "$RUN/pseudo" "pseudo export directory"
    python tools/export_pseudo_views.py "${COMMON[@]}" -m "$A0" --iteration 10000 --a0-checkpoint "$CKPT" --live-pseudo-audit "$PSEUDO_LIVE_GATE" --output-dir "$RUN/pseudo" --max-pseudo-views 32 --seed "$SEED"
    require_file "$PSEUDO_MANIFEST" "pseudo-camera manifest"
    ;;
  pseudo-audit)
    require_recovery_gate
    require_file "$PSEUDO_MANIFEST" "pseudo-camera manifest; run export first"
    run_json_gate "$PSEUDO_GATE" "pseudo-camera provenance gate" \
      python tools/audit_pseudo_camera_manifest.py --manifest "$PSEUDO_MANIFEST" --output "$PSEUDO_GATE"
    stamp_pseudo_gate
    require_pseudo_gate
    ;;
  self-manifest)
    require_pseudo_gate
    require_recovery_gate
    rm -f -- "$RUN/pseudo/self_render_manifest.jsonl"
    python tools/make_self_render_manifest.py --input-manifest "$PSEUDO_MANIFEST" --output-manifest "$RUN/pseudo/self_render_manifest.jsonl"
    ;;
  cache)
    require_pseudo_gate
    require_recovery_gate
    need_difix
    python tools/run_difix_cache.py --manifest "$PSEUDO_MANIFEST" --difix-repo "$DIFIX_REPO" --model-id nvidia/difix_ref --model-revision "$DIFIX_MODEL_REVISION" --dtype fp16 --timestep 199 --guidance-scale 0 --seed "$SEED" --reproducibility-check
    require_nested_passed_json "$PSEUDO_MANIFEST.difix_metadata.json" "reproducibility_check.passed" "Difix pseudo-cache reproducibility gate"
    ;;
  A1)
    require_projection_gate
    require_pseudo_live_gate
    require_smoke_gate
    require_file "$CKPT" "A0 final checkpoint"
    require_recovery_gate
    reset_run_subdir "$A1" "A1 run directory"
    run_logged "$A1/train.log" \
      python train.py "${TRAIN[@]}" -m "$A1" --controlled_ab_role A1 --controlled_ab_checkpoint_iteration 10000 --start_checkpoint "$CKPT" --iterations 12000 --test_iterations 12000 --save_iterations 12000 --checkpoint_iterations 12000
    ;;
  SelfRender)
    require_projection_gate
    require_pseudo_live_gate
    require_smoke_gate
    require_pseudo_gate
    require_file "$CKPT" "A0 final checkpoint"
    require_recovery_gate
    require_file "$RUN/pseudo/self_render_manifest.jsonl" "SelfRender manifest"
    reset_run_subdir "$SELF" "SelfRender run directory"
    run_logged "$SELF/train.log" \
      python train.py "${TRAIN[@]}" -m "$SELF" --controlled_ab_role SelfRender --controlled_ab_checkpoint_iteration 10000 --start_checkpoint "$CKPT" --iterations 12000 --test_iterations 12000 --save_iterations 12000 --checkpoint_iterations 12000 --enable_diffusion_pseudo_rgb --pseudo_rgb_manifest "$RUN/pseudo/self_render_manifest.jsonl" --pseudo_rgb_weight 0.1 --pseudo_rgb_lambda_dssim 0.2 --pseudo_rgb_start 10000 --pseudo_rgb_end 12000 --pseudo_rgb_interval 50 --pseudo_rgb_strict_cache
    ;;
  B)
    require_projection_gate
    require_pseudo_live_gate
    require_smoke_gate
    require_pseudo_gate
    require_file "$CKPT" "A0 final checkpoint"
    require_recovery_gate
    require_nested_passed_json "$PSEUDO_MANIFEST.difix_metadata.json" "reproducibility_check.passed" "Difix pseudo-cache reproducibility gate"
    reset_run_subdir "$B" "B run directory"
    run_logged "$B/train.log" \
      python train.py "${TRAIN[@]}" -m "$B" --controlled_ab_role B --controlled_ab_checkpoint_iteration 10000 --start_checkpoint "$CKPT" --iterations 12000 --test_iterations 12000 --save_iterations 12000 --checkpoint_iterations 12000 --enable_diffusion_pseudo_rgb --pseudo_rgb_manifest "$PSEUDO_MANIFEST" --pseudo_rgb_weight 0.1 --pseudo_rgb_lambda_dssim 0.2 --pseudo_rgb_start 10000 --pseudo_rgb_end 12000 --pseudo_rgb_interval 50 --pseudo_rgb_strict_cache
    ;;
  pair-self)
    audit_pair_self
    ;;
  pair-b)
    audit_pair_b
    ;;
  pair-pseudo)
    audit_pair_self
    audit_pair_b
    audit_pair_pseudo
    ;;
  evidence-export-a1)
    audit_pair_b
    reset_run_subdir "$RUN/evidence_a1" "A1 held-out evidence directory"
    python tools/export_heldout_views.py "${COMMON[@]}" -m "$A1" --iteration 12000 --checkpoint "$A1/chkpnt12000.pth" --dataset "$DATASET" --pair-audit "$RUN/a1_b_pair_audit.json" --output-dir "$RUN/evidence_a1" --seed "$SEED"
    ;;
  evidence-export-b)
    audit_pair_b
    reset_run_subdir "$RUN/evidence_b" "B held-out evidence directory"
    python tools/export_heldout_views.py "${COMMON[@]}" -m "$B" --iteration 12000 --checkpoint "$B/chkpnt12000.pth" --dataset "$DATASET" --pair-audit "$RUN/a1_b_pair_audit.json" --output-dir "$RUN/evidence_b" --seed "$SEED"
    ;;
  evidence-cache-a1|evidence-cache-b)
    audit_pair_b
    need_difix
    E="$RUN/evidence_a1"
    [[ "$STAGE" == evidence-cache-b ]] && E="$RUN/evidence_b"
    python tools/run_difix_cache.py --manifest "$E/difix_manifest.jsonl" --difix-repo "$DIFIX_REPO" --model-id nvidia/difix_ref --model-revision "$DIFIX_MODEL_REVISION" --dtype fp16 --timestep 199 --guidance-scale 0 --seed "$SEED" --reproducibility-check
    require_nested_passed_json "$E/difix_manifest.jsonl.difix_metadata.json" "reproducibility_check.passed" "$STAGE reproducibility gate"
    ;;
  identity-manifest)
    audit_pair_b
    rm -f -- "$PAIRED_IDENTITY_MANIFEST" "$PAIRED_IDENTITY_METADATA"
    python tools/make_paired_identity_manifest.py --a1-manifest "$RUN/evidence_a1/evidence_manifest.jsonl" --b-manifest "$RUN/evidence_b/evidence_manifest.jsonl" --output "$PAIRED_IDENTITY_MANIFEST"
    require_paired_identity_manifest
    ;;
  identity-a1|identity-b)
    : "${DINOV2_REPO:?Set immutable local DINOv2 checkout}"
    : "${DINOV2_WEIGHT_PATH:?Set exact local DINOv2 weights file}"
    require_paired_identity_manifest
    OUT="$RUN/identity_a1"
    [[ "$STAGE" == identity-b ]] && OUT="$RUN/identity_b"
    ARM=A1
    [[ "$STAGE" == identity-b ]] && ARM=B
    reset_run_subdir "$OUT" "$ARM identity diagnostic directory"
    python tools/evaluate_track_evidence.py --track-h5 "$TRACK" --images-dir "$SCENE/images" --target-manifest "$PAIRED_IDENTITY_MANIFEST" --paired-arm "$ARM" --feature-backend dinov2 --dinov2-model dinov2_vits14 --window-radii 16 32 48 --primary-radius 32 --shift-pixels 4 8 16 --hard-negative-radius 32 --temperature 0.07 --output-dir "$OUT"
    python tools/paired_evidence_report.py --csv "$OUT/per_track.csv" --radius 32 --output "$OUT/paired_report.json"
    ;;
  identity-report)
    refresh_identity_association
    ;;
  report)
    require_recovery_gate
    require_paired_identity_manifest
    require_file "$RUN/identity_a1/per_track.csv" "A1 identity per-track report"
    require_file "$RUN/identity_b/per_track.csv" "B identity per-track report"
    python tools/paired_evidence_report.py --csv "$RUN/identity_a1/per_track.csv" --radius 32 --output "$RUN/identity_a1/paired_report.json"
    python tools/paired_evidence_report.py --csv "$RUN/identity_b/per_track.csv" --radius 32 --output "$RUN/identity_b/paired_report.json"
    refresh_identity_association
    ;;
  metrics)
    require_completed_training_runs
    generate_contrast_metrics
    ;;
  integrity)
    require_completed_training_runs
    require_environment_gate
    require_track_gate
    require_projection_gate
    require_pseudo_live_gate
    require_smoke_gate
    require_pseudo_gate
    require_recovery_gate
    generate_contrast_metrics
    refresh_identity_association
    run_json_gate "$INTEGRITY_REPORT" "single-scene/seed final integrity gate" \
      python tools/audit_run_integrity.py --run "$RUN" --dataset "$DATASET" --scene "$SCENE_NAME" --seed "$SEED" --output "$INTEGRITY_REPORT"
    ;;
  *)
    die "Unknown stage: $STAGE"
    ;;
esac
