#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"; export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
mkdir -p docs/validation
python evidence_track/evaluation/smoke_test_phase2_1.py | tee docs/validation/repair_smoke_phase21.log
python -m pytest tests -q | tee docs/validation/repair_regression_tests.log
python evidence_track/evaluation/test_geometry_recovery.py --synthetic-smoke --steps 100 --output docs/validation/repair_synthetic_recovery.json
python evidence_track/evaluation/collect_environment.py --output docs/validation/repair_environment.json
