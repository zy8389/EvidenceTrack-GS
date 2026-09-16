#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"; export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
mkdir -p validation
python tools/smoke_test_phase2_1.py | tee validation/repair_smoke_phase21.log
python -m pytest tests -q | tee validation/repair_regression_tests.log
python tools/test_geometry_recovery.py --synthetic-smoke --steps 100 --output validation/repair_synthetic_recovery.json
python tools/collect_environment.py --output validation/repair_environment.json
