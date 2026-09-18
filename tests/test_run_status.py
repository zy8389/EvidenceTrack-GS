import json
from argparse import Namespace

import pytest

from diffusion_guidance.run_status import fail_closed, write_training_status
from utils.config_utils import ConfigValidator


def test_fail_closed_marks_run_invalid(tmp_path):
    with pytest.raises(RuntimeError, match="marked INVALID"):
        fail_closed(
            tmp_path,
            stage="constraint_forward",
            iteration=17,
            operation=lambda: (_ for _ in ()).throw(ValueError("broken geometry")),
        )

    status = json.loads((tmp_path / "run_status.json").read_text(encoding="utf-8"))
    assert status == {
        "error": "broken geometry",
        "error_type": "ValueError",
        "iteration": 17,
        "schema": "evidencetrack_training_status_v1",
        "stage": "constraint_forward",
        "status": "INVALID",
    }


def test_training_status_can_transition_to_completed(tmp_path):
    write_training_status(tmp_path, status="RUNNING", stage="initialization")
    write_training_status(
        tmp_path,
        status="COMPLETED",
        stage="training_complete",
        iteration=12000,
    )
    status = json.loads((tmp_path / "run_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "COMPLETED"
    assert status["iteration"] == 12000


def test_enabled_constraints_do_not_silently_disable_missing_tracks(tmp_path):
    args = Namespace(
        enable_geometric_constraints=True,
        track_path=str(tmp_path / "missing.h5"),
    )
    assert ConfigValidator.validate_file_paths(args) is False
    assert args.enable_geometric_constraints is True
