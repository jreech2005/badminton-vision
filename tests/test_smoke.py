"""Scaffold smoke checks: package layout, error taxonomy, CLI wiring."""

import pytest

from badminton_vision import paths
from badminton_vision.cli import main
from badminton_vision.errors import (
    BadmintonVisionError,
    DataContractError,
    LabelMapError,
    LeakGuardError,
    ValidationGateError,
)

pytestmark = pytest.mark.unit


def test_repo_root_resolves() -> None:
    assert (paths.REPO_ROOT / "pyproject.toml").is_file()


def test_error_taxonomy() -> None:
    for exc in (DataContractError, LabelMapError, LeakGuardError, ValidationGateError):
        assert issubclass(exc, BadmintonVisionError)


def test_cli_requires_command() -> None:
    with pytest.raises(SystemExit) as excinfo:
        main([])
    assert excinfo.value.code == 2
