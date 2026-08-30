"""Optional slow integration test — run smoke pipeline end-to-end."""

import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO / "scripts" / "smoke_test_pipeline.py"


@pytest.mark.integration
@pytest.mark.timeout(900)
def test_smoke_pipeline_script():
    if not _SCRIPT.is_file():
        pytest.skip("smoke_test_pipeline.py not found")
    fixture = _REPO / "tests" / "fixtures" / "smoke" / "CRC_TMA_expression.csv"
    if not fixture.is_file():
        pytest.skip("smoke fixture missing")

    result = subprocess.run(
        [sys.executable, str(_SCRIPT)],
        cwd=str(_REPO),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
