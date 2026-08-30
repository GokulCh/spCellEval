"""Tests for execution log capture."""

import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src" / "utils") not in sys.path:
    sys.path.insert(0, str(_REPO / "src" / "utils"))

from pipeline_logging import PipelineLogSession, _Tee  # noqa: E402


def test_tee_writes_to_all_streams(tmp_path):
    a = (tmp_path / "a.txt").open("w", encoding="utf-8")
    b = (tmp_path / "b.txt").open("w", encoding="utf-8")
    tee = _Tee(a, b)
    tee.write("hello\n")
    tee.flush()
    a.close()
    b.close()
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "hello\n"
    assert (tmp_path / "b.txt").read_text(encoding="utf-8") == "hello\n"


def test_pipeline_log_session_captures_stdout(tmp_path):
    log_dir = tmp_path / "results" / "TEST" / "logs"
    with PipelineLogSession(
        root=tmp_path,
        dataset="TEST",
        run_name="pipeline",
        argv=["python", "run.py", "--dataset", "TEST"],
    ) as session:
        print("captured line")

    assert session.main_log is not None
    assert session.main_log.is_file()
    text = session.main_log.read_text(encoding="utf-8")
    assert "captured line" in text
    assert "spCellEval run log" in text
    assert (log_dir / "latest.log").is_file()
    manifest = json.loads((log_dir / "latest_manifest.json").read_text(encoding="utf-8"))
    assert manifest["dataset"] == "TEST"
    assert manifest["status"] == "success"
