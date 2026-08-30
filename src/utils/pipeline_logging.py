"""
pipeline_logging.py
===================
Capture full terminal output (stdout/stderr) and Python logging to per-run log files.

Log files are written under ``results/{dataset}/logs/`` by default:

* ``pipeline_YYYYMMDD_HHMMSS.log`` — timestamped run log
* ``latest.log`` — copy of the most recent run
* ``latest_manifest.json`` — run metadata (command, timestamps, status)
"""

from __future__ import annotations

import json
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, IO, List, Optional, TextIO

_LOG_TS = "%Y%m%d_%H%M%S"
logger = logging.getLogger("pipeline_logging")


class _Tee(TextIO):
    """Write all output to multiple streams (console + log file)."""

    def __init__(self, *streams: TextIO) -> None:
        self._streams = streams

    def write(self, data: str) -> int:
        if not data:
            return 0
        for stream in self._streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()

    def isatty(self) -> bool:
        return any(getattr(stream, "isatty", lambda: False)() for stream in self._streams)


def default_log_dir(root: Path, dataset: str) -> Path:
    """Default directory for run logs for a dataset."""
    return root / "results" / dataset / "logs"


class PipelineLogSession:
    """
    Context manager that tees stdout/stderr to a timestamped log file.

    Subprocesses inherit the teed streams, so child script output is captured too.
    """

    def __init__(
        self,
        *,
        root: Path,
        dataset: str,
        run_name: str = "pipeline",
        verbose: bool = False,
        log_dir: Optional[Path] = None,
        argv: Optional[List[str]] = None,
    ) -> None:
        self.root = root
        self.dataset = dataset
        self.run_name = run_name
        self.verbose = verbose
        self.log_dir = log_dir or default_log_dir(root, dataset)
        self.argv = list(argv) if argv is not None else list(sys.argv)

        self.timestamp = datetime.now().astimezone()
        self.main_log: Optional[Path] = None
        self.latest_log: Optional[Path] = None
        self.started_at: Optional[str] = None
        self.finished_at: Optional[str] = None

        self._log_file: Optional[IO[str]] = None
        self._orig_stdout: Optional[TextIO] = None
        self._orig_stderr: Optional[TextIO] = None
        self._extra_manifest: Dict[str, Any] = {}

    def __enter__(self) -> PipelineLogSession:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        ts = self.timestamp.strftime(_LOG_TS)
        self.main_log = self.log_dir / f"{self.run_name}_{ts}.log"
        self.latest_log = self.log_dir / "latest.log"
        self._log_file = self.main_log.open("w", encoding="utf-8", newline="\n")

        header = (
            f"{'=' * 72}\n"
            f"spCellEval run log: {self.run_name}\n"
            f"dataset: {self.dataset}\n"
            f"started: {self.timestamp.isoformat()}\n"
            f"command: {' '.join(self.argv)}\n"
            f"log file: {self.main_log}\n"
            f"{'=' * 72}\n"
        )
        self._log_file.write(header)
        self._log_file.flush()

        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr
        sys.stdout = _Tee(self._orig_stdout, self._log_file)  # type: ignore[assignment]
        sys.stderr = _Tee(self._orig_stderr, self._log_file)  # type: ignore[assignment]

        self.started_at = self.timestamp.isoformat()
        logger.info("Execution log: %s", self.main_log)
        return self

    def configure_logging(self, level: Optional[int] = None) -> None:
        """Configure root logger to emit through stdout (which is teed to the log file)."""
        if level is None:
            level = logging.DEBUG if self.verbose else logging.INFO
        logging.basicConfig(
            format="%(asctime)s [%(levelname)s] %(message)s",
            level=level,
            force=True,
        )

    def add_manifest(self, **fields: Any) -> None:
        """Attach extra key/value pairs to the run manifest written on exit."""
        self._extra_manifest.update(fields)

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.finished_at = datetime.now().astimezone().isoformat()
        status = "FAILED" if exc_type else "SUCCESS"
        footer = (
            f"\n{'=' * 72}\n"
            f"finished: {self.finished_at}\n"
            f"status: {status}\n"
        )
        if exc_type is not None:
            footer += f"error: {exc_type.__name__}: {exc}\n"
        footer += f"{'=' * 72}\n"
        print(footer, end="")

        if self._orig_stdout is not None:
            sys.stdout = self._orig_stdout
        if self._orig_stderr is not None:
            sys.stderr = self._orig_stderr

        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None

        if self.main_log is not None and self.main_log.is_file() and self.latest_log is not None:
            try:
                shutil.copy2(self.main_log, self.latest_log)
            except OSError:
                pass

        self._write_manifest(exc_type, exc)
        return False

    def _write_manifest(self, exc_type, exc) -> None:
        manifest: Dict[str, Any] = {
            "dataset": self.dataset,
            "run_name": self.run_name,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "log_file": str(self.main_log) if self.main_log else None,
            "latest_log": str(self.latest_log) if self.latest_log else None,
            "command": self.argv,
            "status": "failed" if exc_type else "success",
            "error": f"{exc_type.__name__}: {exc}" if exc_type else None,
        }
        manifest.update(self._extra_manifest)
        if self.log_dir is not None:
            manifest_path = self.log_dir / "latest_manifest.json"
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def log_command_banner(stage: str, cmd: List[str]) -> None:
    """Write a visible section header before running a subprocess."""
    logging.getLogger("run_pipeline").info("-" * 60)
    logging.getLogger("run_pipeline").info("STAGE: %s", stage)
    logging.getLogger("run_pipeline").info("COMMAND: %s", " ".join(cmd))
    logging.getLogger("run_pipeline").info("-" * 60)
