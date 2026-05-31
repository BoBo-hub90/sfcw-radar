"""
Unit tests for the centralized logging configuration (utils.logger).

These tests exercise the public helpers (get_logger, set_level) and the global
side effects they install (rotating file handler, logs/ directory, warnings
capture). No hardware is involved.
"""

from __future__ import annotations

import logging
import os
import sys

import pytest

# Make the src/ packages importable without installing the project.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

from utils import logger as logger_mod  # noqa: E402
from utils.logger import get_logger, set_level, LOG_DIR  # noqa: E402


def test_get_logger_returns_logger_instance():
    """get_logger hands back a standard logging.Logger."""
    log = get_logger("test.module")
    assert isinstance(log, logging.Logger)
    assert log.name == "test.module"


def test_set_level_changes_handler_levels():
    """set_level updates the level on the root logger and every handler."""
    get_logger("test.levels")  # ensure handlers are installed

    set_level("DEBUG")
    root = logging.getLogger()
    assert root.level == logging.DEBUG
    assert all(h.level == logging.DEBUG for h in root.handlers)

    set_level("WARNING")
    assert root.level == logging.WARNING
    assert all(h.level == logging.WARNING for h in root.handlers)


def test_set_level_rejects_unknown_level():
    """An unrecognized level name raises ValueError."""
    with pytest.raises(ValueError):
        set_level("VERBOSE")


def test_logs_directory_created_automatically():
    """Importing/using the logger creates the logs/ directory."""
    get_logger("test.logsdir")
    assert os.path.isdir(LOG_DIR)


def test_capture_warnings_is_active():
    """logging.captureWarnings(True) routes Python warnings into logging."""
    get_logger("test.warnings")
    # captureWarnings(True) installs logging's internal showwarning shim.
    assert logging._warnings_showwarning is not None


def test_module_exposes_expected_handlers():
    """After configuration the root logger has a console and a file handler."""
    get_logger("test.handlers")
    handler_types = {type(h).__name__ for h in logging.getLogger().handlers}
    assert "StreamHandler" in handler_types
    assert "RotatingFileHandler" in handler_types
    # Sanity: the file handler points at the configured log file.
    assert os.path.basename(logger_mod.LOG_FILE) == "sfcw_radar.log"
