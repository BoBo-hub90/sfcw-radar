"""
Centralized logging configuration for the sfcw-radar project.

Every module obtains its logger through get_logger(__name__) so that console
and file output share one consistent format and level policy. Handlers are
attached once to the root logger; named module loggers propagate their records
up to those handlers, which keeps a single point of control for formatting and
log levels across the whole project.

Two sinks are configured:
  - console (StreamHandler)     : INFO and above, for live operation.
  - file (RotatingFileHandler)  : DEBUG and above, written to logs/sfcw_radar.log
                                  with rotation (1 MB x 3 backups) for post-run
                                  inspection.

Python's own warnings are routed through logging (captureWarnings) so they land
in the same sinks as everything else.

Usage:
    from utils.logger import get_logger, set_level

    log = get_logger(__name__)
    log.info("radar starting")
    set_level("DEBUG")   # raise verbosity on every handler at once
"""

from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler

# Shared log format and timestamp style for all handlers.
LOG_FORMAT = "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s"
DATE_FORMAT = "%H:%M:%S"

DEFAULT_LEVEL = logging.INFO

# Rotating file handler sizing.
LOG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "logs",
)
LOG_FILE = os.path.join(LOG_DIR, "sfcw_radar.log")
MAX_BYTES = 1_000_000   # 1 MB per file before rotation
BACKUP_COUNT = 3

# Accepted names for set_level().
_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
}

# Set once handlers have been installed on the root logger.
_configured = False


def _configure() -> None:
    """
    Install the console and rotating-file handlers on the root logger once.

    Idempotent: repeated calls (one per module importing get_logger) are no-ops
    after the first, so handlers are never duplicated.
    """
    global _configured
    if _configured:
        return

    os.makedirs(LOG_DIR, exist_ok=True)

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)

    file_handler = RotatingFileHandler(
        LOG_FILE, maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    root = logging.getLogger()
    # Root must pass DEBUG through so the file handler can see debug records;
    # each handler then applies its own threshold.
    root.setLevel(logging.DEBUG)
    root.addHandler(console)
    root.addHandler(file_handler)

    # Route Python warnings (warnings.warn) into the logging system.
    logging.captureWarnings(True)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """
    Return a module logger wired to the shared console and file handlers.

    Args:
        name: Logger name, conventionally the module's __name__.

    Returns:
        A logging.Logger that propagates to the centrally configured handlers.
    """
    _configure()
    return logging.getLogger(name)


def set_level(level: str) -> None:
    """
    Set the logging level on the root logger and every installed handler.

    Args:
        level: One of "DEBUG", "INFO", "WARNING", "ERROR" (case-insensitive).

    Raises:
        ValueError: If the level name is not recognized.
    """
    _configure()

    key = level.upper()
    if key not in _LEVELS:
        raise ValueError(
            f"Unknown level {level!r}; expected one of {sorted(_LEVELS)}"
        )
    numeric = _LEVELS[key]

    root = logging.getLogger()
    root.setLevel(numeric)
    for handler in root.handlers:
        handler.setLevel(numeric)
