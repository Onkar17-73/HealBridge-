import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional


DEFAULT_LOG_DIR = Path("logs")
DEFAULT_LOG_FILE = DEFAULT_LOG_DIR / "app.log"


def setup_logging(log_dir: Optional[Path] = None, level: int = logging.INFO) -> logging.Logger:
    """Configure a rotating file logger for the application and return the root app logger.

    - Writes to logs/app.log by default
    - Rotates at ~5 MB, keeps 5 backups
    - Formats with timestamp, level, logger name and message
    """
    # Normalize to absolute path to avoid duplicate handler checks failing on relative vs absolute
    log_dir = (log_dir or DEFAULT_LOG_DIR)
    log_dir = Path(log_dir).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "app.log"
    log_path_abs = str(log_path)

    logger = logging.getLogger("datasmith")
    logger.setLevel(level)
    # Prevent propagation to root to avoid duplicate writes if root has handlers
    logger.propagate = False

    # Avoid duplicate handlers on repeated setup
    # Check if a file handler for this exact absolute path already exists
    for h in list(logger.handlers):
        if isinstance(h, RotatingFileHandler):
            try:
                if os.path.abspath(getattr(h, "baseFilename", "")) == os.path.abspath(log_path_abs):
                    # Already configured for this file; keep existing handler and return
                    return logger
            except Exception:
                continue

    fmt = logging.Formatter(fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    file_handler = RotatingFileHandler(log_path_abs, maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    # Optional: quieter console handler to avoid noisy interactive output
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        console = logging.StreamHandler(stream=sys.stderr)
        console.setLevel(logging.WARNING)
        console.setFormatter(fmt)
        logger.addHandler(console)

    logger.info("logging initialized: file=%s, level=%s", log_path, logging.getLevelName(level))
    return logger
