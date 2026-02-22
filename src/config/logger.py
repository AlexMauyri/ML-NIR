"""
logger.py – Configure the root logger from logging.yaml.

Call ``setup_logging()`` once at the very start of each entry-point script.
Every other module obtains its own logger with ``logging.getLogger(__name__)``.

Log files are named after the current date: ``logs/dd-mm-yyyy.log``.
A new file is started each calendar day; up to 31 rotated copies are kept.
"""
import logging
import logging.config
import logging.handlers
from datetime import datetime
from pathlib import Path

import yaml


def _today_log_filename(log_dir: Path) -> str:
    """Return an absolute log-file path named ``dd-mm-yyyy.log`` for today."""
    name = datetime.now().strftime("%d-%m-%Y") + ".log"
    return str(log_dir / name)


def setup_logging(
    config_path: Path | None = None,
    log_dir: Path | None = None,
) -> None:
    """
    Load the YAML logging config, inject today's date-stamped log filename,
    and activate the configuration.

    Parameters
    ----------
    config_path : path to logging.yaml  (default: config/logging.yaml)
    log_dir     : directory for log files (default: project/logs/)
    """
    if config_path is None:
        config_path = Path(__file__).parent / "logging.yaml"
    if log_dir is None:
        log_dir = Path(__file__).parent.parent / "logs"

    log_dir.mkdir(parents=True, exist_ok=True)

    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)

    for handler in cfg.get("handlers", {}).values():
        if "filename" in handler:
            handler["filename"] = _today_log_filename(log_dir)

    logging.config.dictConfig(cfg)
