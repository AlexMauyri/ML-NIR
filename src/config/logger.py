"""
logger.py – Configure the root logger from logging.yaml.

Call ``setup_logging()`` once at the very start of each entry-point script.
Every other module obtains its own logger with ``logging.getLogger(__name__)``.
"""
import logging
import logging.config
import logging.handlers
from pathlib import Path

import yaml


def setup_logging(
    config_path: Path | None = None,
    log_dir: Path | None = None,
) -> None:
    """
    Load the YAML logging config and patch the rotating file handler so its
    ``filename`` is resolved relative to *log_dir* (defaults to project/logs/).

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
            handler["filename"] = str(log_dir / Path(handler["filename"]).name)

    logging.config.dictConfig(cfg)
