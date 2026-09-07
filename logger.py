"""
Centralised logging configuration.
build and return a named logger.

"""

import logging
import sys
from functools import lru_cache

# Log format: timestamp | level | module name | message
_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def _configure_root_logger() -> None:
    """
    Configure the root logger once with a StreamHandler to stdout.
    """
    root = logging.getLogger()

    if root.handlers:
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))

    root.addHandler(handler)
    root.setLevel(logging.INFO)


@lru_cache(maxsize=None)
def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger for the calling module.

    """
    _configure_root_logger()
    return logging.getLogger(name)