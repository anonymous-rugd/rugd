"""Terminal progress helpers that keep tqdm bars and persistent logs readable."""
from __future__ import annotations

import logging

from tqdm.auto import tqdm


class TqdmLoggingHandler(logging.Handler):
    """Write log records through tqdm so active progress bars are not corrupted."""

    def emit(self, record):
        try:
            tqdm.write(self.format(record))
        except (OSError, ValueError):
            self.handleError(record)


def progress(iterable, description: str, unit: str, leave: bool = True):
    """Create a consistent, terminal-width-aware progress bar with elapsed time and ETA."""
    return tqdm(iterable, desc=description, unit=unit, dynamic_ncols=True,
                mininterval=0.5, smoothing=0.1, leave=leave)
