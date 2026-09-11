from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager

from ..types import Trace, TraceEvent

logger = logging.getLogger(__name__)


@contextmanager
def trace_timing(trace: Trace | None, stage: str, name: str) -> Iterator[None]:
    started = time.perf_counter()
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("stage.start stage=%s name=%s", stage, name)
    try:
        yield
    finally:
        ms = (time.perf_counter() - started) * 1000.0
        if trace is not None:
            trace.events.append(TraceEvent(stage=stage, name=name, ms=ms))
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "stage.finish stage=%s name=%s elapsed_ms=%.3f",
                stage,
                name,
                ms,
            )
