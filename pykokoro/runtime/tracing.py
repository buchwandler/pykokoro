from __future__ import annotations

import logging
import time
from types import TracebackType
from typing import Literal

from ..types import Trace, TraceEvent

logger = logging.getLogger(__name__)


class _TraceTimingContext:
    def __init__(self, trace: Trace | None, stage: str, name: str) -> None:
        self._trace = trace
        self._stage = stage
        self._name = name
        self._started: float | None = None

    def __enter__(self) -> None:
        self._started = time.perf_counter()
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "stage.start stage=%s name=%s",
                self._stage,
                self._name,
            )

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _traceback: TracebackType | None,
    ) -> Literal[False]:
        started = self._started
        if started is None:
            raise RuntimeError("trace timing context was not entered")

        ms = (time.perf_counter() - started) * 1000.0
        if self._trace is not None:
            self._trace.events.append(
                TraceEvent(
                    stage=self._stage,
                    name=self._name,
                    ms=ms,
                )
            )
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "stage.finish stage=%s name=%s elapsed_ms=%.3f",
                self._stage,
                self._name,
                ms,
            )
        return False


def trace_timing(
    trace: Trace | None,
    stage: str,
    name: str,
) -> _TraceTimingContext:
    return _TraceTimingContext(trace, stage, name)
