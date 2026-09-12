from __future__ import annotations

import logging
from dataclasses import dataclass

from pykokoro.runtime.tracing import trace_timing
from pykokoro.types import Trace


def test_trace_timing_does_not_log_when_debug_is_disabled(caplog) -> None:
    caplog.set_level(logging.INFO, logger="pykokoro.runtime.tracing")

    with trace_timing(Trace(), "g2p", "phonemize"):
        pass

    assert not any("stage." in record.message for record in caplog.records)


def test_trace_timing_logs_start_and_finish_and_records_trace(caplog) -> None:
    trace = Trace()
    caplog.set_level(logging.DEBUG, logger="pykokoro.runtime.tracing")

    with trace_timing(trace, "g2p", "phonemize"):
        pass

    messages = [record.message for record in caplog.records]
    assert any("stage.start stage=g2p name=phonemize" in message for message in messages)
    finish = next(message for message in messages if "stage.finish" in message)
    assert "stage=g2p" in finish
    assert "name=phonemize" in finish
    assert float(finish.rsplit("elapsed_ms=", 1)[1]) >= 0
    assert len(trace.events) == 1
    assert trace.events[0].stage == "g2p"


def test_trace_timing_logs_without_trace(caplog) -> None:
    caplog.set_level(logging.DEBUG, logger="pykokoro.runtime.tracing")

    with trace_timing(None, "runtime", "resolve"):
        pass

    assert any(
        "stage.finish stage=runtime name=resolve" in record.message for record in caplog.records
    )


@dataclass(frozen=True)
class _FrozenTracingError(Exception):
    message: str


def test_trace_timing_preserves_frozen_exception_and_records_trace() -> None:
    trace = Trace()
    expected = _FrozenTracingError("original failure")

    try:
        with trace_timing(trace, "g2p", "phonemize"):
            raise expected
    except _FrozenTracingError as exc:
        assert exc is expected
    else:
        raise AssertionError("expected frozen exception to propagate")

    assert len(trace.events) == 1
    assert trace.events[0].stage == "g2p"
    assert trace.events[0].name == "phonemize"


def test_trace_timing_logs_finish_for_frozen_exception(caplog) -> None:
    trace = Trace()
    caplog.set_level(logging.DEBUG, logger="pykokoro.runtime.tracing")

    try:
        with trace_timing(trace, "g2p", "phonemize"):
            raise _FrozenTracingError("boom")
    except _FrozenTracingError:
        pass

    assert any(
        "stage.finish stage=g2p name=phonemize" in record.message
        for record in caplog.records
    )
