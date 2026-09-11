from __future__ import annotations

import logging

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
