# Hard-case benchmark status

This benchmark guide describes the former document-planning frontend and its SSMD-aware
segment/plan diagnostics. Those are not part of PyKokoro's request-centric synthesis
API. The old harness has not been migrated to `SynthesisSegment` and is not a supported
benchmark entry point for the new engine boundary.

The underlying short-sentence, KokoroG2P, inference, timing, and waveform behavior
remains engine-owned and is covered by request-level tests. Use the
[request lifecycle](pipeline_stages.md) and
[short-sentence synthesis](short_sentence_quality.md) documentation for the supported
runtime contract.
