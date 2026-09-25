# Request lifecycle

The renderer operates on one prepared `SynthesisSegment` and produces one
`RenderedSegment`:

```text
SynthesisSegment
      |
      v
prepared-text KokoroG2P
      |
      v
model-specific phoneme/token preparation
      |
      v
voice/style resolution and ONNX inference
      |
      v
request-local postprocessing and timing reconstruction
      |
      v
RenderedSegment
```

## Prepared-text G2P

PyKokoro forwards the request's text, explicit language, pronunciation overrides,
linguistic token annotations, routing settings, and model target to KokoroG2P's
prepared-text API. It does not parse a document, build a plan, or interpret markup.

## Model and voice preparation

The resolved request configuration selects a compatible model profile and voice/style. A
voice on `SynthesisSegment` overrides the configured default; a `VoiceBlend` can be
supplied as the request voice. The engine prepares model-ready style data and sends
token IDs, speed, and a seed when supported to OnnxVoice.

## Inference and internal chunking

PyKokoro may split a long request into token-limit chunks or make additional calls for
its short-sentence strategy. Such chunks belong to the same public request and are
stitched into that request's one result. Chunk timings are rebased to the stitched
request-local waveform. No cross-request silence, markers, embedded clips, or
caller-visible timeline data is added.

## Result

`RenderedSegment` contains the request ID, mono float32 audio, sample rate, prepared
text, language, resolved voice name when applicable, phonemes, token IDs, diagnostics,
and optional trace and word timings. Timings use sample offsets within the returned
waveform. The caller owns any cross-request playback order, pause policy, resampling,
and composition.
