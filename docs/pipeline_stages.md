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

PyKokoro forwards the request's exact text, explicit language, pronunciation overrides,
and linguistic annotations, including morphology, to KokoroG2P. Supplied tokens take
precedence and prevent internal spaCy analysis. It does not parse a document, build a
plan, or interpret markup.

## Model and voice preparation

The resolved request configuration selects a compatible model profile and voice/style. A
voice on `SynthesisSegment` overrides the configured default; a `VoiceBlend` can be
supplied as the request voice. The engine prepares model-ready style data and sends
token IDs, speed, and a seed when supported to OnnxVoice.

## Atomic inference and capacity

Each request is one indivisible inference unit. After G2P, PyKokoro checks its
model-token count against the resolved profile. Oversized input raises
`SynthesisInputTooLongError` with the source-text length, actual token count, maximum,
and model identity. PyKokoro never splits or chunks a request; the caller owns
segmentation and composition.

`SynthesisConfig.long_text_split` remains only as a migration surface. It accepts
`"none"`; explicit `"sentence"` or `"token"` values raise `ConfigurationError`.

Short-sentence handling is also explicit and disabled when no short-sentence
configuration or enable override is supplied. Callers can opt in with
`GenerationConfig(enable_short_sentence=True)` or `ShortSentenceConfig`. It may use
context and retry inference internally, but returned text, phonemes, and timings remain
request-local. `RenderedSegment.short_sentence_mode` reports the mode actually used
without exposing generated context text.

## Result

`RenderedSegment` contains the request ID, mono float32 audio, sample rate, exact
request text, language, resolved voice name when applicable, phonemes, token IDs,
diagnostics, and word timings. Timings use source-text character ranges and ordered
sample offsets inside the returned waveform. `synthesis_identity` exposes stable
output-affecting settings; `voice_level_applications` records calibration mode, gain,
source, and missing-custom-voice outcomes. The caller owns cross-request playback order,
pause policy, resampling, and composition.
