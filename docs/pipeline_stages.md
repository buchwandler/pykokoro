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

One public synthesis request always returns one `RenderedSegment`. After G2P, PyKokoro
checks the model-token count against the resolved profile. By default,
`long_text_split="none"` raises `SynthesisInputTooLongError` when the request exceeds
capacity and does not import PhraseSplit.

Set `SynthesisConfig.long_text_split="sentence"` to enable internal splitting only for
oversized requests. PhraseSplit loads lazily and packs source-aligned sentence spans
into model-safe chunks, falling back to clauses and safe word boundaries for an
oversized sentence. The chunks are rendered in order and joined into the same request
result. A single word that cannot fit safely still raises `SynthesisInputTooLongError`.
`long_text_use_spacy=False` selects PhraseSplit's simple mode; `None` permits a
compatible local spaCy model with regex fallback, and `True` requires spaCy and a
compatible model. Caller-owned composition still applies between separate synthesis
requests. Short-sentence handling is also explicit and disabled when no short-sentence
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
