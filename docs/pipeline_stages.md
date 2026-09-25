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

For an oversized prepared-text request, `SynthesisConfig.long_text_split` selects
model-input chunking:

- `"sentence"` (default) lazily uses PhraseSplit's regex backend with `use_spacy=False`.
  It packs complete sentences within the model token budget and falls back to token-safe
  splitting when a sentence is too large.
- `"token"` uses token-safe splitting without PhraseSplit.
- `"none"` rejects an oversized request with `SynthesisInputTooLongError`.

The model token limit remains authoritative. This is acoustic inference chunking, not
document parsing or a public sentence plan. All internal chunks belong to the same
public request and are stitched into one `RenderedSegment`. Chunk timings are rebased to
that request-local waveform. No cross-request silence, markers, embedded clips, or
caller-visible timeline data is added.

Separately configured short-sentence handling may also issue extra inference calls; it
does not change the long-text splitting policy.

## Result

`RenderedSegment` contains the request ID, mono float32 audio, sample rate, prepared
text, language, resolved voice name when applicable, phonemes, token IDs, diagnostics,
and optional trace and word timings. Timings use sample offsets within the returned
waveform. The caller owns any cross-request playback order, pause policy, resampling,
and composition.
