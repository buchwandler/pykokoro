# PyKokoro Documentation

PyKokoro is a request-centric Kokoro speech-synthesis engine. It accepts prepared speech
requests and returns independent rendered waveforms. Parsing, speech planning, document
semantics, timeline composition, and final-output mastering belong to the calling
application and its other tools.

```{toctree}
:caption: 'Contents:'
:maxdepth: 2

quickstart
installation
basic_usage
advanced_features
short_sentence_quality
pipeline_stages
api_reference
examples
languages
breaking-change-0.10.0
changelog
```

## Engine responsibilities

- KokoroG2P prepared-text integration with explicit pronunciation languages,
  source-aligned overrides, and linguistic tokens
- Kokoro voice, model, and style selection, including blends
- Model token-capacity validation and opt-in sentence-level splitting for oversized
  requests, plus ONNX inference
- Request-local timing reconstruction, waveform validation, tracing, and optional voice
  calibration
- Independent `RenderedSegment` results and standalone WAV writing

## Feature guides

- [Long-text capacity, splitting, and errors](basic_usage.md#render-longer-text)
- [Direct phonemes, language routing, VoiceBlend, and result metadata]
  (advanced_features.md)
- [Language codes and acoustic-profile discovery](languages.md)
- [Frontends, installed lexicons, cache location, and asset progress](installation.md)
- [Short-sentence configuration and modes](short_sentence_quality.md)

PyKokoro does not parse SSMD or YAML, create UtterPlans or AudioJobs, compose separate
caller requests, insert cross-request silence, or apply document-level effects. The
v0.10.0 breaking boundary and migration examples are described in the
[release note](breaking-change-0.10.0.md).

## Quick example

```python
from pykokoro import GenerationConfig, KokoroSynthesizer, SynthesisConfig

config = SynthesisConfig(
    voice="af_sarah",
    generation=GenerationConfig(lang="en-us"),
)
with KokoroSynthesizer(config) as synthesizer:
    result = synthesizer.synthesize_text("Hello, world.", language="en-us")
result.save_wav("hello.wav")
```

See the [maintained examples](examples.md) for requests with pronunciation context,
linguistic tokens, and independent batch results.
