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
pipeline_stages
api_reference
examples
languages
breaking-change-0.10.0
changelog
```

## Engine responsibilities

- KokoroG2P prepared-text integration with explicit pronunciation languages,
  source-aligned overrides, and linguistic annotations
- Kokoro voice, model, and style selection, including blends
- Model-limit chunking, short-sentence handling, and ONNX inference
- Request-local timing reconstruction, waveform validation, tracing, and optional voice
  calibration
- Independent `RenderedSegment` results and standalone WAV writing

PyKokoro does not parse SSMD or YAML, create UtterPlans or AudioJobs, compose caller
requests, insert cross-request silence, or apply document-level effects. The planned
0.10.0 breaking change and migration boundary are described in the
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

See the [maintained examples](../examples/README.md) for requests with pronunciation
context, annotations, and independent batch results.
