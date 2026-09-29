# Quick start

PyKokoro needs Python 3.10 or newer. Install the package with one supported ONNX Runtime
provider, for example:

```bash
pip install "pykokoro[cpu]"
```

## Synthesize one string

```python
from pykokoro import GenerationConfig, KokoroSynthesizer, SynthesisConfig

config = SynthesisConfig(
    voice="af_bella",
    generation=GenerationConfig(lang="en-us"),
)
with KokoroSynthesizer(config) as synthesizer:
    rendered = synthesizer.synthesize_text(
        "Hello, world.",
        language="en-us",
        voice="af_bella",
    )

rendered.save_wav("hello.wav")
print(rendered.sample_rate, rendered.audio.shape)
```

The default renderer loads model assets lazily. A cached model is reused by the renderer
for requests with the same model configuration. `RenderedSegment.save_wav()` writes mono
float32 WAV audio through `soundfile`.

## Render longer text

The default `long_text_split="none"` mode raises `SynthesisInputTooLongError` when a
request exceeds the model capacity. Set `long_text_split="sentence"` to enable lazy,
model-safe PhraseSplit chunking for oversized text. The chunks produce one
`RenderedSegment`; spaCy is not required by default. See
[long-text configuration](basic_usage.md#render-longer-text).

## Submit a prepared request

Use `SynthesisSegment` when the caller already has an ID, resolved voice, or
pronunciation context. Offsets are half-open Python character ranges into the exact
request text:

```python
from pykokoro import (
    GenerationConfig,
    KokoroSynthesizer,
    LinguisticToken,
    PronunciationOverride,
    SynthesisConfig,
    SynthesisSegment,
)

request = SynthesisSegment(
    id="line-001",
    text="Hello Welt.",
    language="en-us",
    voice="af_bella",
    pronunciation_overrides=(PronunciationOverride(6, 10, language="de"),),
    tokens=(LinguisticToken(0, 5, text="Hello", pos="INTJ"),),
)
config = SynthesisConfig(
    voice="af_bella",
    generation=GenerationConfig(lang="en-us"),
)
with KokoroSynthesizer(config) as synthesizer:
    rendered = synthesizer.synthesize(request)
```

A caller can pass source-aligned context extracted elsewhere without passing planner,
parser, or spaCy objects to PyKokoro.

## Independent requests

```python
from pykokoro import (
    GenerationConfig,
    KokoroSynthesizer,
    SynthesisConfig,
    SynthesisSegment,
)
config = SynthesisConfig(generation=GenerationConfig(lang="en-us"))
requests = (
    SynthesisSegment("one", "First request.", "en-us", voice="af_sarah"),
    SynthesisSegment("two", "Second request.", "en-us", voice="af_bella"),
)
with KokoroSynthesizer(config) as synthesizer:
    for rendered in synthesizer.synthesize_segments(requests):
        rendered.save_wav(f"{rendered.id}.wav")
```

The order and IDs are preserved. Each item is a separate waveform; PyKokoro does not
concatenate them or insert silence between requests. The caller decides whether and how
to compose the resulting audio.

## Where next?

- [Installation and model providers](installation.md)
- [Basic request and configuration patterns](basic_usage.md)
- [Pronunciation context, routing, and calibration](advanced_features.md)
- [Public API reference](api_reference.md)
- [Breaking change and migration note](breaking-change-0.10.0.md)
- [Language codes and supported model profiles](languages.md)
- [Frontends, lexicons, asset progress, and cache installation](installation.md)
- [Maintained categorized examples](examples.md)

- Need long text? → [basic usage](basic_usage.md#render-longer-text)
- Need pronunciation control? → [advanced features](advanced_features.md)
- Need a model/voice? → [language profiles](languages.md) and the
  [discovery example](../examples/models_and_languages.py)
- Need diagnostics? →
  [advanced features](advanced_features.md#result-metadata-and-errors) and the
  [result metadata example](../examples/result_metadata.py)

`SynthesisInputTooLongError` is the expected default outcome for an oversized request;
enable sentence splitting explicitly or catch the typed error and decide how the caller
should continue.
