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
    annotations=(LinguisticToken(0, 5, text="Hello", pos="INTJ"),),
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

## Next steps

- [Installation and model providers](installation.md)
- [Basic request and configuration patterns](basic_usage.md)
- [Pronunciation context, routing, and calibration](advanced_features.md)
- [Public API reference](api_reference.md)
- [Breaking change and migration note](breaking-change-0.10.0.md)
