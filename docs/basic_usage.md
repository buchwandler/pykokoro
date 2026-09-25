# Basic usage

PyKokoro renders one already-prepared speech request at a time. The request's `language`
is explicit; it is not inferred from the voice name or from document metadata.

## Configure and reuse the synthesizer

```python
from pykokoro import GenerationConfig, KokoroSynthesizer, SynthesisConfig

config = SynthesisConfig(
    voice="af_bella",
    generation=GenerationConfig(lang="en-us", speed=1.0),
)
with KokoroSynthesizer(config) as synthesizer:
    first = synthesizer.synthesize_text("Hello.", language="en-us")
    second = synthesizer.synthesize_text("How are you?", language="en-us")
```

`KokoroSynthesizer` reuses its G2P frontend and compatible renderer resources across
requests. Close it explicitly or use it as a context manager.

## Configure a model

Model and voice choices can be supplied to `SynthesisConfig` while the target
pronunciation language remains on the request:

```python
from pykokoro import GenerationConfig, KokoroSynthesizer, SynthesisConfig

config = SynthesisConfig(
    voice="af_sarah",
    model_source="github",
    model_variant="v1.0",
    model_quality="fp32",
    generation=GenerationConfig(lang="en-us", speed=1.05, random_seed=7),
)
with KokoroSynthesizer(config) as synthesizer:
    rendered = synthesizer.synthesize_text("Model-selected speech.", language="en-us")
```

The supported combinations depend on model profiles and available artifacts. Use
`discover_models()` to inspect runtime-ready models, languages, voices, qualities, and
frontends without loading model weights. `model_path`, `voices_path`, and
`model_config_path` can point at local artifacts when appropriate.

## Caller-owned request sizing

Every request is atomic. PyKokoro phonemizes exactly the supplied text and checks
capacity after G2P tokenization. If the request exceeds the model limit,
`SynthesisInputTooLongError` reports the source-text length, model-token count, maximum,
and model identity. The engine never divides the request; callers choose text boundaries
and compose separate results themselves.

`SynthesisConfig.long_text_split` is retained for migration but accepts only `"none"`.
Explicit legacy `"sentence"` or `"token"` values raise `ConfigurationError`; they are
never silently mapped. Split requests before calling PyKokoro, then pass each span as
its own request.

For example, use externally selected boundaries and retain responsibility for combining
the independent results:

```python
from phrasplit import split_with_offsets
from pykokoro import (
    GenerationConfig,
    KokoroSynthesizer,
    SynthesisConfig,
    SynthesisSegment,
)

text = "A long passage. Another sentence."
parts = split_with_offsets(
    text, mode="sentence", use_spacy=True, language="en"
)
requests = [
    SynthesisSegment(str(index), part.text, "en-us", voice="af_sarah")
    for index, part in enumerate(parts)
]
config = SynthesisConfig(
    generation=GenerationConfig(lang="en-us"),
)
with KokoroSynthesizer(config) as synthesizer:
    rendered_parts = list(synthesizer.synthesize_segments(requests))
```

Install PhraseSplit's optional spaCy extra with `pip install "phrasplit[nlp]"` only if
the caller uses its `use_spacy=True` analysis mode. Each external part becomes an
independent request and result. PyKokoro checks each request after G2P and never further
divides it.

## Per-request voice and independent batch output

A request-level voice takes precedence over the configured default:

```python
from pykokoro import GenerationConfig, KokoroSynthesizer, SynthesisConfig, SynthesisSegment

config = SynthesisConfig(generation=GenerationConfig(lang="en-us"))
requests = (
    SynthesisSegment("a", "A first line.", "en-us", voice="af_sarah"),
    SynthesisSegment("b", "A second line.", "en-us", voice="af_bella"),
)
with KokoroSynthesizer(config) as synthesizer:
    rendered_results = list(synthesizer.synthesize_segments(requests))
```

`rendered_results` contains two separate `RenderedSegment` values in input order. Their
audio is not concatenated and no silence is inserted between them. If a complete program
or chapter is needed, compose the results in the application that owns that timeline.

## Save and play

```python
rendered.save_wav("speech.wav")
rendered.play()  # requires: pip install "pykokoro[cpu,playback]"
```

WAV output is mono float32. Direct playback is optional and consumes one rendered
waveform; it is not a composition API.

## Configuration ownership

- `GenerationConfig.speed` controls Kokoro's acoustic inference speed. Editorial
  playback rate belongs to the caller's composition layer.
- `GenerationConfig.random_seed` sets the inference seed when supported by the
  model/runtime.
- `GenerationConfig.enable_short_sentence` or `SynthesisConfig.short_sentence_config`
  explicitly enables short-sentence processing; leaving both unset keeps it off.
- `RenderedSegment.synthesis_identity` and `voice_level_applications` expose cache
  identity and calibration outcomes.
- `SynthesisConfig.voice_level` controls optional engine-local voice calibration, not
  whole-program loudness mastering.
- `SynthesisConfig.return_trace=True` attaches request-local engine trace information to
  the result.
- `SynthesisConfig.waveform_validation` controls engine-local waveform checks.

See [advanced features](advanced_features.md) for source-aligned G2P context and
routing.
