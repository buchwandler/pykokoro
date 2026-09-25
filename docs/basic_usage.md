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

## Render longer text

By default, `long_text_split="none"` keeps each request intact and raises
`SynthesisInputTooLongError` if its prepared token count exceeds the model capacity. To
opt into internal model-safe splitting for oversized requests, configure
`long_text_split="sentence"`:

```python
from pykokoro import GenerationConfig, KokoroSynthesizer, SynthesisConfig

text = "The first sentence is here. The next sentence follows."
config = SynthesisConfig(
    voice="af_sarah",
    generation=GenerationConfig(lang="en-us"),
    long_text_split="sentence",
    long_text_use_spacy=False,
)
with KokoroSynthesizer(config) as synthesizer:
    rendered = synthesizer.synthesize_text(text, language="en-us")

assert rendered.text == text
rendered.save_wav("long-text.wav")
```

Splitting runs only when the model token limit is exceeded. PhraseSplit is loaded
lazily; its simple mode does not require spaCy. `long_text_use_spacy=None` allows
PhraseSplit to use a compatible local spaCy model or fall back to regex, while `True`
requires spaCy and a compatible model. Sentence spans are packed into token-safe chunks.
Oversized sentences fall back to clauses and safe word boundaries. A single word that
cannot fit safely still raises `SynthesisInputTooLongError`. All internal chunks are
rendered as one `RenderedSegment` with the original request text. Separate requests
remain independent, and cross-request composition remains caller-owned.

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
