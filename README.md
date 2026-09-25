# PyKokoro

PyKokoro is a Kokoro speech-synthesis engine. It accepts prepared speech text and
explicit pronunciation context, uses KokoroG2P for model-compatible phonemization, and
uses OnnxVoice for Kokoro ONNX inference. Document parsing, SSMD interpretation, speech
planning, timeline composition, embedded audio, and final-output mastering are
intentionally outside PyKokoro.

## Quick start

Install one ONNX Runtime provider, then synthesize a prepared string:

```bash
pip install "pykokoro[cpu]"
```

```python
from pykokoro import GenerationConfig, KokoroSynthesizer, SynthesisConfig

config = SynthesisConfig(
    voice="af_sarah",
    generation=GenerationConfig(lang="en-us"),
)
with KokoroSynthesizer(config) as synthesizer:
    rendered = synthesizer.synthesize_text(
        "Hello, world.",
        language="en-us",
        voice="af_sarah",
    )

rendered.save_wav("hello.wav")
```

The WAV writer stores mono float32 audio. Model assets are provisioned lazily on first
use.

## Prepared requests and pronunciation context

For orchestration, create a `SynthesisSegment` with an opaque request ID, prepared text,
an explicit pronunciation language, and an actual Kokoro voice. Optional source-aligned
pronunciation instructions and linguistic annotations use offsets into that exact text:

```python
from pykokoro import (
    LinguisticToken,
    PronunciationOverride,
    SynthesisSegment,
)

request = SynthesisSegment(
    id="line-001",
    text="Hello Welt.",
    language="en-us",
    voice="af_sarah",
    pronunciation_overrides=(PronunciationOverride(6, 10, language="de"),),
    annotations=(LinguisticToken(0, 5, text="Hello", pos="INTJ"),),
)
```

Use `synthesize()` for one request or `synthesize_segments()` for an ordered iterable of
independent requests. Each request yields its own `RenderedSegment`; PyKokoro does not
join batch results or insert cross-request silence. The caller can save, play, or pass
each waveform to a separate composition system.

Input text is prepared speech, not a document markup language. PyKokoro does not parse
SSMD, YAML front matter, or say-as/voice/pause directives. `GenerationConfig.speed` is a
Kokoro acoustic inference control, not editorial timeline-rate policy.

## Engine behavior

The engine owns Kokoro-specific G2P integration, request-local voice/model/style
selection, voice blends, token-capacity validation, explicitly configured short-sentence
handling, inference, timing reconstruction, waveform validation, tracing, and optional
voice-level calibration. `discover_models()` and `discover_lexicons()` inspect supported
runtime capabilities and lexicon metadata without loading a synthesis session.

Each call returns one request-local `RenderedSegment`. By default,
`long_text_split="none"` keeps the exact-text path and raises
`SynthesisInputTooLongError` when the prepared text exceeds the model token limit. Set
`SynthesisConfig.long_text_split="sentence"` to enable internal model-safe splitting
only when the request is oversized. PhraseSplit is imported lazily for that path; it
splits on sentence boundaries, then clauses or safe word boundaries as needed, and joins
the audio chunks into one result while preserving the original request text and
source-aligned context. `long_text_use_spacy=False` is the default, so spaCy is not
required. Separate caller requests remain separate results, and cross-request
composition stays with the caller. See
[long-text configuration](docs/basic_usage.md#render-longer-text) for details.

## Installation

Python 3.10 or newer is required. Choose one ONNX Runtime provider extra per
environment:

```bash
pip install "pykokoro[cpu]"        # CPU
pip install "pykokoro[gpu]"        # NVIDIA CUDA
pip install "pykokoro[openvino]"   # OpenVINO
pip install "pykokoro[directml]"  # DirectML
pip install "pykokoro[coreml]"     # Apple CoreML
```

For optional direct playback, install `pykokoro[cpu,playback]`. PyKokoro writes WAV
files through `soundfile`; `RenderedSegment.play()` uses the optional `sounddevice`
dependency. Install `espeak-ng` when selecting an eSpeak frontend or fallback.

## Documentation and examples

- [Quickstart](docs/quickstart.md)
- [Request API and advanced usage](docs/advanced_features.md)
- [API reference](docs/api_reference.md)
- [Language and model profiles](docs/languages.md)
- [Maintained examples](examples/README.md)
- [Breaking change and migration note for v0.10.0](docs/breaking-change-0.10.0.md)
- [Historical changelog](docs/changelog.md)

Run the request-centric examples from the repository root with
`python examples/run_all.py`.

## Development

```bash
pip install -e ".[dev,cpu]"
python -m pytest
```

See `AGENTS.md` for the repository's test, lint, and type-check commands.
