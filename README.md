[![PyPI - Version](https://img.shields.io/pypi/v/pykokoro)](https://pypi.org/project/pykokoro/)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pykokoro)
![PyPI - Downloads](https://img.shields.io/pypi/dm/pykokoro)
[![codecov](https://codecov.io/gh/buchwandler/pykokoro/graph/badge.svg?token=iCHXwbjAXG)](https://codecov.io/gh/buchwandler/pykokoro)

# PyKokoro

A Python library for Kokoro TTS (Text-to-Speech) using ONNX runtime.

## Features

- **ONNX-based TTS**: Fast, efficient text-to-speech using the Kokoro-82M model
- **Multiple Languages**: Support for English, Spanish, French, German, Italian,
  Portuguese, and more
- **Multiple Voices**: 54+ built-in voices (or 103 voices with v1.1-zh model)
- **Voice Blending**: Create custom voices by blending multiple voices
- **Multiple Model Sources**: Download models from HuggingFace or GitHub (v1.0/v1.1-zh)
- **Model Quality Options**: Choose from fp32, fp16, q8, q4, and uint8 quantization
  levels
- **ONNX Execution Providers**: Capability-driven CUDA, NNAPI, XNNPACK, CoreML,
  DirectML, and other runtime-reported providers
- **Phoneme Support**: Advanced phoneme-based generation with kokorog2p
- **Language-Aware spaCy Models**: Shared pipeline-owned Pass-A and Pass-B resources
  with disabled, local fallback, and strict policies
- **Hugging Face Integration**: Automatic model downloading from Hugging Face Hub
- **Explicit Language Planning**: The document language is required before parsing; SSMD
  `lang` spans provide explicit mixed-language runs
- **Text Normalization**: Spokenform owns generic written-to-spoken preparation, including
  semantic say-as behavior when supported by the SSMD/Spokenform contract
- **Maintainer Benchmarking**: PolyNorm-based phoneme regression tooling for the
  PyKokoro frontend path


## v0.9 orchestration contract

PyKokoro v0.9 requires an explicit document language. Set `GenerationConfig(lang="en-us")`
or pass `lang="en-us"` to `run`; the per-call value takes precedence. A voice or model
profile never selects the language. Mixed-language documents must use explicit SSMD language
spans, for example:

```python
config = PipelineConfig(generation=GenerationConfig(lang="en-us"))
result = KokoroPipeline(config).run(
    'Hello [Welt]{lang="de"}.',
    lang="en-us",
 )
```

Integrated requests perform source analysis before Spokenform and fresh prepared-text
analysis afterward. The pipeline reuses loaded local spaCy pipelines, but releases request
documents before returning results. Use `tokenizer_config.use_spacy=False` to disable NLP,
leave it unset for local-only fallback, or set it to `True` for strict model availability.
## Runtime model support

Runtime model selection uses the canonical `catalog/models.json` registry. Model
metadata, voices, frontend IDs, runtime layouts, artifact hashes, provider, and
redistribution policy are not inferred from GitHub release names.

| Status                      | Meaning                                                                 | Examples                                                                                                   |
| --------------------------- | ----------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| ready                       | Registry distribution and PyKokoro frontend/layout are usable           | `v1.0`, `v1.1-zh`, `v1.2-de-martin`, `de-thorsten`, `th-wayu`, `sv-joakim`, `kk-anuarsv`, Zaakirio Russian |
| experimental                | Usable only when explicitly enabled for an experimental frontend        | Profiles marked experimental by the local compatibility policy                                             |
| restricted                  | Runtime is visible but redistribution policy is not ordinary            | `he-hebrew-nc`                                                                                             |
| registry-unavailable        | Registry has no runtime-ready distribution                              | `vi-anphunl`                                                                                               |
| unsupported-layout/frontend | Registry metadata is valid but the local implementation cannot serve it | A future registry layout or frontend                                                                       |

Thai Wayu uses the registry's `split-onnx-v1` layout and selects its prosody, curves,
and decoder components as one distribution. Russian Zaakirio uses pinned upstream-only
distributions and validates raw float32 voice artifacts locally; those bytes are not
mirrored to GitHub.

Use `python examples/models_and_languages.py` to inspect every registry model, language,
provider, voice, quality, frontend, layout, and support status without downloading model
weights. Pass `--model MODEL_ID` to synthesize only one selected model; experimental
frontends additionally require `--include-experimental`.

## Installation

### Basic Installation (CPU only)

```bash
pip install "pykokoro[cpu]"
```

The ONNX Runtime distributions are alternatives. Install exactly one provider extra for
inference: `cpu`, `gpu`, `openvino`, or `directml`. The `coreml` extra uses the macOS
CPU distribution plus CoreML tooling. The `all` extra adds optional playback support
(`sounddevice`) and never installs multiple ONNX Runtime wheels.

### GPU and Accelerator Support

PyKokoro supports multiple hardware accelerators for faster inference:

#### NVIDIA CUDA GPU

```bash
pip install pykokoro[gpu]
```

#### Intel OpenVINO

**Note:** OpenVINO is currently incompatible with Kokoro models due to dynamic rank
tensor requirements. The provider will automatically fall back to CPU if OpenVINO fails.

```bash
pip install pykokoro[openvino]
```

#### DirectML (Windows - AMD/Intel/NVIDIA GPUs)

```bash
pip install pykokoro[directml]
```

#### Apple CoreML (macOS)

```bash
pip install pykokoro[coreml]
```

#### Optional Features

```bash
pip install pykokoro[all]
```

### Direct Playback

For direct playback from memory, install the optional feature extra:

```bash
pip install "pykokoro[cpu,playback]"
```

`AudioResult.play()` sends an already-generated NumPy waveform directly to the system
audio output. Playback is blocking and does not create a WAV file. For long text with
low startup latency, use `pipeline.play_streaming(text, unit="sentence")`; it prepares
the document globally, then generates sentence audio while one persistent bounded stream
plays earlier sentences. Linux-like systems may also need a PortAudio system package.
The older `pykokoro[sounddevice]` extra remains valid.

### Performance Comparison

To find the best provider for your system, run the benchmark:

```bash
python examples/gpu_benchmark.py
```

## Quick Start

The pipeline API is the only supported interface.

```python
from pykokoro import KokoroPipeline, PipelineConfig

pipe = KokoroPipeline(PipelineConfig(voice="af_sarah"))
res = pipe.run("Hello")
audio = res.audio
```

### Managing Result Memory

By default, `AudioResult` retains the raw and processed waveform for each phoneme
segment for diagnostics and callers that inspect segment audio. For long documents,
enable compact result retention when only the final waveform and metadata are needed:

```python
from pykokoro import PipelineConfig, build_pipeline

pipeline = build_pipeline(
    config=PipelineConfig(
        voice="af_heart",
        retain_segment_audio=False,
    )
)

result = pipeline.run("Long text")
result.save_wav("chapter.wav")
result.release_audio()
pipeline.close()
```

Compact mode reduces memory retained by the completed result after generation; it does
not make peak memory independent of input duration because the pipeline still builds the
whole-result concatenated waveform. Use `result.release_segment_audio()` to retain the
final waveform while dropping per-segment arrays, or `result.release_audio()` to drop
both. These methods only release references owned by the result, so arrays held
separately by callers remain valid. Callers that need raw or processed segment waveforms
should keep `retain_segment_audio=True`. Use the paragraph streaming API below for
bounded unit rendering.

### Paragraph-Unit Streaming

Use `prepare_units()` when a document should be prepared once but rendered and stored
one paragraph at a time. Preparation resolves SSMD directives, pauses, markers, voices,
and preprocessing globally; `skip_indices` can omit units already completed by a caller.

```python
from pykokoro import KokoroPipeline, PipelineConfig

pipeline = KokoroPipeline(PipelineConfig(voice="af_sarah"))
with pipeline.prepare_units("First paragraph.\n\nSecond paragraph.") as prepared:
    for unit in prepared.render(skip_indices={0}):
        try:
            save_waveform(unit.audio, unit.sample_rate)
        finally:
            unit.release_audio()
```

`AudioUnitResult.release_audio()` is destructive and idempotent. The iterator also
releases the previous unit before yielding the next one, so consumers should copy or
persist audio inside the loop. Closing the prepared object releases prepared segment
audio but does not close the reusable pipeline backend.

Each descriptor has a zero-based source-order index and a `text_hash` using the
`pykokoro-audio-unit-v1` schema. Store both values in resume manifests; schema changes
must use a new prefix. Hashes include audio-semantic settings such as voice, pauses,
language, prosody, model variant, and explicit `model_identity`, but ignore tracing,
retention, cache-directory, and machine-local runtime toggles. Persist or copy a unit's
waveform before advancing the iterator because advancing releases the previous result.

## Pipeline Stages

The pipeline is built from composable stages so you can swap behavior without rewriting
the whole flow:

`doc_parser (SSMD structure) -> text_preparer (Spokenform) -> sentence_segmenter (Phrasplit) -> g2p (prepared mode) -> phoneme_processing -> audio_generation -> audio_postprocessing`

Stages can be replaced with no-op adapters when you want to disable behavior. See
`examples/pipeline_stage_showcase.py` for a full wiring example.

```python
from pykokoro import KokoroPipeline, PipelineConfig
from pykokoro.stages.doc_parsers.plain import PlainTextDocumentParser

pipe = KokoroPipeline(
    PipelineConfig(voice="af"),
    doc_parser=PlainTextDocumentParser(),
)
res = pipe.run("First paragraph.\n\nSecond paragraph.")
```

### Migration

Old (removed):

```python
# Legacy Kokoro-based API has been removed in favor of the pipeline.
```

New:

```python
from pykokoro import KokoroPipeline, PipelineConfig

pipe = KokoroPipeline(PipelineConfig(voice="af"))
res = pipe.run("Hello")
audio = res.audio
```

### Helper Snippet

```python
from pykokoro import KokoroPipeline, PipelineConfig
from pykokoro.generation_config import GenerationConfig

generation = GenerationConfig(lang="en-us", speed=1.0)
config = PipelineConfig(voice="af_sarah", generation=generation)
pipe = KokoroPipeline(config)
res = pipe.run("Hello")
```

## Hardware Acceleration

### Automatic Provider Selection (Recommended)

```python
# Auto-select by runtime capability (CUDA > NNAPI > OpenVINO > CoreML > DirectML > XNNPACK > CPU)
# The selected accelerator is paired with CPU fallback when the session supports it.
from pykokoro import KokoroPipeline, PipelineConfig

pipe = KokoroPipeline(PipelineConfig(provider="auto", voice="af_sarah"))
res = pipe.run("Hello")
```

### Explicit Provider Selection

```python
# Force specific provider
from pykokoro import KokoroPipeline, PipelineConfig

pipe = KokoroPipeline(PipelineConfig(provider="cuda", voice="af_sarah"))  # NVIDIA CUDA
pipe = KokoroPipeline(PipelineConfig(provider="nnapi", voice="af_sarah"))  # Android NNAPI
pipe = KokoroPipeline(PipelineConfig(provider="xnnpack", voice="af_sarah"))  # XNNPACK
pipe = KokoroPipeline(PipelineConfig(provider="openvino", voice="af_sarah"))  # Intel OpenVINO
pipe = KokoroPipeline(PipelineConfig(provider="directml", voice="af_sarah"))  # Windows DirectML
pipe = KokoroPipeline(PipelineConfig(provider="coreml", voice="af_sarah"))  # Apple CoreML
pipe = KokoroPipeline(PipelineConfig(provider="cpu", voice="af_sarah"))  # CPU only
```

### Check Available Providers

```bash
# See all available providers on your system
python examples/provider_info.py

# Benchmark all providers
python examples/gpu_benchmark.py
```

### Environment Variable Override

```bash
# Force a specific provider via environment variable
export ONNX_PROVIDER="OpenVINOExecutionProvider"
python your_script.py
```

Provider aliases and full names returned by ONNX Runtime are accepted. Inspect and
resolve providers without duplicating platform-specific mappings:

```python
from pykokoro.onnx_session import (
    get_available_execution_providers,
    resolve_execution_provider,
)

print(get_available_execution_providers())
print(resolve_execution_provider("auto"))
```

## Usage Examples

Maintainer benchmark documentation for the PolyNorm phoneme gate lives in
`docs/polynorm_benchmark.md`.

### Basic Text-to-Speech

```python
from pykokoro import KokoroPipeline, PipelineConfig
from pykokoro.generation_config import GenerationConfig

# Create pipeline with GPU acceleration and fp16 model
config = PipelineConfig(
    voice="af_nicole",
    provider="cuda",
    model_quality="fp16",
    generation=GenerationConfig(lang="en-us"),
)
pipe = KokoroPipeline(config)

# Generate audio
res = pipe.run("Hello world")
audio = res.audio
```

### Voice Blending

```python
# Blend two voices (50% each)
from pykokoro import KokoroPipeline, PipelineConfig
from pykokoro.voice_manager import VoiceBlend

blend = VoiceBlend.parse("af_nicole:50,am_michael:50")
pipe = KokoroPipeline(PipelineConfig(voice=blend))
res = pipe.run("Mixed voice")
audio = res.audio
```

### Direct Playback of Generated Chunks

For independent chunks, `AudioResult.play()` plays each generated waveform directly:

```python
from pykokoro import KokoroPipeline, PipelineConfig

pipe = KokoroPipeline(PipelineConfig(voice="af_sarah"))
chunks = ["Long text", "here..."]
for text_chunk in chunks:
    result = pipe.run(text_chunk)
    result.play()
```

For long text with low startup latency, prefer sentence streaming through one persistent
bounded output stream:

```python
from pykokoro import KokoroPipeline, PipelineConfig

text = "First sentence. Second sentence. Third sentence."
with KokoroPipeline(PipelineConfig(voice="af_sarah")) as pipe:
    pipe.play_streaming(text, unit="sentence", queue_size=2)
```

`play_streaming()` performs global document preparation first, then generates and queues
each selected unit as playback consumes the previous one. It creates no temporary WAV
and retains no complete generated waveform. `queue_size` is bounded pending-waveform
capacity, not an exact startup prebuffer count.

For custom consumers or paragraph-sized chunks, use the prepared-unit API directly:

```python
with pipe.prepare_units(text, unit="paragraph") as prepared:
    for result in prepared.render():
        try:
            consume(result.audio, result.sample_rate)
        finally:
            result.release_audio()
```

Use `result.play()` for a short utterance that has already been generated completely.

### Phoneme-Based Generation

```python
from pykokoro import KokoroPipeline, PipelineConfig
from pykokoro.generation_config import GenerationConfig
from pykokoro.tokenizer import Tokenizer

# Create tokenizer
tokenizer = Tokenizer()

# Convert text to phonemes
phonemes = tokenizer.phonemize("Hello world", lang="en-us")
print(phonemes)  # hə'loʊ wɜːld

# Generate from phonemes
config = PipelineConfig(
    voice="af_sarah",
    generation=GenerationConfig(lang="en-us", is_phonemes=True),
)
pipe = KokoroPipeline(config)
res = pipe.run(phonemes)
audio = res.audio
```

### Pause Control

PyKokoro uses SSMD (Speech Synthesis Markdown) syntax for controlling pauses in
generated speech:

#### 1. SSMD Break Markers

Add explicit pauses using SSMD break syntax in your text:

```python
# Use SSMD break markers in your text
text = "Chapter 5 ...p I'm Klaus. ...c Welcome to the show!"

# Breaks are processed automatically
from pykokoro import KokoroPipeline, PipelineConfig

pipe = KokoroPipeline(PipelineConfig(voice="am_michael"))
res = pipe.run(text)
audio = res.audio
```

**SSMD Break Markers:**

- `...n` - No pause (0ms)
- `...w` - Weak pause (150ms by default)
- `...c` - Clause/comma pause (300ms by default)
- `...s` - Sentence pause (600ms by default)
- `...p` - Paragraph pause (1000ms by default)
- `...500ms` - Custom pause (500 milliseconds)
- `...2s` - Custom pause (2 seconds)

**Note:** Bare `...` (ellipsis) is NOT treated as a pause and will be phonemized
normally.

**Custom Pause Durations:**

```python
from pykokoro import KokoroPipeline, PipelineConfig
from pykokoro.generation_config import GenerationConfig

config = PipelineConfig(
    voice="am_michael",
    generation=GenerationConfig(
        pause_mode="manual",
        pause_clause=0.2,  # ...c = 200ms
        pause_sentence=0.5,  # ...s = 500ms
        pause_paragraph=1.5,  # ...p = 1500ms
    ),
)
pipe = KokoroPipeline(config)
res = pipe.run(text)
audio = res.audio
```

#### 2. Automatic Natural Pauses

For more natural speech, enable automatic pause insertion at linguistic boundaries with
`pause_mode="auto"`:

```python
text = """
Artificial intelligence is transforming our world. Machine learning models
are becoming more sophisticated, efficient, and accessible.

Deep learning, a subset of AI, uses neural networks with many layers. These
networks can learn complex patterns from data, enabling breakthroughs in
computer vision, natural language processing, and speech recognition.
"""

# Automatic pauses at clause, sentence, and paragraph boundaries
from pykokoro import KokoroPipeline, PipelineConfig
from pykokoro.generation_config import GenerationConfig

config = PipelineConfig(
    voice="af_sarah",
    generation=GenerationConfig(
        pause_mode="auto",
        pause_clause=0.25,  # Pause after clauses (commas)
        pause_sentence=0.5,  # Pause after sentences
        pause_paragraph=1.0,  # Pause after paragraphs
        pause_variance=0.05,  # Add natural variance (default)
        random_seed=42,  # For reproducible results (optional)
    ),
)
pipe = KokoroPipeline(config)
res = pipe.run(text)
audio = res.audio
```

**Key Features:**

- **Natural boundaries**: Automatically detects clauses, sentences, and paragraphs
- **Variance**: Gaussian variance prevents robotic timing (±100ms by default)
- **Reproducible**: Use `random_seed` for consistent output
- **Composable**: Works with SSMD break markers

**Splitting Behavior:**

- `SsmdDocumentParser` handles paragraph/sentence segmentation using SSMD.
- `PlainTextDocumentParser` uses optional `phrasplit` sentence splitting.

**Pause Variance Options:**

- `pause_variance=0.0` - No variance (exact pauses)
- `pause_variance=0.05` - Default (±100ms at 95% confidence)
- `pause_variance=0.1` - More variation (±200ms at 95% confidence)

**Note:** For sentence splitting with `PlainTextDocumentParser` and spaCy-based G2P
tokenization, install spaCy and at least one language model:

```bash
pip install spacy
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_md
```

If both `TokenizerConfig.spacy_model` and `spacy_model_size` are unset (the default),
PyKokoro asks each spaCy-using backend to select the highest installed compatible model
for the effective language (`trf > lg > md > sm`). No spaCy model is downloaded
automatically. `"auto"` remains accepted as a compatibility alias for unset.

Use `with_spacy_model(size="lg")` or an explicit package when a strict choice is needed.
`lg` and `trf` can improve linguistic quality but require substantially more memory and
startup time than `sm`/`md`. The selected concrete sentence and G2P packages are
available in `AudioResult.document_metadata["spacy_models"]`.

For TTSForge integrations, use the PyKokoro version that provides these
`TokenizerConfig` fields, read concrete sentence and G2P selections from that metadata
path, and rely on the guarantee that plain and SSMD parsing share the same request while
G2P delegates resolution to kokorog2p.

**Combining Both Approaches:**

Use SSMD markers for emphasis metadata and automatic pauses for natural rhythm:

```python
text = "Welcome! ...p Let's discuss AI, machine learning, and deep learning."

config = PipelineConfig(
    voice="af_sarah",
    generation=GenerationConfig(pause_mode="auto", pause_variance=0.05),
)
pipe = KokoroPipeline(config)
res = pipe.run(text)
audio = res.audio
```

See `examples/pauses_demo.py`, `examples/pauses_with_splitting.py`, and
`examples/automatic_pauses_demo.py` for complete examples.

### SSMD emphasis policy

SSMD emphasis is preserved in segment metadata, but PyKokoro defaults to
`SSMDRenderConfig(emphasis_mode="plain")` so ordinary synthesis is not changed
surprisingly. The policy modes are:

- `plain`: preserve emphasis metadata and synthesize unmodified speech silently
- `approximate`: apply deterministic gain-only changes: `strong` `+6dB`, `moderate`
  `+3dB`, and `reduced` `-3dB` by default
- `warn`: synthesize unmodified speech and emit one `ssmd.emphasis_unsupported` trace
  warning per logical source segment
- `error`: reject effectful emphasis before model inference

`emphasis="none"` means ordinary speech and is accepted silently in every mode. Set
`emphasis_gain_scale` on `SSMDRenderConfig` to scale only the automatic gain while
preserving the semantic level. `1.0` is the current/default strength, `0.5` halves the
automatic gain, and `1.5` makes it 50% stronger; the supported range is `0.0..2.0`.
Explicit SSMD `volume` metadata takes precedence over automatic emphasis gain. This
setting does not add automatic pitch or rate changes, and PyKokoro does not provide
TTSForge's user-facing integer emphasis presets.

```python
config = PipelineConfig(
    ssmd=SSMDRenderConfig(
        emphasis_mode="approximate",
        emphasis_gain_scale=1.5,
    )
)
```

### Prosody backend selection

Combined SSMD pitch, rate, and volume effects use one AudioSig speech-effects compositor
pass. PyKokoro defaults to the speech-oriented WSOLA backend:

```python
from pykokoro import PipelineConfig, ProsodyConfig

config = PipelineConfig(
    prosody=ProsodyConfig(method="wsola"),
)
```

ESOLA and TD-PSOLA are experimental alternatives, while `phase_vocoder` remains
available as a compatibility and diagnostic reference. The `psola` spelling is accepted
as an alias for `td_psola`:

```python
config = PipelineConfig(
    prosody=ProsodyConfig(method="td_psola"),
)
```

For an apples-to-apples comparison, disable fallback so unsupported methods cannot be
silently relabeled:

```python
config = PipelineConfig(
    prosody=ProsodyConfig(
        method="esola",
        fallback_methods=(),
        strict=True,
    ),
)
```

WSOLA is the production default. ESOLA validates its computed backend rate in
`0.5..2.0`, and current TD-PSOLA limits are rate `0.75..1.5` and pitch `-6..+6 st`;
non-strict mode can fall back to configured backends after a failure. No backend
guarantees formant preservation, and quality depends on the voice and utterance. Run
`examples/compare_prosody_algorithms.py` before changing a default. Prosody is applied
to isolated rendered segments, so it cannot restore sentence-level coarticulation,
intonation, or spectral continuity lost during separate synthesis.

To request audible approximation explicitly:

```python
from pykokoro import KokoroPipeline, PipelineConfig, SSMDRenderConfig

config = PipelineConfig(ssmd=SSMDRenderConfig(emphasis_mode="approximate"))
result = KokoroPipeline(config).run("This is *moderate emphasis*.")
```

### Voice Switching (SSMD)

You can switch voices per segment using SSMD directives. Block directives use
`<div voice="...">` while inline annotations use `[text]{voice="..."}`.

```python
text = (
    '<div voice="af_sarah">\n'
    "Hello there.\n"
    "</div>\n\n"
    '<div voice="am_michael">\n'
    "General Kenobi.\n"
    "</div>"
)

pipe = KokoroPipeline(
    PipelineConfig(voice="af", generation=GenerationConfig(lang="en-us"))
)
res = pipe.run(text)
```


## Explicit mixed-language text

Automatic language detection is not part of PyKokoro v0.9. Set the document language explicitly
and mark language changes with SSMD spans:

```python
text = 'Hello [Welt]{lang="de"} and [bonjour]{lang="fr"}.'
result = pipe.run(text, lang="en-us")
```

Voice metadata selects a voice only. Applications needing detection must run an external
detector and convert its result to explicit language spans before calling PyKokoro.

### Automatic Spoken-Form Normalization

Spokenform owns ordinary written-to-spoken preparation in the integrated path. Common
abbreviations and structured forms such as dates, times, numbers, currency,
measurements, ordinals, and other supported expressions can be spoken naturally from raw
text:

```python
from pykokoro import GenerationConfig, KokoroPipeline, PipelineConfig

text = (
    "Dr. Smith will see you at 10:30 on 05/20/2023. "
    "The box weighs 5 kg and costs $10.99."
)
pipeline = KokoroPipeline(
    PipelineConfig(generation=GenerationConfig(lang="en-us"))
)
result = pipeline.run(text)
```

The default pipeline owns this order explicitly: SSMD structure is parsed first,
Spokenform prepares the written text, Phrasplit detects sentences in the prepared text,
and kokorog2p receives that prepared text through `phonemize_prepared()`. Segment
offsets therefore refer to the prepared spoken `clean_text`; structural annotations,
events, and preparation provenance remain available in the document metadata used by
downstream stages. Use `examples/german3.py` for a German regression containing dates,
quantities, abbreviations, ordinals, and currency.

### Explicit SSMD Say-As Overrides

Use SSMD (Speech Synthesis Markdown) say-as annotations when the author needs explicit
interpretation or an override. They are not required for common automatic forms:

```python
from pykokoro import KokoroPipeline, PipelineConfig
from pykokoro.generation_config import GenerationConfig

pipe = KokoroPipeline(PipelineConfig(voice="af_sarah"))

# Cardinal numbers
text = 'I have [123]{as="cardinal"} apples'
res = pipe.run(text)
# TTS says: "I have one hundred twenty-three apples"

# Ordinal numbers
text = 'I came in [3]{as="ordinal"} place'
res = pipe.run(text)
# TTS says: "I came in third place"

# Digits (spell out)
text = 'My PIN is [1234]{as="digits"}'
res = pipe.run(text)
# TTS says: "My PIN is one two three four"

# Telephone numbers
text = 'Call [+1-555-0123]{as="telephone"}'
res = pipe.run(text)
# TTS says: "Call plus one five five five oh one two three"

# Dates with custom formatting
text = 'Today is [12/31/2024]{as="date" format="mdy"}'
res = pipe.run(text)
# TTS says: "Today is December thirty-first, two thousand twenty-four"

# Time (12-hour or 24-hour)
text = 'The time is [14:30]{as="time"}'
res = pipe.run(text)
# TTS says: "The time is two thirty PM"

# Characters (spell out)
text = 'The code is [ABC]{as="characters"}'
res = pipe.run(text)
# TTS says: "The code is A B C"

# Fractions
text = 'Add [1/2]{as="fraction"} cup of sugar'
res = pipe.run(text)
# TTS says: "Add one half cup of sugar"

# Units
text = 'The package weighs [5kg]{as="unit"}'
res = pipe.run(text)
# TTS says: "The package weighs five kilograms"
```

**Supported Say-As Types:**

- `cardinal` - Numbers as cardinals: "123" → "one hundred twenty-three"
- `ordinal` - Numbers as ordinals: "3" → "third"
- `digits` - Spell out digits: "123" → "one two three"
- `number` - Alias for cardinal
- `fraction` - Fractions: "1/2" → "one half"
- `characters` - Spell out text: "ABC" → "A B C"
- `telephone` - Phone numbers: "+1-555-0123" → "plus one five five five oh one two
  three"
- `date` - Dates with format support (mdy, dmy, ymd, ym, my, md, dm, d, m, y)
- `time` - Time in 12h or 24h format
- `unit` - Units: "5kg" → "five kilograms"
- `expletive` - Censors to "beep"

**Multi-language Support:**

Say-as works with multiple languages (English, French, German, Spanish, and more):

```python
from pykokoro import KokoroPipeline, PipelineConfig
from pykokoro.generation_config import GenerationConfig

# French cardinal
text = '[123]{as="cardinal"}'
pipe = KokoroPipeline(PipelineConfig(voice="ff_siwis", generation=GenerationConfig(lang="fr-fr")))
res = pipe.run(text)
# TTS says: "cent vingt-trois"

# German ordinal
text = '[3]{as="ordinal"}'
pipe = KokoroPipeline(PipelineConfig(voice="gf_maria", generation=GenerationConfig(lang="de-de")))
res = pipe.run(text)
# TTS says: "dritte"
```

**Combining with Other Features:**

Say-as works seamlessly with all SSMD features:

```python
# With prosody
text = '[100]{as="cardinal" volume="loud"} dollars!'

# With pauses
text = '[First]{as="ordinal"} ...c [second]{as="ordinal"} ...c [third]{as="ordinal"}!'

# With emphasis metadata (audible approximation is opt-in)
text = 'The winner is [1]{as="ordinal" emphasis="moderate"}!'
```

See `examples/say_as_demo.py` for comprehensive examples.

#### 4. Automatic Short Sentence Handling

When processing text, very short sentences (like "Why?" or "Go!") can produce poor audio
quality when processed individually (only 3-8 phonemes each). Pykokoro can add phoneme
context around those short segments before synthesis.

**How It Works:**

1. Short segments are detected based on phoneme token length.
2. Depending on the chosen resolve mode, the segment is wrapped with more context.
   (default resolve mode: `randomized-phrase`)
3. TTS generates audio from the wrapped phoneme sequence.
4. Cut away the extra context and put audio together.

This happens automatically during `pipe.run()` - no configuration needed!

NOTE: Currently, phrase and randomized-phrase mode only support ENGLISH text! (see
"Advanced customization of short-sentence handling" to add support for other languages)

**Customizing the Behavior:**

You can customize the behavior using `ShortSentenceConfig`:

```python
from pykokoro import KokoroPipeline, PipelineConfig
from pykokoro.short_sentence_handler import ShortSentenceConfig

# Less aggressive short sentence handling (also less acurate)
short_sentence_config = ShortSentenceConfig(
    resolve_mode="wrap",
    min_phoneme_length=10,  # Treat segments <10 phoneme tokens as short
    phoneme_pretext="—",  # Add this before and after short phonemes
)

# More advanced short sentence handling (useful for some voices)
short_sentence_config = ShortSentenceConfig(
    resolve_mode="randomized-phrase",
    min_phoneme_length=40,  # Treat segments <40 phoneme tokens as short
)

pipe = KokoroPipeline(PipelineConfig(voice="af_sarah", short_sentence_config=short_sentence_config))
res = pipe.run("Why?")
```

**Default Configuration:**

- `enabled=True`: Short-sentence handling is enabled by default
- `min_phoneme_length=30`: Segments below this token count engage short-sentence
  handling
- `resolve_mode="randomized-phrase"` Chose between `randomized-phrase`(default),
  `phrase`, or `wrap`(fallback)
- `phrase_selection="auto"` Chose which phrase templates to use. `auto`= uses "end", if
  phrase ends with '.', otherwise uses "neutral"
- `phrase_fallback_tries=5`: Phrase modes try up to X alternate phrase templates before
  falling back to wrap mode when a cut lacks confident boundaries.
- `phoneme_pretext="—"`: Phoneme context added in wrap mode before and after short
  segments

```python
from pykokoro.short_sentence_handler import (
    PhraseResolveMode,
    ShortSentenceConfig,
)

short_sentence_config = ShortSentenceConfig(
    resolve_modes={
        "phrase": PhraseResolveMode(
            phrase_selection="end",  # "auto", "neutral", or "end"
        ),
        "randomized-phrase": RandomizedPhraseResolveMode(
            phrase_selection="neutral",  # "auto", "neutral", or "end"
        ),
        "wrap": WrapResolveMode(phoneme_pretext="…"),
    },
    resolve_mode="phrase",
    phrase_fallback_tries=10,
)
```

**Voice Recommendation:**

For phrase-based short-sentence handling, prefer these voices in order: `am_santa`,
`af_nicole`, `bm_lewis`, `bm_george`, `af_bella`, `am_echo`, `af_sky`, `af_sarah`,
`bm_fable`, `af_heart`, `am_michael`, `af_alloy`, `af_nova`, `bf_isabella`, and
`am_adam`. If you prefer one of the less accurate voices, try blending it with one on
this list. E.g. --voice-blend "bf_lily:60,bf_isabella:40"

**Disabling Short Sentence Handling:**

```python
from pykokoro import KokoroPipeline, PipelineConfig
from pykokoro.short_sentence_handler import ShortSentenceConfig

short_sentence_config = ShortSentenceConfig(enabled=False)
pipe = KokoroPipeline(PipelineConfig(voice="af_sarah", short_sentence_config=short_sentence_config))
res = pipe.run("Why?")
```

See `examples/optimal_phoneme_length_demo.py` for a demonstration.

**Advanced customization of short-sentence handling**

You can add custom template phrases used to add context in phrase mode, but THIS IS NOT
RECOMMENDED for most users! However, you can use it to add support for more languages
than just english.

WARNING: The quality of the phrase makes a huge difference. If possible, test the
phrases first, e.g. by using the various short-sentence py scripts in metrics/. All
default phrases have been verified with the
metrics\rank_short_sentence_phrases_across_voice_list.py script to work reliably with
most voices.

```python
from pykokoro.short_sentence_handler import (
    PhraseResolveMode,
    ShortSentenceConfig,
)

short_sentence_config = ShortSentenceConfig(
    resolve_modes={
        "phrase": PhraseResolveMode(
            phrase_selection="end",  # "auto", "neutral", or "end"
            neutral_phrase="The word, {segment}, appears here.",  # Changing this to anything not in the default neutral_phrases list is not recommended
            end_phrase="The word is hello. The word is '{segment}'",  # Changing this to anything not in the default end_phrases list is not recommended
        ),
        "randomized-phrase": RandomizedPhraseResolveMode(
            phrase_selection="neutral",  # "auto", "neutral", or "end"
            neutral_phrases=[  # Adding new untested phrases is not recommended without rigurous testing
                "First {segment} is the word.",
                "Second {segment} is the word.",
                "Third {segment} is the word.",
                "Fourth {segment} is the word.",
            ],
            end_phrases=[  # Adding new untested phrases is not recommended without rigurous testing
                "First {segment}."
            ],
        ),
        "wrap": WrapResolveMode(phoneme_pretext="…"),
    },
    resolve_mode="phrase",
)
```

## Available Voices

The library includes voices across different languages and accents. The number of
available voices depends on the model source:

### HuggingFace & GitHub v1.0 (54 voices)

- **American English**: af_alloy, af_bella, af_sarah, am_adam, am_michael, etc.
- **British English**: bf_alice, bf_emma, bm_george, bm_lewis
- **Spanish**: ef_dora, em_alex
- **French**: ff_siwis
- **Japanese**: jf_alpha, jm_kumo
- **Chinese**: zf_xiaobei, zm_yunxi
- And many more...

### GitHub v1.1-zh (103 voices)

Includes all voices from v1.0 plus additional Chinese voices:

- **English voices**: af_maple, af_sol, bf_vale (confirmed working)
- **Chinese voices**: zf_001 through zf_099, zm_009 through zm_100

**Example - Using v1.1-zh with English:**

```python
from pykokoro import KokoroPipeline, PipelineConfig
from pykokoro.generation_config import GenerationConfig

config = PipelineConfig(
    voice="af_maple",
    model_source="github",
    model_variant="v1.1-zh",
    generation=GenerationConfig(lang="en-us"),
)
pipe = KokoroPipeline(config)
res = pipe.run("Hello world!")
audio = res.audio
```

List all available voices:

```python
from pykokoro import KokoroPipeline, PipelineConfig

pipe = KokoroPipeline(PipelineConfig(voice="af_sarah"))
pipe.run("Hello")
# Voices are loaded lazily by the backend after the first run.
voices = pipe.synth._kokoro.get_voices()
print(voices)
```

## Model Sources

PyKokoro supports downloading models from multiple sources:

### HuggingFace (Default)

HuggingFace is the default source with 54 multi-language voices. It downloads the model,
voice archive, and the vocabulary config required by the HuggingFace profile:

```python
from pykokoro import KokoroPipeline, PipelineConfig

pipe = KokoroPipeline(
    PipelineConfig(
        voice="af_sarah",
        model_source="huggingface",
        model_quality="fp32",  # fp32, fp16, q8, q8f16, q4, q4f16, uint8, uint8f16
    )
)
res = pipe.run("Hello world")
```

### GitHub v1.0

54 voices with additional `fp16-gpu` optimized quality:

```python
from pykokoro import KokoroPipeline, PipelineConfig

pipe = KokoroPipeline(
    PipelineConfig(
        voice="af_sarah",
        model_source="github",
        model_variant="v1.0",
        model_quality="fp16-gpu",  # fp32, fp16, fp16-gpu, q8
    )
)
res = pipe.run("Hello world")
```

### Termux/Android: GitHub v1.0

When HuggingFace downloads are unavailable, select the GitHub `v1.0` source explicitly.
GitHub v1.0 downloads only its ONNX model and voice archive and uses the embedded
standard v1.0 vocabulary, so it does not require a HuggingFace `config.json`:

```python
from pykokoro import KokoroPipeline, PipelineConfig

pipe = KokoroPipeline(
    PipelineConfig(
        voice="af_heart",
        model_source="github",
        model_variant="v1.0",
        model_quality="fp32",
    )
)
```

PyKokoro never silently switches between model sources. Explicit `model_path` and
`voices_path` files are validated in place and are never replaced with managed cache
files. A Termux/Android ONNX Runtime warning is a separate runtime-provider issue; it
does not change model-download or source-selection behavior.

### GitHub v1.1-zh (English + Chinese)

103 voices including English and Chinese speakers:

```python
from pykokoro import KokoroPipeline, PipelineConfig
from pykokoro.generation_config import GenerationConfig

pipe = KokoroPipeline(
    PipelineConfig(
        voice="af_maple",
        model_source="github",
        model_variant="v1.1-zh",
        model_quality="fp32",  # Only fp32 available
        generation=GenerationConfig(lang="en-us"),
    )
)
res = pipe.run("Hello world")
audio = res.audio
```

**Note:** Chinese text generation requires proper phonemization support (currently in
development).

### German Martin v1.2

When `GenerationConfig(lang="de")` (or `de-de`, `de-at`, or `de-ch`) is used without
explicit model settings, PyKokoro selects the GitHub `v1.2-de-martin` profile, its
fp32-only `kokoro-german-martin-v1.2.onnx` model, and the single `martin` voice before
backend and G2P caches are constructed. The first run downloads roughly 311 MB of model
and voice assets into the normal `~/.cache/pykokoro` cache. GitHub downloads are
verified with the published SHA-256 digests and structural checks; invalid managed
cached files are removed and re-downloaded. Explicit `model_path` and `voices_path`
files are validated in place and are never silently replaced.

```python
from pykokoro import GenerationConfig, KokoroPipeline, PipelineConfig

config = PipelineConfig(generation=GenerationConfig(lang="de", speed=1.125))
with KokoroPipeline(config) as pipe:
    result = pipe.run("Das ist ein deutscher Testsatz.")
```

For reproducible configuration, select the profile explicitly:

```python
config = PipelineConfig(
    voice="martin",
    model_source="github",
    model_variant="v1.2-de-martin",
    model_quality="fp32",
    generation=GenerationConfig(lang="de", speed=1.125),
)
```

Martin uses the built-in Kokoro v1.0 vocabulary and does not download a Tundragoon
config. `martin` alone also infers German; custom voice archives may expose additional
voice names when selected explicitly. The profile's suggested speed of `1.125` is
advisory, so applications must set it explicitly when they want it. German
Language-specific automatic spoken-form normalization belongs to the compatible
kokorog2p release. PyKokoro keeps source offsets and segments tied to the original input
text, consumes kokorog2p's prepared G2P result, and owns synthesis.

## Model Quality Options

Available quality options vary by source:

**HuggingFace Models:**

- `fp32`: Full precision (highest quality, largest size)
- `fp16`: Half precision (good quality, smaller size)
- `q8`: 8-bit quantized (fast, small)
- `q8f16`: 8-bit with fp16 (balanced)
- `q4`: 4-bit quantized (fastest, smallest)
- `q4f16`: 4-bit with fp16 (compact)
- `uint8`: Unsigned 8-bit (compatible)
- `uint8f16`: Unsigned 8-bit with fp16

**GitHub v1.0 Models:**

- `fp32`: Full precision
- `fp16`: Half precision
- `fp16-gpu`: GPU-optimized fp16
- `q8`: 8-bit quantized

**GitHub v1.1-zh Models:**

- `fp32`: Full precision only

**GitHub v1.2-de-martin:**

- `fp32`: Full precision only; no fp16 or quantized Martin artifacts are published

```python
from pykokoro import KokoroPipeline, PipelineConfig

# HuggingFace with q8
pipe = KokoroPipeline(
    PipelineConfig(voice="af_sarah", model_source="huggingface", model_quality="q8")
)

# GitHub v1.0 with GPU-optimized fp16
pipe = KokoroPipeline(
    PipelineConfig(
        voice="af_sarah",
        model_source="github",
        model_variant="v1.0",
        model_quality="fp16-gpu",
    )
)
```

### Source-Aware Asset Inspection

Model, config, and voice status checks use the exact `(source, variant, quality)`
configuration. Voice archives use source- and variant-specific names, including
`voices-v1.0.bin`, `voices-v1.1-zh.bin`, and `voices-german-v1.1.bin` for GitHub.

```python
from pykokoro.model_assets import are_models_downloaded, get_model_asset_paths

assets = get_model_asset_paths(
    source="github",
    variant="v1.0",
    quality="fp32",
)
print(assets.model, assets.voices, assets.missing)
print(are_models_downloaded(source="github", variant="v1.0", quality="fp32"))
```

Inspection does not download assets or consult another source, variant, or quality.
Runtime startup performs checksum and structural validation for managed assets.

### Registry and model-cache updates

Managed registry metadata is cached at `~/.cache/pykokoro/registry/models.json`; runtime
artifacts are stored in model and distribution-specific subdirectories below the same
registry cache. Every cached and downloaded artifact is checked against its recorded
size and SHA-256 digest.

When an online load must use the last valid local registry because the remote catalog is
temporarily unavailable, PyKokoro records that fallback and logs a warning. If a newly
downloaded artifact proves that the selected catalog metadata is stale, PyKokoro
bypasses the catalog cache, refreshes the registry once, and retries resolution using
the fresh distribution metadata. Only artifacts that fail validation are replaced.
Integrity verification is never disabled.

Offline mode reads and validates the cached registry and artifacts without network
access. Missing or invalid offline assets fail clearly. Users do not need to delete
`models.json` or an entire model directory after a catalog or model update.

## Configuration

Configuration is stored in a platform-specific directory:

- Linux: `~/.config/pykokoro/config.json`
- macOS: `~/Library/Application Support/pykokoro/config.json`
- Windows: `%APPDATA%\pykokoro\config.json`

```python
from pykokoro.utils import load_config, save_config

# Load config
config = load_config()

# Modify config
config["model_quality"] = "fp16"
config["use_gpu"] = True

# Save config
save_config(config)
```

## Advanced Features

### Custom Phoneme Dictionary

```python
from pykokoro import GenerationConfig, KokoroPipeline, PipelineConfig
from pykokoro.tokenizer import TokenizerConfig

# Create config with custom phoneme dictionary
tokenizer_config = TokenizerConfig(phoneme_dictionary_path="my_pronunciations.json")

pipe = KokoroPipeline(
    PipelineConfig(
        voice="af_sarah",
        generation=GenerationConfig(lang="en-us"),
        tokenizer_config=tokenizer_config,
    )
)
res = pipe.run("Hello")
```

### Explicit Mixed Language Support

Automatic language detection is intentionally not configured in the tokenizer. Use the
document language and explicit SSMD `lang` spans instead:

```python
from pykokoro import GenerationConfig, KokoroPipeline, PipelineConfig

config = PipelineConfig(generation=GenerationConfig(lang="en-us"))
pipe = KokoroPipeline(config)
res = pipe.run("[Ich gehe]{lang=\"de\"} zum Meeting", lang="en-us")
```

### Language-Aware spaCy Model Selection

Use the helper to request highest-available selection, an exact tier, or an exact
package. The transform applies the same request to sentence segmentation and G2P:

```python
from pykokoro import (
    GenerationConfig,
    KokoroPipeline,
    PipelineConfig,
    with_spacy_model,
)

base = PipelineConfig(
    voice="af_sarah",
    generation=GenerationConfig(lang="de"),
)
config = with_spacy_model(size="lg")(base)

# For lang="de", this asks both lower libraries for de_core_news_lg
pipe = KokoroPipeline(config)
res = pipe.run("Guten Tag")

# Or select one exact package:
config = with_spacy_model("de_core_news_sm")(base)
```

You can still force an explicit model package name:

```python
from pykokoro import KokoroPipeline, PipelineConfig
from pykokoro.tokenizer import TokenizerConfig

tokenizer_config = TokenizerConfig(
    spacy_model="fr_core_news_sm",  # explicit package
)
pipe = KokoroPipeline(PipelineConfig(voice="af_sarah", tokenizer_config=tokenizer_config))
```

### Backend Configuration

Control which phonemization backend and dictionaries to use:

```python
from pykokoro import KokoroPipeline, PipelineConfig
from pykokoro.tokenizer import TokenizerConfig

# Default: Full dictionaries with espeak fallback (best quality)
tokenizer_config = TokenizerConfig(
    backend="espeak", load_gold=True, load_silver=True, use_espeak_fallback=True
)

# Memory-optimized: Gold dictionary only
tokenizer_config = TokenizerConfig(
    backend="espeak",
    load_gold=True,
    load_silver=False,  # Saves ~22-31 MB
    use_espeak_fallback=True,
)

# Fastest initialization: Pure espeak
tokenizer_config = TokenizerConfig(
    backend="espeak", load_gold=False, load_silver=False, use_espeak_fallback=True
)

# Alternative backend (requires pygoruut)
tokenizer_config = TokenizerConfig(backend="goruut")

pipe = KokoroPipeline(PipelineConfig(voice="af_sarah", tokenizer_config=tokenizer_config))
res = pipe.run("Hello")
```

**Note**: `use_dictionary` parameter is deprecated. Use `load_gold` and `load_silver`
instead for finer control.

### Named KokoroG2P Lexicons

The native KokoroG2P backend also supports explicit named lexicon selection through
`TokenizerConfig.lexicons`:

```python
from pykokoro.tokenizer import TokenizerConfig

# Compatibility behavior, controlled by KokoroG2P and legacy switches.
default_config = TokenizerConfig()

# German Gold lexicon only.
gold_config = TokenizerConfig(lexicons="gold")

# German Crane lexicon only.
crane_config = TokenizerConfig(lexicons="crane")
```

`lexicons=None` preserves the compatibility behavior. An explicit selection takes
precedence over `load_gold` and `load_silver`, because KokoroG2P owns the named-lexicon
policy. Ordered selections such as `lexicons=("gold", "crane")` are supported for
layered lookup, where the first matching layer wins. That layered lookup is not a
Gold-versus-Crane A/B comparison. For an A/B comparison, render separately with
`("gold",)` and `("crane",)` and combine the results yourself.

The named lexicons are KokoroG2P/G2Lex resources consumed by PyKokoro; they are not
PyKokoro-owned datasets.

**External G2P Libraries**: You can also use external phonemization libraries like
[Misaki](https://github.com/hexgrad/misaki):

```python
from misaki import en, espeak
from pykokoro import KokoroPipeline, PipelineConfig
from pykokoro.generation_config import GenerationConfig

# Misaki G2P with espeak-ng fallback
fallback = espeak.EspeakFallback(british=False)
g2p = en.G2P(trf=False, british=False, fallback=fallback)
phonemes, _ = g2p("Hello, world!")

# Generate audio from phonemes
pipe = KokoroPipeline(
    PipelineConfig(
        voice="af_bella",
        generation=GenerationConfig(is_phonemes=True, lang="en-us"),
    )
)
res = pipe.run(phonemes)
samples = res.audio
```

## SSMD 0.8 portable documents

PyKokoro consumes SSMD 0.8 portable front matter by default. Header metadata is never
spoken: `title` is returned in `AudioResult.document_metadata`, `voice_bindings.kokoro`
maps portable role names to concrete Kokoro voices, and `pause_defaults` controls
implicit sentence, paragraph, and concrete voice-change boundaries. Explicit SSMD breaks
always win over document defaults, and simultaneous defaults use the longest duration.

```python
from dataclasses import replace
from pykokoro import KokoroPipeline, PipelineConfig, SSMDRenderConfig

script = """---
title: Portable review
voice_bindings:
  kokoro:
    host: af_sarah
    guest: af_bella
pause_defaults:
  enabled: true
  sentence: 250ms
  paragraph: 700ms
  voice_change: 350ms
---
<div voice="host">Welcome to the review.</div>

<div voice="guest">The roles remain portable across renderers.</div>
"""
cfg = PipelineConfig(ssmd=SSMDRenderConfig())
result = KokoroPipeline(cfg).run(
    script,
    ssmd=replace(cfg.ssmd, voice_bindings={"kokoro": {"guest": "bf_emma"}}),
)
assert result.document_metadata["title"] == "Portable review"
```

Use `SSMDRenderConfig(parse_header=False)` only when a literal leading `---` block must
remain text. PyKokoro does not read SSMD's user configuration files implicitly. Voice
language, gender, and variant hints are preserved as metadata but do not select voices;
audio annotations require an application-supplied resolver, and unsupported extensions
are rejected for the Kokoro profile.

## Word timings

Timestamp-capable Kokoro ONNX models expose model-derived word timings from named
duration outputs (`pred_dur`, `pred_duration`, or `durations`).
`AudioUnitResult.word_timings` is relative to that unit's waveform, while
`AudioResult.word_timings` is relative to the complete waveform. Each `WordTiming` uses
integer sample offsets into the exact final waveform and clean-text character offsets;
derive seconds with `start_seconds(sample_rate)` and `end_seconds(sample_rate)`. Missing
or incomplete duration output, waveform-only models, and externally replaced audio
produce no fabricated timings. The G2P cache rebuilds schema-incompatible entries after
upgrade, and `release_audio()` preserves timing metadata.

For sentence streaming, see `examples/stream_with_word_timings.py`. Applications can
copy each unit's audio, keep its timing metadata, and highlight
`document.clean_text[word.char_start:word.char_end]` whenever the playback sample cursor
is within `[word.start_sample, word.end_sample)`.

## License

This library is licensed under the Apache License 2.0.

## Credits

- **Kokoro Model**: [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)
- **ONNX Models**:
  [onnx-community/Kokoro-82M-v1.0-ONNX](https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX)
- **Phonemizer**: [kokorog2p](https://github.com/buchwandler/kokorog2p)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Links

- **GitHub**: https://github.com/buchwandler/pykokoro
- **PyPI**: https://pypi.org/project/pykokoro/
- **Documentation**: https://pykokoro.readthedocs.io/
