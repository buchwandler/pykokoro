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
- **Voice Blending**: Create custom voices by blending multiple voices
- **Registry-backed model discovery**: Inspect runnable models, voices, languages,
  qualities, frontends, layouts, and distribution provenance with `discover_models()`
- **Multiple languages and model families**: Supports standard Kokoro families plus
  dedicated language/model checkpoints exposed by the canonical registry
- **Explicit runtime source selection**: Choose supported GitHub or Hugging Face
  distributions where applicable; registry metadata remains the source of truth
- **Explicit Language Planning**: The document language is required before parsing; SSMD
  `lang` spans provide explicit mixed-language runs
- **Text Normalization**: Spokenform owns generic written-to-spoken preparation,
  including semantic say-as behavior when supported by the SSMD/Spokenform contract
- **Maintainer Benchmarking**: PolyNorm-based phoneme regression tooling for the
  PyKokoro frontend path

## Runtime and audio ownership

PyKokoro is the Kokoro producer layer. OnnxVoice owns model installation, cached
artifacts, providers, and ONNX execution. AudioCompose owns batch timeline composition,
explicit silence, marker/span finalization, resampling, complete-output loudness, and
WAV encoding.

The existing `pipeline.run()` and `AudioResult` APIs remain available. Projects that
need to combine generated audio with other engines can use the explicit job boundary:

```python
job = pipeline.to_audio_job("Hello world.", lang="en-us")
job.save("hello.audiojob.json")
```

`run()` composes that same prepared job once. Streaming remains incremental producer
output and does not apply complete-output loudness normalization.

## v0.9 orchestration contract

PyKokoro v0.9 requires an explicit document language. Set
`GenerationConfig(lang="en-us")` or pass `lang="en-us"` to `run`; the per-call value
takes precedence. A voice or model profile never selects the language. Mixed-language
documents must use explicit SSMD language spans, for example:

```python
config = PipelineConfig(generation=GenerationConfig(lang="en-us"))
result = KokoroPipeline(config).run(
    'Hello [Welt]{lang="de"}.',
    lang="en-us",
 )
```

Integrated requests perform source analysis before Spokenform and fresh prepared-text
analysis afterward. The pipeline reuses loaded local spaCy pipelines, but releases
request documents before returning results. Use `tokenizer_config.use_spacy=False` to
disable NLP, leave it unset for local-only fallback, or set it to `True` for strict
model availability.

## UtterPlan debugging workflow

The existing text API remains the simplest entry point:

```python
from pykokoro import GenerationConfig, KokoroPipeline, PipelineConfig

pipeline = KokoroPipeline(
    PipelineConfig(generation=GenerationConfig(lang="en-us"))
)
result = pipeline.run("Hello world.")
```

For a reproducible planning boundary, build and save an immutable `UtterancePlan`, then
render only that plan:

```python
from utterplan import PlannerConfig, UtterancePlan, UtterancePlanner
from pykokoro import KokoroPipeline

planner = UtterancePlanner(PlannerConfig(language="en-us"))
plan = planner.plan("Hello world.")
plan.save("hello.utterplan.json")

loaded = UtterancePlan.load("hello.utterplan.json")
result = pipeline.run_plan(loaded)
```

`run_plan()` does not reparse SSMD or rerun Spokenform, Phrasplit, or planning-time
linguistic analysis. Inspect the UtterPlan when spoken text, segmentation, pauses,
directives, or units are wrong. Inspect PyKokoro when the plan is correct but
pronunciation, model timing, or audio is wrong. The supplied plan is never mutated
during rendering.

## Runtime model support

Runtime model selection uses the canonical `catalog/models.json` registry. Model
metadata, voices, frontend IDs, runtime layouts, artifact hashes, provider, and
redistribution policy are not inferred from GitHub release names.

| Status                 | Meaning                                                                |
| ---------------------- | ---------------------------------------------------------------------- |
| `ready`                | Registry distribution and local frontend/layout are supported          |
| `experimental`         | Requires explicit experimental frontend enablement                     |
| `restricted`           | Visible metadata but ordinary redistribution/runtime use is restricted |
| `registry-unavailable` | Profile exists locally but registry has no runtime-ready distribution  |
| `unsupported-layout`   | Registry layout is newer or unsupported locally                        |
| `unsupported-frontend` | Registry frontend is not implemented locally                           |

Run `python examples/models_and_languages.py` or call `discover_models()` for the
current inventory and exact status of every model. Thai Wayu uses the registry's
`split-onnx-v1` layout and selects its prosody, curves, and decoder components as one
distribution. Russian Zaakirio uses pinned upstream-only distributions and validates raw
float32 voice artifacts locally; those bytes are not mirrored to GitHub.

Use `python examples/models_and_languages.py` to inspect every registry model, language,
provider, voice, quality, frontend, layout, and support status without downloading model
weights. Pass `--model MODEL_ID` to synthesize only one selected model; experimental
frontends additionally require `--include-experimental`.

### Public model capability discovery

Downstream applications can inspect the canonical runtime inventory without importing
the ONNX backend or downloading model assets:

```python
from pykokoro import discover_models

inventory = discover_models()
for model in inventory.models:
    print(model.model_id, model.languages, model.voices, model.status)
```

`discover_models()` reports runtime capabilities, selected distribution provenance,
frontend and G2P metadata, qualities, and verified named lexicons. `offline=True` uses
only the cached registry and forbids network access. `refresh=True` refreshes registry
metadata only, never model or voice assets; the two options cannot be combined.
`registry_source` and `cache_fallback` describe where the inventory came from.

`discover_models()` describes the capabilities and inventory available to the installed
PyKokoro runtime.

### Public pipeline configuration resolution

For preflight, orchestration, dry-run planning, and configuration inspection, resolve
the automatic model choices without constructing the synthesis runtime or loading model
and voice assets:

```python
from pykokoro import GenerationConfig, PipelineConfig, resolve_pipeline_config

requested = PipelineConfig(generation=GenerationConfig(lang="de"))
resolved = resolve_pipeline_config(requested)

print(resolved.model_variant)
print(resolved.model_source)
print(resolved.model_quality)
print(resolved.voice)
```

`resolve_pipeline_config()` is a metadata-only operation. It applies PyKokoro's built-in
model, source, quality, and voice policy without constructing `KokoroPipeline`,
importing ONNX Runtime, creating an ONNX session, loading assets, or synthesizing audio.
It does not replace `discover_models()`, which provides runtime capability and model
inventory metadata.

## Logging and runtime diagnostics

PyKokoro uses standard Python logging for streaming operational visibility. It does not
configure global logging, install handlers or formatters, set logger levels, or add
timestamps. The embedding application owns logging presentation and decides whether
lifecycle records are shown.

For local diagnostics, configure logging before constructing a pipeline:

```python
import logging

logging.basicConfig(level=logging.DEBUG)

from pykokoro import KokoroPipeline
```

Production applications should configure the `pykokoro` logger through their existing
logging setup instead of relying on `basicConfig`:

```python
import logging

logging.getLogger("pykokoro").setLevel(logging.INFO)
```

INFO records cover major milestones such as model and distribution selection, artifact
downloads, ONNX session creation, backend readiness, and voice loading. DEBUG records
add stage timings, cache decisions, inference counts and runtimes, and audio-unit
completion. Records contain no application-provided timestamps, so the host formatter
can apply its own timestamp format.

Streaming logs and `PipelineConfig(return_trace=True)` serve different purposes. Logging
reports lifecycle events while synthesis is running. `return_trace=True` attaches
structured diagnostics to returned audio results for programmatic inspection. Enabling
one does not enable or print the other. Routine lifecycle logs avoid complete user
documents, phoneme streams, audio arrays, credentials, and model contents.

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
python examples/cpu_benchmark.py
```

## Quick Start

The pipeline API is the only supported interface.

```python
from pykokoro import GenerationConfig, KokoroPipeline, PipelineConfig

pipe = KokoroPipeline(PipelineConfig(generation=GenerationConfig(lang="en-us"), voice="af_sarah"))
res = pipe.run("Hello")
audio = res.audio
```

### First-run model downloads

Managed runtime assets are provisioned lazily when synthesis first needs them. On a cold
cache, user-facing examples can show the asset name, expected size, byte progress, and
checksum verification with the built-in reporter:

```python
from pykokoro import ConsoleAssetProgress, GenerationConfig, KokoroPipeline, PipelineConfig

config = PipelineConfig(
    voice="af_sarah",
    generation=GenerationConfig(lang="en-us"),
    asset_progress=ConsoleAssetProgress(),
)
pipe = KokoroPipeline(config)
result = pipe.run("Hello")
```

Applications can receive structured `AssetProgressEvent` values instead by passing a
callable as `asset_progress`. Valid managed assets are reused silently from the local
cache, and offline mode never starts a download.

### Managing Result Memory

By default, `AudioResult` retains the raw and processed waveform for each phoneme
segment for diagnostics and callers that inspect segment audio. For long documents,
enable compact result retention when only the final waveform and metadata are needed:

```python
from pykokoro import GenerationConfig, PipelineConfig, build_pipeline

pipeline = build_pipeline(
    config=PipelineConfig(
        generation=GenerationConfig(lang="en-us"),
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
from pykokoro import GenerationConfig, KokoroPipeline, PipelineConfig

pipeline = KokoroPipeline(PipelineConfig(generation=GenerationConfig(lang="en-us"), voice="af_sarah"))
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

`UtterancePlanner -> UtterancePlan adapter -> g2p -> phoneme_processing -> audio_generation -> audio_postprocessing`

UtterPlan owns SSMD and plain-text parsing, Spokenform preparation, phrase segmentation,
pauses, markers, and directives. PyKokoro owns rendering after the plan is adapted.
No-op adapters can still replace downstream renderer stages:

```python
from pykokoro import GenerationConfig, KokoroPipeline, PipelineConfig
from pykokoro.stages.audio_generation.noop import NoopAudioGenerationAdapter
from pykokoro.stages.audio_postprocessing.noop import NoopAudioPostprocessingAdapter

pipe = KokoroPipeline(
    PipelineConfig(generation=GenerationConfig(lang="en-us"), voice="af_sarah"),
    audio_generation=NoopAudioGenerationAdapter(seconds_per_segment=0.0),
    audio_postprocessing=NoopAudioPostprocessingAdapter(),
)
res = pipe.run("First paragraph.\n\nSecond paragraph.")
```

For an explicitly configured plan, create an `UtterancePlanner`, call
`planner.plan(text)`, and pass the result to `pipe.run_plan(plan)`. The regular
`run(text)` path derives the effective planner configuration from `PipelineConfig`;
`run_plan()` is the handoff for plans compiled elsewhere.

### Migration

Old (removed):

```python
# Legacy Kokoro-based API has been removed in favor of the pipeline.
```

New:

```python
from pykokoro import GenerationConfig, KokoroPipeline, PipelineConfig

pipe = KokoroPipeline(PipelineConfig(generation=GenerationConfig(lang="en-us"), voice="af_sarah"))
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
from pykokoro import GenerationConfig, KokoroPipeline, PipelineConfig

pipe = KokoroPipeline(PipelineConfig(generation=GenerationConfig(lang="en-us"), provider="auto", voice="af_sarah"))
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
python examples/cpu_benchmark.py
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

### Short-sentence benchmark tools

PyKokoro has separate tools for structural diagnosis and listening-based parameter
comparison:

- `benchmarks/short_sentence_latency.py` runs the warm policy/scenario benchmark and
  reports final cases, actual cut strategies, failures, retries, and inference costs. It
  defaults to a human-readable table, supports `--output-format table|jsonl|both`, JSONL
  and summary JSON export, and `--dry-run`. It does not write WAV artifacts.
- `benchmarks/short_sentence_parameter_sweep.py` renders one fixed short segment for an
  ordered parameter sweep. It writes one combined labeled WAV and a JSON manifest.
  Labels use short-sentence handling disabled, phrase retries default to zero, and the
  manifest records reproducibility metadata and audio offsets.

Use the parameter sweep for human listening, not automatic optimization. See
[`docs/short_sentence_quality.md`](docs/short_sentence_quality.md) for CLI examples,
supported parameters, starting ranges, and artifact details.

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
from pykokoro import GenerationConfig, KokoroPipeline, PipelineConfig
from pykokoro.voice_manager import VoiceBlend

blend = VoiceBlend.parse("af_nicole:50,am_michael:50")
pipe = KokoroPipeline(PipelineConfig(generation=GenerationConfig(lang="en-us"), voice=blend))
res = pipe.run("Mixed voice")
audio = res.audio
```

### Direct Playback of Generated Chunks

For independent chunks, `AudioResult.play()` plays each generated waveform directly:

```python
from pykokoro import GenerationConfig, KokoroPipeline, PipelineConfig

pipe = KokoroPipeline(PipelineConfig(generation=GenerationConfig(lang="en-us"), voice="af_sarah"))
chunks = ["Long text", "here..."]
for text_chunk in chunks:
    result = pipe.run(text_chunk)
    result.play()
```

For long text with low startup latency, prefer sentence streaming through one persistent
bounded output stream:

```python
from pykokoro import GenerationConfig, KokoroPipeline, PipelineConfig

text = "First sentence. Second sentence. Third sentence."
with KokoroPipeline(PipelineConfig(generation=GenerationConfig(lang="en-us"), voice="af_sarah")) as pipe:
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
from pykokoro import GenerationConfig, KokoroPipeline, PipelineConfig

pipe = KokoroPipeline(PipelineConfig(generation=GenerationConfig(lang="en-us"), voice="am_michael"))
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
from pykokoro import GenerationConfig, KokoroPipeline, PipelineConfig

config = PipelineConfig(
    voice="am_michael",
    generation=GenerationConfig(
        lang="en-us",
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

# Automatic pauses at sentence/paragraph boundaries, parenthetical asides, and high-confidence clausal commas
from pykokoro import GenerationConfig, KokoroPipeline, PipelineConfig

config = PipelineConfig(
    voice="af_sarah",
    generation=GenerationConfig(
        lang="en-us",
        pause_mode="auto",
        pause_clause=0.25,  # Pause after high-confidence clausal commas
        pause_parenthetical=0.15,  # Short pause around parenthetical asides
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

- **Natural boundaries**: Automatically pauses at sentences, paragraphs, high-confidence
  clausal commas, and parenthetical asides
- **Variance**: Gaussian variance prevents robotic timing (±100ms by default)
- **Reproducible**: Use `random_seed` for consistent output
- **Composable**: Works with SSMD break markers

**Splitting Behavior:**

- `SsmdDocumentParser` handles paragraph/sentence segmentation using SSMD.
- `PlainTextDocumentParser` uses optional `phrasplit` sentence splitting.

- In `pause_mode="auto"`, dependency-aware Phrasplit analysis detects high-confidence
  clausal commas and parenthetical asides. List commas and shared-subject continuations
  remain untouched.

For example,
`It had picked up the sound of a explosion, direction suggested it was behind.` is
refined at the detected comma so the preceding segment receives one deterministic
`pause_clause`. For example, `They changed out their clothes (stained with blood).`
receives a short `pause_parenthetical` before the aside in auto mode. Set
`pause_parenthetical=0.0` to disable only these inferred parenthetical pauses. The
setting is independent from `pause_clause`, `pause_sentence`, and `pause_paragraph`.

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
    generation=GenerationConfig(lang="en-us", pause_mode="auto", pause_variance=0.05),
)
pipe = KokoroPipeline(config)
res = pipe.run(text)
audio = res.audio
```

See `examples/pauses_demo.py`, `examples/automatic_pauses_demo.py`,
`examples/english_clausal_comma_pause.py`, and `examples/english_parenthetical_pause.py`
for examples.

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
from pykokoro import GenerationConfig, KokoroPipeline, PipelineConfig, SSMDRenderConfig

config = PipelineConfig(generation=GenerationConfig(lang="en-us"), ssmd=SSMDRenderConfig(emphasis_mode="approximate"))
result = KokoroPipeline(config).run("This is *moderate emphasis*.")
```

### Audio level and loudness

Kokoro voices are not guaranteed to have identical perceived loudness. PyKokoro keeps
three separate mechanisms:

#### Raw model level

Different voice embeddings and checkpoints naturally produce different output levels.
This is the unmodified audio level from the selected model and voice.

#### Voice leveling

Use a reviewed, static per-voice correction when switching voices should be less
jarring:

```python
from pykokoro import KokoroPipeline, LoudnessConfig, PipelineConfig

pipeline = KokoroPipeline(
    PipelineConfig(
        voice="af_bella",
        loudness=LoudnessConfig(voice_leveling="calibrated"),
    )
)
```

`voice_leveling="calibrated"` applies one fixed gain for the model source, model,
quality, and voice. It preserves intentional relative `volume` and emphasis effects. It
is an offline equalization aid, not a final delivery loudness or peak guarantee for
arbitrary content.

The packaged catalog contains 216 reviewed fp32 voice identities from the
`pykokoro-count-1-to-10-v2` corpus at a reference of `-24 LUFS`. A calibration record
requires the exact model source, model ID, model quality, and voice identity. Missing
records are a safe no-op, so Swedish voices without measurements, non-fp32 qualities,
custom identities, and voice blends are not silently substituted or aliased. Calibrated
leveling remains opt-in and does not claim that every available voice is measured.

The calibration catalog is generated offline from the versioned Spokenform count-to-ten
stimulus. The benchmark verbalizes 1 through 10 independently, repeats the stimulus,
measures integrated loudness and true peak, and produces a candidate for explicit
review. It does not download model assets or overwrite the packaged catalog.

#### Complete-output normalization

To normalize the complete returned waveform, configure a target explicitly:

```python
LoudnessConfig(
    voice_leveling="calibrated",
    target_lufs=-24.0,
    true_peak_ceiling_dbtp=-1.0,
)
```

This measures the completed waveform and applies one final gain subject to the true-peak
policy. It is deliberately separate from static voice leveling and is not available for
true streaming because streaming does not have the complete waveform.

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
    PipelineConfig(voice="af_sarah", generation=GenerationConfig(lang="en-us"))
)
res = pipe.run(text)
```

## Explicit mixed-language text

Automatic routing is optional and remains separate from the explicit document language.
Set the document language explicitly and mark semantic or pronunciation-only language
spans with SSMD:

```python
from pykokoro import (
    GenerationConfig,
    LanguageDetectionConfig,
    KokoroPipeline,
    PipelineConfig,
)

config = PipelineConfig(
    generation=GenerationConfig(lang="de"),
    language_detection=LanguageDetectionConfig(mode="auto", languages=("de", "en")),
)
pipe = KokoroPipeline(config)
text = 'Die [File]{lang="en" scope="pronunciation"} wird gecancelt.'
result = pipe.run(text)
```

`scope="pronunciation"` changes only G2P. Automatic KokoroG2P routing also changes only
pronunciation fragments. The selected PyKokoro acoustic model, voice, ONNX provider, and
document language remain German.

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
Spokenform prepares the written text, Phrasplit detects sentences and high-confidence
clausal commas in the prepared text, structural refinement applies the resulting
boundaries, and kokorog2p receives that prepared text through `phonemize_prepared()`.
Segment offsets therefore refer to the prepared spoken `clean_text`; structural
annotations, events, and preparation provenance remain available in the document
metadata used by downstream stages. Use `examples/german3.py` for a German regression
containing dates, quantities, abbreviations, ordinals, and currency.

### Explicit SSMD Say-As Overrides

Use SSMD (Speech Synthesis Markdown) say-as annotations when the author needs explicit
interpretation or an override. They are not required for common automatic forms:

```python
from pykokoro import GenerationConfig, KokoroPipeline, PipelineConfig
from pykokoro.generation_config import GenerationConfig

pipe = KokoroPipeline(PipelineConfig(generation=GenerationConfig(lang="en-us"), voice="af_sarah"))

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
   (default resolve mode: `randomized-phrase` when the loaded model exposes duration
   timestamps)
3. TTS generates audio from the wrapped phoneme sequence.
4. Cut away the extra context and put audio together.

This happens automatically during `pipe.run()` - no configuration needed! Phrase-based
modes require a model duration/timestamp output. When no explicit short-sentence
configuration is supplied, PyKokoro automatically uses `wrap` for models without that
output. If a phrase mode is explicitly requested for such a model, PyKokoro logs a
warning and falls back to `wrap`. Phrase-based short-sentence handling uses
language-localized carrier catalogs for English, German, Spanish, French, Italian,
Portuguese, European Portuguese, Korean, Japanese, Chinese/Mandarin, Arabic, Hebrew,
Kazakh, Swedish, Thai, Vietnamese, Russian, Hindi, Polish, and Turkish. Unknown
languages never fall back to English carrier phrases. They use `wrap` unless a
user-provided `ShortSentencePhraseSet` is supplied.

NOTE: Carrier quality remains voice and model dependent. Benchmark localized phrases
before relying on them in production.

**Customizing the Behavior:**

You can customize the behavior using `ShortSentenceConfig`:

```python
from pykokoro import GenerationConfig, KokoroPipeline, PipelineConfig
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

pipe = KokoroPipeline(PipelineConfig(generation=GenerationConfig(lang="en-us"), voice="af_sarah", short_sentence_config=short_sentence_config))
res = pipe.run("Why?")
```

**Default Configuration:**

- `enabled=True`: Short-sentence handling is enabled by default
- `min_phoneme_length=30`: Segments below this token count engage short-sentence
  handling
- `resolve_mode="randomized-phrase"` Chooses between `randomized-phrase` (default),
  `phrase`, or `wrap` (fallback). Phrase-based defaults require model timestamps;
  no-timestamp models automatically use `wrap`.
- `phrase_selection="auto"` Chooses which phrase templates to use. `auto` uses "end" if
  the phrase ends with '.', otherwise uses "neutral"
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
from pykokoro import GenerationConfig, KokoroPipeline, PipelineConfig
from pykokoro.short_sentence_handler import ShortSentenceConfig

short_sentence_config = ShortSentenceConfig(enabled=False)
pipe = KokoroPipeline(PipelineConfig(generation=GenerationConfig(lang="en-us"), voice="af_sarah", short_sentence_config=short_sentence_config))
res = pipe.run("Why?")
```

See `examples/optimal_phoneme_length_demo.py` for a demonstration.

**Advanced customization of short-sentence handling**

You can add custom template phrases used to add context in phrase mode, but THIS IS NOT
RECOMMENDED for most users! You can supply a `ShortSentencePhraseSet` when you need
custom carrier wording or language coverage beyond the built-in catalogs.

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

## Discovering Models, Voices, and Lexicons

The canonical registry is the authoritative inventory for current runtime capabilities.
Use `discover_models()` instead of maintaining a hard-coded list of model-specific
voices:

```python
from pykokoro import discover_models

inventory = discover_models()
for model in inventory.models:
    print(
        model.model_id,
        model.status,
        model.languages,
        model.default_voice,
        len(model.voices),
    )
    for voice in model.voice_details:
        print(" ", voice.name, voice.language_label, voice.gender)
```

The inventory includes model IDs, statuses, languages, voices, qualities, frontends,
layouts, distribution provenance, and voice metadata when available. Use
`python examples/models_and_languages.py` or `python examples/all_voices.py --help` for
ready-made inventory and showcase commands. The v1.0 and v1.1-zh profiles below are
illustrative model families, not the complete PyKokoro catalog.

### Discover named lexicons

```python
from pykokoro import discover_lexicons

result = discover_lexicons(language="de")
for item in result.lexicons:
    print(item.selector, item.locale, item.installed, item.phoneme_encoding)
```

`discover_lexicons()` is metadata-only. It does not install model or lexicon assets. Use
`model_variant=...` to narrow the result when the model advertises a known named lexicon
capability.

### Runtime capabilities versus published releases

- `discover_models()` answers what this PyKokoro runtime can use, including model IDs,
  distributions, providers, qualities, voices, and runtime availability.
- Lower-level published model catalog inspection belongs to OnnxVoice, for example
  `OnnxVoice().list("kokoro")`.
- Use `ModelCapabilities` and `ModelDiscoveryResult` for the structured PyKokoro
  discovery result.

For example, inspect the installed capability inventory without network access:

```python
from pykokoro import discover_models

result = discover_models(offline=True)
for model in result.models:
    print(model.model_id, model.distribution_id, model.provider, model.qualities)
```

### v1.0 and v1.1-zh examples

The classic v1.0 profile provides a multilingual Kokoro voice inventory. The v1.1-zh
checkpoint has its own voice inventory: three English voices (`af_maple`, `af_sol`, and
`bf_vale`) plus its numbered Chinese voice set. It is not a superset of the v1.0
archive. Enumerate the exact current inventories with `discover_models()`.

```python
from pykokoro import GenerationConfig, KokoroPipeline, PipelineConfig

config = PipelineConfig(
    voice="af_maple",
    model_source="github",
    model_variant="v1.1-zh",
    generation=GenerationConfig(lang="en-us"),
)
with KokoroPipeline(config) as pipe:
    result = pipe.run("Hello world!")
```

To select the standard v1.0 profile explicitly, set `model_variant="v1.0"` and choose a
voice and quality advertised by `discover_models()`.

### Model discovery and selection

PyKokoro resolves model capabilities from the canonical registry. For the standard
Kokoro profiles, explicit `model_source="github"` and `model_source="huggingface"`
remain available where supported. The built-in default source is GitHub.

Use discovery before selecting a model, voice, or quality:

```python
from pykokoro import discover_models

for model in discover_models().models:
    print(model.model_id, model.qualities)
```

### Explicit model source selection

Hugging Face is an available alternative when the selected registry distribution
supports it:

```python
from pykokoro import GenerationConfig, PipelineConfig

config = PipelineConfig(
    generation=GenerationConfig(lang="en-us"),
    model_source="huggingface",
    model_variant="v1.0",
    voice="af_sarah",
)
```

The GitHub v1.0 distribution can be selected explicitly, including on Termux or Android:

```python
config = PipelineConfig(
    generation=GenerationConfig(lang="en-us"),
    model_source="github",
    model_variant="v1.0",
    voice="af_heart",
)
```

PyKokoro never silently switches between model sources. Explicit `model_path` and
`voices_path` files are validated in place and are never replaced with managed cache
files. A Termux/Android ONNX Runtime warning is a separate provider issue.

### Choosing model quality

Qualities belong to the selected registry distribution and model. Not every model
provides every quality. `model_quality=None` lets PyKokoro resolve the model/profile
default; an explicit unsupported quality fails clearly. Select only a quality advertised
by `discover_models()`, for example:

```python
config = PipelineConfig(
    generation=GenerationConfig(lang="en-us"),
    model_variant="v1.0",
    voice="af_sarah",
    model_quality="q8",
    model_source="github",
 )
```

### Chinese text

For Mandarin text, select a Chinese-capable model such as `v1.1-zh`, use
`GenerationConfig(lang="zh")`, and choose a voice from that model's registry inventory.

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
config. `martin` selects the corresponding model profile, but it does not replace the
required document language. Use `GenerationConfig(lang="de")` (or an appropriate German
locale) explicitly. Custom voice archives may expose additional voice names when
selected explicitly. The profile's suggested speed of `1.125` is advisory, so
applications must set it explicitly when they want it. Language-specific automatic
spoken-form normalization for German text belongs to the compatible kokorog2p release.
PyKokoro keeps source offsets and segments tied to the original input text, consumes
kokorog2p's prepared G2P result, and owns synthesis.

### Model Quality Options

Quality options are selected from the registry distribution for the chosen model.
Discover the current options instead of relying on a manually maintained source/model
matrix:

```python
from pykokoro import discover_models

for model in discover_models().models:
    print(model.model_id, model.qualities)
```

Qualities belong to the selected model and distribution. Not every model provides every
quality. `model_quality=None` allows PyKokoro to resolve the model/profile default,
while an explicit unsupported quality fails clearly. For a model that advertises `q8`,
select it with:

```python
from pykokoro import GenerationConfig, KokoroPipeline, PipelineConfig

pipe = KokoroPipeline(
    PipelineConfig(
        generation=GenerationConfig(lang="en-us"),
        model_variant="v1.0",
        model_source="github",
        model_quality="q8",
        voice="af_sarah",
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

## Pipeline configuration

Configure synthesis through the immutable `PipelineConfig` passed to the pipeline. Set
the document language explicitly and select a provider through `provider` when needed:

```python
from pykokoro import GenerationConfig, PipelineConfig

config = PipelineConfig(
    generation=GenerationConfig(lang="en-us"),
    voice="af_sarah",
    provider="auto",
    model_quality="fp16",
)
```

`provider="auto"` selects the best available execution provider according to the
documented priority. Use `resolve_pipeline_config()` to inspect effective model settings
before constructing a backend. The older `load_config()` and `save_config()` utilities
remain available as legacy backend configuration helpers, but they are not the primary
pipeline configuration interface.

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
from pykokoro import GenerationConfig, KokoroPipeline, PipelineConfig
from pykokoro.tokenizer import TokenizerConfig

# Native KokoroG2P with explicit Gold lexicon and eSpeak provider fallback.
tokenizer_config = TokenizerConfig(
    backend="kokorog2p",
    lexicons=("gold",),
    fallback="espeak",
)

# Native KokoroG2P with no static Lexphon layer and no provider fallback.
tokenizer_config = TokenizerConfig(
    backend="kokorog2p",
    lexicons=(),
    fallback="none",
)

# Primary eSpeak backend. This is not lexicon-first fallback behavior.
tokenizer_config = TokenizerConfig(backend="espeak")

# Primary Goruut backend (requires the Goruut extra/runtime).
tokenizer_config = TokenizerConfig(backend="goruut")
pipe = KokoroPipeline(PipelineConfig(generation=GenerationConfig(lang="en-us"), voice="af_sarah", tokenizer_config=tokenizer_config))
res = pipe.run("Hello")
```

`backend="kokorog2p"` selects the native lexicon-first stack. Its `fallback` chooses the
optional Lexphon provider after selected lexicons miss: `none`, `espeak`, or `goruut`.
`backend="espeak"` and `backend="goruut"` select those engines as the primary backend.

**Note**: `use_dictionary`, `load_gold`, and `load_silver` are legacy compatibility
inputs. New code should use `TokenizerConfig.lexicons`. Explicit named selections take
precedence; the only legacy combination without a faithful current mapping is Gold
disabled with Silver enabled, which raises an actionable error.

### Named KokoroG2P Lexicons

The native KokoroG2P backend also supports explicit named lexicon selection through
`TokenizerConfig.lexicons`:

```python
from pykokoro.tokenizer import TokenizerConfig

# Compatibility behavior. PyKokoro selects de-de:espeak implicitly for German; other languages use their KokoroG2P defaults.
default_config = TokenizerConfig()
# German Gold lexicon only.
gold_config = TokenizerConfig(lexicons="gold")

# German Crane lexicon only.
crane_config = TokenizerConfig(lexicons="crane")
```

`lexicons=None` uses PyKokoro's language defaults. For German, the implicit selection is
the static de-de:espeak lexicon. An explicit selection takes precedence over legacy
dictionary flags. `lexicons=("gold", "crane")` are supported for layered lookup, where
the first matching layer wins. That layered lookup is not a Gold-versus-Crane A/B
comparison. For an A/B comparison, render separately with `("gold",)` and `("crane",)`
and combine the results yourself.

Provider-only operation is explicit with `lexicons=()`; it selects no static Lexphon
layers and can use `fallback="espeak"` or `fallback="goruut"`. A static lexicon named
`espeak` is still a lexical resource and does not mean the dynamic eSpeak provider. The
named lexicons are KokoroG2P/G2Lex resources consumed by PyKokoro; they are not
PyKokoro-owned datasets.

#### Automatic Lexphon data provisioning

Before native KokoroG2P construction, PyKokoro resolves the effective named lexicons for
the routed language and checks the local Lexphon store. In `auto` mode only missing
Lexphon-backed assets are installed. Warm runs require no catalog access or network
access. Provisioning applies only to the native `backend="kokorog2p"` path; primary
eSpeak and Goruut backends do not download static lexicons.

Use `"installed-only"` for offline or pre-provisioned deployments. In that mode PyKokoro
never installs or consults the catalog. A missing asset raises Lexphon's original
installation error. Catalog, download, integrity, alphabet, and other G2P errors are
propagated unchanged.

```python
from pykokoro.tokenizer import TokenizerConfig

automatic = TokenizerConfig(lexicons=("gold",))
offline = TokenizerConfig(
    lexicons=("gold",),
    lexicon_data_policy="installed-only",
)
```

For explicit provisioning, install the selected assets before running PyKokoro:

```bash
lexphon data available de-DE
lexphon data install de-de:gold
lexphon data verify de-de:gold
```

Set `LEXPHON_DATA_HOME` to select the persistent data store. Set `LEXPHON_CATALOG_URL`
to use a pinned local or remote catalog during provisioning. A pre-populated data store
can be copied into an offline runtime; no catalog is needed on warm paths.

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
from pykokoro import GenerationConfig, KokoroPipeline, PipelineConfig, SSMDRenderConfig

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
cfg = PipelineConfig(generation=GenerationConfig(lang="en-us"), ssmd=SSMDRenderConfig())
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
