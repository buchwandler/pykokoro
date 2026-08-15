# Pipeline Usage and Stages

`KokoroPipeline` is the configurable engine behind the high-level `Kokoro` class. Use it
when you want to swap parsing/segmentation stages, run custom G2P logic, or control
model loading at a lower level.

## Pipeline overview

The default pipeline wiring is:

`doc_parser -> g2p -> phoneme_processing -> audio_generation -> audio_postprocessing`

Default stage classes:

- `SsmdDocumentParser`
- `KokoroG2PAdapter`
- `OnnxPhonemeProcessorAdapter`
- `OnnxAudioGenerationAdapter`
- `OnnxAudioPostprocessingAdapter`

If any of the audio stages are omitted, the pipeline builds a `Kokoro` ONNX backend and
wires the missing adapters automatically.

## Quick start

```python
from pykokoro import GenerationConfig, KokoroPipeline, PipelineConfig

config = PipelineConfig(
    voice="af_bella",
    generation=GenerationConfig(speed=1.0),
)
pipeline = KokoroPipeline(config)
result = pipeline.run("Hello from the pipeline.")
result.save_wav("output.wav")

# Inspect intermediates
segments = result.segments
phoneme_segments = result.phoneme_segments

# Enable trace details when needed
traced = pipeline.run("Hello", return_trace=True)
if traced.trace:
    print(traced.trace.warnings)
```

## Configuration

`PipelineConfig` and `GenerationConfig` are frozen dataclasses. Use
`dataclasses.replace` when you want a modified copy.

```python
from dataclasses import replace
from pykokoro import GenerationConfig, KokoroPipeline, PipelineConfig

cfg = PipelineConfig(voice="af_bella")
faster_cfg = replace(cfg, generation=replace(cfg.generation, speed=1.2))
pipeline = KokoroPipeline(faster_cfg)
```

### PipelineConfig fields

#### Core

- `voice`: Default voice name (`str`) or `VoiceBlend` used unless SSMD metadata
  overrides the voice per segment.
- `generation`: `GenerationConfig` instance with speed, language, pause handling, and
  phoneme controls.
- `prosody`: `ProsodyConfig` selecting the AudioSig speech-effects backend. WSOLA is the
  default; `esola` and `td_psola` are experimental, and `psola` aliases `td_psola`.

#### Model and provider

- `model_quality`: `"fp32"`, `"fp16"`, `"fp16-gpu"`, `"q8"`, `"q8f16"`, `"q4"`,
  `"q4f16"`, `"uint8"`, `"uint8f16"`. `None` uses the backend default.
- `model_source`: `"huggingface"` or `"github"`.
- `model_variant`: `"v1.0"` or `"v1.1-zh"`.
- `model_path`: Path to a local ONNX model file. Overrides model download.
- `voices_path`: Path to a local voices file. Overrides voice download.
- `provider`: ONNX provider name (`"auto"`, `"cpu"`, `"cuda"`, `"openvino"`,
  `"directml"`, `"coreml"`).
- `provider_options`: Dict of provider/session options passed to ONNX Runtime.
- `session_options`: Pre-built `onnxruntime.SessionOptions` (advanced use).

#### Tokenizer and phoneme handling

- `tokenizer_config`: `TokenizerConfig` used by SSMD parsing and `kokorog2p`.
- `tokenizer_config.spacy_model`: explicit spaCy package name, or unset. `"auto"` is
  accepted as a compatibility alias for unset.
- `tokenizer_config.spacy_model_size`: exact package tier (`"sm"`, `"md"`, `"lg"`,
  `"trf"`), or unset. With both values unset, each backend selects its highest installed
  compatible model (`trf > lg > md > sm`) without downloading.
- `espeak_config`: Deprecated espeak configuration. Prefer `TokenizerConfig`.
- `short_sentence_config`: `ShortSentenceConfig` for short-sentence handling.
- `overlap_mode`: `"snap"` clips overlapping SSMD spans to segment bounds, `"strict"`
  drops partial spans and emits trace warnings.

Helper for an exact spaCy model request:

```python
from pykokoro import PipelineConfig, with_spacy_model

cfg = PipelineConfig(voice="af_bella")
cfg = with_spacy_model(size="lg")(cfg)
```

#### Other

- `return_trace`: Include `Trace` in `AudioResult` with timings/warnings.
- `enable_deprecation_warnings`: Reserved for compatibility warnings.
- `cache_dir`: Directory for the G2P disk cache (JSON files). Set `None` to disable
  caching.

### GenerationConfig fields

- `speed`: Speech rate multiplier (`1.0` is normal).
- `lang`: Default language code for phonemization (`"en-us"` etc).
- `is_phonemes`: Treat input text as phoneme strings instead of raw text.
- `pause_mode`: `"tts"` keeps natural model pauses, `"manual"` trims segment silence and
  preserves explicit pauses, `"auto"` inserts pauses at sentence/paragraph boundaries
  and trims segment silence.
- `pause_clause`: Default pause for SSMD `...c` breaks (seconds).
- `pause_sentence`: Default pause for SSMD `...s` breaks (seconds).
- `pause_paragraph`: Default pause for SSMD `...p` breaks (seconds).
- `pause_variance`: Stored for compatibility with the `Kokoro` API. The pipeline stages
  do not currently apply variance.
- `random_seed`: Stored for compatibility with the `Kokoro` API. The pipeline stages do
  not currently use the seed.
- `enable_short_sentence`: Override short sentence handling for the run.

## Runtime overrides

`KokoroPipeline.run` accepts overrides for any `PipelineConfig` field. The `lang`
keyword is special-cased to update `generation.lang` for convenience.

```python
from dataclasses import replace
from pykokoro import GenerationConfig

# Override just the language
result = pipeline.run("Bonjour", lang="fr")

# Override generation settings per call
manual = replace(
    pipeline.config.generation,
    pause_mode="manual",
    pause_sentence=0.5,
)
result = pipeline.run("Hello...s world", generation=manual)

# Override model settings per call
result = pipeline.run("Quick test", model_quality="q8")
```

## Stage behavior

### SSMD document parser

`SsmdDocumentParser` uses the SSMD 0.8 public front-matter parser and body-only
segmentation to turn SSMD markup into clean text plus metadata spans, pause boundaries,
and sentence/paragraph segments. Explicit break durations retain their processor
mapping; implicit document defaults are reduced before G2P.

Supported SSMD features include:

- Break markers: `...c`, `...s`, `...p`, `...500ms`
- Language overrides: `[Bonjour]{lang="fr"}`
- IPA phoneme overrides: `[tomato]{ipa="təˈmeɪtoʊ"}`
- Prosody annotations: `[text]{rate="fast" pitch="high" volume="loud"}`
- Inline voice annotations and `<div voice="af_sarah">` directives

The parser attaches SSMD metadata to annotation spans so later stages can select
per-segment language, voices, phonemes, and prosody. Sentence-level `<div>` language,
voice, and prosody directives are inherited by contained segments, while inline
annotations override individual fields.

### Plain text sentence splitting

`PlainTextDocumentParser` uses the optional `phrasplit` package for sentence splitting.
When `phrasplit` is unavailable, it falls back to a single segment. The language model
is derived from `generation.lang` using spaCy package naming rules (for example
`en_core_web_sm` for English).

Split boundaries are forced at SSMD pause boundaries and at spans that contain phoneme
overrides so those overrides are kept intact. Set `PYKOKORO_DEBUG_SEGMENTS=1` to log
segment offsets.

### Kokoro G2P adapter

`KokoroG2PAdapter` uses the `kokorog2p` package to produce phonemes and token IDs.

- `generation.lang` selects the G2P language.
- `generation.is_phonemes` treats input as phonemes and skips text G2P.
- SSMD `ph`/`phonemes` spans override phonemes for that segment.
- `tokenizer_config` is forwarded to `kokorog2p.get_g2p`.
- Unset `spacy_model` and `spacy_model_size` resolve independently per effective
  language; the selected concrete packages are exposed in result metadata.
- `cache_dir` enables on-disk caching of phonemes/tokens.
- Long phoneme token sequences are split into batches of `MAX_PHONEME_LENGTH`.

### Onnx phoneme processing

`OnnxPhonemeProcessorAdapter` calls the ONNX backend to normalize tokens, skip empty
segments, and apply short-sentence handling.

- `short_sentence_config` controls defaults for short sentence handling.
- `generation.enable_short_sentence` can override the config per run.

### Onnx audio generation

`OnnxAudioGenerationAdapter` generates raw audio per phoneme segment.

- `voice` provides the default voice style.
- SSMD voice metadata (`voice`/`voice_name`) overrides the voice per segment.
- `generation.speed` controls synthesis speed.

### Onnx audio postprocessing

`OnnxAudioPostprocessingAdapter` trims silence and concatenates segments.

- `generation.pause_mode"` set to `"manual"` or `"auto"` enables silence trimming before
  inserting explicit pauses.
- SSMD prosody metadata (rate/pitch/volume) is applied to each segment through one
  AudioSig compositor pass. Configured fallbacks are used only in non-strict mode.
- `pause_before`/`pause_after` values from G2P are inserted between segments.

WSOLA is the production default. ESOLA's computed backend rate must be `0.5..2.0`, and
current TD-PSOLA limits are rate `0.75..1.5` and pitch `-6..+6 st`. No backend
guarantees formant preservation; quality depends on the voice and utterance. Because
segments are processed independently, this stage cannot restore sentence-level
coarticulation or pitch continuity.

## Customizing the pipeline

You can replace individual stages or use the provided no-op adapters. The showcase
script demonstrates multiple wiring styles:

`examples/pipeline_stage_showcase.py`

Example with explicit stage wiring:

```python
from pykokoro import GenerationConfig, PipelineConfig, build_pipeline
from pykokoro.stages.audio_generation.noop import NoopAudioGenerationAdapter
from pykokoro.stages.audio_postprocessing.noop import NoopAudioPostprocessingAdapter
from pykokoro.stages.doc_parsers.ssmd import SsmdDocumentParser
from pykokoro.stages.g2p.kokorog2p import KokoroG2PAdapter
from pykokoro.stages.phoneme_processing.noop import NoopPhonemeProcessorAdapter

cfg = PipelineConfig(
    voice="af_heart",
    generation=GenerationConfig(lang="en-us"),
)
pipeline = build_pipeline(
    config=cfg,
    doc_parser=SsmdDocumentParser(),
    g2p=KokoroG2PAdapter(),
    phoneme_processing=NoopPhonemeProcessorAdapter(),
    audio_generation=NoopAudioGenerationAdapter(),
    audio_postprocessing=NoopAudioPostprocessingAdapter(),
)
```

### SSMD 0.8 document controls

`PipelineConfig.ssmd` controls header parsing, provider-scoped API binding overrides,
unknown-header strictness, missing-voice behavior, pause-default overrides, emphasis
policy and gain scaling, and the explicit audio source resolver. Header bindings
override direct logical references, while API bindings override header bindings.
Document pause defaults are reduced before G2P; explicit breaks take precedence and
simultaneous implicit defaults use the maximum duration. `DocumentResult.header`/`body`
and `AudioResult.document_metadata` expose copied metadata.

Emphasis capability policy is evaluated after phoneme processing and before
`audio_generation`. `plain` preserves metadata without modifying audio, `warn` emits one
diagnostic per logical source segment, `error` rejects before inference, and
`approximate` adds deterministic gain metadata (`strong` `+6dB`, `moderate` `+3dB`,
`reduced` `-3dB`) at `emphasis_gain_scale=1.0`. The scale accepts finite values from
`0.0` through `2.0` and changes only automatic gain; semantic emphasis remains intact.
`none` is ordinary speech in every mode. Explicit `volume` metadata is retained with
precedence over approximation, and no automatic rate or pitch metadata is added.

## Local model files and providers

To load local ONNX artifacts, set `model_path` and `voices_path`. You can also select a
specific execution provider.

```python
from pathlib import Path
from pykokoro import KokoroPipeline, PipelineConfig

cfg = PipelineConfig(
    voice="af_bella",
    model_path=Path("/models/kokoro.onnx"),
    voices_path=Path("/models/voices.bin"),
    provider="cuda",
    provider_options={"device_id": 0},
)
pipeline = KokoroPipeline(cfg)
result = pipeline.run("Hello from local files.")
```
