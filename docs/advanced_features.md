# Advanced Features

This guide covers the supported pipeline-first API for controlled generation and
long-form rendering.

## Unit-wise rendering

`prepare_units()` prepares the complete document once, then renders selected paragraph
or sentence units one at a time. This preserves document-global SSMD offsets, voice
bindings, pauses, and marker ownership while bounding live generated waveform memory to
the selected unit:

```python
from pathlib import Path

import soundfile as sf

from pykokoro import KokoroPipeline, PipelineConfig

with KokoroPipeline(PipelineConfig(voice="af_sarah")) as pipeline:
    with pipeline.prepare_units(script, unit="paragraph") as prepared:
        for result in prepared.render(skip_indices={0, 1}):
            try:
                sf.write(
                    Path(f"paragraph-{result.descriptor.index:04d}.wav"),
                    result.audio,
                    result.sample_rate,
                )
            finally:
                result.release_audio()
```

Sentence units provide the low-startup-latency direct-playback path:

```python
with KokoroPipeline(PipelineConfig(voice="af_sarah")) as pipeline:
    pipeline.play_streaming(script, unit="sentence", queue_size=2)
```

Playback starts after the first sentence and uses one persistent bounded output stream.

Descriptors are available before inference and contain source-order indices, clean-text
offsets, segment ownership, marker names, and a `text_hash`. Store the
`pykokoro-audio-unit-v1` schema beside hashes in a resume manifest. Hashes include
audio-semantic settings, not tracing, retention, cache directories, or machine-local
runtime toggles. Set `PipelineConfig(model_identity="model-v1")` to give a local model a
stable resume identity.

`AudioUnitResult.release_audio()` is destructive and idempotent. The iterator releases
the previous result before yielding the next one, so callers must persist or copy its
array inside the loop. Closing prepared units releases prepared segment arrays but does
not close the reusable pipeline backend.

Preparation is global rather than text-streaming: parsing, segmentation, G2P, and
phoneme preprocessing still operate on the full document. The bounded part is unit audio
generation and postprocessing.

## SSMD 0.8 metadata

Portable YAML headers can define logical voices, pause defaults, title metadata, and
markers. Bind logical roles to provider voices through `SSMDRenderConfig` and render
either paragraph or sentence units with the same lifecycle shown above. See
`examples/paragraph_ssmd_voices.py` for a complete script and marker offsets.

## Generation and pauses

```python
from pykokoro import GenerationConfig, KokoroPipeline, PipelineConfig

generation = GenerationConfig(
    lang="en-us",
    speed=1.05,
    pause_mode="manual",
    pause_clause=0.2,
    pause_sentence=0.5,
    pause_paragraph=1.0,
)

with KokoroPipeline(PipelineConfig(voice="af_bella", generation=generation)) as pipeline:
    result = pipeline.run("A short sentence ...s followed by another.")
    result.save_wav("pauses.wav")
    result.release_audio()
```

Automatic pauses use local SSMD/header defaults when present and otherwise the
configured `GenerationConfig` values. Explicit break events take precedence; explicit
zero durations remain zero.

## Voice blending

Voice blending is represented by `VoiceBlend` from the concrete voice-manager module and
passed through `PipelineConfig`:

```python
import soundfile as sf

from pykokoro import KokoroPipeline, PipelineConfig
from pykokoro.voice_manager import VoiceBlend

blend = VoiceBlend.parse("af_bella:50,af_sarah:50")
with KokoroPipeline(PipelineConfig(voice=blend)) as pipeline:
    result = pipeline.run("This is a blended voice.")
    sf.write("blended.wav", result.audio, result.sample_rate)
    result.release_audio()
```

## Composable stages

The pipeline stages follow this order:

`doc_parser -> g2p -> phoneme_processing -> audio_generation -> audio_postprocessing`

Custom stages can be injected into `KokoroPipeline` for tests, experiments, and
dependency-light processing. Importing the pipeline and running fully custom stages does
not require ONNX Runtime. Default audio stages require one of the provider extras.

## spaCy policy

`TokenizerConfig(use_spacy=...)` is tri-state and local-only:

- `False`: never use spaCy;
- `None`: choose the best compatible installed model or fall back;
- `True`: require a compatible local model;
- explicit model or size: require that exact local request.

No path downloads a spaCy model automatically. Selection metadata is retained in the
prepared document metadata for both sentence splitting and G2P.

## Migration from the removed API

The old single-object API is not part of the current package. Use the pipeline
lifecycle:

```python
from pykokoro import KokoroPipeline, PipelineConfig

with KokoroPipeline(PipelineConfig(voice="af_bella")) as pipeline:
    result = pipeline.run(text)
    audio = result.audio.copy()
    sample_rate = result.sample_rate
    result.release_audio()
```

Historical scripts are retained under `examples/legacy/` for reference only and are not
indexed or tested as maintained usage examples.
