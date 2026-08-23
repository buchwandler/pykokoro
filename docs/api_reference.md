# API Reference

This page provides API documentation for the supported pipeline-first interface.

## Main Classes

### KokoroPipeline

```{eval-rst}
.. autoclass:: pykokoro.KokoroPipeline
   :members:
   :undoc-members:
   :show-inheritance:
```

**Basic Example:**

```python
from pykokoro import KokoroPipeline, PipelineConfig

pipe = KokoroPipeline(PipelineConfig(voice="af_bella"))
result = pipe.run("Hello, world!")
print(result.sample_rate)
```

### Unit streaming

```{eval-rst}
.. autoclass:: pykokoro.PreparedAudioUnits
   :members:
   :undoc-members:
```

```{eval-rst}
.. autoclass:: pykokoro.AudioUnitDescriptor
   :members:
   :undoc-members:
```

```{eval-rst}
.. autoclass:: pykokoro.AudioUnitResult
   :members:
   :undoc-members:
```
```{eval-rst}
.. autoclass:: pykokoro.WordTiming
   :members:
   :undoc-members:
```

`AudioUnitResult.word_timings` uses sample offsets relative to that unit's `audio`; `AudioResult.word_timings` uses offsets relative to the complete result waveform. `char_start` and `char_end` refer to clean-text positions. Timing lists are preserved by `release_audio()`.
`AudioUnitKind` supports `"paragraph"` and `"sentence"`; paragraph remains the default
for `prepare_units()` and `iter_units()`. `unit_kind` and `sentence_idx` identify the
selected grouping on each descriptor. Preparation parses, phonemizes, and preprocesses
the complete document once, then defers audio generation until a selected unit is
rendered. Generated waveform memory is bounded by the selected unit size and the
playback queue, while document metadata remains global.

```python
with pipe.prepare_units(script, unit="sentence") as prepared:
    for result in prepared.render(skip_indices=completed_indices):
        try:
            consume(result.audio, result.sample_rate)
        finally:
            result.release_audio()
```

### Streaming playback

`KokoroPipeline.play_streaming(text, unit="sentence", device=None, queue_size=2)` uses
one persistent `SoundDevicePlayer`. It begins playback after the first rendered unit,
queues copied waveforms with bounded backpressure, and blocks until the final queued
unit has finished. It creates no temporary WAV and does not concatenate a final
waveform. The queue capacity is pending waveform capacity in addition to the actively
written waveform, not an exact startup prebuffer count. Empty input opens no device.

`play_prepared_units()` is the lower-level helper for callers that already own a
prepared unit lifecycle. `SoundDevicePlayer` is also available for custom consumers; its
`submit()` copy makes releasing a source `AudioUnitResult` after submission safe.
Synthesis remains sequential on the caller thread; playback is the only worker-thread
operation.

`AudioResult.play()` and `AudioUnitResult.play()` remain blocking helpers for an already
generated single waveform.

### PipelineConfig

```{eval-rst}
.. autoclass:: pykokoro.PipelineConfig
   :members:
   :undoc-members:
   :show-inheritance:
```

### GenerationConfig

```{eval-rst}
.. autoclass:: pykokoro.GenerationConfig
   :members:
   :undoc-members:
   :show-inheritance:
```

### ProsodyConfig

```{eval-rst}
.. autoclass:: pykokoro.ProsodyConfig
   :members:
   :undoc-members:
   :show-inheritance:
```

`ProsodyConfig(method="wsola")` selects the production speech-oriented default. The
supported methods are `wsola`, experimental `esola` and `td_psola`, compatibility
`phase_vocoder`, and the `psola` alias for `td_psola`. Strict mode prevents fallback;
non-strict mode follows `fallback_methods`. No method guarantees formant preservation.

## Pipeline Helpers

```{eval-rst}
.. autofunction:: pykokoro.build_pipeline
```

```{eval-rst}
.. autofunction:: pykokoro.with_spacy_model
```

```{eval-rst}
.. autofunction:: pykokoro.with_spacy_model_size
```

## Result and Data Classes

### AudioResult

```{eval-rst}
.. autoclass:: pykokoro.types.AudioResult
   :members:
   :undoc-members:
   :show-inheritance:
```

`AudioResult` owns references to its final waveform and any raw or processed per-segment
waveforms. `AudioResult.release_segment_audio()` destructively and idempotently releases
only segment arrays, while `AudioResult.release_audio()` also replaces the final
waveform with an empty array of the same dtype. Metadata, markers, trace data, segments,
and sample rate remain available. Callers should copy or retain `result.audio`
separately before releasing it if they need that array afterward.

`AudioResult.play()` and `AudioUnitResult.play()` provide optional direct system
playback. They import `sounddevice` lazily, block until playback completes, accept an
optional `device=` selector, and never create a temporary file. Install playback support
with `pip install "pykokoro[playback]"`; Linux-like systems may also require PortAudio.
Calling either method after `release_audio()` raises a clear empty/released-audio error.

Set `PipelineConfig(retain_segment_audio=False)` for compact results when segment
waveforms are not needed. This reduces retained memory after generation, but
whole-result concatenation still occurs and peak memory remains dependent on input
duration. `run()` retains whole-result concatenation semantics; use `prepare_units()` or
`iter_units()` when generated waveform memory should remain bounded to the selected unit
size plus the bounded playback queue.

Prepared unit descriptor hashes use the `pykokoro-audio-unit-v1` schema. Store the
schema alongside each hash for resumable exporters. Indices are zero-based source order,
hashes include audio-semantic configuration, and advancing a render iterator releases
the previous result's waveform unless the caller copied or persisted it first.

### Segment

```{eval-rst}
.. autoclass:: pykokoro.types.Segment
   :members:
   :undoc-members:
   :show-inheritance:
```

### PhonemeSegment

```{eval-rst}
.. autoclass:: pykokoro.types.PhonemeSegment
   :members:
   :undoc-members:
   :show-inheritance:
```

### Trace

```{eval-rst}
.. autoclass:: pykokoro.types.Trace
   :members:
   :undoc-members:
   :show-inheritance:
```

## Voice Blending

### VoiceBlend

```{eval-rst}
.. autoclass:: pykokoro.onnx_backend.VoiceBlend
   :members:
   :undoc-members:
   :show-inheritance:
```

```python
from pykokoro import KokoroPipeline, PipelineConfig
from pykokoro.onnx_backend import VoiceBlend

blend = VoiceBlend.parse("af_bella:60,af_sarah:40")
pipe = KokoroPipeline(PipelineConfig(voice=blend))
result = pipe.run("Blended voice example")
```

## Tokenizer

### Tokenizer

```{eval-rst}
.. autoclass:: pykokoro.tokenizer.Tokenizer
   :members:
   :undoc-members:
   :show-inheritance:
```

### TokenizerConfig

```{eval-rst}
.. autoclass:: pykokoro.tokenizer.TokenizerConfig
   :members:
   :undoc-members:
   :show-inheritance:
```

### PhonemeResult

```{eval-rst}
.. autoclass:: pykokoro.tokenizer.PhonemeResult
   :members:
   :undoc-members:
   :show-inheritance:
```

**Tokenizer Example:**

```python
from pykokoro.tokenizer import Tokenizer

tokenizer = Tokenizer()
phonemes = tokenizer.phonemize("Hello", lang="en-us")
print(phonemes)
```

## Model and Voice Utilities

These utilities live in `pykokoro.onnx_backend` and are used for model and voice
management.

HuggingFace is the default model source. For Termux/Android installations where
HuggingFace downloads are unavailable, select GitHub v1.0 explicitly:

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

GitHub v1.0 uses the embedded standard v1.0 vocabulary and does not require a
HuggingFace `config.json`. Model sources are never silently switched. Explicit
`model_path` and `voices_path` files remain validated in place, and Android ONNX Runtime
warnings are independent of model asset selection.

```{eval-rst}
.. autofunction:: pykokoro.onnx_backend.download_model
```

```{eval-rst}
.. autofunction:: pykokoro.onnx_backend.download_voice
```

```{eval-rst}
.. autofunction:: pykokoro.onnx_backend.download_all_models
```

```{eval-rst}
.. autofunction:: pykokoro.onnx_backend.download_all_voices
```

```{eval-rst}
.. autofunction:: pykokoro.onnx_backend.download_config
```

```{eval-rst}
.. autofunction:: pykokoro.onnx_backend.get_model_path
```

```{eval-rst}
.. autofunction:: pykokoro.onnx_backend.get_voice_path
```

## Configuration Helpers

```{eval-rst}
.. autofunction:: pykokoro.utils.load_config
```

```{eval-rst}
.. autofunction:: pykokoro.utils.save_config
```

```{eval-rst}
.. autofunction:: pykokoro.utils.get_user_cache_path
```

```{eval-rst}
.. autofunction:: pykokoro.utils.get_user_config_path
```

## SSMD 0.8 API

The public renderer configuration is `pykokoro.ssmd_config.SSMDRenderConfig` with
`SSMDPauseOverrides`. `PipelineConfig(ssmd=...)` sets defaults, and `run(..., ssmd=...)`
accepts a per-render replacement. `AudioResult.document_metadata` contains copied title,
binding, and pause metadata; `AudioResult.markers` contains structured marker sample
offsets. Audio annotations require an explicit resolver and fall back to `alt` text;
Kokoro extensions are rejected by profile validation.

`SSMDRenderConfig.emphasis_mode` defaults to `"plain"`, preserving emphasis metadata
without changing generated audio. `"approximate"` applies core gain-only mappings
(`strong` `+6dB`, `moderate` `+3dB`, `reduced` `-3dB`) at `emphasis_gain_scale=1.0`. The
scale accepts finite values from `0.0` through `2.0`; `0.5` halves automatic gain and
`1.5` makes it 50% stronger without changing semantic emphasis. Explicit `volume` values
win. `"warn"` preserves audio and adds one `ssmd.emphasis_unsupported` warning per
logical source segment, while `"error"` rejects effectful emphasis before inference. The
`none` level is always a silent no-op. Scaling is gain-only: it adds no automatic `rate`
or `pitch` fields. Explicit SSMD prosody remains independent and no prosody extra is
required.

## See Also

- {doc}`basic_usage` - Fundamental usage patterns
- {doc}`examples` - Practical pipeline examples
