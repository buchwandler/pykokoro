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

Set `PipelineConfig(retain_segment_audio=False)` for compact results when segment
waveforms are not needed. This reduces retained memory after generation, but
whole-result concatenation still occurs and peak memory remains dependent on input
duration. A future streaming API is required for bounded peak memory.

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
without changing generated audio. `"approximate"` applies core volume-only mappings
(`strong` `+6dB`, `moderate` `+3dB`, `reduced` `-3dB`); explicit prosody values win.
`"warn"` preserves audio and adds one `ssmd.emphasis_unsupported` warning per logical
source segment, while `"error"` rejects effectful emphasis before inference. The `none`
level is always a silent no-op. Approximation is processed by the core AudioSig
dependency; no prosody extra is required.

## See Also

- {doc}`basic_usage` - Fundamental usage patterns
- {doc}`examples` - Practical pipeline examples
