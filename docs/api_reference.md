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

These utilities live in `pykokoro.onnx_backend` and are used for model and
voice management.

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

## See Also

- {doc}`basic_usage` - Fundamental usage patterns
- {doc}`examples` - Practical pipeline examples
