# Basic Usage

This guide covers the fundamental usage patterns of PyKokoro.

```{eval-rst}
.. note::

   PyKokoro uses ``KokoroPipeline`` as the supported API. The pipeline wraps all
   stages (document parsing, splitting, G2P, and synthesis) behind one call.
```

## Initializing the Pipeline

The main entry point is the `KokoroPipeline` class:

```python
from pykokoro import GenerationConfig, KokoroPipeline, PipelineConfig

# Initialize with defaults (HuggingFace v1.0)
pipe = KokoroPipeline(PipelineConfig(voice="af_bella"))

# Specify model source and variant
pipe = KokoroPipeline(
    PipelineConfig(
        voice="af_bella",
        model_source="huggingface",
        model_variant="v1.0",
    )
)

# GitHub source
pipe = KokoroPipeline(
    PipelineConfig(
        voice="af_bella",
        model_source="github",
        model_variant="v1.0",
    )
)

# GitHub v1.0 is the explicit Termux-friendly source when HuggingFace is unavailable.
# It uses the embedded v1.0 vocabulary and does not need config.json from HuggingFace.

# Custom generation settings
generation = GenerationConfig(lang="en-us", speed=1.1)
pipe = KokoroPipeline(PipelineConfig(voice="af_bella", generation=generation))
```

HuggingFace is the default source, and source selection is deterministic: PyKokoro does
not silently fall back to another source. Explicit `model_path` and `voices_path` files
continue to be validated and used in place. ONNX Runtime Android/provider warnings are
separate from model downloads.

### Reusing the Pipeline

Create a pipeline once and reuse it across runs:

```python
from pykokoro import KokoroPipeline, PipelineConfig

pipe = KokoroPipeline(PipelineConfig(voice="af_bella"))
result = pipe.run("Hello!")
print(result.sample_rate)
```

### Using Local Model Files

If you already have the ONNX model and voices files locally, pass their paths through
`PipelineConfig`:

```python
from pathlib import Path

from pykokoro import GenerationConfig, KokoroPipeline, PipelineConfig

config = PipelineConfig(
    voice="af_bella",
    generation=GenerationConfig(lang="en-us"),
    model_path=Path("/models/kokoro.onnx"),
    voices_path=Path("/models/voices.bin.npz"),
)
pipe = KokoroPipeline(config)
result = pipe.run("Using local model files.")
```

### Model Quality Options

Available quality options vary by model source and variant:

**HuggingFace (Default Source):**

Both v1.0 and v1.1-zh variants support:

- `fp32` - Full precision (highest quality, largest size)
- `fp16` - Half precision (good balance)
- `q8` - 8-bit quantized (default, good balance)
- `q8f16` - 8-bit with fp16
- `q4` - 4-bit quantized (smallest, faster)
- `q4f16` - 4-bit with fp16
- `uint8` - Unsigned 8-bit
- `uint8f16` - Unsigned 8-bit with fp16

**GitHub v1.0:**

- `fp32` - Full precision
- `fp16` - Half precision
- `fp16-gpu` - GPU-optimized fp16
- `q8` - 8-bit quantized

**GitHub v1.1-zh:**

- `fp32` - Full precision only

```python
from pykokoro import KokoroPipeline, PipelineConfig

# HuggingFace v1.0 with fp16
pipe = KokoroPipeline(PipelineConfig(voice="af_bella", model_quality="fp16"))

# GitHub v1.0 with GPU optimization
pipe = KokoroPipeline(
    PipelineConfig(
        voice="af_bella",
        model_source="github",
        model_variant="v1.0",
        model_quality="fp16-gpu",
    )
)
```

## Generating Speech

### Basic Text-to-Speech

```python
from pykokoro import KokoroPipeline, PipelineConfig

pipe = KokoroPipeline(PipelineConfig(voice="af_bella"))
result = pipe.run("Hello, world!")
audio = result.audio
sample_rate = result.sample_rate
```

### Saving Audio

Using soundfile (recommended):

```python
import soundfile as sf

from pykokoro import KokoroPipeline, PipelineConfig

pipe = KokoroPipeline(PipelineConfig(voice="af_bella"))
result = pipe.run("Hello!")
sf.write("output.wav", result.audio, result.sample_rate)
```

## Voice Selection

Voice names follow the pattern: `{accent}_{gender}_{name}`

- **Accent**: `af` (American Female), `am` (American Male), `bf` (British Female), `bm`
  (British Male)
- **Gender**: `f` (female), `m` (male)
- **Name**: Specific voice identifier

Use the voice name in `PipelineConfig`:

```python
from pykokoro import KokoroPipeline, PipelineConfig

pipe = KokoroPipeline(PipelineConfig(voice="bf_emma"))
result = pipe.run("Hello from the UK!")
```

## Language Settings

PyKokoro defaults language from the voice prefix, but you can override it:

```python
from pykokoro import GenerationConfig, KokoroPipeline, PipelineConfig

generation = GenerationConfig(lang="fr")
pipe = KokoroPipeline(PipelineConfig(voice="af_sarah", generation=generation))
result = pipe.run("Bonjour le monde")
```

Supported languages: `en-us`, `en-gb`, `es`, `fr`, `de`, `it`, `pt`, `hi`, `ja`, `zh`

## Language-Aware spaCy Models

When both spaCy settings are unset (the default), PyKokoro asks each backend to select
the highest installed compatible model for the effective language
(`trf > lg > md > sm`). No model is downloaded automatically; `"auto"` remains an
accepted alias for unset.

Use `with_spacy_model` to request an exact tier or package consistently across sentence
segmentation and G2P:

```python
from pykokoro import (
    GenerationConfig,
    KokoroPipeline,
    PipelineConfig,
    with_spacy_model,
)

base = PipelineConfig(
    voice="af_bella",
    generation=GenerationConfig(lang="de"),
)
cfg = with_spacy_model(size="lg")(base)

# For lang="de", both components request de_core_news_lg
pipe = KokoroPipeline(cfg)
result = pipe.run("Guten Tag")
```

You can still force a specific spaCy package if needed:

```python
from pykokoro import KokoroPipeline, PipelineConfig
from pykokoro.tokenizer import TokenizerConfig

tokenizer_config = TokenizerConfig(spacy_model="fr_core_news_sm")
pipe = KokoroPipeline(PipelineConfig(voice="af_bella", tokenizer_config=tokenizer_config))
```

`result.document_metadata["spacy_models"]` reports the concrete sentence and G2P
packages selected. `lg` and `trf` generally improve linguistic quality but use more
memory and take longer to initialize than `sm` and `md`.

## Speech Speed Control

Adjust the speaking rate with `GenerationConfig.speed`:

```python
from pykokoro import GenerationConfig, KokoroPipeline, PipelineConfig

generation = GenerationConfig(speed=1.5)
pipe = KokoroPipeline(PipelineConfig(voice="af_bella", generation=generation))
result = pipe.run("Fast speech")
```

Recommended range: 0.5 to 2.0

## Prosody Backend Selection

SSMD rate, pitch, and volume metadata is composed in one AudioSig speech-effects pass.
The default backend is WSOLA:

```python
from pykokoro import PipelineConfig, ProsodyConfig

config = PipelineConfig(
    voice="af_bella",
    prosody=ProsodyConfig(method="wsola"),
)
```

Use `td_psola` (or its `psola` alias) and `esola` only as experimental choices. Current
TD-PSOLA limits are rate `0.75..1.5` and pitch `-6..+6 st`; ESOLA requires its computed
backend rate to remain in `0.5..2.0`. `phase_vocoder` remains available as a reference
path. Strict comparison mode disables fallback:

```python
config = PipelineConfig(
    prosody=ProsodyConfig(
        method="esola",
        fallback_methods=(),
        strict=True,
    ),
)
```

No backend guarantees formant preservation. Results vary by voice and utterance, and
segment-level processing cannot restore sentence-level coarticulation. Use
`examples/compare_prosody_algorithms.py` to compare identical source audio before
changing the default; its objective metrics are diagnostic rather than naturalness
scores.

## Pause Control

### Manual Pause Markers

Add explicit pauses using SSMD break markers:

- `...c` - Short/comma pause
- `...s` - Medium/sentence pause
- `...p` - Long/paragraph pause
- `...500ms` - Custom duration pause

```python
from pykokoro import GenerationConfig, KokoroPipeline, PipelineConfig

text = "Hello! ...c This is a short pause. ...s And now a longer pause."
generation = GenerationConfig(pause_mode="manual")
pipe = KokoroPipeline(PipelineConfig(voice="af_bella", generation=generation))
result = pipe.run(text)
```

### Automatic Natural Pauses

For natural rhythm, let the pipeline insert pauses at boundaries:

```python
from pykokoro import GenerationConfig, KokoroPipeline, PipelineConfig

text = """
Artificial intelligence is transforming our world. Machine learning
models are becoming more sophisticated and accessible.

Deep learning uses neural networks with many layers.
"""

generation = GenerationConfig(
    pause_mode="auto",
    pause_clause=0.25,
    pause_sentence=0.5,
    pause_paragraph=1.0,
    pause_variance=0.05,
    random_seed=42,
)
pipe = KokoroPipeline(PipelineConfig(voice="af_sarah", generation=generation))
result = pipe.run(text)
```

## Text Normalization (Say-As)

SSMD say-as syntax converts numbers, dates, and other formats:

```python
from pykokoro import KokoroPipeline, PipelineConfig

text = 'I have [123]{as="cardinal"} apples and [12/31/2024]{as="date" format="mdy"}.'
pipe = KokoroPipeline(PipelineConfig(voice="af_sarah"))
result = pipe.run(text)
```

## Error Handling

```python
from pykokoro import KokoroPipeline, PipelineConfig

try:
    pipe = KokoroPipeline(PipelineConfig(voice="invalid_voice"))
    pipe.run("Hello!")
except Exception as exc:
    print(f"Pipeline error: {exc}")
```

## Batch Processing

Process multiple texts efficiently:

```python
import soundfile as sf

from pykokoro import KokoroPipeline, PipelineConfig

texts = [
    ("Welcome", "welcome.wav"),
    ("Thank you", "thanks.wav"),
    ("Goodbye", "goodbye.wav"),
]

pipe = KokoroPipeline(PipelineConfig(voice="af_bella"))
for text, filename in texts:
    result = pipe.run(text)
    sf.write(filename, result.audio, result.sample_rate)
```

## SSMD 0.8 portable metadata

SSMD front matter is parsed by default and removed before sentence parsing. Use logical
roles in the body and bind them in the document header. API bindings override document
bindings, explicit breaks override implicit defaults, and `parse_header=False` preserves
literal leading delimiters. PyKokoro does not load SSMD user configuration files; audio
annotations require an explicit resolver.

### Emphasis behavior

The default `SSMDRenderConfig(emphasis_mode="plain")` preserves emphasis metadata but
leaves speech unmodified. Use `emphasis_mode="approximate"` to opt into the core
volume-only mapping `strong` `+6dB`, `moderate` `+3dB`, and `reduced` `-3dB`. `warn`
keeps ordinary speech and reports one trace warning per logical source segment; `error`
rejects effectful emphasis before inference. `emphasis="none"` is silently accepted in
every mode, and explicit prosody values take precedence.

```python
from pykokoro import KokoroPipeline, PipelineConfig, SSMDRenderConfig

script = """---
title: Portable podcast
voice_bindings:
  kokoro:
    host: af_sarah
pause_defaults:
  enabled: true
  paragraph: 700ms
---
<div voice="host">Welcome to the portable podcast.</div>
"""
result = KokoroPipeline(PipelineConfig(ssmd=SSMDRenderConfig())).run(script)
print(result.document_metadata["title"])
```

## Next Steps

- {doc}`advanced_features` - Voice blending, phoneme control, and more
- {doc}`examples` - Real-world examples
- {doc}`api_reference` - Complete API documentation
