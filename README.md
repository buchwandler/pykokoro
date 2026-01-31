[![PyPI - Version](https://img.shields.io/pypi/v/pykokoro)](https://pypi.org/project/pykokoro/)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pykokoro)
![PyPI - Downloads](https://img.shields.io/pypi/dm/pykokoro)
[![codecov](https://codecov.io/gh/holgern/pykokoro/graph/badge.svg?token=iCHXwbjAXG)](https://codecov.io/gh/holgern/pykokoro)

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
- **GPU Acceleration**: Optional CUDA, CoreML, or DirectML support
- **Phoneme Support**: Advanced phoneme-based generation with kokorog2p
- **Hugging Face Integration**: Automatic model downloading from Hugging Face Hub
- **Text Normalization**: Automatic say-as support for numbers, dates, phone numbers,
  and more using SSMD markup

## Installation

### Basic Installation (CPU only)

```bash
pip install pykokoro
```

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

#### All Accelerators

```bash
pip install pykokoro[all]
```

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

## Pipeline Stages

The pipeline is built from composable stages so you can swap behavior without rewriting
the whole flow:

`doc_parser (includes segmentation) -> g2p -> phoneme_processing -> audio_generation -> audio_postprocessing`

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
# Auto-select best available provider (CUDA > CoreML > DirectML > CPU)
# Note: OpenVINO is attempted but will fall back to next priority if incompatible
from pykokoro import KokoroPipeline, PipelineConfig

pipe = KokoroPipeline(PipelineConfig(provider="auto", voice="af_sarah"))
res = pipe.run("Hello")
```

### Explicit Provider Selection

```python
# Force specific provider
from pykokoro import KokoroPipeline, PipelineConfig

pipe = KokoroPipeline(PipelineConfig(provider="cuda", voice="af_sarah"))      # NVIDIA CUDA
pipe = KokoroPipeline(PipelineConfig(provider="openvino", voice="af_sarah"))  # Intel OpenVINO
pipe = KokoroPipeline(PipelineConfig(provider="directml", voice="af_sarah"))  # Windows DirectML
pipe = KokoroPipeline(PipelineConfig(provider="coreml", voice="af_sarah"))    # Apple CoreML
pipe = KokoroPipeline(PipelineConfig(provider="cpu", voice="af_sarah"))       # CPU only
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

## Usage Examples

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

### Streaming Generation

```python
from pykokoro import KokoroPipeline, PipelineConfig

pipe = KokoroPipeline(PipelineConfig(voice="af_sarah"))
chunks = ["Long text", "here..."]
for text_chunk in chunks:
    res = pipe.run(text_chunk)
    play_audio(res.audio, res.sample_rate)
```

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
        pause_clause=0.2,      # ...c = 200ms
        pause_sentence=0.5,    # ...s = 500ms
        pause_paragraph=1.5,   # ...p = 1500ms
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
        pause_clause=0.25,        # Pause after clauses (commas)
        pause_sentence=0.5,       # Pause after sentences
        pause_paragraph=1.0,      # Pause after paragraphs
        pause_variance=0.05,      # Add natural variance (default)
        random_seed=42,           # For reproducible results (optional)
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

**Note:** For sentence splitting with `PlainTextDocumentParser`, install spaCy:

```bash
pip install spacy
python -m spacy download en_core_web_sm
```

**Combining Both Approaches:**

Use SSMD markers for special emphasis and automatic pauses for natural rhythm:

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

### Voice Switching (SSMD)

You can switch voices per segment using SSMD directives. Block directives use
`<div voice="...">` while inline annotations use `[text]{voice="..."}`.

```python
text = (
    '<div voice="af_sarah">Hello there.</div>\n\n'
    '<div voice="am_michael">General Kenobi.</div>'
)

pipe = KokoroPipeline(PipelineConfig(voice="af"))
res = pipe.run(text)
```

```python
text = "[Hello]{voice='af_sarah'} ...s [World]{voice='am_michael'}"
res = pipe.run(text)
```

### Text Normalization (Say-As)

PyKokoro supports automatic text normalization using SSMD (Speech Synthesis Markdown)
syntax. Convert numbers, dates, phone numbers, and more into speakable text:

```python
from pykokoro import KokoroPipeline, PipelineConfig
from pykokoro.generation_config import GenerationConfig

pipe = KokoroPipeline(PipelineConfig(voice="af_sarah"))

# Cardinal numbers
text = "I have [123](as: cardinal) apples"
res = pipe.run(text)
# TTS says: "I have one hundred twenty-three apples"

# Ordinal numbers
text = "I came in [3](as: ordinal) place"
res = pipe.run(text)
# TTS says: "I came in third place"

# Digits (spell out)
text = "My PIN is [1234](as: digits)"
res = pipe.run(text)
# TTS says: "My PIN is one two three four"

# Telephone numbers
text = "Call [+1-555-0123](as: telephone)"
res = pipe.run(text)
# TTS says: "Call plus one five five five oh one two three"

# Dates with custom formatting
text = "Today is [12/31/2024](as: date, format: mdy)"
res = pipe.run(text)
# TTS says: "Today is December thirty-first, two thousand twenty-four"

# Time (12-hour or 24-hour)
text = "The time is [14:30](as: time)"
res = pipe.run(text)
# TTS says: "The time is two thirty PM"

# Characters (spell out)
text = "The code is [ABC](as: characters)"
res = pipe.run(text)
# TTS says: "The code is A B C"

# Fractions
text = "Add [1/2](as: fraction) cup of sugar"
res = pipe.run(text)
# TTS says: "Add one half cup of sugar"

# Units
text = "The package weighs [5kg](as: unit)"
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
text = "[123](as: cardinal)"
pipe = KokoroPipeline(
    PipelineConfig(voice="ff_siwis", generation=GenerationConfig(lang="fr-fr"))
)
res = pipe.run(text)
# TTS says: "cent vingt-trois"

# German ordinal
text = "[3](as: ordinal)"
pipe = KokoroPipeline(
    PipelineConfig(voice="gf_maria", generation=GenerationConfig(lang="de-de"))
)
res = pipe.run(text)
# TTS says: "dritte"
```

**Combining with Other Features:**

Say-as works seamlessly with all SSMD features:

```python
# With prosody
text = "[100](as: cardinal) +loud+ dollars!"

# With pauses
text = "[First](as: ordinal) ...c [second](as: ordinal) ...c [third](as: ordinal)!"

# With emphasis
text = "The winner is *[1](as: ordinal)*!"
```

See `examples/say_as_demo.py` for comprehensive examples.

#### 4. Automatic Short Sentence Handling

When processing text, very short sentences (like "Why?" or "Go!") can produce poor audio
quality when processed individually (only 3-8 phonemes each). Pykokoro automatically
handles this using a "repeat-and-cut" technique:

**How It Works:**

1. Short segments are detected based on phoneme length (default: <30 phonemes)
2. The sentence is repeated: "Why?" → "Why? Why? Why?"
3. TTS generates audio with more context (better prosody)
4. Audio is trimmed to extract only the first instance

This happens automatically during `pipe.run()` - no configuration needed!

**Customizing the Behavior:**

You can customize the thresholds using `ShortSentenceConfig`:

```python
from pykokoro import KokoroPipeline, PipelineConfig
from pykokoro.short_sentence_handler import ShortSentenceConfig

# More aggressive short sentence handling
short_sentence_config = ShortSentenceConfig(
    min_phoneme_length=50,    # Treat segments <50 phonemes as short
    target_phoneme_length=150, # Repeat until ~150 phonemes
    max_repetitions=7,         # Allow up to 7 repetitions
)

pipe = KokoroPipeline(
    PipelineConfig(voice="af_sarah", short_sentence_config=short_sentence_config)
)
res = pipe.run("Why?")
```

**Default Configuration:**

- `min_phoneme_length=30`: Segments below this use repeat-and-cut
- `target_phoneme_length=100`: Target length for repeated text
- `max_repetitions=5`: Maximum times to repeat

**Disabling Short Sentence Handling:**

```python
from pykokoro import KokoroPipeline, PipelineConfig
from pykokoro.short_sentence_handler import ShortSentenceConfig

short_sentence_config = ShortSentenceConfig(min_phoneme_length=0)
pipe = KokoroPipeline(
    PipelineConfig(voice="af_sarah", short_sentence_config=short_sentence_config)
)
res = pipe.run("Why?")
```

See `examples/optimal_phoneme_length_demo.py` for a demonstration.

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

The default source with 54 multi-language voices:

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
from pykokoro import KokoroPipeline, PipelineConfig
from pykokoro.tokenizer import TokenizerConfig

# Create config with custom phoneme dictionary
tokenizer_config = TokenizerConfig(
    phoneme_dictionary_path="my_pronunciations.json"
)

pipe = KokoroPipeline(
    PipelineConfig(voice="af_sarah", tokenizer_config=tokenizer_config)
)
res = pipe.run("Hello")
```

### Mixed Language Support

```python
from pykokoro import KokoroPipeline, PipelineConfig
from pykokoro.tokenizer import TokenizerConfig

tokenizer_config = TokenizerConfig(
    use_mixed_language=True,
    mixed_language_primary="en-us",
    mixed_language_allowed=["en-us", "de", "fr"]
)

pipe = KokoroPipeline(
    PipelineConfig(voice="af_sarah", tokenizer_config=tokenizer_config)
)
res = pipe.run("Ich gehe zum Meeting")
```

### Backend Configuration

Control which phonemization backend and dictionaries to use:

```python
from pykokoro import KokoroPipeline, PipelineConfig
from pykokoro.tokenizer import TokenizerConfig

# Default: Full dictionaries with espeak fallback (best quality)
tokenizer_config = TokenizerConfig(
    backend="espeak",
    load_gold=True,
    load_silver=True,
    use_espeak_fallback=True
)

# Memory-optimized: Gold dictionary only
tokenizer_config = TokenizerConfig(
    backend="espeak",
    load_gold=True,
    load_silver=False,  # Saves ~22-31 MB
    use_espeak_fallback=True
)

# Fastest initialization: Pure espeak
tokenizer_config = TokenizerConfig(
    backend="espeak",
    load_gold=False,
    load_silver=False,
    use_espeak_fallback=True
)

# Alternative backend (requires pygoruut)
tokenizer_config = TokenizerConfig(
    backend="goruut"
)

pipe = KokoroPipeline(
    PipelineConfig(voice="af_sarah", tokenizer_config=tokenizer_config)
)
res = pipe.run("Hello")
```

**Note**: `use_dictionary` parameter is deprecated. Use `load_gold` and `load_silver`
instead for finer control.

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

## License

This library is licensed under the Apache License 2.0.

## Credits

- **Kokoro Model**: [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)
- **ONNX Models**:
  [onnx-community/Kokoro-82M-v1.0-ONNX](https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX)
- **Phonemizer**: [kokorog2p](https://github.com/remyxai/kokorog2p)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Links

- **GitHub**: https://github.com/holgern/pykokoro
- **PyPI**: https://pypi.org/project/pykokoro/
- **Documentation**: https://pykokoro.readthedocs.io/
