# Examples

This page provides practical examples for common use cases.

```{eval-rst}
.. note::

   The supported interface is ``KokoroPipeline``. If you see legacy ``Kokoro``
   snippets in older examples, update them to the pipeline style shown below.
```

## Pipeline Stage Showcase

Use the stage showcase script to see how the new pipeline stages fit together:

`examples/pipeline_stage_showcase.py`

## Spoken-Form Normalization

Run:

`python examples/spokenform_showcase.py`

The example feeds unannotated text containing abbreviations, dates, times, currency,
measurements, and other structured expressions through Spokenform's automatic
spoken-form preparation. It prints the prepared text and phonemes, then synthesizes the
same raw source with PyKokoro. Use `--inspect-only` to inspect the front-end result
without loading a synthesis model.

## Hello World

The simplest example:

```python
from pykokoro import GenerationConfig, KokoroPipeline, PipelineConfig

pipe = KokoroPipeline(PipelineConfig(generation=GenerationConfig(lang="en-us"), voice="af_bella"))
result = pipe.run("Hello, world!")

# Direct system playback, using the optional playback dependency:
result.play()

# Or persist the waveform:
result.save_wav("hello.wav")
```

## Multi-Voice Demo

Generate the same text with different voices:

```python
import soundfile as sf
from pykokoro import GenerationConfig, KokoroPipeline, PipelineConfig

text = "This is a demonstration of different voices."

voices = [
    ("af_bella", "American Female - Bella"),
    ("am_adam", "American Male - Adam"),
    ("bf_emma", "British Female - Emma"),
    ("bm_george", "British Male - George"),
]

for voice_name, description in voices:
    print(f"Generating: {description}")
    pipe = KokoroPipeline(PipelineConfig(generation=GenerationConfig(lang="en-us"), voice=voice_name))
    result = pipe.run(text)
    sf.write(f"voice_{voice_name}.wav", result.audio, result.sample_rate)
```

## Pause Markers Demo

Demonstrate different pause durations:

```python
import soundfile as sf

from pykokoro import GenerationConfig, KokoroPipeline, PipelineConfig

text = """
This is a sentence with a short pause ...c
Now a medium pause ...s
And finally a long pause ...p
Back to normal.
"""

generation = GenerationConfig(pause_mode="manual")
pipe = KokoroPipeline(PipelineConfig(generation=GenerationConfig(lang="en-us"), voice="af_bella", generation=generation))
result = pipe.run(text)
sf.write("pauses_demo.wav", result.audio, result.sample_rate)
```

### Custom Pause Durations

```python
import soundfile as sf

from pykokoro import GenerationConfig, KokoroPipeline, PipelineConfig

text = "Custom ...c pauses ...s here ...p"

generation = GenerationConfig(
    pause_mode="manual",
    pause_clause=0.2,
    pause_sentence=0.5,
    pause_paragraph=1.0,
)
pipe = KokoroPipeline(PipelineConfig(generation=GenerationConfig(lang="en-us"), voice="af_bella", generation=generation))
result = pipe.run(text)
sf.write("custom_pauses.wav", result.audio, result.sample_rate)
```

## Voice Blending

### Simple Blend

```python
import soundfile as sf

from pykokoro import GenerationConfig, KokoroPipeline, PipelineConfig
from pykokoro.onnx_backend import VoiceBlend

blend = VoiceBlend.parse("af_bella:50,af_sarah:50")
pipe = KokoroPipeline(PipelineConfig(generation=GenerationConfig(lang="en-us"), voice=blend))
result = pipe.run("This is a blended voice")
sf.write("blended.wav", result.audio, result.sample_rate)
```

### Weighted Blend

```python
import soundfile as sf

from pykokoro import GenerationConfig, KokoroPipeline, PipelineConfig
from pykokoro.onnx_backend import VoiceBlend

blend = VoiceBlend.parse("af_bella:70,af_sarah:30")
pipe = KokoroPipeline(PipelineConfig(generation=GenerationConfig(lang="en-us"), voice=blend))
result = pipe.run("Weighted blend example")
sf.write("weighted_blend.wav", result.audio, result.sample_rate)
```

## Multi-Language Support

### Spanish

```python
import soundfile as sf

from pykokoro import GenerationConfig, KokoroPipeline, PipelineConfig

text = "Hola, como estas? Este es un ejemplo en espanol."
generation = GenerationConfig(lang="es")
pipe = KokoroPipeline(PipelineConfig(voice="af_nicole", generation=generation))
result = pipe.run(text)
sf.write("spanish.wav", result.audio, result.sample_rate)
```

### French

```python
import soundfile as sf

from pykokoro import GenerationConfig, KokoroPipeline, PipelineConfig

text = "Bonjour! Ceci est un exemple en francais."
generation = GenerationConfig(lang="fr")
pipe = KokoroPipeline(PipelineConfig(voice="af_sarah", generation=generation))
result = pipe.run(text)
sf.write("french.wav", result.audio, result.sample_rate)
```

## Long Text Processing

For longer text, reuse a pipeline and let the document parser handle segmentation:

```python
import soundfile as sf

from pykokoro import GenerationConfig, KokoroPipeline, PipelineConfig

long_text = """
This is a long passage of text that demonstrates automatic processing.
Each sentence will be processed separately for better quality.

This is a new paragraph. It will also be handled efficiently.
"""

generation = GenerationConfig(pause_mode="manual")
pipe = KokoroPipeline(PipelineConfig(generation=GenerationConfig(lang="en-us"), voice="af_bella", generation=generation))
result = pipe.run(long_text)
sf.write("long_text.wav", result.audio, result.sample_rate)
```

## Prosody Backend Comparison

Use `examples/prosody_algorithm_selection.py` for a small, validated diagnostic
comparison. It synthesizes once, or accepts a known-good WAV, and applies WSOLA, ESOLA,
TD-PSOLA, and phase vocoder to the exact same reference. On Termux/Android, use the
input-WAV path so source synthesis is isolated from the AudioSig comparison:

```bash
python examples/prosody_algorithm_selection.py \
  --input-wav reference.wav \
  --output-dir build/prosody-selection
```

The tool writes a neutral `reference.wav`, explicit mono `PCM_16` outputs, and
`metrics.json`. It validates shape, finite values, peaks, RMS, adjacent-sample jumps,
WAV headers, and decoded frame counts. Positive gain is not part of the default
comparison because it can exceed full scale and cause common PCM clipping; over-range
audio is rejected instead of being silently clipped.

For a reproducible blind set, run the full comparison harness:

```bash
python examples/compare_prosody_algorithms.py \
  --input-wav input.wav \
  --output-dir build/prosody-comparison
```

The full comparison script renders rate-only, pitch-only, emphasis, and combined presets
from identical source audio, then writes WAV files, CSV/JSON diagnostic metrics,
randomized blind copies, a private key, and a manifest. Objective metrics do not measure
naturalness. WSOLA is the production default; ESOLA and TD-PSOLA remain experimental. No
backend guarantees formant preservation, and isolated segment processing cannot restore
sentence-level coarticulation.

## Batch Processing

### Process Multiple Files

```python
import soundfile as sf
from pathlib import Path

from pykokoro import GenerationConfig, KokoroPipeline, PipelineConfig

scripts = {
    "intro": "Welcome to our podcast!",
    "segment1": "This is the first segment.",
    "segment2": "This is the second segment.",
    "outro": "Thank you for listening!",
}

output_dir = Path("podcast_segments")
output_dir.mkdir(exist_ok=True)

pipe = KokoroPipeline(PipelineConfig(generation=GenerationConfig(lang="en-us"), voice="af_bella"))
for filename, text in scripts.items():
    print(f"Generating {filename}...")
    result = pipe.run(text)
    output_path = output_dir / f"{filename}.wav"
    sf.write(output_path, result.audio, result.sample_rate)
```

## Portable SSMD 0.8 podcast

Prefer stable logical roles in portable documents. See
`examples/ssmd_080_portable_podcast.py` for a runnable example using `host`, `cohost`,
and `guest` bindings, document pause defaults, and an API override. The body remains
portable while a renderer selects concrete Kokoro voice IDs.

## See Also

- {doc}`basic_usage` - Fundamental usage patterns
- {doc}`advanced_features` - Advanced features and techniques
- {doc}`api_reference` - API documentation
