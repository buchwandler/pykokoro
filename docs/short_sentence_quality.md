# Short-sentence synthesis

Short-sentence processing is **disabled by default**. Enable it explicitly with
`SynthesisConfig.short_sentence_config` or the per-request
`GenerationConfig.enable_short_sentence` override. It is an engine-local inference aid
for short utterances, not editorial pause insertion, cross-request composition, or a
timeline API. The result remains one `RenderedSegment` aligned to the original request
text.

## Public configuration

`ShortSentenceConfig` supports these root-API choices:

```python
from pykokoro import ShortSentenceConfig

ShortSentenceConfig(enabled=False)                 # Explicitly off
ShortSentenceConfig(resolve_mode="wrap")           # Fast phoneme-context wrapping
ShortSentenceConfig(resolve_mode="phrase")         # Phrase context and audio cutting
ShortSentenceConfig(resolve_mode="randomized-phrase")  # Varied phrase context and cutting
```

The default `ShortSentenceConfig()` is enabled and uses `randomized-phrase`; it does not
become active unless the application supplies that config or turns on the generation
override. The default `min_phoneme_length=30` threshold determines which prepared
utterances are treated as short. The `wrap` strategy surrounds the target phonemes with
configured phoneme pretext. Phrase strategies synthesize a surrounding sentence and cut
the target region when timing geometry supports it; `randomized-phrase` varies the
surrounding phrase. Phrase selection's built-in `auto` policy chooses a context style
from the target form.

Phrase-based strategies need a model ONNX output with duration/timestamp information. If
that is unavailable, an explicitly configured phrase mode warns and falls back to `wrap`
for the run. `wrap` does not require phrase timestamps. Phrase processing can require
additional inference work; use `enabled=False` when that work is not appropriate.

## Configure for a synthesizer

```python
from pykokoro import (
    GenerationConfig,
    KokoroSynthesizer,
    ShortSentenceConfig,
    SynthesisConfig,
)

config = SynthesisConfig(
    generation=GenerationConfig(lang="en-us"),
    short_sentence_config=ShortSentenceConfig(resolve_mode="phrase"),
)
with KokoroSynthesizer(config) as synthesizer:
    result = synthesizer.synthesize_text("Yes!", language="en-us")

print(result.short_sentence_mode)
```

`GenerationConfig.enable_short_sentence=True` is an explicit per-request enable
override; `False` disables processing even if a config is present. If left unset, the
`ShortSentenceConfig` (when supplied) determines whether it is enabled. Keep the random
seed fixed when reproducibility across randomized phrase choices matters.

PyKokoro may use internal context, retries, and audio cutting, but it returns only the
original request's waveform and metadata. The short-sentence mode used is available
through `RenderedSegment.short_sentence_mode`; original text and word timings remain
source-aligned. For diagnostics, opt into `return_trace=True` and inspect the
request-local `trace`.

See the public-API-only [`short_sentence_demo.py`](../examples/short_sentence_demo.py)
for the supported configuration forms. For speech outside this engine feature, such as
pauses between requests, use the caller's composition layer.
