# Advanced request features

All text passed to PyKokoro is prepared speech text. The engine applies the KokoroG2P
prepared-text behavior but does not interpret SSMD, YAML, or document-level directives.

## Source-aligned pronunciation overrides

`PronunciationOverride` carries an already-resolved pronunciation instruction for a
half-open range `[start, end)` in `SynthesisSegment.text`. A language override selects
the pronunciation language for that range:

```python
from pykokoro import PronunciationOverride, SynthesisSegment

request = SynthesisSegment(
    id="code-switch",
    text="Hello Welt.",
    language="en-us",
    voice="af_sarah",
    pronunciation_overrides=(PronunciationOverride(6, 10, language="de"),),
)
```

A direct phoneme override uses the same source-coordinate contract. Supply a phoneme
sequence that is valid for the selected Kokoro frontend and model vocabulary:

```python
from pykokoro import PronunciationOverride, SynthesisSegment

request = SynthesisSegment(
    id="direct-phonemes",
    text="Hello",
    language="en-us",
    voice="af_sarah",
    pronunciation_overrides=(PronunciationOverride(0, 5, phonemes="hˈɛloʊ"),),
)
```

Whole-request direct input can instead be set with `SynthesisSegment.phonemes`; it
cannot be combined with span-level direct-phoneme overrides. Ambiguous overlapping
direct overrides are rejected. Offsets refer to the exact prepared request string, not
an earlier source file or markup document.

## Caller-provided linguistic annotations

`LinguisticToken` supplies source-aligned POS, tag, lemma, morphology, and optional
language context to KokoroG2P. The caller owns text analysis and passes only these
simple values:

```python
from pykokoro import LinguisticToken, SynthesisSegment

request = SynthesisSegment(
    id="homograph",
    text="I read the note yesterday.",
    language="en-us",
    voice="af_sarah",
    tokens=(
        LinguisticToken(
            2, 6, text="read", pos="VERB", tag="VBD", lemma="read", morph="Tense=Past"
        ),
    ),
)
```

PyKokoro does not require Utterplan, spaCy documents, or planner node objects. Supplied
tokens, including `morph`, take precedence and bypass engine-side spaCy. `annotations`
remains available as a compatibility alias for `tokens`. Annotation ranges are validated
against the request text; if token text is present, it must match the slice at those
offsets.

## Automatic pronunciation-language routing

The request still requires an explicit main language. Optional routing asks KokoroG2P to
select among configured pronunciation candidates:

```python
from pykokoro import LanguageRoutingConfig, SynthesisConfig

config = SynthesisConfig(
    language_routing=LanguageRoutingConfig(mode="auto", languages=("en", "de")),
)
```

Explicit request language overrides and token language annotations are passed through
the prepared-text API; routing does not change the selected acoustic model, voice, or
runtime.

## Explicit short-sentence handling

Short-sentence context is disabled when neither `short_sentence_config` nor the
generation override is supplied. Opt in with a `ShortSentenceConfig`, or set
`GenerationConfig(enable_short_sentence=True)`. PyKokoro may use generated context
internally, but the result text and word timings remain aligned to the original request.
The `RenderedSegment.short_sentence_mode` field reports the mode used without exposing
the context.

## Tracing and voice calibration

Set `return_trace=True` for request-local engine trace information. `voice_level` can
enable engine-local voice calibration:

```python
from pykokoro import SynthesisConfig, VoiceLevelConfig

config = SynthesisConfig(
    return_trace=True,
    voice_level=VoiceLevelConfig(mode="calibrated"),
)
```

Calibration is distinct from mastering a complete program or audiobook. Each result's
`voice_level_applications` records the applied gain, calibration source, and a
structured `calibration_not_found` outcome for custom voices.
`synthesis_identity.cache_key` provides a stable digest of output-affecting engine
settings. For capacity errors and model profiles, see
[the request lifecycle](pipeline_stages.md) and [language profiles](languages.md).
