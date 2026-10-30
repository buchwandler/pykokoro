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

## Character say-as spans

`SayAsSpan` requests character-by-character reading for an exact, half-open range in the
prepared request text. For example, PyKokoro may send `A R T` to KokoroG2P for the
marked canonical text `A R T`; the caller's text is not rewritten:

```python
from pykokoro import SayAsSpan, SynthesisSegment

text = "Because someone called A R T."
start = text.index("A R T")
request = SynthesisSegment(
    id="character-say-as",
    text=text,
    language="en-us",
    say_as=(SayAsSpan(start, start + len("A R T"), interpret_as="characters"),),
)
```

Character `say_as` uses a private frontend projection while keeping the caller's
canonical request text unchanged. Compact character sequences are separated with spaces
(`ART` → `A R T`); already separated sequences remain unchanged (`A R T` → `A R T`). If
a detached possessive suffix immediately follows a marked character span, the shadow
frontend attaches it (`A R T 's` → `A R T's`) so the frontend can process the suffix as
part of the final spelled character. Public request text, offsets, alignments, and word
timings remain canonical. Only explicitly marked spans are transformed, and `characters`
is the supported interpretation. The marked phrase stays inside the caller's request:
say-as does not add requests or pauses. The normal token-capacity policy still applies;
opt-in long-text splitting protects marked spans from being cut. `measure_request()`
uses the same projected frontend as synthesis.

## Caller-provided linguistic tokens

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
remains available as a compatibility alias for `tokens`. Token ranges are validated
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

Explicit request language overrides and token-level language information are passed
through the prepared-text API; routing does not change the selected acoustic model,
voice, or runtime.

For a deterministic comparison, keep the main request language and apply a
`PronunciationOverride(language=...)` to the exact source span; automatic routing and
the deterministic variant are contrasted in
[`language_routing.py`](../examples/language_routing.py).

## Explicit short-sentence handling

Short-sentence context is disabled when neither `short_sentence_config` nor the
generation override is supplied. Opt in with a `ShortSentenceConfig`, or set
`GenerationConfig(enable_short_sentence=True)`. PyKokoro may use generated context
internally, but the result text and word timings remain aligned to the original request.
The `RenderedSegment.short_sentence_mode` field reports the mode used without exposing
the context.

## Short-sentence modes

This feature is off unless enabled through `ShortSentenceConfig` or
`GenerationConfig.enable_short_sentence`. The public modes are `wrap`, `phrase`, and
`randomized-phrase`; `ShortSentenceConfig(enabled=False)` explicitly disables it.
Phrase-based modes require timestamp-capable model output and may fall back to `wrap`.
See [short-sentence configuration](short_sentence_quality.md) and
[`short_sentence_demo.py`](../examples/short_sentence_demo.py).

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

## Voice blends

Pass a real voice ID or an explicit `VoiceBlend` as the request/config voice. The
structured form makes interpolation and weights visible; the compact form uses
percentages:

```python
from pykokoro import VoiceBlend

blend = VoiceBlend(
    voices=[("af_sarah", 0.6), ("af_bella", 0.4)],
    interpolation="linear",
)
compact = VoiceBlend.parse("af_sarah=60,af_bella=40")
slerp = VoiceBlend.parse("af_sarah=60,af_bella=40@slerp")
multi = VoiceBlend.parse("af_sarah=50,af_bella=30,af_nicole=20")
```

Linear interpolation supports one or more voices. SLERP requires exactly two voices, and
the second voice's weight is its interpolation parameter. Voice IDs and valid
combinations come from model profiles; inspect `discover_models()` instead of assuming a
voice is available for every language or model. See
[`voice_blend.py`](../examples/voice_blend.py).

## Prepared phoneme input

A request may carry whole-request `phonemes` when the caller already prepared compatible
Kokoro phonemes. Whole-request phonemes cannot be combined with span-level
direct-phoneme overrides and must match the selected frontend, model vocabulary, and
source-alignment requirements. Invalid or oversized input raises a typed request error.

Call `synthesizer.prepare(request)` to inspect frontend output without ONNX inference:

```python
prepared = synthesizer.prepare(request)
print(prepared.phonemes)
print(prepared.token_ids)
print(prepared.diagnostics)
```

`PreparedSynthesis` also reports the request ID, text, language, voice, and alignment
tokens.

## Result metadata and errors

`RenderedSegment` carries the original request ID/text, audio and sample rate,
request-local word timings, optional trace, `synthesis_identity`, short-sentence mode,
and `voice_level_applications`. `build_synthesis_identity()` creates identity metadata
without synthesizing. See [`result_metadata.py`](../examples/result_metadata.py) for a
result-output walkthrough. For expected invalid-input, language, or voice failures,
catch the narrow public exceptions rather than catching `Exception`; see
[`error_handling.py`](../examples/error_handling.py).

Use the canonical `SynthesisSegment.tokens` field for caller-provided linguistic tokens.
Routing, G2P languages, and model/voice profiles are distinct: see
[language support](languages.md).

## Related recipes

- [Long text and capacity behavior](basic_usage.md#render-longer-text)
- [Short-sentence quality and explicit modes](short_sentence_quality.md)
- [Frontend, lexicon, cache, and asset progress configuration](installation.md)
- [All maintained example groups](examples.md)
- [English reference voice enrollment and reuse](reference_voice.md)
