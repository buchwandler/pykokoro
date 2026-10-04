# Languages and model profiles

Language support has two independent parts: a prepared-text G2P language supported by
KokoroG2P, and an acoustic model/voice profile that can render the request. A language
code being accepted for phonemization does **not** imply every model or voice supports
it. The request language is always explicit; PyKokoro does not infer it from a voice ID,
document headers, or text detection.

## Prepared-text language codes

The current KokoroG2P language-code contract accepts these canonical codes:

| Language              | Canonical code |
| --------------------- | -------------- |
| Arabic                | `ar`           |
| Czech                 | `cs-cz`        |
| German                | `de-de`        |
| English (UK)          | `en-gb`        |
| English (US)          | `en-us`        |
| Spanish               | `es-es`        |
| French                | `fr-fr`        |
| Hebrew                | `he`           |
| Hindi                 | `hi-in`        |
| Italian               | `it-it`        |
| Japanese              | `ja-jp`        |
| Kazakh                | `kk`           |
| Korean                | `ko-kr`        |
| Portuguese (Brazil)   | `pt-br`        |
| Portuguese (Portugal) | `pt-pt`        |
| Russian               | `ru-ru`        |
| Swedish               | `sv-se`        |
| Thai                  | `th-th`        |
| Vietnamese            | `vi-vn`        |
| Chinese (Mandarin)    | `zh`           |

Common shorthand aliases are normalized by KokoroG2P before routing: `en` → `en-us`,
`es` → `es-es`, `fr` → `fr-fr`, `de` → `de-de`, `it` → `it-it`, `pt` → `pt-br`, `vi` →
`vi-vn`, `sv` → `sv-se`, `ru` → `ru-ru`, `hi` → `hi-in`, `ko` → `ko-kr`, `ja` → `ja-jp`,
`th` → `th-th`, and `cs` → `cs-cz`. Both `zh` and `cmn` normalize to `zh`. Prefer
canonical values in application data so the selected language is unambiguous.

This table documents the G2P/frontend contract, not a guarantee about model artifacts. A
frontend backend may have additional runtime requirements, such as an installed eSpeak
NG executable. See [installation](installation.md#frontends-and-lexicons).

## Explicit language and pronunciation spans

`SynthesisSegment.language` is required and describes the request's main pronunciation
language. A span override can select a different G2P language for a source-aligned
range; it does not switch the acoustic model or voice:

```python
from pykokoro import PronunciationOverride, SynthesisSegment

request = SynthesisSegment(
    id="mixed",
    text="Hello Welt.",
    language="en-us",
    voice="af_sarah",
    pronunciation_overrides=(PronunciationOverride(6, 10, language="de-de"),),
)
```

Offsets are half-open character ranges into the exact prepared request text. For
complete multilingual utterances requiring different acoustic profiles, submit separate
requests and let the caller manage any composition.

## Automatic pronunciation routing

`LanguageRoutingConfig(mode="auto", languages=(...))` asks KokoroG2P to consider a
bounded candidate set for prepared-text pronunciation. The request still has its
explicit main language; routing does not detect a document's language, change the
acoustic model, or pick a voice. Explicit source-aligned request overrides and token
language values remain available:

```python
from pykokoro import LanguageRoutingConfig, SynthesisConfig

config = SynthesisConfig(
    language_routing=LanguageRoutingConfig(mode="auto", languages=("en", "de")),
)
# Normalized candidate codes: ("en-us", "de-de")
```

See [`language_routing.py`](../examples/language_routing.py) for an automatic request
beside a deterministic, explicitly annotated request.

## Discover compatible model profiles and voices

Use metadata-only discovery to check the profiles that the installed runtime can
resolve. Do not hardcode voice compatibility from a voice name or from the G2P language
table:

```python
from pykokoro import discover_models

inventory = discover_models(offline=True)
for profile in inventory.models:
    print(
        profile.model_id,
        profile.languages,
        profile.voices,
        profile.voice_mode,
        profile.supports_reference_enrollment,
        profile.speed_supported,
        profile.status,
    )
```

Discovery does not load model weights or create an ONNX inference session. It reports
model profile metadata (including language and voice compatibility) and runtime status;
offline mode avoids refreshing remote metadata. The result is a capability inventory,
not a synthesis guarantee if the required model assets are not installed or cannot be
reached. See [`models_and_languages.py`](../examples/models_and_languages.py) for
inventory and optional selected-model synthesis.

Reference-only profiles report `voice_mode="reference"`, an empty `voices` tuple, no
default voice, `supports_reference_enrollment=True`, and `speed_supported=False`. Use
`KokoroSynthesizer.enroll_voice()` before synthesis with such a profile. See the
[English reference voice guide](reference_voice.md).
