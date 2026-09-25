# Languages and model profiles

Every `SynthesisSegment` supplies an explicit pronunciation language.
`GenerationConfig.lang` can provide the default for `synthesize_text()`, but a voice
name does not select language and PyKokoro does not inspect document headers or markup
for it.

## Prepared-text languages

The native KokoroG2P path supports these language codes and aliases:

- English: `en-us`, `en-gb`, and `en`
- Spanish: `es`
- French: `fr`, `fr-fr`
- German: `de`
- Italian: `it`
- Portuguese: `pt`, `pt-pt`
- Korean: `ko`
- Japanese: `ja`
- Chinese: `zh`, `cmn`
- Arabic: `ar`
- Hebrew: `he`
- Kazakh: `kk`
- Swedish: `sv`
- Thai: `th`
- Vietnamese: `vi`

Additional G2P languages may be available when the caller explicitly selects an
appropriate frontend backend, for example eSpeak or Goruut. The acoustic model must
still be compatible with the requested language and voice.

## Pronunciation-language spans

Use source-aligned overrides when one prepared string contains pronunciation material
that should use another language:

```python
from pykokoro import PronunciationOverride, SynthesisSegment

request = SynthesisSegment(
    id="mixed",
    text="Hello Welt.",
    language="en-us",
    voice="af_sarah",
    pronunciation_overrides=(PronunciationOverride(6, 10, language="de"),),
)
```

This affects G2P for the indicated text range. It does not switch the acoustic model or
voice. For complete multilingual utterances that need different acoustic profiles,
submit separate requests and let the caller manage any composition.

## Automatic pronunciation routing

`LanguageRoutingConfig(mode="auto", languages=(...))` lets KokoroG2P consider an
explicit candidate set while the request retains its required main language. Routing is
optional; it is not document-language detection, voice inference, or an acoustic-model
switch. Explicit source-aligned language information remains attached to the request.

## Model and voice discovery

Model defaults are resolved from the request's language and selected voice. To inspect
available profiles before synthesis, use the metadata-only discovery API:

```python
from pykokoro import discover_models

for model in discover_models(offline=True).models:
    print(model.model_id, model.languages, model.voices, model.status)
```

The inventory reports runtime status, voices, languages, qualities, frontend
information, and distribution metadata without loading model weights. See the
[model discovery example](../examples/models_and_languages.py) for optional synthesis of
a selected model.
