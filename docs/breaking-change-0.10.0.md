# Breaking change: request-centric synthesis (v0.10.0)

This note describes PyKokoro's request-centric API boundary for v0.10.0. PyKokoro
accepts prepared speech requests and returns independent rendered waveforms. Document
parsing, speech planning, and cross-request composition belong to the caller. The old
pipeline and document APIs are removed; the deliberate compatibility aliases retained by
the new API are listed below.

## Responsibility change

| Previous ownership                                                 | New ownership                                                                             |
| ------------------------------------------------------------------ | ----------------------------------------------------------------------------------------- |
| PyKokoro parsed documents, SSMD, and planned speech units          | The caller prepares speakable text and resolves document/planning semantics               |
| PyKokoro resolved logical voice roles                              | The caller supplies an actual Kokoro voice or `VoiceBlend`                                |
| PyKokoro applied document pauses, emphasis, and integrated prosody | The caller's speech plan and composition layer own editorial effects and timeline silence |
| PyKokoro returned composed document audio / `AudioJob`             | PyKokoro returns one `RenderedSegment`; the caller decides if and how to compose results  |
| PyKokoro depended on Utterplan and AudioCompose                    | Neither package is a PyKokoro runtime dependency                                          |

SSMD text, front matter, and directives are no longer interpreted. Text passed to
PyKokoro is prepared speech text; it may undergo only the normalizations owned by
KokoroG2P's prepared-text contract. An input string that happens to contain markup
remains ordinary supplied text.

## API mapping

| Removed API                                                   | Replacement                                                                           |
| ------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| `KokoroPipeline`                                              | `KokoroSynthesizer`                                                                   |
| `PipelineConfig`                                              | `SynthesisConfig`                                                                     |
| `run(text)`                                                   | `synthesize_text(text, language=..., voice=...)` for one plain request                |
| `run_plan(plan)` / plan adapters                              | Convert each upstream prepared unit to a `SynthesisSegment`, then call `synthesize()` |
| Document `AudioResult`                                        | One request-local `RenderedSegment`                                                   |
| `AudioJob`, composer wrappers, and document streaming results | Caller-owned orchestration and composition outside PyKokoro                           |
| SSMD pause, emphasis, and prosody configuration               | Upstream speech-plan and final-composition policy                                     |

The deprecated `pykokoro.onnx_session` and `pykokoro.provider_config` modules are
removed. Configure provider and session options through `SynthesisConfig`; OnnxVoice
owns provider selection and ONNX session creation.

## Deliberate compatibility aliases

The new API keeps `SynthesisSegment` as an alias for `SynthesisRequest` and accepts
`annotations` as a compatibility alias for `tokens`. These aliases do not restore the
removed pipeline, document parsing, SSMD, planning, or composition APIs. Example:

```python
from pykokoro import GenerationConfig, KokoroSynthesizer, SynthesisConfig

config = SynthesisConfig(
    voice="af_sarah",
    generation=GenerationConfig(lang="en-us"),
)
with KokoroSynthesizer(config) as synthesizer:
    result = synthesizer.synthesize_text(
        "Hello world.", language="en-us", voice="af_sarah"
    )
result.save_wav("hello.wav")
```

For an orchestrated caller, provide a request ID, prepared text, explicit pronunciation
language, actual voice, and optional source-aligned pronunciation overrides or
linguistic tokens in `SynthesisSegment`. `synthesize_segments()` preserves input order
and returns independent results. It does not concatenate audio or add cross-request
silence.

## Behavior and configuration changes

- Request `language` is explicit; a voice name never supplies it.
- `PronunciationOverride` and `LinguisticToken` offsets refer to the exact request text.
- `long_text_split="none"` is the default. It never imports PhraseSplit and raises
  `SynthesisInputTooLongError` when the model-token limit is exceeded.
- `long_text_split="sentence"` loads PhraseSplit lazily and only splits oversized
  requests. It packs sentence spans to the model limit, falling back to clauses and then
  safe word boundaries as necessary. Internal chunk audio is joined into the request's
  single result.
- `long_text_use_spacy=False` selects PhraseSplit's simple mode without spaCy. `None`
  allows a compatible local spaCy model or regex fallback; `True` requires spaCy and a
  compatible model.
- A single word that cannot fit safely, or oversized whole-request phonemes that cannot
  retain source alignment, still raise `SynthesisInputTooLongError`.
- Explicit short-sentence handling remains local to one synthesis request.
- `GenerationConfig.speed` remains an acoustic inference control. Editorial playback
  rate, pauses, pitch, gain, mastering, and timeline assembly belong to the caller.
- `RenderedSegment.save_wav()` writes one mono float32 WAV without AudioCompose.
- Removed configuration, serialization fields, and old names are rejected by absence;
  they are not accepted and ignored.

## Dependency changes

PyKokoro does not declare Utterplan, AudioCompose, SSMD, or text-file encoding detectors
as runtime dependencies. KokoroG2P owns prepared-text phonemization; OnnxVoice owns
model installation, resolution, and ONNX session concerns; AudioSig supplies DSP
primitives. PyKokoro retains request rendering, voice/model profile selection, timing
reconstruction, and metadata-only `discover_models()` and `discover_lexicons()` APIs.

The supported integration floors are KokoroG2P 0.9.9, Lexphon 0.2.3, PhraseSplit 0.3.9,
AudioSig 0.1.4, and OnnxVoice 0.1.7. PhraseSplit remains a runtime dependency for opt-in
long-text splitting, but neither it nor PyKokoro requires spaCy for the default simple
path. See the [quickstart](quickstart.md), [request examples](examples.md), and
[API reference](api_reference.md) for the new boundary.
