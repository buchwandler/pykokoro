# Breaking change: request-centric synthesis (planned 0.10.0)

**Status: planned / unreleased.** The upcoming breaking release reduces PyKokoro to a
Kokoro speech-synthesis engine. It accepts prepared speech requests and returns
independent rendered waveforms. This is an architectural break; no compatibility
aliases, legacy request parsing, or migration adapters will be provided inside PyKokoro.

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

Example:

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
- Model-limit chunks and short-sentence retries remain internal to one synthesis request
  and are stitched into its single result.
- `GenerationConfig.speed` remains an acoustic inference control. Editorial playback
  rate, pauses, pitch, gain, mastering, and timeline assembly belong to the caller.
- `RenderedSegment.save_wav()` writes one mono float32 WAV without AudioCompose.
- Removed configuration, serialization fields, and old names are rejected by absence;
  they are not accepted and ignored.

## Dependency changes

The package no longer declares Utterplan, AudioCompose, or text-file encoding detectors
as runtime dependencies. KokoroG2P, OnnxVoice, Lexphon, NumPy, audiosig, and soundfile
remain for the engine paths that use them. The supported installed package does not
require SSMD, Utterplan, or AudioCompose to import and use the synthesis API.

See the [quickstart](quickstart.md), [request examples](examples.md), and
[API reference](api_reference.md) for the new boundary.
