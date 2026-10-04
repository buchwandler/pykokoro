# API reference

PyKokoro's supported API is exported by `pykokoro` and listed in `pykokoro.__all__`. It
models prepared synthesis requests and their independent rendered results; document
parsing, speech planning, and composition between requests remain caller-owned.

```python
from pykokoro import (
    GenerationConfig,
    InnoEnrollmentOptions,
    KokoroVoicePack,
    ReferenceVoice,
    VoiceEnrollerSpec,
    KokoroSynthesizer,
    SynthesisConfig,
    SynthesisRequest,
)
```

## Public root symbols

| Area                      | Root exports                                                                                                                                                                                                                                                                                                                                                                        |
| ------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Synthesis                 | `KokoroSynthesizer`, `SynthesisRequest`, `SynthesisSegment`, `RenderedSegment`                                                                                                                                                                                                                                                                                                      |
| Source context and timing | `PronunciationOverride`, `LinguisticToken`, `WordTiming`                                                                                                                                                                                                                                                                                                                            |
| Configuration             | `SynthesisConfig`, `GenerationConfig`, `LanguageRoutingConfig`, `TokenizerConfig`, `ShortSentenceConfig`, `LongTextSplitMode`                                                                                                                                                                                                                                                       |
| Voice level               | `VoiceLevelConfig`, `VoiceLevelApplication`, `VoiceBlend`                                                                                                                                                                                                                                                                                                                           |
| Voice enrollment          | `KokoroVoicePack`, `ReferenceVoice`, `InnoEnrollmentOptions`, `KokoroSynthesizer.enroll_voice()`                                                                                                                                                                                                                                                                                    |
| Model discovery           | `ModelCapabilities`, `ModelDiscoveryResult`, `VoiceCapabilities`, `VoiceEnrollerSpec`, `discover_models`                                                                                                                                                                                                                                                                            |
| Lexicon discovery         | `LexiconCapabilities`, `LexiconDiscoveryResult`, `discover_lexicons`                                                                                                                                                                                                                                                                                                                |
| Asset progress            | `AssetProgressEvent`, `AssetProgressCallback`, `ConsoleAssetProgress`                                                                                                                                                                                                                                                                                                               |
| Synthesis identity        | `SynthesisIdentity`, `build_synthesis_identity`                                                                                                                                                                                                                                                                                                                                     |
| Errors                    | `KokoroError`, `PyKokoroError`, `SynthesisError`, `ConfigurationError`, `InvalidRequestError`, `EmptyTextError`, `InvalidLanguageError`, `InvalidVoiceError`, `InvalidModelError`, `InvalidPronunciationError`, `InvalidLinguisticTokensError`, `UnsupportedFeatureError`, `CapabilityError`, `SynthesisStateError`, `AlignmentError`, `BackendError`, `SynthesisInputTooLongError` |
| Version                   | `__version__`, `__version_tuple__`                                                                                                                                                                                                                                                                                                                                                  |

`SynthesisSegment` is an alias of `SynthesisRequest`, and `PyKokoroError` is a
compatibility alias of `KokoroError`. `AssetProgressCallback`, `LongTextSplitMode`,
`VoiceConditioning`, and `VoiceEnrollmentEngine` are type aliases; they describe
accepted Python values rather than runtime record classes.

## Synthesizer

```{eval-rst}
.. autoclass:: pykokoro.KokoroSynthesizer
   :members:
   :undoc-members:
```

Use it as a context manager or call `close()` when finished. `synthesize()` renders one
request, `synthesize_text()` constructs and renders one prepared string, and
`synthesize_segments()` yields independent results in input order. `prepare()` runs the
frontend and returns `PreparedSynthesis` (request ID, text, language, voice, phonemes,
token IDs, alignment tokens, and diagnostics) without running ONNX inference.

## Requests, results, and source alignment

```{eval-rst}
.. autoclass:: pykokoro.SynthesisRequest
   :members:
   :undoc-members:

.. autoclass:: pykokoro.PronunciationOverride
   :members:
   :undoc-members:

.. autoclass:: pykokoro.LinguisticToken
   :members:
   :undoc-members:

.. autoclass:: pykokoro.RenderedSegment
   :members:
   :undoc-members:

.. autoclass:: pykokoro.WordTiming
   :members:
   :undoc-members:
```

`SynthesisSegment` is the supported compatibility spelling for `SynthesisRequest`.
Request language and voice are explicit. `PronunciationOverride` and `LinguisticToken`
offsets are half-open character ranges into the exact `text` carried by that request;
`tokens` is the canonical `SynthesisRequest` field.

`RenderedSegment` contains one request's waveform and metadata. `save_wav()` writes mono
float32 WAV audio. `play()` uses the optional playback dependency. Results from a batch
remain separate: this API does not join independent requests or insert silence between
them.

## Configuration

```{eval-rst}
.. autoclass:: pykokoro.SynthesisConfig
   :members:
   :undoc-members:

.. autoclass:: pykokoro.GenerationConfig
   :members:
   :undoc-members:

.. autoclass:: pykokoro.LanguageRoutingConfig
   :members:
   :undoc-members:

.. autoclass:: pykokoro.TokenizerConfig
   :members:
   :undoc-members:

.. autoclass:: pykokoro.ShortSentenceConfig
   :members:
   :undoc-members:
```

## Voice enrollment and packs

```{eval-rst}
.. autoclass:: pykokoro.KokoroVoicePack
   :members:
   :undoc-members:

.. autoclass:: pykokoro.InnoEnrollmentOptions
   :members:
   :undoc-members:

.. autoclass:: pykokoro.ReferenceVoice
   :members:
   :undoc-members:
```

`KokoroSynthesizer.enroll_voice()` defaults to `engine="inno"`. It accepts reference
audio without a transcript and returns a portable `KokoroVoicePack` for ordinary static
Kokoro synthesis. The versioned NPZ save/load format is pickle-free. Inno enrollment
requires a runtime-supported `inno-v0.2` model capability. `VoiceEnrollerSpec` discovery
metadata reports its requirements and output format without loading model weights.

AkinVox remains a separate, explicit path. Use `engine="akinvox"` and provide the exact
transcript to return a model-bound `ReferenceVoice`. The voice pack and reference state
are not interchangeable. See the [voice enrollment guide](reference_voice.md) for engine
requirements, audio limits, persistence, calibration limitations, privacy, and examples.

`LongTextSplitMode` is `"none" | "sentence"`. The default `"none"` path raises
`SynthesisInputTooLongError` for an oversized request. `"sentence"` lazily loads
PhraseSplit only when a request exceeds capacity, then returns one result for the
original request. `long_text_use_spacy=False` selects PhraseSplit's simple mode without
spaCy.

`GenerationConfig` controls Kokoro acoustic speed, the default language for
`synthesize_text()`, inference random seed, and the per-request short-sentence override.
Short-sentence handling is disabled unless explicitly enabled with `ShortSentenceConfig`
or `GenerationConfig.enable_short_sentence`.

## Voice blending and calibration

```{eval-rst}
.. autoclass:: pykokoro.VoiceBlend
   :members:
   :undoc-members:

.. autoclass:: pykokoro.VoiceLevelConfig
   :members:
   :undoc-members:

.. autoclass:: pykokoro.VoiceLevelApplication
   :members:
   :undoc-members:
```

A `VoiceBlend` combines supported voice IDs for one request/model profile. Voice-level
calibration is an engine-local option; it is not whole-program loudness mastering.

## Model and lexicon discovery

```{eval-rst}
.. autoclass:: pykokoro.ModelCapabilities
   :members:
   :undoc-members:

.. autoclass:: pykokoro.VoiceCapabilities
   :members:
   :undoc-members:

.. autoclass:: pykokoro.VoiceEnrollerSpec
   :members:
   :undoc-members:

.. autoclass:: pykokoro.ModelDiscoveryResult
   :members:
   :undoc-members:

.. autofunction:: pykokoro.discover_models

.. autoclass:: pykokoro.LexiconCapabilities
   :members:
   :undoc-members:

.. autoclass:: pykokoro.LexiconDiscoveryResult
   :members:
   :undoc-members:

.. autofunction:: pykokoro.discover_lexicons
```

Discovery reports metadata and runtime capability, including `VoiceEnrollerSpec`,
without loading synthesis weights or creating an ONNX session. Offline discovery uses
available local metadata; it does not install model or lexicon assets. Use the inventory
to select an available model/language/voice rather than inferring compatibility from a
voice name.

## Asset progress

```{eval-rst}
.. autoclass:: pykokoro.AssetProgressEvent
   :members:
   :undoc-members:

.. autoclass:: pykokoro.ConsoleAssetProgress
   :members:
   :undoc-members:
```

`AssetProgressCallback` is a callable receiving `AssetProgressEvent` values. Pass a
callback or `ConsoleAssetProgress()` with `SynthesisConfig.asset_progress`. Progress
notifications cover managed model assets, not lexicon data provisioning.

## Synthesis identity

```{eval-rst}
.. autoclass:: pykokoro.SynthesisIdentity
   :members:
   :undoc-members:

.. autofunction:: pykokoro.build_synthesis_identity
```

The identity captures output-affecting synthesis settings and exposes a stable cache
key. Building identity metadata does not run inference.

## Exceptions

```{eval-rst}
.. autoclass:: pykokoro.KokoroError
   :members:
   :undoc-members:

.. autoclass:: pykokoro.SynthesisError
   :members:
   :undoc-members:

.. autoclass:: pykokoro.ConfigurationError
   :members:
   :undoc-members:

.. autoclass:: pykokoro.InvalidRequestError
   :members:
   :undoc-members:

.. autoclass:: pykokoro.EmptyTextError
   :members:
   :undoc-members:

.. autoclass:: pykokoro.InvalidLanguageError
   :members:
   :undoc-members:

.. autoclass:: pykokoro.InvalidVoiceError
   :members:
   :undoc-members:

.. autoclass:: pykokoro.InvalidModelError
   :members:
   :undoc-members:

.. autoclass:: pykokoro.InvalidPronunciationError
   :members:
   :undoc-members:

.. autoclass:: pykokoro.InvalidLinguisticTokensError
   :members:
   :undoc-members:

.. autoclass:: pykokoro.UnsupportedFeatureError
   :members:
   :undoc-members:

.. autoclass:: pykokoro.CapabilityError
   :members:
   :undoc-members:

.. autoclass:: pykokoro.SynthesisStateError
   :members:
   :undoc-members:

.. autoclass:: pykokoro.AlignmentError
   :members:
   :undoc-members:

.. autoclass:: pykokoro.BackendError
   :members:
   :undoc-members:

.. autoclass:: pykokoro.SynthesisInputTooLongError
   :members:
   :undoc-members:
```

Catch the narrow exception that an application can act on; avoid treating every failure
as a recoverable input error. `PyKokoroError` is the retained alias for `KokoroError`.

## Version

`pykokoro.__version__` is the package's version string and `pykokoro.__version_tuple__`
is its parsed tuple. The package version is generated by setuptools-scm.

## Related guides

- [Quickstart](quickstart.md)
- [Configuration and request behavior](basic_usage.md)
- [Prepared text, routing, and result metadata](advanced_features.md)
- [Languages and model profiles](languages.md)
- [Installation and asset progress](installation.md)
