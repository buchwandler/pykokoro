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
    RequestMeasure,
    VoiceEnrollerSpec,
    KokoroSynthesizer,
    SynthesisConfig,
    SayAsSpan,
    SynthesisRequest,
)
```

## Public root symbols

| Area                      | Root exports                                                                                                                                                                                                                                                                                                                                                                        |
| ------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Synthesis                 | `KokoroSynthesizer`, `SynthesisRequest`, `SynthesisSegment`, `RenderedSegment`                                                                                                                                                                                                                                                                                                      |
| Request measurement       | `RequestMeasure`, `KokoroSynthesizer.measure_request()`                                                                                                                                                                                                                                                                                                                             |
| Source context and timing | `PronunciationOverride`, `LinguisticToken`, `SayAsSpan`, `WordTiming`                                                                                                                                                                                                                                                                                                               |
| Configuration             | `SynthesisConfig`, `GenerationConfig`, `LanguageRoutingConfig`, `TokenizerConfig`, `ShortSentenceConfig`, `LongTextSplitMode`                                                                                                                                                                                                                                                       |
| Voice level               | `VoiceLevelConfig`, `VoiceLevelApplication`, `VoiceBlend`                                                                                                                                                                                                                                                                                                                           |
| Voice enrollment          | `KokoroVoicePack`, `ReferenceVoice`, `InnoEnrollmentOptions`, `KokoroSynthesizer.enroll_voice()`                                                                                                                                                                                                                                                                                    |
| Model discovery           | `ModelCapabilities`, `ModelDiscoveryResult`, `VoiceCapabilities`, `VoiceEnrollerSpec`, `discover_models`                                                                                                                                                                                                                                                                            |
| Lexicon discovery         | `LexiconCapabilities`, `LexiconDiscoveryResult`, `discover_lexicons`                                                                                                                                                                                                                                                                                                                |
| Local G2Lex inputs        | `G2LexFileRef`, `G2LexFileInspection`, `inspect_g2lex_file`                                                                                                                                                                                                                                                                                                                         |
| Asset progress            | `AssetProgressEvent`, `AssetProgressCallback`, `ConsoleAssetProgress`                                                                                                                                                                                                                                                                                                               |
| Synthesis identity        | `SynthesisIdentity`, `build_synthesis_identity`                                                                                                                                                                                                                                                                                                                                     |
| API contract              | `REQUEST_API_VERSION`, `RequestApiContract`, `request_api_contract`                                                                                                                                                                                                                                                                                                                 |
| Errors                    | `KokoroError`, `PyKokoroError`, `SynthesisError`, `ConfigurationError`, `InvalidRequestError`, `EmptyTextError`, `InvalidLanguageError`, `InvalidVoiceError`, `InvalidModelError`, `InvalidPronunciationError`, `InvalidLinguisticTokensError`, `UnsupportedFeatureError`, `CapabilityError`, `SynthesisStateError`, `AlignmentError`, `BackendError`, `SynthesisInputTooLongError` |
| Error diagnostics         | `ERROR_DIAGNOSTICS_VERSION`, `error_diagnostic()`                                                                                                                                                                                                                                                                                                                                   |
| Version                   | `__version__`, `__version_tuple__`                                                                                                                                                                                                                                                                                                                                                  |

`SynthesisSegment` is an alias of `SynthesisRequest`, and `PyKokoroError` is a
compatibility alias of `KokoroError`. `AssetProgressCallback`, `LongTextSplitMode`,
`VoiceConditioning`, and `VoiceEnrollmentEngine` are type aliases; they describe
accepted Python values rather than runtime record classes.

## Request API compatibility contract

`pykokoro.REQUEST_API_VERSION` identifies the request API contract family. Call
`pykokoro.request_api_contract()` to inspect its stable capability declaration before
constructing an engine:

```python
import pykokoro

contract = pykokoro.request_api_contract()
assert contract.version == pykokoro.REQUEST_API_VERSION == 2
assert contract.request_type == "SynthesisRequest"
assert contract.result_type == "RenderedSegment"
assert contract.measurement_method == "measure_request"
assert contract.voice_enrollment_method == "enroll_voice"
assert contract.supports_request_measurement and contract.supports_reference_voice
assert contract.supports_say_as and contract.say_as_interpretations == ("characters",)
assert contract.supports_voice_enrollment
```

The declaration is dependency-light: it does not initialize a G2P frontend, model
registry, ONNX session, or audio backend. Resolving `pykokoro.KokoroSynthesizer` loads
the class without constructing its G2P adapter; constructing the synthesizer or
rendering a request is a separate runtime operation. This versioned declaration gives
integrations a stable compatibility probe instead of requiring them to infer a contract
from implementation details.

Each `SynthesisRequest` is one caller-prepared atomic text boundary. Parsing, shaping,
and splitting documents remain caller-owned; the default strict token-capacity behavior
raises `SynthesisInputTooLongError` instead of silently changing that boundary.

## Synthesizer

```{eval-rst}
.. autoclass:: pykokoro.KokoroSynthesizer
   :members:
   :undoc-members:
```

Use it as a context manager or call `close()` when finished. `synthesize()` renders one
request, `synthesize_text()` constructs and renders one prepared string, and
`synthesize_segments()` yields independent results in input order. `measure_request()`
runs the request frontend and returns a `RequestMeasure` token count without rendering;
unlike `prepare()`, it can resolve a reference-only target without a `ReferenceVoice`.
`prepare()` runs the frontend and returns `PreparedSynthesis` (request ID, text,
language, voice, phonemes, token IDs, alignment tokens, and diagnostics) without running
ONNX inference.

## Requests, results, and source alignment

```{eval-rst}
.. autoclass:: pykokoro.SayAsSpan
   :members:
   :undoc-members:


.. autoclass:: pykokoro.SynthesisRequest
   :members:
   :undoc-members:


.. autoclass:: pykokoro.RequestMeasure
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

`SayAsSpan` marks an exact half-open range in `SynthesisRequest.text`. The currently
supported interpretation is `characters`; only marked spans are projected to KokoroG2P's
shadow frontend (for example, canonical `A R T` may be sent as `A- R- T`). The request,
prepared text, public alignment, and word timings stay on canonical text and offsets. A
marked phrase remains inside the caller's request; say-as does not add pauses or itself
split text. `RequestMeasure.amount` is the exact frontend token count. `maximum` is an
optional caller-provided limit, and `fits` is true when the amount is within that limit,
false when it exceeds it, and `None` when no limit was supplied. Measurement does not
imply that reference audio is valid or enrollment will succeed.

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

.. autoclass:: pykokoro.G2LexFileRef
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
By default, `random_seed=None` requests no specific seed and allows the selected runtime
to apply its own default; an explicit integer requests that seed where supported.
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

.. autoclass:: pykokoro.G2LexFileInspection :members: :undoc-members:

.. autofunction:: pykokoro.inspect_g2lex_file

Discovery reports metadata and runtime capability, including `VoiceEnrollerSpec`,
without loading synthesis weights or creating an ONNX session. Offline discovery uses
available local metadata; it does not install model or lexicon assets. Use the inventory
to select an available model/language/voice rather than inferring compatibility from a
voice name.

`discover_lexicons()` covers registered selectors. For a local file, call
`inspect_g2lex_file(path)` to validate and read its manifest without traversing the
pronunciation entries; `inspection.to_ref()` creates an immutable reference pinned to
the content and logical hashes. Add one or more references through
`TokenizerConfig(g2lex_files=(ref,))`. Local sources are forwarded to KokoroG2P and
never enter Lexphon's named-asset installer. Prepared alignment metadata retains the
contextual heteronym decision and its reason when one was selected.

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

## Structured error diagnostics

`ERROR_DIAGNOSTICS_VERSION` versions this additive error-reporting API independently of
`REQUEST_API_VERSION`. `error_diagnostic(error, max_depth=8)` returns a bounded,
JSON-compatible mapping with exception type, module, message, optional semantic stage
and details, native error fields for `PyKokoroError`, and a cause-first `cause` or
fallback `context` chain. It detects cycles and marks a chain `truncated` when the depth
limit is reached. It does not serialize tracebacks or local variables.

PyKokoro uses the stable stages `frontend`, `model_install`, `model_open`,
`reference_enrollment`, `voice_enrollment`, `inference`, `postprocess`, `synthesis`, and
`close`. Engine-generated details use request/model identifiers, operation names, and
reference fingerprints rather than raw request text, transcripts, audio arrays,
phonemes, or token arrays. Native exception messages are retained and may themselves
contain a local path.

.. autodata:: pykokoro.ERROR_DIAGNOSTICS_VERSION

.. autofunction:: pykokoro.error_diagnostic

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
