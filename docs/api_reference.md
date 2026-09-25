# API reference

The supported public API represents one prepared synthesis request and one independent
rendered result. Document planning and composition APIs are intentionally not part of
the package.

## Synthesizer

```{eval-rst}
.. autoclass:: pykokoro.synthesizer.KokoroSynthesizer
   :members:
   :undoc-members:
```

The main methods are:

- `synthesize(request)` renders one `SynthesisRequest` (`SynthesisSegment` remains a
  compatibility alias).
- `synthesize_text(text, language=..., voice=...)` creates and renders one plain
  prepared text request.
- `synthesize_segments(segments)` yields independent results in input order.
- `prepare(segment)` returns model-ready Kokoro frontend data for diagnostics and
  advanced integrations.
- `close()` releases the owned renderer resources.

## Request and result types

```{eval-rst}
.. autoclass:: pykokoro.synthesis_types.SynthesisRequest
   :members:
   :undoc-members:


.. autoclass:: pykokoro.synthesis_types.SynthesisSegment
   :members:
   :undoc-members:

.. autoclass:: pykokoro.synthesis_types.PronunciationOverride
   :members:
   :undoc-members:

.. autoclass:: pykokoro.synthesis_types.LinguisticToken
   :members:
   :undoc-members:

.. autoclass:: pykokoro.synthesis_types.RenderedSegment
   :members:
   :undoc-members:

.. autoclass:: pykokoro.types.WordTiming
   :members:
   :undoc-members:
```

`SynthesisSegment` offsets are half-open Python character ranges into its exact `text`.
`RenderedSegment.audio` is one-dimensional float32 audio, `sample_rate` is positive, and
word timings are local to that result. `save_wav(path)` writes mono float32 WAV audio.

## Configuration

```{eval-rst}
.. autoclass:: pykokoro.synthesis_config.SynthesisConfig
   :members:
   :undoc-members:

.. autoclass:: pykokoro.synthesis_identity.SynthesisIdentity
   :members:
   :undoc-members:


.. autoclass:: pykokoro.exceptions.SynthesisInputTooLongError
   :members:
   :undoc-members:

.. autoclass:: pykokoro.generation_config.GenerationConfig
   :members:
   :undoc-members:

.. autoclass:: pykokoro.language_routing.LanguageRoutingConfig
   :members:
   :undoc-members:

.. autoclass:: pykokoro.voice_level.VoiceLevelConfig
   :members:
   :undoc-members:

.. autoclass:: pykokoro.voice_level.VoiceLevelApplication
   :members:
   :undoc-members:

.. autoclass:: pykokoro.short_sentence_handler.ShortSentenceConfig
   :members:
   :undoc-members:
```

Each synthesis request is atomic. Capacity is checked after frontend tokenization;
oversized input raises `SynthesisInputTooLongError`. `SynthesisConfig.long_text_split`
is migration-only, accepts only `"none"`, and rejects legacy splitting modes.
`RenderedSegment.synthesis_identity` and `voice_level_applications` expose resolved
output identity and calibration outcomes. `GenerationConfig.speed` is the acoustic
inference speed passed to Kokoro, not an application-level playback-rate effect.
`SynthesisConfig.voice_level` is engine-local voice calibration rather than whole-output
mastering.

## Voice blends and discovery

```{eval-rst}
.. autoclass:: pykokoro.voice_manager.VoiceBlend
   :members:
   :undoc-members:

.. autofunction:: pykokoro.discovery.discover_models

.. autofunction:: pykokoro.lexicon_discovery.discover_lexicons
```

`discover_models()` describes runtime capabilities without initializing ONNX inference
or downloading model weights. `discover_lexicons()` describes available named G2P
lexicons.

## Frontend configuration and model asset progress

```{eval-rst}
.. autoclass:: pykokoro.tokenizer.TokenizerConfig
   :members:
   :undoc-members:

.. autoclass:: pykokoro.tokenizer.EspeakConfig
   :members:
   :undoc-members:

.. autoclass:: pykokoro.asset_progress.AssetProgressEvent
   :members:
   :undoc-members:
```

For installation, request examples, and the breaking migration boundary, see the
[quickstart](quickstart.md), [advanced features](advanced_features.md), and
[release note](breaking-change-0.10.0.md).
