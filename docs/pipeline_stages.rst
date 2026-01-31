Pipeline Usage and Stages
=========================

``KokoroPipeline`` is the configurable engine behind the high-level
``Kokoro`` class. Use it when you want to swap parsing/segmentation stages,
run custom G2P logic, or control model loading at a lower level.

Pipeline overview
-----------------

The default pipeline wiring is:

``doc_parser -> g2p -> phoneme_processing -> audio_generation -> audio_postprocessing``

Default stage classes:

* ``SsmdDocumentParser``
* ``KokoroG2PAdapter``
* ``OnnxPhonemeProcessorAdapter``
* ``OnnxAudioGenerationAdapter``
* ``OnnxAudioPostprocessingAdapter``

If any of the audio stages are omitted, the pipeline builds a ``Kokoro`` ONNX
backend and wires the missing adapters automatically.

Quick start
-----------

.. code-block:: python

   from pykokoro import GenerationConfig, KokoroPipeline, PipelineConfig

   config = PipelineConfig(
       voice="af_bella",
       generation=GenerationConfig(speed=1.0),
   )
   pipeline = KokoroPipeline(config)
   result = pipeline.run("Hello from the pipeline.")
   result.save_wav("output.wav")

   # Inspect intermediates
   segments = result.segments
   phoneme_segments = result.phoneme_segments

   # Enable trace details when needed
   traced = pipeline.run("Hello", return_trace=True)
   if traced.trace:
       print(traced.trace.warnings)

Configuration
-------------

``PipelineConfig`` and ``GenerationConfig`` are frozen dataclasses. Use
``dataclasses.replace`` when you want a modified copy.

.. code-block:: python

   from dataclasses import replace
   from pykokoro import GenerationConfig, KokoroPipeline, PipelineConfig

   cfg = PipelineConfig(voice="af_bella")
   faster_cfg = replace(cfg, generation=replace(cfg.generation, speed=1.2))
   pipeline = KokoroPipeline(faster_cfg)

PipelineConfig fields
~~~~~~~~~~~~~~~~~~~~~

Core
^^^^

* ``voice``: Default voice name (``str``) or ``VoiceBlend`` used unless SSMD
  metadata overrides the voice per segment.
* ``generation``: ``GenerationConfig`` instance with speed, language, pause
  handling, and phoneme controls.

Model and provider
^^^^^^^^^^^^^^^^^^

* ``model_quality``: ``"fp32"``, ``"fp16"``, ``"fp16-gpu"``, ``"q8"``,
  ``"q8f16"``, ``"q4"``, ``"q4f16"``, ``"uint8"``, ``"uint8f16"``.
  ``None`` uses the backend default.
* ``model_source``: ``"huggingface"`` or ``"github"``.
* ``model_variant``: ``"v1.0"`` or ``"v1.1-zh"``.
* ``model_path``: Path to a local ONNX model file. Overrides model download.
* ``voices_path``: Path to a local voices file. Overrides voice download.
* ``provider``: ONNX provider name (``"auto"``, ``"cpu"``, ``"cuda"``,
  ``"openvino"``, ``"directml"``, ``"coreml"``).
* ``provider_options``: Dict of provider/session options passed to ONNX Runtime.
* ``session_options``: Pre-built ``onnxruntime.SessionOptions`` (advanced use).

Tokenizer and phoneme handling
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* ``tokenizer_config``: ``TokenizerConfig`` used by SSMD parsing and ``kokorog2p``.
* ``espeak_config``: Deprecated espeak configuration. Prefer ``TokenizerConfig``.
* ``short_sentence_config``: ``ShortSentenceConfig`` for short-sentence handling.
* ``overlap_mode``: ``"snap"`` clips overlapping SSMD spans to segment bounds,
  ``"strict"`` drops partial spans and emits trace warnings.

Other
^^^^^

* ``return_trace``: Include ``Trace`` in ``AudioResult`` with timings/warnings.
* ``enable_deprecation_warnings``: Reserved for compatibility warnings.
* ``cache_dir``: Directory for the G2P disk cache (JSON files). Set ``None``
  to disable caching.

GenerationConfig fields
~~~~~~~~~~~~~~~~~~~~~~~

* ``speed``: Speech rate multiplier (``1.0`` is normal).
* ``lang``: Default language code for phonemization (``"en-us"`` etc).
* ``is_phonemes``: Treat input text as phoneme strings instead of raw text.
* ``pause_mode``: ``"tts"`` keeps natural model pauses, ``"manual"`` trims
  segment silence and preserves explicit pauses, ``"auto"`` inserts pauses
  at sentence/paragraph boundaries and trims segment silence.
* ``pause_clause``: Default pause for SSMD ``...c`` breaks (seconds).
* ``pause_sentence``: Default pause for SSMD ``...s`` breaks (seconds).
* ``pause_paragraph``: Default pause for SSMD ``...p`` breaks (seconds).
* ``pause_variance``: Stored for compatibility with the ``Kokoro`` API.
  The pipeline stages do not currently apply variance.
* ``random_seed``: Stored for compatibility with the ``Kokoro`` API.
  The pipeline stages do not currently use the seed.
* ``enable_short_sentence``: Override short sentence handling for the run.

Runtime overrides
-----------------

``KokoroPipeline.run`` accepts overrides for any ``PipelineConfig`` field. The
``lang`` keyword is special-cased to update ``generation.lang`` for convenience.

.. code-block:: python

   from dataclasses import replace
   from pykokoro import GenerationConfig

   # Override just the language
   result = pipeline.run("Bonjour", lang="fr")

   # Override generation settings per call
   manual = replace(
       pipeline.config.generation,
       pause_mode="manual",
       pause_sentence=0.5,
   )
   result = pipeline.run("Hello...s world", generation=manual)

   # Override model settings per call
   result = pipeline.run("Quick test", model_quality="q8")

Stage behavior
--------------

SSMD document parser
~~~~~~~~~~~~~~~~~~~~

``SsmdDocumentParser`` uses ``parse_ssmd_to_segments`` to turn SSMD markup into
clean text plus metadata spans, pause boundaries, and sentence/paragraph
segments. It honors ``generation.pause_*`` values when converting break
strengths into durations.

Supported SSMD features include:

* Break markers: ``...c``, ``...s``, ``...p``, ``...500ms``
* Language overrides: ``[Bonjour](fr)``
* Phoneme overrides: ``[tomato](ph: t eh m aa t ow)``
* Prosody markup (rate/pitch/volume) and emphasis
* Voice markers: ``[Hello]{voice="af_sarah"}``

The parser attaches SSMD metadata to annotation spans so later stages can
select per-segment voices and prosody.

Plain text sentence splitting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``PlainTextDocumentParser`` uses the optional ``phrasplit`` package for
sentence splitting. When ``phrasplit`` is unavailable, it falls back to a
single segment. The language model is derived from ``generation.lang`` (for
example ``en_core_web_sm`` for English).

Split boundaries are forced at SSMD pause boundaries and at spans that contain
phoneme overrides so those overrides are kept intact. Set
``PYKOKORO_DEBUG_SEGMENTS=1`` to log segment offsets.

Kokoro G2P adapter
~~~~~~~~~~~~~~~~~~

``KokoroG2PAdapter`` uses the ``kokorog2p`` package to produce phonemes and
token IDs.

* ``generation.lang`` selects the G2P language.
* ``generation.is_phonemes`` treats input as phonemes and skips text G2P.
* SSMD ``ph``/``phonemes`` spans override phonemes for that segment.
* ``tokenizer_config`` is forwarded to ``kokorog2p.get_g2p``.
* ``cache_dir`` enables on-disk caching of phonemes/tokens.
* Long phoneme token sequences are split into batches of
  ``MAX_PHONEME_LENGTH``.

Onnx phoneme processing
~~~~~~~~~~~~~~~~~~~~~~~

``OnnxPhonemeProcessorAdapter`` calls the ONNX backend to normalize tokens,
skip empty segments, and apply short-sentence handling.

* ``short_sentence_config`` controls defaults for short sentence handling.
* ``generation.enable_short_sentence`` can override the config per run.

Onnx audio generation
~~~~~~~~~~~~~~~~~~~~~

``OnnxAudioGenerationAdapter`` generates raw audio per phoneme segment.

* ``voice`` provides the default voice style.
* SSMD voice metadata (``voice``/``voice_name``) overrides the voice per segment.
* ``generation.speed`` controls synthesis speed.

Onnx audio postprocessing
~~~~~~~~~~~~~~~~~~~~~~~~~

``OnnxAudioPostprocessingAdapter`` trims silence and concatenates segments.

* ``generation.pause_mode"`` set to ``"manual"`` or ``"auto"`` enables silence
  trimming before inserting explicit pauses.
* SSMD prosody metadata (rate/pitch/volume) is applied to each segment.
* ``pause_before``/``pause_after`` values from G2P are inserted between segments.

Customizing the pipeline
------------------------

You can replace individual stages or use the provided no-op adapters.
The showcase script demonstrates multiple wiring styles:

``examples/pipeline_stage_showcase.py``

Example with explicit stage wiring:

.. code-block:: python

   from pykokoro import GenerationConfig, KokoroPipeline, PipelineConfig
   from pykokoro.onnx_backend import Kokoro
   from pykokoro.stages.audio_generation.onnx import OnnxAudioGenerationAdapter
   from pykokoro.stages.audio_postprocessing.onnx import OnnxAudioPostprocessingAdapter
   from pykokoro.stages.doc_parsers.ssmd import SsmdDocumentParser
   from pykokoro.stages.g2p.kokorog2p import KokoroG2PAdapter
   from pykokoro.stages.phoneme_processing.onnx import OnnxPhonemeProcessorAdapter

   cfg = PipelineConfig(
       voice="af_heart",
       generation=GenerationConfig(lang="en-us"),
   )
   kokoro = Kokoro(model_quality=cfg.model_quality)

   pipeline = KokoroPipeline(
       cfg,
       doc_parser=SsmdDocumentParser(),
       g2p=KokoroG2PAdapter(),
       phoneme_processing=OnnxPhonemeProcessorAdapter(kokoro),
       audio_generation=OnnxAudioGenerationAdapter(kokoro),
       audio_postprocessing=OnnxAudioPostprocessingAdapter(kokoro),
   )

Local model files and providers
-------------------------------

To load local ONNX artifacts, set ``model_path`` and ``voices_path``.
You can also select a specific execution provider.

.. code-block:: python

   from pathlib import Path
   from pykokoro import KokoroPipeline, PipelineConfig

   cfg = PipelineConfig(
       voice="af_bella",
       model_path=Path("/models/kokoro.onnx"),
       voices_path=Path("/models/voices.bin"),
       provider="cuda",
       provider_options={"device_id": 0},
   )
   pipeline = KokoroPipeline(cfg)
   result = pipeline.run("Hello from local files.")
