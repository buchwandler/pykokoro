Basic Usage
===========

This guide covers the fundamental usage patterns of PyKokoro.

.. note::

   PyKokoro uses ``KokoroPipeline`` as the supported API. The pipeline wraps all
   stages (document parsing, splitting, G2P, and synthesis) behind one call.

Initializing the Pipeline
-------------------------

The main entry point is the ``KokoroPipeline`` class:

.. code-block:: python

   from pykokoro import GenerationConfig, KokoroPipeline, PipelineConfig

   # Initialize with defaults (HuggingFace v1.0)
   pipe = KokoroPipeline(PipelineConfig(voice="af_bella"))

   # Specify model source and variant
   pipe = KokoroPipeline(
       PipelineConfig(
           voice="af_bella",
           model_source="huggingface",
           model_variant="v1.0",
       )
   )

   # GitHub source
   pipe = KokoroPipeline(
       PipelineConfig(
           voice="af_bella",
           model_source="github",
           model_variant="v1.0",
       )
   )

   # Custom generation settings
   generation = GenerationConfig(lang="en-us", speed=1.1)
   pipe = KokoroPipeline(PipelineConfig(voice="af_bella", generation=generation))

Reusing the Pipeline
~~~~~~~~~~~~~~~~~~~~

Create a pipeline once and reuse it across runs:

.. code-block:: python

   from pykokoro import KokoroPipeline, PipelineConfig

   pipe = KokoroPipeline(PipelineConfig(voice="af_bella"))
   result = pipe.run("Hello!")
   print(result.sample_rate)

Using Local Model Files
~~~~~~~~~~~~~~~~~~~~~~~

If you already have the ONNX model and voices files locally, pass their paths
through ``PipelineConfig``:

.. code-block:: python

   from pathlib import Path

   from pykokoro import GenerationConfig, KokoroPipeline, PipelineConfig

   config = PipelineConfig(
       voice="af_bella",
       generation=GenerationConfig(lang="en-us"),
       model_path=Path("/models/kokoro.onnx"),
       voices_path=Path("/models/voices.bin.npz"),
   )
   pipe = KokoroPipeline(config)
   result = pipe.run("Using local model files.")

Model Quality Options
~~~~~~~~~~~~~~~~~~~~~

Available quality options vary by model source and variant:

**HuggingFace (Default Source):**

Both v1.0 and v1.1-zh variants support:

* ``fp32`` - Full precision (highest quality, largest size)
* ``fp16`` - Half precision (good balance)
* ``q8`` - 8-bit quantized (default, good balance)
* ``q8f16`` - 8-bit with fp16
* ``q4`` - 4-bit quantized (smallest, faster)
* ``q4f16`` - 4-bit with fp16
* ``uint8`` - Unsigned 8-bit
* ``uint8f16`` - Unsigned 8-bit with fp16

**GitHub v1.0:**

* ``fp32`` - Full precision
* ``fp16`` - Half precision
* ``fp16-gpu`` - GPU-optimized fp16
* ``q8`` - 8-bit quantized

**GitHub v1.1-zh:**

* ``fp32`` - Full precision only

.. code-block:: python

   from pykokoro import KokoroPipeline, PipelineConfig

   # HuggingFace v1.0 with fp16
   pipe = KokoroPipeline(PipelineConfig(voice="af_bella", model_quality="fp16"))

   # GitHub v1.0 with GPU optimization
   pipe = KokoroPipeline(
       PipelineConfig(
           voice="af_bella",
           model_source="github",
           model_variant="v1.0",
           model_quality="fp16-gpu",
       )
   )

Generating Speech
-----------------

Basic Text-to-Speech
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pykokoro import KokoroPipeline, PipelineConfig

   pipe = KokoroPipeline(PipelineConfig(voice="af_bella"))
   result = pipe.run("Hello, world!")
   audio = result.audio
   sample_rate = result.sample_rate

Saving Audio
~~~~~~~~~~~~

Using soundfile (recommended):

.. code-block:: python

   import soundfile as sf

   from pykokoro import KokoroPipeline, PipelineConfig

   pipe = KokoroPipeline(PipelineConfig(voice="af_bella"))
   result = pipe.run("Hello!")
   sf.write("output.wav", result.audio, result.sample_rate)

Using scipy:

.. code-block:: python

   from scipy.io import wavfile

   from pykokoro import KokoroPipeline, PipelineConfig

   pipe = KokoroPipeline(PipelineConfig(voice="af_bella"))
   result = pipe.run("Hello!")
   audio_int16 = (result.audio * 32767).astype("int16")
   wavfile.write("output.wav", result.sample_rate, audio_int16)

Voice Selection
---------------

Voice names follow the pattern: ``{accent}_{gender}_{name}``

* **Accent**: ``af`` (American Female), ``am`` (American Male), ``bf`` (British Female), ``bm`` (British Male)
* **Gender**: ``f`` (female), ``m`` (male)
* **Name**: Specific voice identifier

Use the voice name in ``PipelineConfig``:

.. code-block:: python

   from pykokoro import KokoroPipeline, PipelineConfig

   pipe = KokoroPipeline(PipelineConfig(voice="bf_emma"))
   result = pipe.run("Hello from the UK!")

Language Settings
-----------------

PyKokoro defaults language from the voice prefix, but you can override it:

.. code-block:: python

   from pykokoro import GenerationConfig, KokoroPipeline, PipelineConfig

   generation = GenerationConfig(lang="fr")
   pipe = KokoroPipeline(PipelineConfig(voice="af_sarah", generation=generation))
   result = pipe.run("Bonjour le monde")

Supported languages: ``en-us``, ``en-gb``, ``es``, ``fr``, ``de``, ``it``, ``pt``, ``hi``, ``ja``, ``zh``

Speech Speed Control
--------------------

Adjust the speaking rate with ``GenerationConfig.speed``:

.. code-block:: python

   from pykokoro import GenerationConfig, KokoroPipeline, PipelineConfig

   generation = GenerationConfig(speed=1.5)
   pipe = KokoroPipeline(PipelineConfig(voice="af_bella", generation=generation))
   result = pipe.run("Fast speech")

Recommended range: 0.5 to 2.0

Pause Control
-------------

Manual Pause Markers
~~~~~~~~~~~~~~~~~~~~

Add explicit pauses using SSMD break markers:

* ``...c`` - Short/comma pause
* ``...s`` - Medium/sentence pause
* ``...p`` - Long/paragraph pause
* ``...500ms`` - Custom duration pause

.. code-block:: python

   from pykokoro import GenerationConfig, KokoroPipeline, PipelineConfig

   text = "Hello! ...c This is a short pause. ...s And now a longer pause."
   generation = GenerationConfig(pause_mode="manual")
   pipe = KokoroPipeline(PipelineConfig(voice="af_bella", generation=generation))
   result = pipe.run(text)

Automatic Natural Pauses
~~~~~~~~~~~~~~~~~~~~~~~~

For natural rhythm, let the pipeline insert pauses at boundaries:

.. code-block:: python

   from pykokoro import GenerationConfig, KokoroPipeline, PipelineConfig

   text = """
   Artificial intelligence is transforming our world. Machine learning
   models are becoming more sophisticated and accessible.

   Deep learning uses neural networks with many layers.
   """

   generation = GenerationConfig(
       pause_mode="auto",
       pause_clause=0.25,
       pause_sentence=0.5,
       pause_paragraph=1.0,
       pause_variance=0.05,
       random_seed=42,
   )
   pipe = KokoroPipeline(PipelineConfig(voice="af_sarah", generation=generation))
   result = pipe.run(text)

Text Normalization (Say-As)
---------------------------

SSMD say-as syntax converts numbers, dates, and other formats:

.. code-block:: python

   from pykokoro import KokoroPipeline, PipelineConfig

   text = "I have [123](as: cardinal) apples and [12/31/2024](as: date, format: mdy)."
   pipe = KokoroPipeline(PipelineConfig(voice="af_sarah"))
   result = pipe.run(text)

Error Handling
--------------

.. code-block:: python

   from pykokoro import KokoroPipeline, PipelineConfig

   try:
       pipe = KokoroPipeline(PipelineConfig(voice="invalid_voice"))
       pipe.run("Hello!")
   except Exception as exc:
       print(f"Pipeline error: {exc}")

Batch Processing
----------------

Process multiple texts efficiently:

.. code-block:: python

   import soundfile as sf

   from pykokoro import KokoroPipeline, PipelineConfig

   texts = [
       ("Welcome", "welcome.wav"),
       ("Thank you", "thanks.wav"),
       ("Goodbye", "goodbye.wav"),
   ]

   pipe = KokoroPipeline(PipelineConfig(voice="af_bella"))
   for text, filename in texts:
       result = pipe.run(text)
       sf.write(filename, result.audio, result.sample_rate)

Next Steps
----------

* :doc:`advanced_features` - Voice blending, phoneme control, and more
* :doc:`examples` - Real-world examples
* :doc:`api_reference` - Complete API documentation
