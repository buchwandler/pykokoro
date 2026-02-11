API Reference
=============

This page provides API documentation for the supported pipeline-first interface.

Main Classes
------------

KokoroPipeline
~~~~~~~~~~~~~~

.. autoclass:: pykokoro.KokoroPipeline
   :members:
   :undoc-members:
   :show-inheritance:

**Basic Example:**

.. code-block:: python

   from pykokoro import KokoroPipeline, PipelineConfig

   pipe = KokoroPipeline(PipelineConfig(voice="af_bella"))
   result = pipe.run("Hello, world!")
   print(result.sample_rate)

PipelineConfig
~~~~~~~~~~~~~~

.. autoclass:: pykokoro.PipelineConfig
   :members:
   :undoc-members:
   :show-inheritance:

GenerationConfig
~~~~~~~~~~~~~~~~

.. autoclass:: pykokoro.GenerationConfig
   :members:
   :undoc-members:
   :show-inheritance:

Pipeline Helpers
----------------

.. autofunction:: pykokoro.build_pipeline
.. autofunction:: pykokoro.with_spacy_model_size

Result and Data Classes
-----------------------

AudioResult
~~~~~~~~~~~

.. autoclass:: pykokoro.types.AudioResult
   :members:
   :undoc-members:
   :show-inheritance:

Segment
~~~~~~~

.. autoclass:: pykokoro.types.Segment
   :members:
   :undoc-members:
   :show-inheritance:

PhonemeSegment
~~~~~~~~~~~~~~

.. autoclass:: pykokoro.types.PhonemeSegment
   :members:
   :undoc-members:
   :show-inheritance:

Trace
~~~~~

.. autoclass:: pykokoro.types.Trace
   :members:
   :undoc-members:
   :show-inheritance:

Voice Blending
--------------

VoiceBlend
~~~~~~~~~~

.. autoclass:: pykokoro.onnx_backend.VoiceBlend
   :members:
   :undoc-members:
   :show-inheritance:

.. code-block:: python

   from pykokoro import KokoroPipeline, PipelineConfig
   from pykokoro.onnx_backend import VoiceBlend

   blend = VoiceBlend.parse("af_bella:60,af_sarah:40")
   pipe = KokoroPipeline(PipelineConfig(voice=blend))
   result = pipe.run("Blended voice example")

Tokenizer
---------

Tokenizer
~~~~~~~~~

.. autoclass:: pykokoro.tokenizer.Tokenizer
   :members:
   :undoc-members:
   :show-inheritance:

TokenizerConfig
~~~~~~~~~~~~~~~

.. autoclass:: pykokoro.tokenizer.TokenizerConfig
   :members:
   :undoc-members:
   :show-inheritance:

PhonemeResult
~~~~~~~~~~~~~

.. autoclass:: pykokoro.tokenizer.PhonemeResult
   :members:
   :undoc-members:
   :show-inheritance:

**Tokenizer Example:**

.. code-block:: python

   from pykokoro.tokenizer import Tokenizer

   tokenizer = Tokenizer()
   phonemes = tokenizer.phonemize("Hello", lang="en-us")
   print(phonemes)

Model and Voice Utilities
-------------------------

These utilities live in ``pykokoro.onnx_backend`` and are used for model and
voice management.

.. autofunction:: pykokoro.onnx_backend.download_model
.. autofunction:: pykokoro.onnx_backend.download_voice
.. autofunction:: pykokoro.onnx_backend.download_all_models
.. autofunction:: pykokoro.onnx_backend.download_all_voices
.. autofunction:: pykokoro.onnx_backend.download_config
.. autofunction:: pykokoro.onnx_backend.get_model_path
.. autofunction:: pykokoro.onnx_backend.get_voice_path

Configuration Helpers
---------------------

.. autofunction:: pykokoro.utils.load_config
.. autofunction:: pykokoro.utils.save_config
.. autofunction:: pykokoro.utils.get_user_cache_path
.. autofunction:: pykokoro.utils.get_user_config_path

See Also
--------

* :doc:`basic_usage` - Fundamental usage patterns
* :doc:`examples` - Practical pipeline examples
