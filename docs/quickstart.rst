Quick Start Guide
=================

This guide will get you up and running with PyKokoro in just a few minutes.

First Steps
-----------

1. **Install PyKokoro**

   .. code-block:: bash

      pip install pykokoro

2. **Verify Installation**

   .. code-block:: python

      import pykokoro
      print(pykokoro.__version__)

Basic Usage
-----------

Generate Your First Audio
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import soundfile as sf
   from pykokoro import GenerationConfig, KokoroPipeline, PipelineConfig

   config = PipelineConfig(
       voice="af_bella",  # American Female voice
       generation=GenerationConfig(lang="en-us"),
   )
   pipe = KokoroPipeline(config)
   result = pipe.run("Hello! Welcome to PyKokoro text-to-speech.")

   # Save to a WAV file
   sf.write("hello.wav", result.audio, result.sample_rate)

That's it! You've generated your first audio file.

Choosing a Voice
~~~~~~~~~~~~~~~~

PyKokoro comes with 54 voices (v1.0) or 103 voices (v1.1-zh). Here are some popular ones:

.. code-block:: python

   from pykokoro import GenerationConfig, KokoroPipeline, PipelineConfig

   # American English
   pipe = KokoroPipeline(PipelineConfig(voice="af_bella"))
   audio1 = pipe.run("American female voice").audio
   pipe = KokoroPipeline(PipelineConfig(voice="am_adam"))
   audio2 = pipe.run("American male voice").audio

   # British English
   pipe = KokoroPipeline(PipelineConfig(voice="bf_emma"))
   audio3 = pipe.run("British female voice").audio
   pipe = KokoroPipeline(PipelineConfig(voice="bm_george"))
   audio4 = pipe.run("British male voice").audio

   # Other languages
   pipe = KokoroPipeline(
       PipelineConfig(
           voice="af_nicole",
           generation=GenerationConfig(lang="es"),
       )
   )
   audio5 = pipe.run("Hola, mundo").audio
   pipe = KokoroPipeline(
       PipelineConfig(
           voice="af_sarah",
           generation=GenerationConfig(lang="fr"),
       )
   )
   audio6 = pipe.run("Bonjour le monde").audio

To see all available voices, check the README or use the voice listing examples
in ``examples/voices.py``.

Adjusting Speech Speed
~~~~~~~~~~~~~~~~~~~~~~

Control how fast or slow the speech is:

.. code-block:: python

   from pykokoro import GenerationConfig, KokoroPipeline, PipelineConfig

   # Normal speed (default)
   pipe = KokoroPipeline(
       PipelineConfig(voice="af_bella", generation=GenerationConfig(speed=1.0))
   )
   audio1 = pipe.run("Normal speed").audio

   # Slower (0.5x)
   pipe = KokoroPipeline(
       PipelineConfig(voice="af_bella", generation=GenerationConfig(speed=0.5))
   )
   audio2 = pipe.run("Slower speech").audio

   # Faster (1.5x)
   pipe = KokoroPipeline(
       PipelineConfig(voice="af_bella", generation=GenerationConfig(speed=1.5))
   )
   audio3 = pipe.run("Faster speech").audio

Adding Pauses
~~~~~~~~~~~~~

Add natural pauses in your speech using SSMD break syntax:

.. code-block:: python

   from pykokoro import KokoroPipeline, PipelineConfig

   text = """
   Welcome to the tutorial ...c
   This is a short pause ...s
   And this is a longer pause ...p
   These pauses make speech sound more natural.
   """

   pipe = KokoroPipeline(PipelineConfig(voice="af_bella"))
   result = pipe.run(text)

   import soundfile as sf
   sf.write("with_pauses.wav", result.audio, result.sample_rate)

Pause syntax (SSMD breaks):
* ``...c`` - Short/comma pause (0.3 seconds, default)
* ``...s`` - Medium/sentence pause (0.6 seconds, default)
* ``...p`` - Long/paragraph pause (1.0 seconds, default)
* ``...500ms`` - Custom duration pause (e.g., 500 milliseconds)

Reusing the Pipeline
~~~~~~~~~~~~~~~~~~~~

Reuse a pipeline instance for multiple runs:

.. code-block:: python

   from pykokoro import KokoroPipeline, PipelineConfig

   pipe = KokoroPipeline(PipelineConfig(voice="af_bella"))
   for sentence in ["Hello", "How are you?", "Goodbye!"]:
       result = pipe.run(sentence)
       print(result.audio.shape)

Processing Long Text
~~~~~~~~~~~~~~~~~~~~

For long text, PyKokoro automatically handles splitting at natural boundaries:

.. code-block:: python

   from pykokoro import GenerationConfig, KokoroPipeline, PipelineConfig

   long_text = """
   This is a long passage of text. It has multiple sentences.
   The text will be processed intelligently.

   This is a new paragraph. It will be processed efficiently.
   """

   pipe = KokoroPipeline(PipelineConfig(voice="af_bella"))
   result = pipe.run(long_text)

   # Or let PyKokoro insert boundary pauses
   auto_pipe = KokoroPipeline(
       PipelineConfig(
           voice="af_bella",
           generation=GenerationConfig(pause_mode="auto"),
       )
   )
   auto_result = auto_pipe.run(long_text)

   import soundfile as sf
   sf.write("long_text.wav", auto_result.audio, auto_result.sample_rate)

Complete Example
----------------

Here's a complete example putting it all together:

.. code-block:: python

   from pykokoro import GenerationConfig, KokoroPipeline, PipelineConfig
   import soundfile as sf

   def text_to_speech(text, output_file, voice="af_bella", speed=1.0):
       """Convert text to speech and save to file."""
       config = PipelineConfig(
           voice=voice,
           generation=GenerationConfig(speed=speed),
       )
       pipe = KokoroPipeline(config)
       result = pipe.run(text)
       sf.write(output_file, result.audio, result.sample_rate)
       print(f"Saved audio to {output_file}")

   # Example usage
   text = """
   Welcome to PyKokoro! ...s

   This library makes text-to-speech generation simple ...c
   You can control voice, speed, and add natural pauses ...s

   Enjoy creating audio content!
   """

   text_to_speech(text, "welcome.wav", voice="af_bella", speed=1.0)

Next Steps
----------

Now that you know the basics, explore:

* :doc:`basic_usage` - Detailed usage guide
* :doc:`advanced_features` - Voice blending, phoneme control, and more
* :doc:`examples` - More examples and use cases
* :doc:`api_reference` - Complete API documentation
