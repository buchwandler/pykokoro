Advanced Features
=================

This guide covers advanced features of PyKokoro for power users.

.. note::

   Use ``KokoroPipeline`` as the supported interface. Legacy ``Kokoro`` snippets
   can be updated by replacing ``kokoro.create`` with ``pipe.run``.

Voice Blending
--------------

Create custom voices by blending multiple voices together.

Basic Voice Blending
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pykokoro import Kokoro, VoiceBlend

   with Kokoro() as kokoro:
       # Blend two voices equally
       blend = VoiceBlend.parse("af_bella + af_sarah")

       audio, sr = kokoro.create(
           "This is a blended voice",
           voice=blend
       )

Weighted Blending
~~~~~~~~~~~~~~~~~

Control the contribution of each voice:

.. code-block:: python

   from pykokoro import Kokoro, VoiceBlend

   with Kokoro() as kokoro:
       # 70% bella, 30% sarah
       blend = VoiceBlend.parse("af_bella*0.7 + af_sarah*0.3")

       audio, sr = kokoro.create(
           "Weighted blend",
           voice=blend
       )

       # Percentage notation (normalized automatically)
       blend2 = VoiceBlend.parse("af_bella*70% + af_sarah*30%")

Multiple Voice Blending
~~~~~~~~~~~~~~~~~~~~~~~~

Blend more than two voices:

.. code-block:: python

   from pykokoro import Kokoro, VoiceBlend

   with Kokoro() as kokoro:
       # Three-way blend
       blend = VoiceBlend.parse(
           "af_bella*0.5 + af_sarah*0.3 + af_nicole*0.2"
       )

       audio, sr = kokoro.create(
           "Complex blend",
           voice=blend
       )

Creating Blended Voice Programmatically
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pykokoro import Kokoro, VoiceBlend

   # Create blend object directly
   blend = VoiceBlend(
       voices=["af_bella", "af_sarah", "am_adam"],
       weights=[0.4, 0.4, 0.2]
   )

   with Kokoro() as kokoro:
       audio, sr = kokoro.create(
           "Custom blend",
           voice=blend
       )

Phoneme-Based Generation
-------------------------

For precise control, generate speech directly from phonemes.

Using create_from_phonemes()
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pykokoro import Kokoro

   with Kokoro() as kokoro:
       # Get phonemes for text
       phonemes = kokoro.tokenizer.phonemize("Hello, world!")

       # Generate from phonemes
       audio, sr = kokoro.create_from_phonemes(
           phonemes,
           voice="af_bella",
           speed=1.0
       )

Text to Phonemes
~~~~~~~~~~~~~~~~

Convert text to phonemes:

.. code-block:: python

   from pykokoro import Kokoro

   with Kokoro() as kokoro:
       # Get phonemes
       phonemes = kokoro.tokenizer.phonemize(
           "Hello, world!",
           lang="en-us"
       )
       print(f"Phonemes: {phonemes}")

       # Get detailed phoneme info
       result = kokoro.tokenizer.text_to_phonemes(
           "Hello",
           lang="en-us",
           with_words=True
       )
       print(result)

PhonemeSegment Processing
~~~~~~~~~~~~~~~~~~~~~~~~~~

Work with phoneme segments for batch processing:

.. code-block:: python

   from pykokoro import phonemize_text_list, create_tokenizer

   tokenizer = create_tokenizer()

   texts = ["Hello", "World", "How are you?"]
   segments = phonemize_text_list(texts, tokenizer, lang="en-us")

   for segment in segments:
       print(f"Text: {segment.text}")
       print(f"Phonemes: {segment.phonemes}")
       print(f"Tokens: {segment.tokens}")

Advanced Text Splitting
------------------------

Split and Phonemize in One Step (Legacy API)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For advanced text processing with automatic splitting and phoneme generation.

The ``split_and_phonemize_text`` function intelligently handles long text by:

1. Splitting text using your chosen mode (paragraph, sentence, or clause)
2. Phonemizing each segment
3. **Automatically cascading to finer split modes** if segments exceed the phoneme limit
4. Only truncating as last resort (when even individual words are too long)

**Cascade behavior:**

- ``paragraph`` mode → ``sentence`` → ``clause`` → ``word`` → truncate
- ``sentence`` mode → ``clause`` → ``word`` → truncate
- ``clause`` mode → ``word`` → truncate
- ``word`` mode → truncate (with warning)

.. code-block:: python

   from pykokoro import split_and_phonemize_text, create_tokenizer

   tokenizer = create_tokenizer()

   long_text = """
   This is the first sentence. This is the second.

   This is a new paragraph.
   """

   # The function automatically ensures all segments stay within limit
   segments = split_and_phonemize_text(
       long_text,
       tokenizer,
       lang="en-us",
       split_mode="sentence",  # Will cascade to clause/word if needed
       max_phoneme_length=510  # Kokoro's maximum
   )

   for segment in segments:
       print(f"Paragraph {segment.paragraph}, Sentence {segment.sentence}")
       print(f"Text: {segment.text}")
       print(f"Phonemes: {segment.phonemes[:50]}...")
       # segment.phonemes is guaranteed to be <= 510

Split Modes in Detail (Legacy API)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Paragraph Mode:**

.. code-block:: python

   segments = split_and_phonemize_text(
       text,
       tokenizer,
       split_mode="paragraph"  # Splits on double newlines
   )

**Sentence Mode:**

Requires spaCy for sentence boundary detection:

.. code-block:: python

   segments = split_and_phonemize_text(
       text,
       tokenizer,
       split_mode="sentence"  # Splits on sentence boundaries
   )

**Clause Mode:**

Splits on both sentences and commas for finer control:

.. code-block:: python

   segments = split_and_phonemize_text(
       text,
       tokenizer,
       split_mode="clause"  # Splits on sentences and commas
   )

Pause Mode
~~~~~~~~~~

The modern ``Kokoro.create()`` API uses ``pause_mode`` for controlling pause behavior:

.. code-block:: python

   from pykokoro import Kokoro

   with Kokoro() as kokoro:
       # Default: TTS controls pauses naturally
       audio, sr = kokoro.create(
           long_text,
           voice="af_sarah"
       )

       # Auto mode: PyKokoro inserts boundary pauses
       audio, sr = kokoro.create(
           long_text,
           voice="af_sarah",
           pause_mode="auto",
           pause_clause=0.25,           # Clause boundaries
           pause_sentence=0.5,          # Sentence boundaries
           pause_paragraph=1.0,         # Paragraph boundaries
           pause_variance=0.05,         # Natural variance (Gaussian)
           random_seed=42               # For reproducible results
       )

       # Manual mode: PyKokoro trims and preserves explicit pauses
       audio, sr = kokoro.create(
           long_text,
           voice="af_sarah",
           pause_mode="manual",         # Preserve explicit pauses
           pause_clause=0.25,           # Clause boundaries
           pause_sentence=0.5,          # Sentence boundaries
           pause_paragraph=1.0,         # Paragraph boundaries
           pause_variance=0.05,         # Natural variance (Gaussian)
           random_seed=42               # For reproducible results
       )

**Pause Variance Details:**

The ``pause_variance`` parameter adds Gaussian (normal distribution) variance to
pause durations, making speech sound more natural:

* **0.0** - No variance, exact pause durations
* **0.05** - Default, ±100ms at 95% confidence interval
* **0.1** - Higher variance, ±200ms at 95% confidence

The variance ensures that pauses are never exactly the same length, mimicking
natural human speech rhythm.

**Reproducibility:**

Use ``random_seed`` for consistent output across runs:

.. code-block:: python

   # Same output every time
   audio1, sr = kokoro.create(text, voice="af_sarah",
                              pause_mode="auto",
                              random_seed=42)

   audio2, sr = kokoro.create(text, voice="af_sarah",
                              pause_mode="auto",
                              random_seed=42)

   # audio1 and audio2 are identical

   # Different output each time
   audio3, sr = kokoro.create(text, voice="af_sarah",
                              pause_mode="auto",
                              random_seed=None)  # or omit parameter

Custom Warning Callbacks
~~~~~~~~~~~~~~~~~~~~~~~~~

Handle warnings during phoneme generation:

.. code-block:: python

   from pykokoro import split_and_phonemize_text, create_tokenizer

   def my_warning_handler(message):
       print(f"WARNING: {message}")

   tokenizer = create_tokenizer()
   segments = split_and_phonemize_text(
       very_long_text,
       tokenizer,
       warn_callback=my_warning_handler
   )

GPU Acceleration
----------------

Automatic GPU Detection
~~~~~~~~~~~~~~~~~~~~~~~

PyKokoro automatically uses GPU if available:

.. code-block:: python

   from pykokoro import Kokoro, get_device

   # Check available device
   device = get_device()
   print(f"Using device: {device}")

   # Kokoro will use GPU automatically
   with Kokoro() as kokoro:
       audio, sr = kokoro.create("Hello!", voice="af_bella")

Forcing Specific Device
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pykokoro import Kokoro

   # Force CPU
   kokoro_cpu = Kokoro(device="cpu")

   # Force CUDA (NVIDIA)
   kokoro_gpu = Kokoro(device="cuda")

   # Force ROCm (AMD)
   kokoro_rocm = Kokoro(device="rocm")

GPU Information
~~~~~~~~~~~~~~~

.. code-block:: python

   from pykokoro import get_gpu_info

   info = get_gpu_info()
   print(f"Device: {info['device']}")
   print(f"Providers: {info['providers']}")

Custom Model Paths
------------------

Model Sources
~~~~~~~~~~~~~

PyKokoro supports multiple model sources:

**HuggingFace (Default):**

.. code-block:: python

   from pykokoro import Kokoro

   # Default: HuggingFace with 54 voices
   kokoro = Kokoro(model_source="huggingface", model_quality="fp32")

**HuggingFace v1.0 (Default - 54 voices, 8 quality options):**

.. code-block:: python

   # Default: HuggingFace v1.0
   kokoro = Kokoro(model_quality="q8")  # Recommended default

**HuggingFace v1.1-zh (103 voices, 8 quality options):**

.. code-block:: python

   # HuggingFace v1.1-zh with English + Chinese voices
   # Supports all quantization levels: fp32, fp16, q8, q8f16, q4, q4f16, uint8, uint8f16
   kokoro = Kokoro(
       model_variant="v1.1-zh",
       model_quality="q8"  # All qualities available
   )

   # Use English voices
   audio, sr = kokoro.create(
       "Hello world!",
       voice="af_maple",  # v1.1-zh English voice
       lang="en-us"
   )

**GitHub v1.0 (54 voices, 4 quality options):**

.. code-block:: python

   # GitHub v1.0 with GPU-optimized fp16
   kokoro = Kokoro(
       model_source="github",
       model_variant="v1.0",
       model_quality="fp16-gpu"  # Options: fp32, fp16, fp16-gpu, q8
   )

**GitHub v1.1-zh (103 voices, fp32 only):**

.. code-block:: python

   # GitHub v1.1-zh with English + Chinese voices
   kokoro = Kokoro(
       model_source="github",
       model_variant="v1.1-zh",
       model_quality="fp32"  # Only fp32 available
   )

   # Use English voices
   audio, sr = kokoro.create(
       "Hello world!",
       voice="af_maple",  # v1.1-zh English voice
       lang="en-us"
   )

**Note:** Chinese text generation is currently in development. Use English voices from
v1.1-zh with English text for now.

Use Custom Model Files
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pykokoro import Kokoro

   kokoro = Kokoro(
       model_path="/path/to/custom/model.onnx",
       voices_path="/path/to/voices.bin"
   )

Download Models Manually
~~~~~~~~~~~~~~~~~~~~~~~~~

**HuggingFace Models:**

.. code-block:: python

   from pykokoro import download_model, download_voice, download_all_models

   # Download specific model quality (v1.0 by default)
   download_model(variant="v1.0", quality="q8")

   # Download specific voice
   download_voice(voice_name="af_bella", variant="v1.0")

   # Download all models
   download_all_models(variant="v1.0")

**GitHub Models:**

.. code-block:: python

   from pykokoro.onnx_backend import (
       download_model_github,
       download_voices_github,
       download_all_models_github
   )

   # Download GitHub v1.0 model
   download_model_github(variant="v1.0", quality="fp16-gpu")

   # Download GitHub v1.0 voices
   download_voices_github(variant="v1.0")

   # Download all GitHub v1.1-zh files
   download_all_models_github(
       variant="v1.1-zh",
       quality="fp32",
       progress_callback=lambda msg: print(msg)
   )

**HuggingFace v1.1-zh Models:**

.. code-block:: python

   from pykokoro import (
       download_model,
       download_all_voices,
       download_all_models,
       download_config
   )

   # Download HuggingFace v1.1-zh model (with quantization)
   download_model(variant="v1.1-zh", quality="q8")

   # Download all 103 voices for v1.1-zh
   def progress(voice_name, current, total):
       print(f"Downloading {current}/{total}: {voice_name}")

   download_all_voices(variant="v1.1-zh", progress_callback=progress)

   # Download configuration for v1.1-zh
   download_config(variant="v1.1-zh")

   # Download everything (model + config + all voices)
   download_all_models(
       variant="v1.1-zh",
       quality="q8",
       progress_callback=lambda msg: print(msg)
   )

**Available Quality Options by Source:**

* **HuggingFace v1.0**: fp32, fp16, q8, q8f16, q4, q4f16, uint8, uint8f16
* **HuggingFace v1.1-zh**: fp32, fp16, q8, q8f16, q4, q4f16, uint8, uint8f16
* **GitHub v1.0**: fp32, fp16, fp16-gpu, q8
* **GitHub v1.1-zh**: fp32 only

Get Model Paths
~~~~~~~~~~~~~~~

.. code-block:: python

   from pykokoro import get_model_path, get_voice_path

   # HuggingFace model paths
   model_path = get_model_path(quality="q8")
   voice_path = get_voice_path()

   print(f"Model: {model_path}")
   print(f"Voices: {voice_path}")

   # GitHub model paths are stored in variant-specific subdirectories:
   # ~/.cache/pykokoro/models/onnx/v1.0/kokoro-v1.0.onnx
   # ~/.cache/pykokoro/models/onnx/v1.1-zh/kokoro-v1.1-zh.onnx
   # ~/.cache/pykokoro/voices/v1.0/voices-v1.0.bin
   # ~/.cache/pykokoro/voices/v1.1-zh/voices-v1.1-zh.bin

Advanced Tokenizer Configuration
---------------------------------

Custom Tokenizer Settings
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pykokoro import create_tokenizer, TokenizerConfig

   # Custom tokenizer config
   tokenizer_config = TokenizerConfig(
       vocab_path="/path/to/vocab.txt",
       espeak_config=espeak_config
   )

   tokenizer = create_tokenizer(config=tokenizer_config)

Mixed Language Support
~~~~~~~~~~~~~~~~~~~~~~

For text with multiple languages:

.. code-block:: python

   from pykokoro import create_tokenizer, TokenizerConfig

   config = TokenizerConfig(
       enable_mixed_language=True,
       primary_language="en-us",
       allowed_languages=["en-us", "es", "fr"],
       language_confidence_threshold=0.7
   )

   tokenizer = create_tokenizer(config=config)

Audio Trimming
--------------

Trim Silence from Audio
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pykokoro import trim

   # Generate audio with silence
   with Kokoro() as kokoro:
       audio, sr = kokoro.create("Hello!", voice="af_bella")

   # Trim silence
   trimmed_audio, trim_info = trim(audio)

   print(f"Original: {len(audio)} samples")
   print(f"Trimmed: {len(trimmed_audio)} samples")
   print(f"Trim info: {trim_info}")

Short Sentence Handling
-----------------------

PyKokoro improves short, single-word sentences by surrounding the word with a
pause marker. You can tune settins via ``ShortSentenceConfig``:

.. code-block:: python

   from pykokoro.short_sentence_handler import ShortSentenceConfig

   short_config = ShortSentenceConfig(
       phoneme_pretext="…",
   )

Configuration Management
------------------------

Save and Load Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pykokoro import save_config, load_config

   # Save configuration
   config = {
       "default_voice": "af_bella",
       "default_speed": 1.0,
       "model_quality": "q8"
   }
   save_config(config, "my_config.json")

   # Load configuration
   loaded_config = load_config("my_config.json")

Get Cache Paths
~~~~~~~~~~~~~~~

.. code-block:: python

   from pykokoro import get_user_cache_path, get_user_config_path

   cache_path = get_user_cache_path()
   config_path = get_user_config_path()

   print(f"Cache: {cache_path}")
   print(f"Config: {config_path}")

Performance Tips
----------------

1. **Reuse Kokoro Instance**

   Don't create a new ``Kokoro()`` for each request - initialize once and reuse.

2. **Use GPU When Available**

   GPU acceleration provides 3-10x speedup.

3. **Batch Processing**

   Process multiple texts in one session to avoid initialization overhead.

4. **Choose Appropriate Model Quality**

   Use ``q6`` or ``q8`` for production; ``fp16`` only when quality is critical.

5. **Use pause_mode for Long Text**

   Using ``pause_mode="auto"`` with appropriate pause settings improves quality for long text.

Internal Architecture
---------------------

Understanding PyKokoro's Internal Structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PyKokoro uses a modular architecture with specialized manager classes for different responsibilities:

**OnnxSessionManager** (``pykokoro/onnx_session.py``)

Manages ONNX Runtime session creation and configuration:

* Automatic provider selection (CUDA → ROCm → CPU)
* Session options and execution providers
* Fallback handling when GPU is unavailable

**VoiceManager** (``pykokoro/voice_manager.py``)

Handles voice loading and blending:

* Loads voice embeddings from binary files
* Implements voice blending with weighted combinations
* Validates voice availability across model variants

**AudioGenerator** (``pykokoro/audio_generator.py``)

Manages the audio generation pipeline:

* Converts phonemes to tokens
* Runs ONNX inference for audio generation
* Handles speed adjustment and audio post-processing

**MixedLanguageHandler** (``pykokoro/mixed_language_handler.py``)

Automatic language detection for multilingual text:

* Detects language boundaries in mixed-language text
* Routes text segments to appropriate language models
* Configurable confidence thresholds

**PhonemeDictionary** (``pykokoro/phoneme_dictionary.py``)

Custom word-to-phoneme mappings:

* Override default pronunciation for specific words
* Support for context-aware phoneme substitution
* JSON-based dictionary format

Using Manager Classes Directly
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

While most users interact with the high-level ``Kokoro`` API, advanced users can work with manager classes directly:

.. code-block:: python

   from pykokoro.onnx_session import OnnxSessionManager
   from pykokoro.voice_manager import VoiceManager
   from pykokoro.audio_generator import AudioGenerator

   # Create ONNX session with custom options
   session_manager = OnnxSessionManager(
       device="cuda",
       providers=["CUDAExecutionProvider"],
       user_session_options={"intra_op_num_threads": 4}
   )

   session = session_manager.create_session(
       model_path="/path/to/model.onnx"
   )

   # Load voices with custom blend
   voice_manager = VoiceManager(model_source="huggingface")
   voice_manager.load_voices("/path/to/voices.bin")
   voice_data = voice_manager.get_blended_voice("af_bella*0.7 + af_sarah*0.3")

   # Generate audio
   audio_gen = AudioGenerator(
       session=session,
       sample_rate=24000,
       lang="en-us"
   )

   audio = audio_gen.generate_audio_from_phonemes(
       phonemes="həˈloʊ wɝld",
       voice_data=voice_data,
       speed=1.0
   )

Custom Phoneme Dictionaries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create custom pronunciation mappings:

.. code-block:: python

   from pykokoro.phoneme_dictionary import PhonemeDictionary

   # Create dictionary
   dictionary = PhonemeDictionary()

   # Add custom pronunciations
   dictionary.add_word("PyKokoro", "paɪ kəˈkɔɹoʊ")
   dictionary.add_word("ONNX", "ɑnɪks")

   # Save to file
   dictionary.save("custom_pronunciations.json")

   # Load and use
   loaded_dict = PhonemeDictionary.load("custom_pronunciations.json")

   # Apply to tokenizer
   from pykokoro import create_tokenizer
   tokenizer = create_tokenizer()
   tokenizer.phoneme_dictionary = loaded_dict

   # Now "PyKokoro" will use custom pronunciation
   phonemes = tokenizer.phonemize("Welcome to PyKokoro!")

Next Steps
----------

* :doc:`examples` - Real-world usage examples
* :doc:`api_reference` - Complete API documentation
