Changelog
=========

Version 0.0.4 (TBD)
-------------------

**Bug Fixes:**

* Fixed deprecated ``local_dir_use_symlinks`` parameter in HuggingFace downloads (removes deprecation warnings)
* Fixed Windows CI test failures related to temporary file permissions in encoding detection tests
* Properly close temporary files before deletion to support Windows file locking behavior

**Breaking Changes:**

* Removed redundant ``v1.1-zh-hf`` model variant - Use ``model_variant="v1.1-zh"`` with ``model_source="huggingface"`` instead
* Changed cache directory structure to ``~/.cache/pykokoro/{models|voices}/{source}/{variant}/``
* Removed deprecated functions: ``download_model_hf_v11zh()``, ``download_voices_hf_v11zh()``, ``download_all_models_hf_v11zh()``
* Function signatures updated: ``download_model()``, ``download_voice()``, ``download_all_voices()``, and ``download_all_models()`` now require ``variant`` parameter
* Path helper functions ``get_model_dir()`` and ``get_voices_dir()`` now require ``source`` and ``variant`` parameters
* **API Simplification:** Replaced ``split_mode`` and ``trim_silence`` parameters with single ``pause_mode`` parameter in ``Kokoro.create()``

**Migration Guide:**

For users upgrading from v0.0.3:

.. code-block:: python

    # Old (v0.3.x) - NO LONGER WORKS
    kokoro = Kokoro(model_variant="v1.1-zh-hf")
    download_model_hf_v11zh(quality="fp16")

    # New (v0.4.0+)
    kokoro = Kokoro(
        model_source="huggingface",
        model_variant="v1.1-zh"
    )
    download_model(variant="v1.1-zh", quality="fp16")

    # Old pause control (v0.3.x) - NO LONGER WORKS
    audio, sr = kokoro.create(
        text,
        voice="af_bella",
        split_mode="clause",
        trim_silence=True,
        pause_clause=0.25
    )

    # New pause control (v0.4.0+)
    # Default: TTS controls pauses naturally
    audio, sr = kokoro.create(text, voice="af_bella")

    # Manual pause control
    audio, sr = kokoro.create(
        text,
        voice="af_bella",
        pause_mode="manual",  # PyKokoro controls pauses precisely
        pause_clause=0.25,
        pause_sentence=0.5,
        pause_paragraph=1.0
    )

**pause_mode Parameter:**

The new ``pause_mode`` parameter simplifies pause control:

* ``pause_mode="tts"`` (default): TTS generates pauses naturally. Best for most content.
* ``pause_mode="manual"``: PyKokoro controls pauses precisely. Best for podcasts, voice switching, and precise timing.
* ``pause_mode="auto"``: PyKokoro inserts pauses at sentence/paragraph boundaries and trims silence.

**Improvements:**

* Unified path structure across all model sources and variants
* Consolidated duplicate download functions for cleaner API
* Shared configuration files between HuggingFace and GitHub sources
* Improved code maintainability with consistent path handling
* All quantization levels supported for v1.1-zh: fp32, fp16, q8, q8f16, q4, q4f16, uint8, uint8f16
* Simplified API with clearer parameter semantics

**New Features:**

* Added support for HuggingFace Kokoro v1.1-zh model (``onnx-community/Kokoro-82M-v1.1-zh-ONNX``)
* Added 103 voices for v1.1-zh variant
* Voice files automatically combined into efficient .npz format
* Progress callbacks for voice downloads
* Added ``pause_mode`` parameter for simplified pause control, including ``"auto"`` boundary pauses

**Documentation:**

* Updated advanced features guide with unified variant usage
* Updated ``examples/hf_v11zh_demo.py`` demonstration script
* Added migration guide for v0.4.0 breaking changes
* Updated all documentation to use new ``pause_mode`` API

Version 0.0.3 (2026-01-07)
--------------------------

**Major Refactoring:**

* Extracted internal manager classes for better code organization
* Reduced codebase complexity by ~706 lines (12% reduction)
* Improved maintainability with better separation of concerns
* 100% backward compatibility maintained - no breaking changes

**New Internal Classes:**

* Added ``OnnxSessionManager`` class for ONNX Runtime session management
* Added ``VoiceManager`` class for voice loading and blending operations
* Added ``AudioGenerator`` class for audio generation pipeline
* Added ``MixedLanguageHandler`` class for automatic language detection
* Added ``PhonemeDictionary`` class for custom word-to-phoneme mappings

**Code Quality:**

* Reduced ``onnx_backend.py`` by 436 lines
* Reduced ``tokenizer.py`` by 270 lines
* Added comprehensive test coverage for new manager classes
* All pre-commit hooks passing (ruff, ruff-format)
* 98.7% test pass rate (312/316 tests)

**Architecture Improvements:**

* Delegate pattern implementation for backward compatibility
* Better separation of session management, voice handling, and audio generation
* Improved modularity for easier testing and maintenance
* Enhanced error handling and validation

**Documentation:**

* Added API documentation for new internal manager classes
* Added internal architecture section to advanced features guide
* Updated changelog with refactoring details

Version 0.0.1 (2025-01-06)
--------------------------

**Breaking Changes:**

* Removed ``PhonemeBook`` class - moved to separate ebook package
* Removed ``PhonemeChapter`` class - moved to separate ebook package
* Removed ``create_phoneme_book_from_chapters()`` function
* Removed ``FORMAT_VERSION`` constant
* Deleted ``examples/phoneme_export.py`` example

**New Features:**

* Added ``split_and_phonemize_text()`` function for standalone text processing
* Added ``enable_pauses`` parameter to ``create()`` method for pause marker support
* Added pause markers: ``(.)``, ``(..)``, ``(...)`` for controlling speech pauses (DEPRECATED - use SSMD break syntax instead: ``...c``, ``...s``, ``...p``)
* Added ``pause_short``, ``pause_medium``, ``pause_long`` parameters for custom pause durations
* Added ``split_mode`` parameter to ``create()`` for intelligent text splitting (DEPRECATED in v0.4.0 - use ``pause_mode`` instead)
* Added ``trim_silence`` parameter for removing silence between segments (DEPRECATED in v0.4.0 - use ``pause_mode="manual"`` instead)
* Added ``pause_after`` field to ``PhonemeSegment`` class

**Improvements:**

* Refactored ``_process_with_split_mode()`` to use standalone function
* Improved phoneme-based generation with automatic length checking
* Enhanced documentation with comprehensive examples
* Better error handling and validation
* Optimized text splitting for long passages

**Bug Fixes:**

* Fixed floating point precision in pause duration tests
* Improved backward compatibility for PhonemeSegment serialization
* Better handling of empty and whitespace-only text

**Documentation:**

* Added complete Sphinx documentation
* Added quick start guide
* Added installation guide
* Added basic usage guide
* Added advanced features guide
* Added comprehensive examples
* Added API reference

Version 0.0.1 (Initial Release)
-------------------------------

**Initial Features:**

* Text-to-speech synthesis using Kokoro model
* Support for 54 voices (v1.0) across multiple languages
* Support for English (US/GB), Spanish, French, German, Italian, Portuguese, Hindi, Japanese, Korean, Chinese
* Voice blending capabilities
* Phoneme-based generation
* GPU acceleration support (CUDA/ROCm)
* Model quality options (fp16, q8, q6)
* Speed control
* Basic tokenizer functionality
* Audio trimming utilities
* Configuration management
* Model and voice downloading
* PhonemeBook and PhonemeChapter classes for document processing
* spaCy integration for sentence splitting
* Mixed language support
