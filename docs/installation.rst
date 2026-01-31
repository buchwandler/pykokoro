Installation Guide
==================

PyKokoro can be installed using pip and requires Python 3.9 or higher.

Basic Installation
------------------

Install the latest stable version from PyPI:

.. code-block:: bash

   pip install pykokoro

This will install PyKokoro with CPU support using ONNX Runtime.

GPU Support
-----------

For GPU acceleration, install with the GPU extras:

NVIDIA CUDA
~~~~~~~~~~~

.. code-block:: bash

   pip install pykokoro[gpu]

This installs ``onnxruntime-gpu`` for NVIDIA CUDA support.

AMD ROCm
~~~~~~~~

For AMD GPUs with ROCm:

.. code-block:: bash

   pip install pykokoro
   pip install onnxruntime-rocm

Custom ONNX Runtime
~~~~~~~~~~~~~~~~~~~

You can also install a specific ONNX Runtime version separately:

.. code-block:: bash

   pip install pykokoro
   pip install onnxruntime-gpu==1.19.2  # or your preferred version

System Requirements
-------------------

Python Version
~~~~~~~~~~~~~~

* Python 3.9 or higher
* Tested on Python 3.9, 3.10, 3.11, 3.12, and 3.13

Dependencies
~~~~~~~~~~~~

Core dependencies (automatically installed):

* ``numpy`` - Array operations
* ``onnxruntime`` - Model inference
* ``espeak-ng`` - Phoneme generation (via ``piper-phonemize``)
* ``piper-phonemize`` - Text-to-phoneme conversion
* ``requests`` - Model downloading
* ``tqdm`` - Progress bars
* ``kokorog2p`` - Enhanced phoneme dictionary
* ``phrasplit`` - Intelligent text splitting

Optional dependencies:

* ``soundfile`` - For saving audio to WAV files (recommended)
* ``onnxruntime-gpu`` - For GPU acceleration
* ``spacy`` - For sentence/clause splitting (used with ``pause_mode="auto"`` and legacy ``split_mode``)

Installing espeak-ng
~~~~~~~~~~~~~~~~~~~~

PyKokoro requires ``espeak-ng`` to be installed on your system.

**Ubuntu/Debian:**

.. code-block:: bash

   sudo apt-get install espeak-ng

**macOS (Homebrew):**

.. code-block:: bash

   brew install espeak-ng

**Windows:**

Download and install from: https://github.com/espeak-ng/espeak-ng/releases

Or use Chocolatey:

.. code-block:: bash

   choco install espeak-ng

Installing spaCy (Optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

For advanced text splitting with ``pause_mode="auto"`` or the legacy ``split_and_phonemize_text()`` function:

.. code-block:: bash

   pip install spacy
   python -m spacy download en_core_web_sm

Development Installation
------------------------

To install from source for development:

.. code-block:: bash

   git clone https://github.com/remixer-dec/pykokoro.git
   cd pykokoro
   pip install -e ".[dev]"

This installs PyKokoro in editable mode with development dependencies.

Verifying Installation
----------------------

Test your installation:

.. code-block:: python

   import pykokoro
   print(pykokoro.__version__)

   # Quick test
   kokoro = pykokoro.Kokoro()
   audio, sr = kokoro.create("Hello, world!", voice="af_bella")
   print(f"Generated {len(audio)} audio samples at {sr} Hz")
   kokoro.close()

Troubleshooting
---------------

Import Errors
~~~~~~~~~~~~~

If you get import errors, ensure all dependencies are installed:

.. code-block:: bash

   pip install --upgrade pykokoro

espeak-ng Not Found
~~~~~~~~~~~~~~~~~~~

If you get errors about espeak-ng not being found:

1. Verify espeak-ng is installed: ``espeak-ng --version``
2. Ensure it's in your system PATH
3. On Windows, you may need to restart your terminal after installation

GPU Not Detected
~~~~~~~~~~~~~~~~

If GPU acceleration isn't working:

1. Verify CUDA/ROCm is installed: ``nvidia-smi`` (NVIDIA) or ``rocm-smi`` (AMD)
2. Check ONNX Runtime GPU: ``python -c "import onnxruntime; print(onnxruntime.get_available_providers())"``
3. Ensure you have the correct ONNX Runtime version for your CUDA version

Model Download Issues
~~~~~~~~~~~~~~~~~~~~~

If model downloads fail:

1. Check your internet connection
2. Verify you have write permissions to the cache directory
3. Try downloading manually and placing in ``~/.cache/pykokoro/``
4. For GitHub models, ensure the release URLs are accessible

**Manual Model Download:**

PyKokoro automatically downloads models on first use, but you can trigger downloads manually:

.. code-block:: python

   from pykokoro import Kokoro

   # HuggingFace v1.0 (default - 54 voices, 8 quality options)
   kokoro = Kokoro(model_quality="fp16")  # Auto-downloads from HuggingFace

   # HuggingFace v1.1-zh (103 voices, 8 quality options)
   kokoro = Kokoro(
       model_variant="v1.1-zh",
       model_quality="q8"  # Auto-downloads from HuggingFace
   )

   # GitHub v1.0 (54 voices, 4 quality options)
   kokoro = Kokoro(
       model_source="github",
       model_variant="v1.0",
       model_quality="fp16-gpu"  # Auto-downloads from GitHub
   )

   # GitHub v1.1-zh (103 voices, fp32 only)
   kokoro = Kokoro(
       model_source="github",
       model_variant="v1.1-zh",
       model_quality="fp32"  # Auto-downloads from GitHub
   )

Models are cached in:

* **HuggingFace v1.0**: ``~/.cache/pykokoro/models/huggingface/v1.0/`` and ``~/.cache/pykokoro/voices/huggingface/v1.0/``
* **HuggingFace v1.1-zh**: ``~/.cache/pykokoro/models/huggingface/v1.1-zh/`` and ``~/.cache/pykokoro/voices/huggingface/v1.1-zh/``
* **GitHub v1.0**: ``~/.cache/pykokoro/models/github/v1.0/`` and ``~/.cache/pykokoro/voices/github/v1.0/``
* **GitHub v1.1-zh**: ``~/.cache/pykokoro/models/github/v1.1-zh/`` and ``~/.cache/pykokoro/voices/github/v1.1-zh/``
