# Installation Guide

PyKokoro can be installed using pip and requires Python 3.10 or higher.

## Basic Installation

Install the latest stable version from PyPI:

```bash
pip install "pykokoro[cpu]"
```

The `cpu`, `gpu`, `openvino`, and `directml` extras are alternative ONNX Runtime
distributions. Install exactly one provider extra per environment.

### Android/Termux Providers

PyKokoro accepts any provider spelling reported by the installed ONNX Runtime. For the
Android/Termux runtime, both aliases and runtime names are valid:

```python
from pykokoro import Kokoro

kokoro = Kokoro(provider="nnapi")
kokoro = Kokoro(provider="xnnpack")
```

Use `provider="auto"` to select the highest-priority available provider by capability.
PyKokoro does not infer availability from platform names and does not suppress ONNX
Runtime warnings.

## GPU Support

For GPU acceleration, install with the GPU extras:

### NVIDIA CUDA

```bash
pip install "pykokoro[gpu]"
```

This installs `onnxruntime-gpu` for NVIDIA CUDA support.

### AMD ROCm

For AMD GPUs with ROCm:

```bash
pip install "pykokoro[cpu]"
pip install onnxruntime-rocm
```

### Custom ONNX Runtime

You can also install a specific ONNX Runtime version separately:

```bash
pip install pykokoro
pip install onnxruntime-gpu==1.19.2  # or your preferred version
```

## System Requirements

### Python Version

- Python 3.10 or higher
- Tested on Python 3.10, 3.11, 3.12, and 3.13

### Dependencies

Core dependencies (automatically installed):

- `kokorog2p` - Text-to-phoneme conversion
- `phrasplit` - Intelligent text splitting
- `numpy` - Model tensors, voice vectors, and array operations
- `audiosig` - Audio signal processing for trim, VAD, resampling, gain, pitch, and rate
- `soundfile` - Audio file I/O
- `platformdirs`, `chardet`, `charset-normalizer`, `huggingface-hub`, `ssmd`, `num2words`,
  `babel`, and `typing_extensions` - runtime support

Optional dependencies:

- `onnxruntime-gpu` - For GPU acceleration
- `spacy` - For sentence/clause splitting and spaCy-aware G2P tokenization

SSMD volume, pitch, and rate processing is provided by core AudioSig. No librosa,
SciPy, audiomentations, signalsmith-stretch, or Python-stretch package is required.

### Installing espeak-ng

PyKokoro requires `espeak-ng` to be installed on your system.

**Ubuntu/Debian:**

```bash
sudo apt-get install espeak-ng
```

**macOS (Homebrew):**

```bash
brew install espeak-ng
```

**Windows:**

Download and install from: <https://github.com/espeak-ng/espeak-ng/releases>

Or use Chocolatey:

```bash
choco install espeak-ng
```

### Installing spaCy (Optional)

For advanced text splitting with `pause_mode="auto"` and language-aware spaCy
tokenization in G2P:

```bash
pip install spacy
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_md
```

```{eval-rst}
.. note::

   ``TokenizerConfig.spacy_model`` defaults to ``"auto"`` and resolves package
   names from language + size (default size: ``md``). For example:
   ``en-us -> en_core_web_md`` and ``de -> de_core_news_md``.
```

## Development Installation

To install from source for development:

```bash
git clone https://github.com/remixer-dec/pykokoro.git
cd pykokoro
pip install -e ".[dev]"
```

This installs PyKokoro in editable mode with development dependencies.

## Verifying Installation

Test your installation:

```python
import pykokoro

print(pykokoro.__version__)

# Quick test
kokoro = pykokoro.Kokoro()
audio, sr = kokoro.create("Hello, world!", voice="af_bella")
print(f"Generated {len(audio)} audio samples at {sr} Hz")
kokoro.close()
```

## Troubleshooting

### Import Errors

If you get import errors, ensure all dependencies are installed:

```bash
pip install --upgrade pykokoro
```

### espeak-ng Not Found

If you get errors about espeak-ng not being found:

1. Verify espeak-ng is installed: `espeak-ng --version`
2. Ensure it's in your system PATH
3. On Windows, you may need to restart your terminal after installation

### GPU Not Detected

If GPU acceleration isn't working:

1. Verify CUDA/ROCm is installed: `nvidia-smi` (NVIDIA) or `rocm-smi` (AMD)
2. Check ONNX Runtime GPU:
   `python -c "import onnxruntime; print(onnxruntime.get_available_providers())"`
3. Ensure you have the correct ONNX Runtime version for your CUDA version

### Model Download Issues

If model downloads fail:

1. Check your internet connection
2. Verify you have write permissions to the cache directory
3. Try downloading manually and placing in `~/.cache/pykokoro/`
4. For GitHub models, ensure the release URLs are accessible

**Manual Model Download:**

PyKokoro automatically downloads models on first use, but you can trigger downloads
manually:

```python
from pykokoro import Kokoro

# HuggingFace v1.0 (default - 54 voices, 8 quality options)
kokoro = Kokoro(model_quality="fp16")  # Auto-downloads from HuggingFace

# HuggingFace v1.1-zh (103 voices, 8 quality options)
kokoro = Kokoro(
    model_variant="v1.1-zh",
    model_quality="q8",  # Auto-downloads from HuggingFace
)

# GitHub v1.0 (54 voices, 4 quality options)
kokoro = Kokoro(
    model_source="github",
    model_variant="v1.0",
    model_quality="fp16-gpu",  # Auto-downloads from GitHub
)

# GitHub v1.1-zh (103 voices, fp32 only)
kokoro = Kokoro(
    model_source="github",
    model_variant="v1.1-zh",
    model_quality="fp32",  # Auto-downloads from GitHub
)
```

Models are cached in:

- **HuggingFace v1.0**: `~/.cache/pykokoro/models/huggingface/v1.0/` and
  `~/.cache/pykokoro/voices/huggingface/v1.0/`
- **HuggingFace v1.1-zh**: `~/.cache/pykokoro/models/huggingface/v1.1-zh/` and
  `~/.cache/pykokoro/voices/huggingface/v1.1-zh/`
- **GitHub v1.0**: `~/.cache/pykokoro/models/github/v1.0/` and
  `~/.cache/pykokoro/voices/github/v1.0/`
- **GitHub v1.1-zh**: `~/.cache/pykokoro/models/github/v1.1-zh/` and
  `~/.cache/pykokoro/voices/github/v1.1-zh/`

The config is stored under `~/.cache/pykokoro/config/{variant}/config.json`. A
source/variant/quality asset set is complete only when its config, model, and exact
voice archive are all nonempty regular files. Downstream integrations can inspect the
same paths without importing ONNX Runtime:

```python
from pykokoro.model_assets import get_model_asset_paths

assets = get_model_asset_paths(source="github", variant="v1.0", quality="fp32")
if not assets.complete:
    print("Missing:", assets.missing)
```
