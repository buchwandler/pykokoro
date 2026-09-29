# Installation

PyKokoro requires Python 3.10 or newer. Install the package with exactly one supported
ONNX Runtime provider extra:

```bash
pip install "pykokoro[cpu]"        # CPU
pip install "pykokoro[gpu]"        # NVIDIA CUDA
pip install "pykokoro[openvino]"   # OpenVINO
pip install "pykokoro[directml]"  # DirectML
pip install "pykokoro[coreml]"     # Apple CoreML
```

These provider extras select alternative ONNX Runtime distributions. Do not install
multiple provider extras into one environment unless you deliberately manage the
resulting runtime packages yourself.

## Optional playback

WAV writing is available through the required `soundfile` dependency. Direct playback is
optional:

```bash
pip install "pykokoro[cpu,playback]"
```

`RenderedSegment.play()` plays that one waveform using `sounddevice`; it does not
compose several requests. Linux systems may also need a PortAudio system package.

## Frontends and lexicons

PyKokoro uses KokoroG2P for prepared-text phonemization. Install `espeak-ng` when using
an espeak frontend or fallback. Named lexicon data can be provisioned by Lexphon on
first use; for offline operation, install the required data ahead of time and select an
installed-only lexicon policy in `TokenizerConfig`.

```python
from pykokoro import TokenizerConfig, discover_lexicons

inventory = discover_lexicons(language="de-de", offline=True)
for lexicon in inventory.lexicons:
    print(lexicon.selector, lexicon.installed, lexicon.model_support)

tokenizer = TokenizerConfig(
    backend="kokorog2p",
    fallback="espeak",
    lexicons=("gold",),
    lexicon_data_policy="installed-only",
    use_spacy=False,  # no spaCy model required for this recipe
)
```

Discovery inspects metadata; it does not download or install lexicon data. The
`installed-only` policy prevents implicit lexicon data provisioning during synthesis.
See [`frontend_and_lexicons.py`](../examples/frontend_and_lexicons.py).

```bash
lexphon data available de-DE
lexphon data install de-de:gold
lexphon data verify de-de:gold
```

Set `LEXPHON_DATA_HOME` to choose a persistent data directory. `discover_lexicons()`
lists lexicon metadata without running synthesis.

## Model assets and discovery

Model and voice assets are resolved lazily when synthesis first needs them.
`SynthesisConfig.cache_dir` chooses the cache directory passed to OnnxVoice for managed
model/voice assets. The default `None` delegates to the backend's normal cache.
Caller-owned custom model and voice files are not moved or replaced.

```python
from pathlib import Path
from pykokoro import SynthesisConfig

config = SynthesisConfig(cache_dir=Path.home() / ".cache" / "pykokoro")
```

For metadata-only runtime inventory, call `discover_models(offline=True)`:

```python
from pykokoro import discover_models

inventory = discover_models(offline=True)
for model in inventory.models:
    print(model.model_id, model.languages, model.voices, model.status)
```

This does not load model weights or create an ONNX session.

## Managed asset progress

Attach `ConsoleAssetProgress()` for a ready-made terminal reporter, or provide a
callback that receives typed `AssetProgressEvent` values. Notifications cover managed
model assets; they do not report lexicon provisioning. First-time synthesis can download
assets unless they are already cached. The example script may require network access and
model storage:

```python
from pykokoro import ConsoleAssetProgress, SynthesisConfig

config = SynthesisConfig(asset_progress=ConsoleAssetProgress())
```

For a custom callback, inspect `event.phase`, `filename`, and byte counts. See
[`asset_progress.py`](../examples/asset_progress.py).

A custom callback can use the public event fields directly:

```python
from pykokoro import AssetProgressEvent, SynthesisConfig

def report_progress(event: AssetProgressEvent) -> None:
    print(event.phase, event.filename, event.bytes_done, event.bytes_total)

config = SynthesisConfig(asset_progress=report_progress)
```

Explicit custom `model_path` and `voices_path` values are used in place and validated;
they are not silently replaced with managed downloads.

## Development install

```bash
git clone https://github.com/buchwandler/pykokoro.git
cd pykokoro
pip install -e ".[dev,cpu]"
```

## Dependency boundary

The PyKokoro runtime depends on KokoroG2P and OnnxVoice, plus the libraries used by its
request-local audio and asset paths. Utterplan, SSMD, and AudioCompose are outside this
package's runtime dependency boundary. Applications can prepare text and resolve speech
plans before calling PyKokoro, then compose returned waveforms in their own output
layer.
