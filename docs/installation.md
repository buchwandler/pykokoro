# Installation Guide

PyKokoro requires Python 3.10 or newer. Install the CPU provider extra for the standard
ONNX Runtime setup:

```bash
pip install "pykokoro[cpu]"
```

The `cpu`, `gpu`, `openvino`, and `directml` extras are alternative ONNX Runtime
distributions; install exactly one provider extra per environment. Importing the
pipeline and using fully custom stages does not require ONNX Runtime, but the default
audio stages do.

## Android/Termux providers

Use a provider name exposed by the installed ONNX Runtime build:

```python
from pykokoro import GenerationConfig, KokoroPipeline, PipelineConfig

with KokoroPipeline(PipelineConfig(generation=GenerationConfig(lang="en-us"), provider="nnapi")) as pipeline:
    result = pipeline.run("Hello from Android.")
```

`provider="auto"` selects the highest-priority available provider. PyKokoro does not
infer provider availability from platform names.

### Termux/Android model assets

HuggingFace remains the default model source. If HuggingFace downloads are unavailable
in Termux, select the self-contained GitHub v1.0 profile explicitly:

```python
from pykokoro import KokoroPipeline, PipelineConfig

pipeline = KokoroPipeline(
    PipelineConfig(
        voice="af_heart",
        model_source="github",
        model_variant="v1.0",
        model_quality="fp32",
    )
)
```

GitHub v1.0 uses the embedded standard v1.0 vocabulary and does not download HuggingFace
`config.json`. PyKokoro never silently changes the configured source. If `model_path`
and `voices_path` are supplied, each file is validated and used in place; missing custom
files do not trigger a managed-cache download. The `Unsupported platform (android)`
warning printed by some ONNX Runtime packages is independent of this model-source and
asset fix.

## Other providers

```bash
pip install "pykokoro[gpu]"       # NVIDIA CUDA
pip install "pykokoro[openvino]"  # OpenVINO Runtime
pip install "pykokoro[directml]"  # DirectML
```

For Lexphon Goruut provider fallback, install the supported extra:

```bash
pip install "pykokoro[goruut]"
```

This enables `fallback="goruut"` for the native `backend="kokorog2p"` path. It is
distinct from `backend="goruut"`, which selects Goruut as the primary backend. For a
custom ONNX Runtime distribution, install the base package and the provider package
separately:

```bash
pip install pykokoro
pip install onnxruntime-gpu==1.19.2
```

## Dependencies and optional spaCy

PyKokoro v0.9 requires `kokorog2p[espeak,en]>=0.9.4,<1.0`, `lexphon>=0.2.3,<0.3`,
`phrasplit>=0.3.9,<0.4`, `ssmd>=0.8.7,<0.9`, and `spokenform>=0.4.2,<0.5`. The native
KokoroG2P frontend uses named lexicons: `lexicons=None` selects KokoroG2P language
defaults, while `lexicons=()` disables static Lexphon layers. New code should use named
lexicons rather than the legacy `use_dictionary`, `load_gold`, and `load_silver`
compatibility inputs.

### Lexphon data provisioning

Before native `backend="kokorog2p"` construction, PyKokoro resolves the effective named
lexicons for the routed language and checks the local Lexphon store. The default
`lexicon_data_policy="auto"` installs only missing Lexphon-backed assets. Warm runs do
not consult the catalog or network. Primary `backend="espeak"` and `backend="goruut"`
paths never download static lexicons.

Use `lexicon_data_policy="installed-only"` for offline or pre-provisioned deployments.
PyKokoro will not install or consult the catalog in that mode, and a missing asset
raises Lexphon's original installation error. Set `LEXPHON_DATA_HOME` for a persistent
store and `LEXPHON_CATALOG_URL` for a pinned local or remote catalog. The document
language is explicit: pass `GenerationConfig(lang="en-us")` or `run(..., lang="en-us")`.
Voice and profile selection never supplies the document language. SSMD `lang` spans are
the supported mechanism for explicit mixed-language documents.

The pipeline owns reusable spaCy resources for integrated Pass A and Pass B analysis:

- `use_spacy=False` disables spaCy;
- `use_spacy=None` selects the best compatible installed local model and falls back
  without downloading when none is available;
- `use_spacy=True`, an explicit model, or an exact model size is strict and remains
  offline.

Install a model only when you want spaCy-aware splitting or G2P:

```bash
pip install spacy
python -m spacy download en_core_web_sm
```

No spaCy model is downloaded automatically. The native kokorog2p backend supports the
languages declared by `pykokoro.constants.SUPPORTED_LANGUAGES`; languages in
`ESPEAK_ONLY_LANGUAGES` require an explicit fallback backend.

## German Martin assets

German runs automatically select the single-speaker GitHub `v1.2-de-martin` profile when
no model or voice is supplied. It provides only `fp32` and downloads approximately 311
MB for the ONNX model plus a 522,506-byte `martin` voice archive on first use. Both
artifacts are checked against their published SHA-256 digests before being cached under
`~/.cache/pykokoro`.

Custom `model_path` and `voices_path` are never replaced by automatic selection; missing
custom files fail directly rather than triggering a download to the shared cache.
Managed cache hits are checksum and structure checked before use. `offline=True` when a
valid managed cache is required. Interrupted GitHub transfers retain a temporary `.part`
file and resume with HTTP Range requests when the release host supports them; completed
files are still checked for exact size, SHA-256, and structure before replacement.

The registry catalog is cached at `~/.cache/pykokoro/registry/models.json`, with runtime
artifacts kept under model and distribution-specific directories. PyKokoro revalidates
cached files and replaces invalid artifacts individually. If a downloaded file reveals
stale catalog metadata, the catalog is refreshed once without falling back to the stale
cache, then asset resolution is retried. Offline mode never refreshes over the network,
and manual deletion of the registry catalog or model directory is not required.

### Observing first-run downloads

Managed runtime assets are provisioned lazily during the first synthesis. Pass
`asset_progress=ConsoleAssetProgress()` to show cold-cache downloads, byte progress, and
verification, or pass a callable to `PipelineConfig.asset_progress` to consume
structured `AssetProgressEvent` values. Valid cache hits remain quiet. Offline mode
raises the existing cache error without emitting a download-start event.

## Model capability discovery

Use the public discovery API to inspect models, voices, languages, qualities, frontends,
and runtime status without downloading model or voice assets:

```python
from pykokoro import discover_models

inventory = discover_models(offline=True)
for model in inventory.models:
    print(model.model_id, model.languages, model.voices, model.status)
```

The default call follows the registry network and cache policy. `offline=True` forbids
network access and requires cached registry metadata. `refresh=True` forces a registry
metadata refresh and may report `cache_fallback=True` if the existing policy uses a
valid cache after a failed refresh. Refresh never downloads model assets, and
`offline=True, refresh=True` is invalid. `registry_source` identifies the registry or
cache used.

`discover_models()` is runtime capability discovery. `available_model_releases()`
remains the API for published release and artifact discovery.

## System requirements

Install `espeak-ng` when using the espeak fallback or backend.

**Ubuntu/Debian:** `sudo apt-get install espeak-ng`

**macOS:** `brew install espeak-ng`

**Windows:** install a release from <https://github.com/espeak-ng/espeak-ng/releases> or
use `choco install espeak-ng`.

## Verify installation

The public API is pipeline-first:

```python
import pykokoro

from pykokoro import KokoroPipeline, PipelineConfig
from pykokoro.generation_config import GenerationConfig

print(pykokoro.__version__)
with KokoroPipeline(
    PipelineConfig(voice="af_bella", generation=GenerationConfig(lang="en-us"))
) as pipeline:
    result = pipeline.run("Hello, world!")
    print(f"Generated {len(result.audio)} samples at {result.sample_rate} Hz")
    result.release_audio()
```

For long-form output, use `prepare_units()` or `iter_units()` so only one paragraph
waveform is rendered at a time. Preparation still parses and phonemizes the complete
document globally. See `examples/paragraph_wave_export.py` for a resumable manifest.

## Development installation

```bash
git clone https://github.com/remixer-dec/pykokoro.git
cd pykokoro
pip install -e ".[dev]"
```

## Troubleshooting

If the default pipeline reports that ONNX Runtime is missing, install one provider
extra, for example `pip install "pykokoro[cpu]"`. If model loading fails, verify that
the provider is available and that the model/voice assets can be downloaded or supplied
through `PipelineConfig(model_path=..., voices_path=...)`.

For dependency-light diagnostics:

```python
from pykokoro.model_assets import get_model_asset_paths

assets = get_model_asset_paths(source="huggingface", variant="v1.0", quality="fp32")
print(assets.missing if not assets.complete else "model assets are ready")
```
