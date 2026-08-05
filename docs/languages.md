# Language and model profiles

PyKokoro resolves omitted model settings from the requested language before creating
the ONNX backend or G2P cache. Non-German runs retain the existing Hugging Face v1.0
default; Chinese automatic selection retains the GitHub v1.1-zh policy.

## German

`de`, `de-de`, `de-at`, and `de-ch` automatically select:

- source: GitHub
- variant: `v1.2-de-martin`
- quality: `fp32`
- voice: `martin`
- vocabulary: built-in Kokoro v1.0

The Martin profile is a single-speaker model. Its ONNX file is
`kokoro-german-martin-v1.2.onnx`; its voice archive is
`voices-german-martin-v1.2.bin`. Both downloads are checked with SHA-256 and the
voice archive must contain `martin`. No model-specific config file is downloaded.

The legacy Eva/Bernd profile remains explicit:

```python
PipelineConfig(
    voice="df_eva",
    model_source="github",
    model_variant="v1.1-de",
    generation=GenerationConfig(lang="de"),
)
```

German structured normalization is owned by the kokorog2p dependency. The compatible
kokorog2p release should expand dates, times, decimal commas, measurements, ordinals,
durations, Euro amounts, and abbreviations such as `Prof.`, `ggf.`, `ca.`, and `zzgl.`
before G2P. Sentence segmentation and source offsets continue to refer to the original
document text.

Managed Martin cache hits are checked against the pinned SHA-256 digests before use;
invalid files are removed and downloaded again. Explicit custom paths are validated in
place and are never replaced by managed downloads. The profile's `suggested_speed`
metadata is advisory only, so callers should set `GenerationConfig(speed=1.125)` when
they want the demonstration speed.
