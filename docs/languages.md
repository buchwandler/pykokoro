# Language and model profiles

PyKokoro resolves omitted model settings from the requested language before creating the
ONNX backend or G2P cache. Non-German runs retain the existing Hugging Face v1.0
default; Chinese automatic selection retains the GitHub v1.1-zh policy.

## German

`de`, `de-de`, `de-at`, and `de-ch` automatically select:

- source: GitHub
- variant: `v1.2-de-martin`
- quality: `fp32`
- voice: `martin`
- vocabulary: built-in Kokoro v1.0

The Martin profile is a single-speaker model. Its ONNX file is
`kokoro-german-martin-v1.2.onnx`; its voice archive is `voices-german-martin-v1.2.bin`.
Both downloads are checked with SHA-256 and the voice archive must contain `martin`. No
model-specific config file is downloaded.


Automatic written-to-spoken preparation is owned by the compatible kokorog2p 0.8.x
dependency across the language pipelines where kokorog2p supports it. This includes
structured forms such as dates, times, decimal commas, measurements, ordinals,
durations, currency amounts, and abbreviations before G2P. Sentence segmentation and
source offsets continue to refer to the original document text. PyKokoro's supported
language and model list remains authoritative; kokorog2p support alone does not add a
new synthesizer language.

Managed Martin cache hits are checked against the pinned SHA-256 digests before use;
invalid files are removed and downloaded again. Explicit custom paths are validated in
place and are never replaced by managed downloads. The profile's `suggested_speed`
metadata is advisory only, so callers should set `GenerationConfig(speed=1.125)` when
they want the demonstration speed.
