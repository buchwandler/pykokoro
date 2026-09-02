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

| Model           | Variant          | Voice      | Status        | Selection                                     |
| --------------- | ---------------- | ---------- | ------------- | --------------------------------------------- |
| Martin          | `v1.2-de-martin` | `martin`   | ready/default | automatic for `lang="de"`                     |
| Kerstin / Crane | `de-crane`       | `default`  | experimental  | explicit + `allow_experimental_frontend=True` |
| Thorsten        | `de-thorsten`    | `thorsten` | ready         | explicit                                      |

The document language is explicit in v0.9. Pass `GenerationConfig(lang="de")` or
`run(..., lang="de")`; voice and model profiles never infer it. Martin is the default
German acoustic model when that language is selected. Kerstin/Crane and Thorsten are
explicit alternative acoustic models; Crane currently requires experimental frontend
opt-in. Compare them with the maintained examples `examples/german.py`,
`examples/german2.py`, and `examples/german3.py`.

The integrated pipeline performs Spokenform preparation before sentence segmentation and
G2P. Spokenform owns written-to-spoken forms such as dates, measurements, ordinals,
currency amounts, and abbreviations. Explicit mixed-language text uses SSMD `lang`
spans. Prepared segment offsets refer to the prepared document text and remain exact
half-open slices.

Managed Martin cache hits are checked against the pinned SHA-256 digests before use;
invalid files are removed and downloaded again. Explicit custom paths are validated in
place and are never replaced by managed downloads. The profile's `suggested_speed`
metadata is advisory only, so callers should set `GenerationConfig(speed=1.125)` when
they want the demonstration speed.
