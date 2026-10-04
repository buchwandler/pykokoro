# Voice enrollment

PyKokoro exposes two distinct enrollment engines through
`KokoroSynthesizer.enroll_voice()`. Inno creates a normal static Kokoro voice pack from
reference audio only. AkinVox creates a transcript-conditioned `ReferenceVoice` for its
cloning model. These outputs have separate formats and synthesis paths.

## Inno voice tuning

Inno v0.2 is a zero-shot voice tuner, not a strong identity cloner. It requires
reference audio but no transcript, and returns a standard `[510, 1, 256]`
`KokoroVoicePack` usable immediately with ordinary static Kokoro synthesis:

```python
from pykokoro import KokoroSynthesizer, KokoroVoicePack

with KokoroSynthesizer() as synthesizer:
    pack = synthesizer.enroll_voice("reference.wav", engine="inno", language="en")
    pack.save("speaker.npz")

    rendered = synthesizer.synthesize_text(
        "New words in the tuned voice.", language="en-us", voice=pack
    )
    rendered.save_wav("tuned.wav")

reloaded = KokoroVoicePack.load("speaker.npz")
```

The default engine is Inno, so omit `reference_text`. Supplying a transcript to Inno is
an error. `InnoEnrollmentOptions(fmax=...)` can forward the supported `fmax` tuning
option. The reference audio is read, downmixed to finite mono float32, and passed with
its original sample rate. No raw audio is stored in the voice pack.

Enrollment requires a model whose runtime metadata advertises the `inno-v0.2` enroller.
The default selection is Kokoro v1.0. For an explicitly selected model, enrollment is
rejected unless that model advertises a compatible Inno capability. Inspect this
metadata without loading model weights:

```python
from pykokoro import discover_models

for model in discover_models(offline=True).models:
    for enroller in model.voice_enrollers:
        print(model.model_id, enroller.id, enroller.min_seconds, enroller.max_seconds)
```

Enroller metadata includes transcript requirements, supported reference duration, and
output format. The installed OnnxVoice distribution must implement the advertised
enrollment contract. PyKokoro does not depend on the Inno source package, PyTorch,
torchaudio, or Transformers.

`KokoroVoicePack.save()` writes a versioned, pickle-free NPZ archive with the style
data, format, base-model, engine, metadata, and fingerprint. `KokoroVoicePack.load()`
validates the archive and fingerprint. `KokoroVoicePack.from_array()` imports a plain
compatible NumPy array. A pack is not a `ReferenceVoice`, and the two formats are not
interchangeable.

Generated packs have no catalog voice calibration. Voice-level mode `off` and an
explicit gain override are supported. Automatic named-voice calibration is unavailable.
Dynamic-pack blending and legacy `.pt` import are not provided.

## AkinVox reference cloning

AkinVox requires reference audio and its exact transcript. Select it explicitly. The
result is a model-bound `ReferenceVoice` state, not a static voice pack:

```python
from pykokoro import KokoroSynthesizer, ReferenceVoice

with KokoroSynthesizer() as synthesizer:
    voice = synthesizer.enroll_voice(
        "reference.wav",
        "The exact words spoken in the reference.",
        engine="akinvox",
        language="en-us",
        name="speaker",
    )
    voice.save("speaker-reference.npz")

    reusable = ReferenceVoice.load("speaker-reference.npz")
    rendered = synthesizer.synthesize_text(
        "New words with reference conditioning.", language="en-us", voice=reusable
    )
    rendered.save_wav("cloned.wav")
```

The maintained [`reference_voice.py` example](../examples/reference_voice.py) accepts a
reference-audio path, its exact transcript, and target text:

```bash
python examples/reference_voice.py reference.wav \
  "The exact words spoken in the reference." \
  "New words spoken with the enrolled voice."
```

AkinVox enrollment or synthesis may provision the cloning model through OnnxVoice. Reuse
a saved state with `ReferenceVoice.load()`; reference audio is not needed again.

### Reference requirements and limits

- Version 1 supports English only. Use `en` or `en-us` for enrollment and synthesis.
- Reference audio must be 3 to 30 seconds long, mono or multichannel, and supplied as a
  WAV/file path or NumPy waveform with an explicit sample rate.
- Use a clean, intelligible recording of one speaker. Avoid background speech, music,
  strong reverberation, silence, and clipping.
- Audio is downmixed to mono and resampled with SoXR high-quality mode. The parity test
  against the upstream resampler checks relative RMS error below `5e-4` and maximum
  absolute error below `3e-4` on its deterministic test signal.
- Supply the exact words spoken in the reference. The transcript is phonemized by the
  English frontend to prepare enrollment tokens; it is not inferred by ASR.
- `ReferenceVoice` is bound to its cloning model ID and model fingerprint. A state
  cannot be used with a different model build. Keep the saved state when reusing the
  same model.
- Reference synthesis currently requires `GenerationConfig.speed == 1.0`.
- Automatic static-voice calibration is unavailable for a reference voice. Voice-level
  mode `off` and an explicit gain override are supported.

## Saved state and privacy

A `ReferenceVoice` NPZ stores model-bound conditioning tensors, their fingerprint, and
optional SHA-256 metadata. It does not contain the original audio or transcript. A
`KokoroVoicePack` stores only a normal static Kokoro style and metadata, not its source
audio. Conditioning state is derived from a person's voice and should be handled as
sensitive data. Protect and share it only with the speaker's permission. You must have
permission to use reference voices for synthesis.

The runtime uses OnnxVoice and ONNX Runtime. PyKokoro does not require PyTorch,
torchaudio, Transformers, or the AkinVox Python package for these features. Reference
audio and transcript content are not written into synthesis traces by default.

See the [API reference](api_reference.md) for public enrollment types and
[model discovery](languages.md) for runtime capability metadata.
