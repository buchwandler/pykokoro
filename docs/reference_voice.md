# English reference voice cloning

PyKokoro can enroll a reusable English reference voice and synthesize new text with it
through the ONNX Runtime backend. Ordinary synthesis without a `ReferenceVoice`
continues to use the existing static Kokoro voice profiles.

## Enroll, save, and reuse

```python
from pykokoro import KokoroSynthesizer, ReferenceVoice

with KokoroSynthesizer() as synthesizer:
    voice = synthesizer.enroll_voice(
        "reference.wav",
        "The exact words spoken in the reference.",
        language="en-us",
        name="speaker",
    )
    voice.save("speaker.npz")

    reusable_voice = ReferenceVoice.load("speaker.npz")
    result = synthesizer.synthesize_text(
        "New words spoken with the enrolled voice.",
        language="en-us",
        voice=reusable_voice,
    )
    result.save_wav("cloned.wav")
```

The maintained [`reference_voice.py` example](../examples/reference_voice.py) accepts a
reference-audio path, its exact transcript, and the target text:

```bash
python examples/reference_voice.py reference.wav \
  "The exact words spoken in the reference." \
  "New words spoken with the enrolled voice."
```

The first enrollment or synthesis may provision the cloning model through OnnxVoice. To
reuse an already saved state in a later process, load it with `ReferenceVoice.load()`
and pass it as `voice`; reference audio is not needed again.

## Reference requirements and limits

- Version 1 supports English only. Use `en` or `en-us` for enrollment and synthesis.
- Reference audio must be 3 to 30 seconds long, mono or multichannel, and supplied as a
  WAV/file path or NumPy waveform with an explicit sample rate.
- Use a clean, intelligible recording of one speaker. Avoid background speech, music,
  strong reverberation, silence, and clipping.
- Audio is downmixed to mono and resampled with SoXR high-quality mode. The parity test
  against the upstream resampler checks relative RMS error below `5e-4` and maximum
  absolute error below `3e-4` on its deterministic test signal.
- Supply the exact words spoken in the reference. The transcript is phonemized by the
  English frontend to prepare the enrollment tokens; it is not inferred by ASR.
- `ReferenceVoice` is bound to its cloning model ID and model fingerprint. A state
  cannot be used with a different model build. Keep the saved state when reusing the
  same model.
- Reference synthesis currently requires `GenerationConfig.speed == 1.0`.
- Automatic static-voice calibration is unavailable for a reference voice. Voice-level
  mode `off` and an explicit gain override are supported.

## Saved state and privacy

The NPZ file stores the model-bound conditioning tensors, their fingerprint, and
optional SHA-256 metadata. It does not contain the original audio or transcript.
Conditioning state is derived from a person's voice and should be handled as sensitive
data. Protect and share it only with the speaker's permission. You must have permission
to use the reference voice for synthesis.

The runtime uses OnnxVoice and ONNX Runtime. PyKokoro does not require PyTorch,
Torchaudio, Transformers, or the AkinVox Python package for this feature. Reference
audio and transcript content are not written into synthesis traces by default.

See [API reference](api_reference.md) for the public `ReferenceVoice` type and
[model discovery](languages.md) for profile capability metadata.
