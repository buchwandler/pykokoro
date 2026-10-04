from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from examples import (
    asset_progress,
    error_handling,
    frontend_and_lexicons,
    language_routing,
    long_text,
    reference_voice,
    result_metadata,
    voice_blend,
)
from pykokoro import AssetProgressEvent, ConsoleAssetProgress, ReferenceVoice


def test_long_text_example_keeps_one_unmodified_request() -> None:
    request = long_text.make_request()
    config = long_text.make_config()

    assert request.text == long_text.LONG_TEXT
    assert request.text.count(".") == 8
    assert request.language == "en-us"
    assert config.long_text_split == "sentence"
    assert config.long_text_use_spacy is False
    assert config.return_trace is True


def test_voice_blend_example_shows_linear_and_slerp_construction() -> None:
    explicit = voice_blend.explicit_blend()
    parsed = voice_blend.parsed_blend()
    spherical = voice_blend.slerp_blend()
    config = voice_blend.make_config(parsed)

    assert explicit.voices == [("af_sarah", 0.6), ("af_bella", 0.4)]
    assert explicit.interpolation == "linear"
    assert parsed.voices == explicit.voices
    assert parsed.interpolation == "linear"
    assert spherical.interpolation == "slerp"
    assert spherical.voices == parsed.voices
    assert ":" not in voice_blend.LINEAR_BLEND
    assert ":" not in voice_blend.SLERP_BLEND
    assert config.voice is parsed


def test_language_routing_example_keeps_acoustic_request_explicit() -> None:
    config = language_routing.make_config()
    automatic, deterministic = language_routing.make_requests()

    assert config.language_routing is not None
    assert config.language_routing.mode == "auto"
    assert config.language_routing.languages == ("en-us", "de-de")
    assert automatic.language == deterministic.language == "en-us"
    assert automatic.voice == deterministic.voice == "af_sarah"
    override = deterministic.pronunciation_overrides[0]
    assert deterministic.text[override.start : override.end] == "Welt"
    assert override.language == "de-de"


def test_frontend_example_separates_discovery_from_provisioning() -> None:
    tokenizer = frontend_and_lexicons.make_tokenizer_config()
    config = frontend_and_lexicons.make_synthesis_config()

    assert tokenizer.backend == "kokorog2p"
    assert tokenizer.fallback == "espeak"
    assert tokenizer.lexicons == ("gold",)
    assert tokenizer.lexicon_data_policy == "installed-only"
    assert tokenizer.use_spacy is False
    assert config.tokenizer_config is tokenizer or config.tokenizer_config == tokenizer


def test_frontend_example_uses_offline_discovery_only(monkeypatch, capsys) -> None:
    calls: list[dict[str, object]] = []

    def discover(*, language: str, offline: bool):
        calls.append({"language": language, "offline": offline})
        return SimpleNamespace(lexicons=())

    monkeypatch.setattr(frontend_and_lexicons, "discover_lexicons", discover)
    frontend_and_lexicons.main()

    assert calls == [{"language": "de", "offline": True}]
    assert "no lexicon data or model assets are installed" in capsys.readouterr().out


def test_asset_progress_supports_console_and_custom_callback(capsys) -> None:
    assert isinstance(asset_progress.make_config().asset_progress, ConsoleAssetProgress)
    callback = asset_progress.make_config(custom_callback=True).asset_progress
    assert callback is asset_progress.progress
    callback(
        AssetProgressEvent(
            phase="download-progress",
            model_id="v1.0",
            distribution_id="cpu",
            artifact_id="model.onnx",
            role="model",
            filename="model.onnx",
            bytes_done=10,
            bytes_total=20,
            target="cache",
        )
    )
    assert "download-progress model.onnx 10 20" in capsys.readouterr().out


def test_result_metadata_and_error_examples_use_public_configuration() -> None:
    config = result_metadata.make_config()
    request = error_handling.make_request()

    assert config.generation.random_seed == 7
    assert config.return_trace is True
    assert request.text[6:10] == "Welt"
    assert request.pronunciation_overrides[0].language == "de-de"


class _FakeResult:
    def __init__(self, *, id: str, text: str, language: str, voice: object) -> None:
        self.id = id
        self.text = text
        self.language = language
        self.voice = voice
        self.audio = np.zeros(4, dtype=np.float32)
        self.sample_rate = 24_000
        self.phonemes = "test-phonemes"
        self.token_ids = (1, 2, 3)
        self.diagnostics = ()
        self.short_sentence_mode = "off"
        self.voice_level_applications = ()
        self.synthesis_identity = SimpleNamespace(cache_key="offline-test-key")
        self.word_timings = ()
        self.trace = SimpleNamespace(
            model={"acoustic_chunk_count": 1},
            inference_summary=lambda: "offline test trace",
        )

    def save_wav(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"fake wav")


class _FakeSynthesizer:
    def __init__(self, config: object) -> None:
        self.config = config

    def __enter__(self) -> _FakeSynthesizer:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def synthesize_text(self, text: str, *, language: str, voice: object = None) -> _FakeResult:
        selected_voice = voice if voice is not None else getattr(self.config, "voice", None)
        return _FakeResult(id="fake", text=text, language=language, voice=selected_voice)

    def synthesize(self, request) -> _FakeResult:
        return _FakeResult(
            id=request.id,
            text=request.text,
            language=request.language,
            voice=request.voice,
        )

    def synthesize_segments(self, requests) -> Iterator[_FakeResult]:
        for request in requests:
            yield self.synthesize(request)


def test_reference_voice_example_reloads_saved_state_before_synthesis(
    monkeypatch, tmp_path: Path
) -> None:
    voice = ReferenceVoice(
        style=np.zeros((1, 256), dtype=np.float32),
        memory=np.ones((1, 2, 192), dtype=np.float32),
        memory_mask=np.ones((1, 2), dtype=np.bool_),
        model_id="en-akinvox-cloning-v1",
        model_fingerprint="example-model-fingerprint",
    )
    calls: list[tuple[object, ...]] = []

    class FakeSynthesizer:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def enroll_voice(self, audio, transcript, **kwargs):
            calls.append(("enroll", audio, transcript, kwargs))
            return voice

        def synthesize_text(self, text, *, language, voice):
            calls.append(("synthesize", text, language, voice.fingerprint))
            return _FakeResult(id="reference", text=text, language=language, voice=voice)

    monkeypatch.setattr(reference_voice, "KokoroSynthesizer", FakeSynthesizer)
    state_path = tmp_path / "nested" / "speaker.npz"
    output_path = tmp_path / "audio" / "cloned.wav"

    reference_voice.run(
        tmp_path / "reference.wav",
        "Exact reference transcript.",
        "New target text.",
        state_path,
        output_path,
    )

    loaded = ReferenceVoice.load(state_path)
    assert loaded.fingerprint == voice.fingerprint
    assert calls[0] == (
        "enroll",
        tmp_path / "reference.wav",
        "Exact reference transcript.",
        {"engine": "akinvox", "language": "en-us", "name": "speaker"},
    )
    assert calls[1] == ("synthesize", "New target text.", "en-us", voice.fingerprint)
    assert output_path.read_bytes() == b"fake wav"


def test_new_synthesis_examples_write_expected_outputs_without_inference(
    monkeypatch, tmp_path: Path
) -> None:
    examples = (
        asset_progress,
        error_handling,
        language_routing,
        long_text,
        result_metadata,
        voice_blend,
    )
    for module in examples:
        monkeypatch.setattr(module, "KokoroSynthesizer", _FakeSynthesizer)
        monkeypatch.setattr(
            module,
            "artifact_path",
            lambda name, module=module: tmp_path / module.__name__.removeprefix("examples.") / name,
        )

    asset_progress.main([])
    error_handling.main()
    language_routing.main()
    long_text.main()
    result_metadata.main()
    voice_blend.main()

    expected = {
        "asset_progress/asset_progress.wav",
        "error_handling/error_handling.wav",
        "language_routing/automatic-routing.wav",
        "language_routing/explicit-german-span.wav",
        "long_text/long_text.wav",
        "result_metadata/result_metadata.wav",
        "voice_blend/voice_blend.wav",
        "voice_blend/voice_blend_slerp.wav",
    }
    actual = {path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*.wav")}
    assert actual == expected
