from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from examples import german_tts_samples as example


def test_canonical_samples_are_exact() -> None:
    assert example.SAMPLES == (
        (
            "sentence_1",
            "An den Wochenenden bin ich jetzt immer nach Hause gefahren und habe Agnes "
            "besucht. Dabei war eigentlich immer sehr schönes Wetter gewesen.",
        ),
        (
            "sentence_2",
            "Dr. A. Smithe von der NATO (und nicht vom CIA) versorgt z.B. - meines "
            "Wissens nach - die Heroin seit dem 15.3.00 tgl. mit 13,84 Gramm Heroin "
            "zu 1,04 DM das Gramm.",
        ),
        (
            "sentence_3",
            "Die Manpowerdiskussion wird gecancelt, du kannst das File vom Server downloaden.",
        ),
    )

    sentence_2 = dict(example.SAMPLES)["sentence_2"]
    sentence_3 = dict(example.SAMPLES)["sentence_3"]
    for fragment in (
        "Dr. A. Smithe",
        "NATO",
        "CIA",
        "z.B. - meines Wissens nach -",
        "15.3.00",
        "tgl.",
        "13,84",
        "1,04 DM",
    ):
        assert fragment in sentence_2
    for fragment in ("Manpowerdiskussion", "gecancelt", "File", "Server", "downloaden"):
        assert fragment in sentence_3


def test_model_inventory_includes_current_german_models() -> None:
    assert set(example.available_german_models()) >= {
        "v1.2-de-martin",
        "de-crane",
        "de-thorsten",
    }
    assert example.available_german_models()[0] == example.DEFAULT_MODEL


def test_help_lists_all_model_and_lexicon_choices() -> None:
    help_text = example.build_parser().format_help()

    for model in example.available_german_models():
        assert model in help_text
    for lexicon in example.LEXICON_CHOICES:
        assert lexicon in help_text
    assert "Martin v1.2" in help_text
    assert "Kerstin / Crane" in help_text
    assert "Thorsten" in help_text


def test_default_cli_selection() -> None:
    args = example.build_parser().parse_args([])

    assert args.model == "v1.2-de-martin"
    assert args.lexicon == "gold"


@pytest.mark.parametrize(
    "arguments",
    [
        ["--model", "unknown-model"],
        ["--lexicon", "unknown-lexicon"],
    ],
)
def test_invalid_choices_fail_in_argparse(arguments: list[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        example.build_parser().parse_args(arguments)

    assert exc_info.value.code == 2


@pytest.mark.parametrize(
    ("model_id", "voice", "experimental"),
    [
        ("v1.2-de-martin", "martin", False),
        ("de-crane", "default", True),
        ("de-thorsten", "thorsten", False),
    ],
)
def test_model_configuration(model_id: str, voice: str, experimental: bool) -> None:
    config = example.make_config(model_id=model_id, lexicon="olaph")

    assert config.voice == voice
    assert config.generation.lang == "de"
    assert config.generation.speed == 1.0
    assert config.model_variant == model_id
    assert config.model_source == "github"
    assert config.model_quality == "fp32"
    assert config.allow_experimental_frontend is experimental
    assert config.tokenizer_config is not None
    assert config.tokenizer_config.lexicons == ("olaph",)


def test_help_does_not_load_remote_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_if_called() -> None:
        raise AssertionError("remote registry must not be loaded for --help")

    from pykokoro import model_registry

    monkeypatch.setattr(model_registry, "load_registry", fail_if_called)

    help_text = example.build_parser().format_help()
    assert "v1.2-de-martin" in help_text


def test_one_pipeline_runs_three_samples_and_writes_three_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pipeline_instances: list[object] = []
    runs: list[str] = []
    writes: list[tuple[Path, object, int]] = []

    class FakePipeline:
        def __init__(self, config: object) -> None:
            self.config = config
            self.enter_count = 0
            self.exit_count = 0
            pipeline_instances.append(self)

        def __enter__(self) -> FakePipeline:
            self.enter_count += 1
            return self

        def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
            self.exit_count += 1

        def run(self, text: str) -> SimpleNamespace:
            runs.append(text)
            audio = np.array([len(runs)], dtype=np.float32)
            return SimpleNamespace(audio=audio, sample_rate=22050, trace=None)

    def fake_write(path: Path, audio: object, sample_rate: int) -> None:
        writes.append((path, audio, sample_rate))

    monkeypatch.setattr(example, "KokoroPipeline", FakePipeline)
    monkeypatch.setattr(example.sf, "write", fake_write)
    monkeypatch.setenv("PYKOKORO_EXAMPLE_OUTPUT_DIR", str(tmp_path))

    outputs = example.synthesize_samples(model_id="de-crane", lexicon="olaph")

    assert len(pipeline_instances) == 1
    pipeline = pipeline_instances[0]
    assert pipeline.enter_count == 1
    assert pipeline.exit_count == 1
    assert runs == [text for _sample_id, text in example.SAMPLES]
    assert len(writes) == 3
    assert [sample_rate for _path, _audio, sample_rate in writes] == [22050] * 3
    assert [path.name for path, _audio, _sample_rate in writes] == [
        "sentence_1.wav",
        "sentence_2.wav",
        "sentence_3.wav",
    ]
    assert [audio.tolist() for _path, audio, _sample_rate in writes] == [[1.0], [2.0], [3.0]]
    assert all(
        path.parent == tmp_path / "german_tts_samples" / "de-crane" / "olaph"
        for path, _audio, _sample_rate in writes
    )
    assert [path for path, _audio, _sample_rate in writes] == outputs
