from __future__ import annotations

import os
import subprocess

from examples import run_all


def test_default_selection_is_lightweight_core_examples() -> None:
    names = {path.name for path in run_all._example_paths()}
    assert {
        "simple_synthesis.py",
        "request_batch.py",
        "pronunciation_overrides.py",
        "linguistic_tokens.py",
        "long_text.py",
        "voice_blend.py",
        "result_metadata.py",
        "models_and_languages.py",
    } == names
    assert "all_voices.py" not in names
    assert "short_sentence_demo.py" not in names
    assert "french.py" not in names


def test_feature_and_language_groups_are_separate() -> None:
    feature = {path.name for path in run_all._example_paths(group="feature")}
    language = {path.name for path in run_all._example_paths(group="language-showcase")}

    assert feature == {
        "english.py",
        "language_routing.py",
        "frontend_and_lexicons.py",
        "asset_progress.py",
        "error_handling.py",
    }
    assert language == {
        "chinese.py",
        "contractions.py",
        "french.py",
        "italian.py",
        "japanese.py",
        "korean.py",
        "portuguese.py",
        "spanish.py",
    }
    assert "simple_synthesis.py" not in language


def test_optional_selection_adds_heavy_showcases() -> None:
    names = {path.name for path in run_all._example_paths(include_optional=True)}
    assert {"all_voices.py", "short_sentence_demo.py"} <= names
    explicit = {path.name for path in run_all._example_paths(group="optional-heavy")}
    assert explicit == {"all_voices.py", "short_sentence_demo.py"}


def test_run_examples_continues_and_reports_failures(monkeypatch, tmp_path) -> None:
    paths = [
        run_all.PROJECT_ROOT / "examples" / "first.py",
        run_all.PROJECT_ROOT / "examples" / "second.py",
    ]
    calls: list[dict[str, object]] = []

    def fake_runner(command, **kwargs):
        calls.append(kwargs)
        return subprocess.CompletedProcess(command, 1 if len(calls) == 1 else 0)

    monkeypatch.setattr(run_all, "ARTIFACT_DIR", tmp_path / "example-artifacts")

    failures = run_all.run_examples(paths, runner=fake_runner)

    assert failures == 1
    assert len(calls) == 2
    assert all(call["cwd"] == run_all.PROJECT_ROOT for call in calls)
    assert all(call["check"] is False for call in calls)
    assert all(call["text"] is True for call in calls)
    environments = [call["env"] for call in calls]
    assert all(isinstance(environment, dict) for environment in environments)
    assert all("PYKOKORO_EXAMPLE_OUTPUT_DIR" in environment for environment in environments)
    assert os.fspath(environments[0]["PYKOKORO_EXAMPLE_OUTPUT_DIR"]).endswith("examples__first")
    assert os.fspath(environments[1]["PYKOKORO_EXAMPLE_OUTPUT_DIR"]).endswith("examples__second")


def test_list_mode_does_not_run(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        run_all, "run_examples", lambda paths: (_ for _ in ()).throw(AssertionError())
    )

    assert run_all.main(["--list"]) == 0
    output = capsys.readouterr().out
    assert "examples/simple_synthesis.py" in output
    assert "all_voices.py" not in output
