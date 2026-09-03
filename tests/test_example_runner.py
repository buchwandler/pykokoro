from __future__ import annotations

import os
import subprocess

from examples import run_all


def test_default_selection_excludes_opt_in_categories() -> None:
    paths = run_all._example_paths()
    names = {path.name for path in paths}

    assert "play_audio.py" not in names
    assert "cpu_benchmark.py" not in names
    assert not any(path.parent.name == "legacy" for path in paths)


def test_selection_flags_add_opt_in_categories() -> None:
    paths = run_all._example_paths(
        include_legacy=True, include_playback=True, include_optional=True
    )
    names = {path.name for path in paths}

    assert "play_audio.py" in names
    assert "cpu_benchmark.py" in names
    assert any(path.parent.name == "legacy" for path in paths)


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
    monkeypatch.setattr(run_all.sys, "argv", ["run_all.py", "--list"])

    assert run_all.main() == 0
    assert "examples/abbreviations.py" in capsys.readouterr().out
