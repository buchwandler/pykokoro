import pytest

from pykokoro.runtime.language_plan import LanguageRun, build_language_plan
from pykokoro.types import AnnotationSpan


def test_language_plan_uses_default_language() -> None:
    assert build_language_plan("Hallo", (), default_language="de-DE") == (
        LanguageRun(0, 5, "de-de"),
    )


def test_language_plan_splits_and_merges_explicit_runs() -> None:
    text = "Deutsch English Deutsch"
    annotations = [AnnotationSpan(8, 15, {"lang": "en-US"})]
    assert build_language_plan(text, annotations, default_language="de") == (
        LanguageRun(0, 8, "de"),
        LanguageRun(8, 15, "en-us"),
        LanguageRun(15, len(text), "de"),
    )


def test_language_plan_merges_adjacent_equal_language_spans() -> None:
    text = "one two"
    annotations = [
        AnnotationSpan(0, 3, {"lang": "en-us"}),
        AnnotationSpan(3, len(text), {"lang": "en-US"}),
    ]
    assert build_language_plan(text, annotations, default_language="de") == (
        LanguageRun(0, len(text), "en-us"),
    )


@pytest.mark.parametrize(
    "annotation",
    [AnnotationSpan(-1, 2, {"lang": "de"}), AnnotationSpan(0, 20, {"lang": "de"})],
)
def test_language_plan_rejects_out_of_bounds(annotation: AnnotationSpan) -> None:
    with pytest.raises(ValueError, match="outside"):
        build_language_plan("short", [annotation], default_language="en-us")


def test_language_plan_rejects_conflicting_overlap() -> None:
    with pytest.raises(ValueError, match="Conflicting"):
        build_language_plan(
            "overlap",
            [AnnotationSpan(0, 5, {"lang": "de"}), AnnotationSpan(3, 7, {"lang": "en-us"})],
            default_language="de",
        )


def test_language_plan_rejects_unknown_language() -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        build_language_plan("text", (), default_language="xx-invalid")


def _languages(plan: tuple[LanguageRun, ...]) -> list[tuple[int, int, str]]:
    return [(run.char_start, run.char_end, run.language) for run in plan]


def test_language_plan_allows_nested_override() -> None:
    text = "x" * 20
    annotations = [
        AnnotationSpan(0, 20, {"lang": "en-us"}),
        AnnotationSpan(5, 10, {"lang": "en-gb"}),
    ]
    assert _languages(build_language_plan(text, annotations, default_language="de-de")) == [
        (0, 5, "en-us"),
        (5, 10, "en-gb"),
        (10, 20, "en-us"),
    ]


def test_language_plan_allows_multiple_nested_overrides() -> None:
    text = "x" * 30
    annotations = [
        AnnotationSpan(0, 30, {"lang": "en-us"}),
        AnnotationSpan(5, 25, {"lang": "de-de"}),
        AnnotationSpan(10, 20, {"lang": "fr-fr"}),
    ]
    assert _languages(build_language_plan(text, annotations, default_language="ja")) == [
        (0, 5, "en-us"),
        (5, 10, "de-de"),
        (10, 20, "fr-fr"),
        (20, 25, "de-de"),
        (25, 30, "en-us"),
    ]


@pytest.mark.parametrize(
    "annotations",
    [
        [AnnotationSpan(0, 10, {"lang": "en-us"}), AnnotationSpan(0, 10, {"lang": "de-de"})],
        [AnnotationSpan(0, 10, {"lang": "en-us"}), AnnotationSpan(5, 15, {"lang": "de-de"})],
    ],
    ids=["equal-range", "crossing"],
)
def test_language_plan_rejects_ambiguous_different_language_overlap(
    annotations: list[AnnotationSpan],
) -> None:
    with pytest.raises(ValueError, match="Conflicting"):
        build_language_plan("x" * 20, annotations, default_language="en-us")


def test_language_plan_allows_same_language_overlap() -> None:
    text = "x" * 10
    annotations = [
        AnnotationSpan(0, 8, {"lang": "en-us"}),
        AnnotationSpan(3, 10, {"lang": "en-us"}),
    ]
    assert _languages(build_language_plan(text, annotations, default_language="de-de")) == [
        (0, 10, "en-us"),
    ]


def test_language_plan_accepts_hindi_as_espeak_fallback() -> None:
    assert _languages(build_language_plan("नमस्ते", (), default_language="hi")) == [
        (0, len("नमस्ते"), "hi"),
    ]
