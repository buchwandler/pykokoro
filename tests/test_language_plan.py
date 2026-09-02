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
