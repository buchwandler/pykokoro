from utterplan import PlannerConfig, UtterancePlanner

from pykokoro.phoneme_dictionary import PhonemeDictionary


def _plan_ssmd(text: str):
    planner = UtterancePlanner(
        PlannerConfig(
            language="en-us",
            document_format="ssmd",
            text_preparation="identity",
        )
    )
    try:
        return planner.plan(text)
    finally:
        planner.close()


def _make_dictionary(entries: dict[str, str]) -> PhonemeDictionary:
    dictionary = PhonemeDictionary()
    dictionary._dictionary = entries
    return dictionary


def test_apply_emits_unescaped_braces():
    dictionary = _make_dictionary({"Hello": "heh-loh"})
    result = dictionary.apply("Hello world")

    assert '[Hello]{ph="heh-loh"}' in result
    assert "\\{" not in result


def test_apply_round_trips_through_ssmd_parser():
    dictionary = _make_dictionary({"Hello": "heh-loh"})
    plan = _plan_ssmd(dictionary.apply("Hello"))
    pronunciation = plan.segments[0].directives.pronunciation
    assert pronunciation is not None
    assert pronunciation.phonemes == "heh-loh"


def test_apply_case_insensitive_preserves_casing():
    dictionary = _make_dictionary({"hello": "heh-loh"})
    result = dictionary.apply("HELLO")

    assert result == '[HELLO]{ph="heh-loh"}'


def test_apply_word_boundaries_avoid_substrings():
    dictionary = _make_dictionary({"he": "H"})
    result = dictionary.apply("the")

    assert result == "the"


def test_apply_multi_word_entries():
    dictionary = _make_dictionary({"New York": "ny"})
    result = dictionary.apply("Welcome to New York")

    assert '[New York]{ph="ny"}' in result


def test_apply_punctuation_and_hyphenated_phrases():
    dictionary = _make_dictionary({"Hello": "hi", "New York": "ny"})
    result = dictionary.apply("Hello, New-York.")

    assert result == '[Hello]{ph="hi"}, [New-York]{ph="ny"}.'


def test_apply_escapes_ssmd_attribute_characters():
    phoneme = 'a\\b"c{d}'
    dictionary = _make_dictionary({"Hello": phoneme})
    plan = _plan_ssmd(dictionary.apply("Hello"))
    pronunciation = plan.segments[0].directives.pronunciation
    assert pronunciation is not None
    assert pronunciation.phonemes == phoneme
