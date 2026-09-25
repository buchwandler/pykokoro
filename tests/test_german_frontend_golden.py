import kokorog2p
import spokenform

GERMAN_GOLDEN_CASES = {
    "14.05.2026": (
        "vierzehnte Mai zweitausendsechsundzwanzig",
        "fˈiːɾʦeːntə mˈI ʦvItWzɛnʣeːksʊndʦvanʦɪç",
    ),
    "18:20 Uhr": (
        "achtzehn Uhr zwanzig",
        "ˈaxʦeːn ˈuːɾ ʦvˈanʦɪç",
    ),
    "1,5 kg": (
        "eins Komma fünf Kilogramm",
        "ˈIns kˈɔmɑː fˈynf kˌiːlɔɡɾˈam",
    ),
    "500 g": (
        "Fünfhundert Gramm",
        "fˈynfhʊndɜt ɡɾˈam",
    ),
    "1 ltr.": (
        "Ein Liter.",
        "ˈIn lˈiːtɜ.",
    ),
    "45 Min.": (
        "Fünfundvierzig Minuten.",
        "fˈynfʊndvˌiːɾʦɪç miːnˈuːtən.",
    ),
    "12,80 EUR": (
        "Zwölf Euro achtzig Cent",
        "ʦvˈœlf ˈɔøroː ˈaxʦɪç sˈɛnt",
    ),
    "Prof.": ("Professor", "pɾoːfˈɛsoːɾ"),
    "zzgl.": ("zuzüglich", "ʦuːʦˈyːklɪç"),
}


def test_german_normalization_and_phonemes_match_golden_cases() -> None:
    for text, (normalized, phonemes) in GERMAN_GOLDEN_CASES.items():
        prepared = spokenform.prepare_for_kokorog2p(text, language="de").spoken_text
        result = kokorog2p.phonemize(
            prepared,
            language="de",
            return_phonemes=True,
            return_ids=True,
        )
        assert prepared.casefold() == normalized.casefold()
        assert result.phonemes == phonemes
