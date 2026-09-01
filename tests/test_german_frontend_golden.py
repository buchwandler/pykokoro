import kokorog2p
import spokenform

GERMAN_GOLDEN_CASES = {
    "14.05.2026": (
        "vierzehnte Mai zweitausendsechsundzwanzig",
        "fiːɾʦeːntə mI tsvˈaɪtaʊzˌɛnʣɛksˌʊnttsvantsˌɪç",
    ),
    "18:20 Uhr": (
        "achtzehn Uhr zwanzig",
        "axʦeːn uːɾ ʦvanʦɪç",
    ),
    "1,5 kg": (
        "eins Komma fünf Kilogramm",
        "Ins kɔmɑː fynf kiːlɔɡɾam",
    ),
    "500 g": (
        "fünfhundert Gramm",
        "fynfhʊndɜt ɡɾam",
    ),
    "1 ltr.": (
        "ein Liter.",
        "In liːtɜ.",
    ),
    "45 Min.": (
        "fünfundvierzig Minuten.",
        "fynfʊndviːɾʦɪç miːnuːtən.",
    ),
    "12,80 EUR": (
        "zwölf Euro achtzig Cent",
        "ʦvœlf ɔøroː axʦɪç sɛnt",
    ),
    "Prof.": ("Professor", "pɾoːfɛsoːɾ"),
    "zzgl.": ("zuzüglich", "ʦuːʦyːklɪç"),
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
        assert prepared == normalized
        assert result.phonemes == phonemes
