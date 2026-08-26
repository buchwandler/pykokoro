import kokorog2p

GERMAN_GOLDEN_CASES = {
    "14.05.2026": (
        "vierzehnte Mai zweitausendsechsundzwanzig",
        "fiːɾtseːntə maɪ tsvˈaɪtaʊzˌɛndzɛksˌʊnttsvantsˌɪç",
    ),
    "18:20 Uhr": (
        "achtzehn Uhr zwanzig",
        "axtseːn uːɾ tsvantsɪç",
    ),
    "1,5 kg": (
        "eins Komma fünf Kilogramm",
        "aɪns kɔmɑː fynf kiːlɔɡɾam",
    ),
    "500 g": (
        "fünfhundert Gramm",
        "fynfhʊndɜt ɡɾam",
    ),
    "1 ltr.": (
        "ein Liter.",
        "aɪn liːtɜ.",
    ),
    "45 Min.": (
        "fünfundvierzig Minuten.",
        "fynfʊndviːɾtsɪç miːnuːtən.",
    ),
    "12,80 EUR": (
        "zwölf Euro achtzig",
        "tsvœlf ɔøroː axtsɪç",
    ),
    "Prof.": ("Professor", "pɾoːfɛsoːɾ"),
    "zzgl.": ("zuzüglich", "tsuːtsyːklɪç"),
}


def test_german_normalization_and_phonemes_match_golden_cases() -> None:
    for text, (normalized, phonemes) in GERMAN_GOLDEN_CASES.items():
        result = kokorog2p.phonemize(
            text,
            language="de",
            return_phonemes=True,
            return_ids=True,
        )
        normalized_result = "".join(
            (getattr(token, "meta", None) or {}).get("_extended_text", getattr(token, "text", ""))
            for token in getattr(result, "tokens", [])
        )

        assert normalized_result == normalized
        assert result.phonemes == phonemes
