"""Language-aware lexical context catalogs for short-sentence synthesis."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ShortSentencePhraseSet:
    """Validated lexical context templates for one canonical language."""

    language: str
    neutral: tuple[str, ...] = ()
    declarative: tuple[str, ...] = ()
    question: tuple[str, ...] = ()
    exclamation: tuple[str, ...] = ()
    ellipsis: tuple[str, ...] = ()
    fragment: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.language.strip():
            raise ValueError("phrase catalog language must not be empty")
        for category in self.categories():
            if any(not template.strip() for template in category):
                raise ValueError("phrase catalog templates must not be empty")
            if any(template.count("{segment}") != 1 for template in category):
                raise ValueError("phrase catalog templates must contain exactly one '{segment}'")

    def categories(self) -> tuple[tuple[str, ...], ...]:
        return (
            self.neutral,
            self.declarative,
            self.question,
            self.exclamation,
            self.ellipsis,
            self.fragment,
        )


ENGLISH_SHORT_SENTENCE_PHRASES = ShortSentencePhraseSet(
    language="en",
    neutral=(
        "The conversation stopped, {segment}, before someone answered.",
        "The hallway went quiet; {segment}; then footsteps resumed.",
        "The radio paused; {segment}; then the broadcast continued.",
        "The student thought, {segment}, before the teacher continued.",
        "He paused…: {segment}…? … Is that you?",
        "The screen changed; {segment}; then the next slide appeared.",
        "The transcript paused…: {segment}; the next entry followed.",
        "The music stopped: {segment}. Then the singer continued.",
        "He paused…: {segment}? Is that you?",
        "The clerk paused, {segment}, before the next name was called.",
    ),
    declarative=(
        "The conversation stopped after one last reply: {segment}",
        "The teacher waited for a response. {segment}",
        "The announcement ended like this: {segment}",
        "There was a pause before the answer came: {segment}",
        "The report concludes with this note: {segment}",
        "The recording trails off after the words, … {segment}",
        "The lesson ended when the teacher asked, {segment}",
        "The host asked again, more quietly this time: {segment}",
        "The letter closed with this unfinished thought — {segment}",
        "The note on the desk simply said, {segment}",
        "At last, the guide called out, {segment}",
    ),
    question=("The question was asked plainly: {segment}", "A quiet voice asked: {segment}"),
    exclamation=("The speaker called out: {segment}", "The announcement ended with: {segment}"),
    ellipsis=("The thought trailed off with: {segment}", "The unfinished sentence was: {segment}"),
    fragment=(
        "The note contained only these words: {segment}",
        "The short message read: {segment}",
    ),
)

GERMAN_SHORT_SENTENCE_PHRASES = ShortSentencePhraseSet(
    language="de",
    declarative=("Die kurze Nachricht lautete: {segment}", "Am Ende stand: {segment}"),
    question=("Die Frage lautete: {segment}", "Dann kam die Frage: {segment}"),
    exclamation=("Der Ausruf lautete: {segment}", "Dann rief jemand: {segment}"),
    ellipsis=("Der unvollständige Gedanke lautete: {segment}",),
    fragment=("Die kurze Notiz lautete: {segment}", "Auf dem Zettel stand: {segment}"),
)

BUILTIN_SHORT_SENTENCE_PHRASES: dict[str, ShortSentencePhraseSet] = {
    "en": ENGLISH_SHORT_SENTENCE_PHRASES,
    "de": GERMAN_SHORT_SENTENCE_PHRASES,
}

SHORT_SENTENCE_PHRASE_LANGUAGE_ALIASES = {
    "en-us": "en",
    "en-gb": "en",
    "de-de": "de",
    "de-at": "de",
    "de-ch": "de",
}


def _normalized_language(language: str) -> str:
    return language.strip().lower().replace("_", "-")


def resolve_short_sentence_phrase_language(
    language: str,
    catalog: dict[str, ShortSentencePhraseSet] | None = None,
) -> str | None:
    """Resolve a segment language to a language present in the phrase catalog."""
    if not isinstance(language, str) or not language.strip():
        return None
    available = BUILTIN_SHORT_SENTENCE_PHRASES if catalog is None else catalog
    normalized_catalog = {_normalized_language(key): key for key in available}
    normalized = _normalized_language(language)
    if normalized in normalized_catalog:
        return normalized_catalog[normalized]
    alias = SHORT_SENTENCE_PHRASE_LANGUAGE_ALIASES.get(normalized)
    if alias in normalized_catalog:
        return normalized_catalog[alias]
    base = normalized.split("-", 1)[0]
    if base in normalized_catalog:
        return normalized_catalog[base]
    return None


def resolve_short_sentence_phrase_set(
    language: str,
    overlay: dict[str, ShortSentencePhraseSet] | None = None,
) -> tuple[ShortSentencePhraseSet | None, str | None, str]:
    """Return a phrase set, canonical language, and source for a segment language."""
    catalog = dict(BUILTIN_SHORT_SENTENCE_PHRASES)
    source = "builtin"
    if overlay:
        catalog.update(overlay)
        source = "user" if resolve_short_sentence_phrase_language(language, overlay) else "builtin"
    resolved = resolve_short_sentence_phrase_language(language, catalog)
    if resolved is None:
        return None, None, source
    return catalog[resolved], resolved, source
