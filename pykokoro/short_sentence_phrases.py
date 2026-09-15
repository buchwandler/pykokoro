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

SPANISH_SHORT_SENTENCE_PHRASES = ShortSentencePhraseSet(
    language="es",
    declarative=(
        "El mensaje decía: {segment}",
        "Al final decía: {segment}",
    ),
    question=(
        "La pregunta era: {segment}",
        "Entonces alguien preguntó: {segment}",
    ),
    exclamation=(
        "Alguien gritó: {segment}",
        "El aviso terminó así: {segment}",
    ),
    ellipsis=(
        "La frase quedó incompleta: {segment}",
        "Las palabras se interrumpieron así: {segment}",
    ),
    fragment=(
        "La nota decía: {segment}",
        "El mensaje corto decía: {segment}",
    ),
)

FRENCH_SHORT_SENTENCE_PHRASES = ShortSentencePhraseSet(
    language="fr",
    declarative=(
        "Le message disait: {segment}",
        "À la fin il était écrit: {segment}",
    ),
    question=(
        "La question était: {segment}",
        "Puis quelqu'un a demandé: {segment}",
    ),
    exclamation=(
        "Quelqu'un a crié: {segment}",
        "L'annonce s'est terminée ainsi: {segment}",
    ),
    ellipsis=(
        "La phrase est restée inachevée: {segment}",
        "Les mots se sont interrompus ainsi: {segment}",
    ),
    fragment=(
        "La note disait: {segment}",
        "Le court message disait: {segment}",
    ),
)

ITALIAN_SHORT_SENTENCE_PHRASES = ShortSentencePhraseSet(
    language="it",
    declarative=(
        "Il messaggio diceva: {segment}",
        "Alla fine c'era scritto: {segment}",
    ),
    question=(
        "La domanda era: {segment}",
        "Poi qualcuno chiese: {segment}",
    ),
    exclamation=(
        "Qualcuno gridò: {segment}",
        "L'annuncio finì così: {segment}",
    ),
    ellipsis=(
        "La frase rimase incompleta: {segment}",
        "Le parole si interruppero così: {segment}",
    ),
    fragment=(
        "Il biglietto diceva: {segment}",
        "Il breve messaggio diceva: {segment}",
    ),
)

PORTUGUESE_SHORT_SENTENCE_PHRASES = ShortSentencePhraseSet(
    language="pt",
    declarative=(
        "A mensagem dizia: {segment}",
        "No final estava escrito: {segment}",
    ),
    question=(
        "A pergunta era: {segment}",
        "Então alguém perguntou: {segment}",
    ),
    exclamation=(
        "Alguém gritou: {segment}",
        "O anúncio terminou assim: {segment}",
    ),
    ellipsis=(
        "A frase ficou incompleta: {segment}",
        "As palavras pararam assim: {segment}",
    ),
    fragment=(
        "O bilhete dizia: {segment}",
        "A mensagem curta dizia: {segment}",
    ),
)

EUROPEAN_PORTUGUESE_SHORT_SENTENCE_PHRASES = ShortSentencePhraseSet(
    language="pt-pt",
    declarative=(
        "A mensagem dizia: {segment}",
        "No fim estava escrito: {segment}",
    ),
    question=(
        "A pergunta era: {segment}",
        "Depois alguém perguntou: {segment}",
    ),
    exclamation=(
        "Alguém gritou: {segment}",
        "O anúncio terminou assim: {segment}",
    ),
    ellipsis=(
        "A frase ficou incompleta: {segment}",
        "As palavras ficaram por acabar: {segment}",
    ),
    fragment=(
        "O bilhete dizia: {segment}",
        "A mensagem curta dizia: {segment}",
    ),
)

KOREAN_SHORT_SENTENCE_PHRASES = ShortSentencePhraseSet(
    language="ko",
    declarative=(
        "메시지에는 이렇게 적혀 있었습니다: {segment}",
        "마지막에는 이렇게 적혀 있었습니다: {segment}",
    ),
    question=(
        "질문은 이랬습니다: {segment}",
        "누군가 물었습니다: {segment}",
    ),
    exclamation=(
        "누군가 외쳤습니다: {segment}",
        "마지막으로 이렇게 외쳤습니다: {segment}",
    ),
    ellipsis=(
        "문장은 여기서 끊겼습니다: {segment}",
        "말은 여기서 흐려졌습니다: {segment}",
    ),
    fragment=(
        "짧은 메모에는 이렇게 적혀 있었습니다: {segment}",
        "쪽지에는 이렇게 적혀 있었습니다: {segment}",
    ),
)

JAPANESE_SHORT_SENTENCE_PHRASES = ShortSentencePhraseSet(
    language="ja",
    declarative=(
        "メッセージにはこうありました: {segment}",
        "最後にはこうありました: {segment}",
    ),
    question=(
        "質問はこうでした: {segment}",
        "そして誰かが尋ねました: {segment}",
    ),
    exclamation=(
        "誰かが叫びました: {segment}",
        "最後にこう叫びました: {segment}",
    ),
    ellipsis=(
        "文はここで途切れました: {segment}",
        "言葉はここで途切れました: {segment}",
    ),
    fragment=(
        "短いメモにはこうありました: {segment}",
        "メモにはこう書かれていました: {segment}",
    ),
)

CHINESE_SHORT_SENTENCE_PHRASES = ShortSentencePhraseSet(
    language="zh",
    declarative=(
        "消息里写着: {segment}",
        "最后写着: {segment}",
    ),
    question=(
        "问题是: {segment}",
        "接着有人问: {segment}",
    ),
    exclamation=(
        "有人喊道: {segment}",
        "最后有人喊道: {segment}",
    ),
    ellipsis=(
        "这句话没有说完: {segment}",
        "话说到这里停了: {segment}",
    ),
    fragment=(
        "纸条上写着: {segment}",
        "短消息写着: {segment}",
    ),
)

ARABIC_SHORT_SENTENCE_PHRASES = ShortSentencePhraseSet(
    language="ar",
    declarative=(
        "نص الرسالة هو: {segment}",
        "في النهاية ورد: {segment}",
    ),
    question=(
        "كان السؤال: {segment}",
        "ثم جاء السؤال: {segment}",
    ),
    exclamation=(
        "صاح أحدهم: {segment}",
        "ثم جاء النداء: {segment}",
    ),
    ellipsis=(
        "توقفت العبارة عند: {segment}",
        "انقطعت الكلمات عند: {segment}",
    ),
    fragment=(
        "نص الملاحظة هو: {segment}",
        "الرسالة القصيرة هي: {segment}",
    ),
)

HEBREW_SHORT_SENTENCE_PHRASES = ShortSentencePhraseSet(
    language="he",
    declarative=(
        "בהודעה היה כתוב: {segment}",
        "בסוף היה כתוב: {segment}",
    ),
    question=(
        "השאלה הייתה: {segment}",
        "ואז מישהו שאל: {segment}",
    ),
    exclamation=(
        "מישהו קרא: {segment}",
        "ואז נשמעה הקריאה: {segment}",
    ),
    ellipsis=(
        "המשפט נקטע כך: {segment}",
        "המילים נקטעו כך: {segment}",
    ),
    fragment=(
        "בפתק היה כתוב: {segment}",
        "בהודעה הקצרה היה כתוב: {segment}",
    ),
)

KAZAKH_SHORT_SENTENCE_PHRASES = ShortSentencePhraseSet(
    language="kk",
    declarative=(
        "Хабарламада былай жазылған: {segment}",
        "Соңында былай жазылған: {segment}",
    ),
    question=(
        "Сұрақ былай қойылды: {segment}",
        "Сосын біреу сұрады: {segment}",
    ),
    exclamation=(
        "Біреу айқайлады: {segment}",
        "Соңында біреу дауыстады: {segment}",
    ),
    ellipsis=(
        "Сөйлем осылай үзіліп қалды: {segment}",
        "Сөз осылай үзіліп қалды: {segment}",
    ),
    fragment=(
        "Жазбада былай жазылған: {segment}",
        "Қысқа хабарламада былай жазылған: {segment}",
    ),
)

SWEDISH_SHORT_SENTENCE_PHRASES = ShortSentencePhraseSet(
    language="sv",
    declarative=(
        "Meddelandet löd: {segment}",
        "Till sist stod det: {segment}",
    ),
    question=(
        "Frågan löd: {segment}",
        "Sedan frågade någon: {segment}",
    ),
    exclamation=(
        "Någon ropade: {segment}",
        "Till sist ropade någon: {segment}",
    ),
    ellipsis=(
        "Meningen avbröts så här: {segment}",
        "Orden tog slut så här: {segment}",
    ),
    fragment=(
        "På lappen stod: {segment}",
        "Det korta meddelandet löd: {segment}",
    ),
)

THAI_SHORT_SENTENCE_PHRASES = ShortSentencePhraseSet(
    language="th",
    declarative=(
        "ข้อความเขียนว่า: {segment}",
        "ตอนท้ายเขียนว่า: {segment}",
    ),
    question=(
        "คำถามคือ: {segment}",
        "จากนั้นมีคนถามว่า: {segment}",
    ),
    exclamation=(
        "มีคนตะโกนว่า: {segment}",
        "ตอนท้ายมีคนตะโกนว่า: {segment}",
    ),
    ellipsis=(
        "ประโยคค้างไว้ว่า: {segment}",
        "คำพูดหยุดลงตรงนี้: {segment}",
    ),
    fragment=(
        "ในโน้ตเขียนว่า: {segment}",
        "ข้อความสั้นเขียนว่า: {segment}",
    ),
)

VIETNAMESE_SHORT_SENTENCE_PHRASES = ShortSentencePhraseSet(
    language="vi",
    declarative=(
        "Tin nhắn ghi: {segment}",
        "Cuối cùng có dòng: {segment}",
    ),
    question=(
        "Câu hỏi là: {segment}",
        "Rồi có người hỏi: {segment}",
    ),
    exclamation=(
        "Có người gọi lớn: {segment}",
        "Cuối cùng có tiếng gọi: {segment}",
    ),
    ellipsis=(
        "Câu nói còn dang dở: {segment}",
        "Lời nói dừng lại ở: {segment}",
    ),
    fragment=(
        "Mẩu giấy ghi: {segment}",
        "Tin nhắn ngắn ghi: {segment}",
    ),
)

RUSSIAN_SHORT_SENTENCE_PHRASES = ShortSentencePhraseSet(
    language="ru",
    declarative=(
        "В сообщении было написано: {segment}",
        "В конце было написано: {segment}",
    ),
    question=(
        "Вопрос был такой: {segment}",
        "Потом кто-то спросил: {segment}",
    ),
    exclamation=(
        "Кто-то крикнул: {segment}",
        "В конце кто-то крикнул: {segment}",
    ),
    ellipsis=(
        "Фраза оборвалась на словах: {segment}",
        "Слова оборвались так: {segment}",
    ),
    fragment=(
        "В записке было написано: {segment}",
        "Короткое сообщение гласило: {segment}",
    ),
)

HINDI_SHORT_SENTENCE_PHRASES = ShortSentencePhraseSet(
    language="hi",
    declarative=(
        "संदेश में लिखा था: {segment}",
        "अंत में लिखा था: {segment}",
    ),
    question=(
        "सवाल था: {segment}",
        "फिर किसी ने पूछा: {segment}",
    ),
    exclamation=(
        "किसी ने पुकारा: {segment}",
        "अंत में किसी ने पुकारा: {segment}",
    ),
    ellipsis=(
        "वाक्य अधूरा रह गया: {segment}",
        "बात यहीं अधूरी रह गई: {segment}",
    ),
    fragment=(
        "पर्ची पर लिखा था: {segment}",
        "छोटे संदेश में लिखा था: {segment}",
    ),
)

POLISH_SHORT_SENTENCE_PHRASES = ShortSentencePhraseSet(
    language="pl",
    declarative=(
        "Wiadomość brzmiała: {segment}",
        "Na końcu było napisane: {segment}",
    ),
    question=(
        "Pytanie brzmiało: {segment}",
        "Potem ktoś zapytał: {segment}",
    ),
    exclamation=(
        "Ktoś zawołał: {segment}",
        "Na końcu ktoś krzyknął: {segment}",
    ),
    ellipsis=(
        "Zdanie pozostało niedokończone: {segment}",
        "Słowa urwały się tak: {segment}",
    ),
    fragment=(
        "Na kartce było napisane: {segment}",
        "Krótka wiadomość brzmiała: {segment}",
    ),
)

TURKISH_SHORT_SENTENCE_PHRASES = ShortSentencePhraseSet(
    language="tr",
    declarative=(
        "Mesajda şöyle yazıyordu: {segment}",
        "Sonunda şöyle yazıyordu: {segment}",
    ),
    question=(
        "Soru şuydu: {segment}",
        "Sonra biri sordu: {segment}",
    ),
    exclamation=(
        "Biri seslendi: {segment}",
        "Sonunda biri bağırdı: {segment}",
    ),
    ellipsis=(
        "Cümle yarım kaldı: {segment}",
        "Sözler burada kesildi: {segment}",
    ),
    fragment=(
        "Notta şöyle yazıyordu: {segment}",
        "Kısa mesajda şöyle yazıyordu: {segment}",
    ),
)

BUILTIN_SHORT_SENTENCE_PHRASES: dict[str, ShortSentencePhraseSet] = {
    "en": ENGLISH_SHORT_SENTENCE_PHRASES,
    "de": GERMAN_SHORT_SENTENCE_PHRASES,
    "es": SPANISH_SHORT_SENTENCE_PHRASES,
    "fr": FRENCH_SHORT_SENTENCE_PHRASES,
    "it": ITALIAN_SHORT_SENTENCE_PHRASES,
    "pt": PORTUGUESE_SHORT_SENTENCE_PHRASES,
    "pt-pt": EUROPEAN_PORTUGUESE_SHORT_SENTENCE_PHRASES,
    "ko": KOREAN_SHORT_SENTENCE_PHRASES,
    "ja": JAPANESE_SHORT_SENTENCE_PHRASES,
    "zh": CHINESE_SHORT_SENTENCE_PHRASES,
    "ar": ARABIC_SHORT_SENTENCE_PHRASES,
    "he": HEBREW_SHORT_SENTENCE_PHRASES,
    "kk": KAZAKH_SHORT_SENTENCE_PHRASES,
    "sv": SWEDISH_SHORT_SENTENCE_PHRASES,
    "th": THAI_SHORT_SENTENCE_PHRASES,
    "vi": VIETNAMESE_SHORT_SENTENCE_PHRASES,
    "ru": RUSSIAN_SHORT_SENTENCE_PHRASES,
    "hi": HINDI_SHORT_SENTENCE_PHRASES,
    "pl": POLISH_SHORT_SENTENCE_PHRASES,
    "tr": TURKISH_SHORT_SENTENCE_PHRASES,
}

SHORT_SENTENCE_PHRASE_LANGUAGE_ALIASES = {
    "en-us": "en",
    "en-gb": "en",
    "de-de": "de",
    "de-at": "de",
    "de-ch": "de",
    "cmn": "zh",
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
