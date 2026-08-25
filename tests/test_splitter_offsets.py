import sys
from types import ModuleType

from pykokoro.generation_config import GenerationConfig
from pykokoro.pipeline_config import PipelineConfig
from pykokoro.stages.doc_parsers.plain import PhrasplitSentenceSplitter
from pykokoro.stages.protocols import DocumentResult
from pykokoro.tokenizer import TokenizerConfig
from pykokoro.types import Trace


def test_phrasplit_splitter_handles_missing_offsets():
    text = "Hello world. How are you?"
    cfg = PipelineConfig(generation=GenerationConfig(lang="en-us"))
    doc = DocumentResult(clean_text=text)

    class DummySplitter(PhrasplitSentenceSplitter):
        def _split_with_offsets(self, phrasplit_module, text, language_model):
            return [
                ("Hello world.", None, None, None, None, None),
                ("How are you?", None, None, None, None, None),
            ]

    splitter = DummySplitter()
    dummy_module = ModuleType("phrasplit")

    with patch_sys_modules({"phrasplit": dummy_module}):
        segments = splitter.split(doc, cfg, Trace())

    assert len(segments) == 2
    assert segments[0].text == "Hello world."
    assert segments[1].text == "How are you?"
    assert segments[0].char_start == 0
    assert segments[1].char_start == segments[0].char_end + 1


def test_phrasplit_splitter_reads_char_offsets():
    text = "Hello world. Goodbye world."
    cfg = PipelineConfig(generation=GenerationConfig(lang="en-us"))
    doc = DocumentResult(clean_text=text)

    class FakeSegment:
        def __init__(self, text: str, start: int, end: int) -> None:
            self.text = text
            self.char_start = start
            self.char_end = end
            self.paragraph_idx = 0
            self.sentence_idx = 0
            self.clause_idx = 0

    first = FakeSegment("Hello world.", 0, 12)
    second = FakeSegment("Goodbye world.", 13, 27)
    dummy_module = ModuleType("phrasplit")
    dummy_module.split_with_offsets = lambda *_args, **_kwargs: [first, second]

    splitter = PhrasplitSentenceSplitter()

    with patch_sys_modules({"phrasplit": dummy_module}):
        segments = splitter.split(doc, cfg, Trace())

    assert [segment.text for segment in segments] == ["Hello world.", "Goodbye world."]
    assert [segment.char_start for segment in segments] == [0, 13]
    assert [segment.char_end for segment in segments] == [12, 27]


def test_phrasplit_splitter_repair_non_whitespace_gap():
    text = "Hello People like you."
    cfg = PipelineConfig(generation=GenerationConfig(lang="en-us"))
    doc = DocumentResult(clean_text=text)

    class DummySplitter(PhrasplitSentenceSplitter):
        def _split_with_offsets(self, phrasplit_module, text, language_model):
            _ = phrasplit_module, text, language_model
            return [
                ("Hello", 0, 5, None, None, None),
                ("eople like you.", 7, 22, None, None, None),
            ]

    splitter = DummySplitter()
    dummy_module = ModuleType("phrasplit")

    with patch_sys_modules({"phrasplit": dummy_module}):
        segments = splitter.split(doc, cfg, Trace())

    assert segments[1].text.startswith("People")
    assert segments[1].char_start == text.index("People")


def test_real_phrasplit_detailed_offsets_are_exact_slices() -> None:
    import phrasplit

    text = "First sentence. Second sentence."
    detailed = phrasplit.split_with_offsets_with_diagnostics(
        text,
        mode="sentence",
        use_spacy=False,
        language="en",
    )
    assert detailed.diagnostics.backend == "regex"
    for segment in detailed.segments:
        assert segment.text == text[segment.char_start : segment.char_end]

    doc = DocumentResult(clean_text=text)
    segments = PhrasplitSentenceSplitter().split(
        doc,
        PipelineConfig(
            generation=GenerationConfig(lang="en-us"),
            tokenizer_config=TokenizerConfig(use_spacy=False),
        ),
        Trace(),
    )

    for segment in segments:
        assert segment.text == text[segment.char_start : segment.char_end]
    assert doc.metadata["spacy_models"]["sentence"]["selected_model"] is None


class patch_sys_modules:
    def __init__(self, updates: dict[str, ModuleType]) -> None:
        self._updates = updates
        self._originals: dict[str, ModuleType | None] = {}

    def __enter__(self):
        for key, value in self._updates.items():
            self._originals[key] = sys.modules.get(key)
            sys.modules[key] = value
        return self

    def __exit__(self, exc_type, exc, tb):
        for key, original in self._originals.items():
            if original is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = original
        return False
