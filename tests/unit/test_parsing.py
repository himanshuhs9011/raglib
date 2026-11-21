"""Tolerant JSON parsing of LLM replies, including replies cut off at max_tokens."""

from __future__ import annotations

from raglib.core.parsing import parse_json
from raglib.graphrag.extractor import GraphExtractor


def test_complete_json_is_unchanged() -> None:
    assert parse_json('{"a": [1, 2]}') == {"a": [1, 2]}
    assert parse_json('Sure:\n```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json("no json here", default="x") == "x"


def test_truncated_reply_keeps_complete_items() -> None:
    cut = '{"entities": [{"name": "Apple", "type": "ORG"}, {"name": "Tim Cook", "type": "PER"}, {"name": "Cup'
    assert parse_json(cut) == {"entities": [{"name": "Apple", "type": "ORG"}, {"name": "Tim Cook", "type": "PER"}]}


def test_truncated_in_second_list() -> None:
    cut = ('```json\n{"entities": [{"name": "A"}], "relations": [{"source": "A", "target": "B"}, '
           '{"source": "A", "target": "C", "description": "a long sentence that never')
    assert parse_json(cut) == {"entities": [{"name": "A"}], "relations": [{"source": "A", "target": "B"}]}


def test_braces_inside_strings_do_not_confuse_the_salvage() -> None:
    cut = '{"entities": [{"name": "x}]{", "d": "quote \\" here"}, {"name": "y'
    assert parse_json(cut) == {"entities": [{"name": "x}]{", "d": 'quote " here'}]}


def test_nothing_complete_or_malformed_gives_default() -> None:
    assert parse_json('{"entities": [{"name": "Ap', default=None) is None
    assert parse_json('{"a": [1}', default="bad") == "bad"


class _LoopingLLM:
    """A model that loops until max_tokens: its reply stops in the middle of an entity."""

    calls = 0

    def complete_json(self, prompt, system="", default=None):
        self.calls += 1
        return parse_json('{"entities": [{"name": "Acme", "type": "ORGANIZATION", "description": "A firm."}, '
                          '{"name": "Acme", "type": "ORGANIZATION", "description": "A fi', default=default)


def test_extractor_keeps_a_truncated_extraction_instead_of_retrying() -> None:
    from raglib.schemas import Document

    llm = _LoopingLLM()
    extractor = GraphExtractor(llm)
    result = extractor.extract_one(Document(id="c1", content="Acme is a firm."))
    assert [entity.name for entity in result.entities] == ["Acme"]
    assert "c1" not in extractor.failed_chunk_ids  # stored, so it is not paid for again
    extractor.extract_one(Document(id="c1", content="Acme is a firm."))
    assert llm.calls == 1  # served from the cache
