from __future__ import annotations

import pytest

from benchmarks.hard_cases.schema import HardCase, HardCaseError, load_jsonl


def test_schema_round_trip_and_expectations() -> None:
    case = HardCase.from_dict(
        {
            "schema_version": 1,
            "id": "en_demo_001",
            "language": "en",
            "locale": "en-US",
            "category": "normalization",
            "text": "$12.50",
            "tags": ["currency"],
            "expect": {
                "spoken_text": "twelve dollars and fifty cents",
                "acoustic_constraints": {"max_duration_s": 3},
            },
        }
    )
    assert HardCase.from_dict(case.to_dict()) == case
    assert case.expect.acoustic_constraints.max_duration_s == 3


@pytest.mark.parametrize("field", ["human_reference", "reference_audio", "human_speaker_id"])
def test_human_reference_fields_are_rejected(field: str) -> None:
    with pytest.raises(HardCaseError, match="human-reference"):
        HardCase.from_dict(
            {
                "schema_version": 1,
                "id": "en_demo_001",
                "language": "en",
                "locale": "en-US",
                "category": "normalization",
                "text": "x",
                field: "not allowed",
            }
        )


def test_jsonl_duplicate_ids_are_rejected(tmp_path) -> None:
    path = tmp_path / "cases.jsonl"
    row = '{"schema_version":1,"id":"en_demo_001","language":"en","locale":null,"category":"normalization","text":"x"}'
    path.write_text(row + "\n" + row + "\n", encoding="utf-8")
    with pytest.raises(HardCaseError, match="duplicate"):
        load_jsonl(path)
