from __future__ import annotations

from examples import german


def test_german_example_contract():
    assert german.OUTPUT_FILE == "german_martin_v1_2.wav"
    assert "14.05.2026" in german.TEXT
    assert "1 ltr." in german.TEXT
    assert "Prof." in german.TEXT
    assert "Tundragoon" not in german.TEXT
    assert "English-trained" not in german.__doc__
