#!/usr/bin/env python3
"""Rank end template phrases for short-sentence phrase modes.

This only test 1 voice! For more thorough testing with multiple voices use rank_short_sentence_phrases_across_voice_list.py
"""

from __future__ import annotations

import find_short_sentence_phrase_candidates as phrase_candidates

PHRASE_CANDIDATES = [
    "The message on the screen changed to {segment}",
    "The line in the script ends with, {segment}",
    "The recording trails off after the words, … {segment}",
    "The message was brief — just {segment}",
    "Is that … {segment}?",
    "The transcript closes with the words, {segment}",
    "The lesson ended when the teacher asked, {segment}",
    "The letter closed with this unfinished thought — {segment}",
    "At last, the guide called out, {segment}",
    "The report concludes with this note: {segment}",
    "The note on the desk simply said, {segment}",
    "The final caption on the screen read, {segment}",
    "The teacher waited for a response. {segment}",
    "The announcement ended like this: {segment}",
    "There was a pause before the answer came: {segment}",
    "The final word is hello. The final word is '{segment}'",
    "The host asked again, more quietly this time: {segment}",
    "The conversation stopped after one last reply: {segment}",
]


def main() -> None:
    phrase_candidates.PHRASE_CANDIDATES = PHRASE_CANDIDATES
    phrase_candidates.REQUIRE_AFTER_BOUNDARY = False
    phrase_candidates.main()


if __name__ == "__main__":
    main()
