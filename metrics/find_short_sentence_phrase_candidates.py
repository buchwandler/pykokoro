#!/usr/bin/env python3
"""Rank neutral template phrases for short-sentence phrase modes.

This only test 1 voice! For more thorough testing with multiple voices use rank_short_sentence_phrases_across_voice_list.py"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean

import numpy as np

from pykokoro.audio_generator import _join_timestamps
from pykokoro.constants import SAMPLE_RATE
from pykokoro.onnx_backend import Kokoro
from pykokoro.short_sentence_handler import phonemize_short_sentence_phrase
from pykokoro.trim import energy_based_vad
from pykokoro.types import PhonemeSegment

VOICE = "bf_lily"
LANG = "en-gb"
SPEED = 0.83
FRAME_DURATION_MS = 5
ENERGY_THRESHOLD = 0.05
MIN_SILENCE_SECONDS = 0.02
MAX_BOUNDARY_GAP_SECONDS = 0.30
REQUIRE_BEFORE_BOUNDARY = True
REQUIRE_AFTER_BOUNDARY = True

PHRASE_CANDIDATES = [
    "She looked up…: {segment}. The conversation resumed.",
    "The room fell quiet … {segment} … afterward, they moved on.",
    "He paused…: {segment}…? … Is that you?",
    "He paused…: {segment}? Is that you?",
    "He paused. … {segment} Then he continued.",
    "The student thought, {segment}, before the teacher continued.",
    "The transcript paused…: {segment}; the next entry followed.",
    "The hallway went quiet; {segment}; then footsteps resumed.",
    "The response stopped — {segment} — then the prompt continued.",
    "The clerk paused, {segment}, before the next name was called.",
    "The narrator paused — {segment} — then the chapter continued.",
    "The music stopped: {segment}. Then the singer continued.",
    "The bell rang — {segment}. Then the class resumed.",
    "The door closed … {segment}. Then the room settled.",
    "The page turned, {segment}, and the reading continued.",
    "The screen changed; {segment}; then the next slide appeared.",
    "The scene ended — {segment} — then the story continued.",
    "The crowd quieted … {segment} … then the speech resumed.",
    "The phone went silent: {segment}. Then the caller spoke.",
    "The radio paused; {segment}; then the broadcast continued.",
    "The recording stopped — {segment}. Then the next clip played.",
    "The lesson paused … {segment}. Then the teacher continued.",
    "The conversation stopped, {segment}, before someone answered.",
    "The final line ended: {segment}. Then silence followed.",
    "The word, {segment}, appears here.",
    "The line says, {segment}, before continuing.",
    "The entry reads, {segment}, —in this place.",
    "It could be said, {segment}, —in all its glory—, might have been better.",
    "He waited a moment, — {segment} — then spoke again."
]

# Keep this in sync with examples/short_sentence_demo.py.
DEMO_SEGMENTS = [
    "Hi!",
    "Why?",
    "No.",
    "No!",
    "Yes!",
    "Help!",
    "Oh!",
    "Stop!",
    "What?",
    "Don't!",
    "One … step. In front.",
    "Of.",
    "The other.",
    "Ah.",
    "Hush.",
    "Go on.",
    "Not yet.",
    "I know.",
    "Mm-hm.",
    "Mr. Vale.",
    "St. John.",
    "'Tis.",
    "Chapter IV.",
    "Hermione."
]


@dataclass
class CandidateScore:
    phrase: str
    successes: int
    attempts: int
    mean_before_gap_seconds: float
    mean_after_gap_seconds: float
    mean_before_pause_seconds: float
    mean_after_pause_seconds: float

    @property
    def success_ratio(self) -> float:
        return self.successes / self.attempts if self.attempts else 0.0

    @property
    def mean_gap_seconds(self) -> float:
        return mean([self.mean_before_gap_seconds, self.mean_after_gap_seconds])

    @property
    def mean_pause_seconds(self) -> float:
        return mean([self.mean_before_pause_seconds, self.mean_after_pause_seconds])


def main() -> None:
    kokoro = Kokoro(provider="cuda")
    try:
        kokoro._init_kokoro()
        assert kokoro._session is not None
        assert kokoro._audio_generator is not None
        output_names = [output.name for output in kokoro._session.get_outputs()]
        if len(output_names) < 2:
            raise RuntimeError(
                "Loaded model has no timestamp output. "
                "Use the timestamped ONNX model before running this script."
            )

        voice_style = kokoro.resolve_voice_style(VOICE)
        scores = []
        for phrase in PHRASE_CANDIDATES:
            score = score_phrase(
                phrase,
                audio_generator=kokoro._audio_generator,
                voice_style=voice_style,
            )
            scores.append(score)
            print(
                f"Processed {score.phrase!r} | "
                f"success={score.successes}/{score.attempts} "
                f"({score.success_ratio:.0%}) | "
                f"{format_score_metrics(score)}"
            )
    finally:
        kokoro.close()

    ranked_before = sorted(
        scores,
        key=lambda score: (
            score.success_ratio,
            score.successes,
            -score.mean_before_gap_seconds,
            score.mean_before_pause_seconds,
        ),
        reverse=True,
    )
    print_scores("Top 5 candidate phrases by pause before segment", ranked_before)
    if REQUIRE_AFTER_BOUNDARY:
        ranked_after = sorted(
            scores,
            key=lambda score: (
                score.success_ratio,
                score.successes,
                -score.mean_after_gap_seconds,
                score.mean_after_pause_seconds,
            ),
            reverse=True,
        )
        ranked_combined = sorted(
            scores,
            key=lambda score: (
                score.success_ratio,
                score.successes,
                -score.mean_gap_seconds,
                score.mean_pause_seconds,
            ),
            reverse=True,
        )
        print_scores("Top 5 candidate phrases by pause after segment", ranked_after)
        print_scores("Top 5 candidate phrases by combined pause metrics", ranked_combined)


def print_scores(title: str, scores: list[CandidateScore]) -> None:
    print("##########################")
    print(title)
    for index, score in enumerate(scores[:5], start=1):
        print(
            f"{index}. {score.phrase!r} | "
            f"success={score.successes}/{score.attempts} "
            f"({score.success_ratio:.0%}) | "
            f"{format_score_metrics(score)}"
        )


def format_score_metrics(score: CandidateScore) -> str:
    before_metrics = (
        f"before_gap={score.mean_before_gap_seconds:.3f}s | "
        f"before_pause={score.mean_before_pause_seconds:.3f}s"
    )
    if not REQUIRE_AFTER_BOUNDARY:
        return f"{before_metrics} | after_gap=n/a | after_pause=n/a"
    return (
        f"{before_metrics} | "
        f"after_gap={score.mean_after_gap_seconds:.3f}s | "
        f"after_pause={score.mean_after_pause_seconds:.3f}s"
    )


def score_phrase(
    phrase: str,
    *,
    audio_generator: object,
    voice_style: np.ndarray,
) -> CandidateScore:
    before_gaps: list[float] = []
    after_gaps: list[float] = []
    before_pause_lengths: list[float] = []
    after_pause_lengths: list[float] = []
    successes = 0

    for index, text in enumerate(DEMO_SEGMENTS):
        segment = PhonemeSegment(
            id=f"candidate_{index}",
            segment_id=f"candidate_{index}",
            phoneme_id=0,
            text=text,
            phonemes="",
            tokens=[],
            lang=LANG,
        )
        phonemes, _, timing_tokens = phonemize_short_sentence_phrase(segment, phrase)
        audio, pred_dur = audio_generator._run_onnx(phonemes, voice_style, SPEED)
        if pred_dur is None:
            continue

        timestamps = _join_timestamps(timing_tokens, pred_dur)
        target_tokens = [
            token
            for token in timestamps
            if token.get("is_target")
            and isinstance(token.get("start_ts"), (int, float))
            and isinstance(token.get("end_ts"), (int, float))
        ]
        if not target_tokens:
            continue

        target_start = int(min(float(token["start_ts"]) for token in target_tokens) * SAMPLE_RATE)
        target_end = int(max(float(token["end_ts"]) for token in target_tokens) * SAMPLE_RATE)
        runs = quiet_runs(audio)
        before = nearest_run_before(runs, target_start) if REQUIRE_BEFORE_BOUNDARY else None
        after = nearest_run_after(runs, target_end) if REQUIRE_AFTER_BOUNDARY else None
        if REQUIRE_BEFORE_BOUNDARY and before is None:
            continue
        if REQUIRE_AFTER_BOUNDARY and after is None:
            continue

        before_gap = (
            max(0, target_start - before[1]) / SAMPLE_RATE if before is not None else 0.0
        )
        after_gap = (
            max(0, after[0] - target_end) / SAMPLE_RATE if after is not None else 0.0
        )
        if (
            before_gap > MAX_BOUNDARY_GAP_SECONDS
            or after_gap > MAX_BOUNDARY_GAP_SECONDS
        ):
            continue

        successes += 1
        before_gaps.append(before_gap)
        after_gaps.append(after_gap)
        if before is not None:
            before_pause_lengths.append((before[1] - before[0]) / SAMPLE_RATE)
        if after is not None:
            after_pause_lengths.append((after[1] - after[0]) / SAMPLE_RATE)

    return CandidateScore(
        phrase=phrase,
        successes=successes,
        attempts=len(DEMO_SEGMENTS),
        mean_before_gap_seconds=mean(before_gaps) if before_gaps else float("inf"),
        mean_after_gap_seconds=mean(after_gaps) if after_gaps else float("inf"),
        mean_before_pause_seconds=mean(before_pause_lengths) if before_pause_lengths else 0.0,
        mean_after_pause_seconds=mean(after_pause_lengths) if after_pause_lengths else 0.0,
    )


def quiet_runs(audio: np.ndarray) -> list[tuple[int, int]]:
    speech_frames = energy_based_vad(
        audio,
        SAMPLE_RATE,
        frame_duration_ms=FRAME_DURATION_MS,
        energy_threshold=ENERGY_THRESHOLD,
    )
    quiet_frames = ~speech_frames
    samples_per_frame = max(1, int(SAMPLE_RATE * FRAME_DURATION_MS / 1000))
    min_frames = max(1, int(MIN_SILENCE_SECONDS * 1000 / FRAME_DURATION_MS))
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, is_quiet in enumerate(quiet_frames):
        if is_quiet and start is None:
            start = index
        elif not is_quiet and start is not None:
            if index - start >= min_frames:
                runs.append((start * samples_per_frame, index * samples_per_frame))
            start = None
    if start is not None and len(quiet_frames) - start >= min_frames:
        runs.append((start * samples_per_frame, len(audio)))
    return runs


def nearest_run_before(
    runs: list[tuple[int, int]],
    anchor: int,
) -> tuple[int, int] | None:
    candidates = [run for run in runs if run[1] <= anchor]
    return max(candidates, key=lambda run: run[1], default=None)


def nearest_run_after(
    runs: list[tuple[int, int]],
    anchor: int,
) -> tuple[int, int] | None:
    candidates = [run for run in runs if run[0] >= anchor]
    return min(candidates, key=lambda run: run[0], default=None)


if __name__ == "__main__":
    main()
