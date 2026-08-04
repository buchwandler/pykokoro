#!/usr/bin/env python3
"""
Demonstrate SSMD break markers combined with pause_mode="manual".

This example shows how to use SSMD break markers together with manual
pause control for better prosody in long texts.

Usage:
    python examples/pauses_with_splitting.py

Output:
    pauses_splitting_demo.wav - Long text with pauses and sentence splitting
"""

from pathlib import Path

import numpy as np
import soundfile as sf

from pykokoro import PipelineConfig
from pykokoro.constants import SAMPLE_RATE
from pykokoro.generation_config import GenerationConfig
from pykokoro.onnx_backend import Kokoro
from pykokoro.stages.audio_generation.onnx import OnnxAudioGenerationAdapter
from pykokoro.stages.doc_parsers.ssmd import SsmdDocumentParser
from pykokoro.stages.g2p.kokorog2p import KokoroG2PAdapter
from pykokoro.stages.phoneme_processing.onnx import OnnxPhonemeProcessorAdapter
from pykokoro.types import Trace


def _format_segment_text(text: str, limit: int = 80) -> str:
    cleaned = " ".join(text.strip().split())
    if len(cleaned) <= limit:
        return cleaned
    return f"{cleaned[:limit]}..."


def _print_phoneme_segments(segments, limit: int = 5) -> None:
    print("First phoneme segments after phoneme_processing:")
    for segment in segments[:limit]:
        summary = _format_segment_text(segment.text)
        print(
            "  - "
            f"id={segment.id} "
            f"para={segment.paragraph_idx} sent={segment.sentence_idx} "
            f"pause_before={segment.pause_before:.3f}s "
            f"pause_after={segment.pause_after:.3f}s "
            f"text='{summary}'"
        )
    print()


def _print_paragraph_pause_debug(segments: list) -> None:
    paragraph_zero = [segment for segment in segments if segment.paragraph_idx == 0]
    if not paragraph_zero:
        print("No paragraph 0 segments found.")
        print()
        return
    last_segment = paragraph_zero[-1]
    print(
        "Paragraph 0 last segment pause_after: "
        f"{last_segment.pause_after:.3f}s (id={last_segment.id})"
    )
    print()


def _plot_segments(samples: np.ndarray, sample_rate: int, segments: list) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore[import-not-found]
    except Exception as exc:
        print(f"Skipping plot (matplotlib unavailable): {exc}")
        print()
        return

    time_axis = np.arange(len(samples)) / sample_rate
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(time_axis, samples, linewidth=0.6, color="#333333")

    cursor = 0.0
    for segment in segments:
        if segment.pause_before > 0:
            ax.axvspan(cursor, cursor + segment.pause_before, color="#f4a261", alpha=0.3)
            cursor += segment.pause_before
        if segment.processed_audio is not None:
            cursor += len(segment.processed_audio) / sample_rate
        if segment.pause_after > 0:
            ax.axvspan(cursor, cursor + segment.pause_after, color="#f4a261", alpha=0.3)
            cursor += segment.pause_after

    ax.set_title("Waveform with pause spans (orange)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    fig.tight_layout()
    plt.show()


def main():
    """Generate example with pauses and text splitting."""
    print("Initializing TTS engine...")
    generation = GenerationConfig(
        lang="en-us",
        pause_mode="manual",
        pause_clause=0.3,
        pause_sentence=0.6,
        pause_paragraph=1.2,
        pause_variance=0.05,
        random_seed=42,
    )
    cfg = PipelineConfig(voice="af_sarah", generation=generation)
    kokoro = Kokoro(
        model_path=Path(cfg.model_path) if cfg.model_path else None,
        voices_path=Path(cfg.voices_path) if cfg.voices_path else None,
        model_quality=cfg.model_quality,
        model_source=cfg.model_source,
        model_variant=cfg.model_variant,
        provider=cfg.provider,
        provider_options=cfg.provider_options,
        session_options=cfg.session_options,
        tokenizer_config=cfg.tokenizer_config,
        espeak_config=cfg.espeak_config,
        short_sentence_config=cfg.short_sentence_config,
    )
    doc_parser = SsmdDocumentParser()
    g2p = KokoroG2PAdapter()
    phoneme_processing = OnnxPhonemeProcessorAdapter(kokoro)
    audio_generation = OnnxAudioGenerationAdapter(kokoro)

    # Long text with SSMD pauses and natural sentence breaks
    text = """
    Welcome to our podcast!

    The Future of AI.

    Today's episode covers three main topics. ...c First, we'll explore neural
    networks and how they learn from data to make increasingly accurate predictions.
    ...s Second, we'll dive into deep learning architectures that power modern AI
    systems, including transformers and convolutional neural networks. ...p And
    third, we'll examine real-world applications transforming industries worldwide,
    from healthcare to autonomous vehicles.

    Each of these topics represents a fascinating area of research and development.
    ...c Neural networks, inspired by biological neurons, process information through
    interconnected layers. ...s Deep learning takes this further by adding many
    layers, enabling the system to learn hierarchical representations.

    Let's dive into these fascinating subjects! ...p
    """

    print("=" * 70)
    print("Generating with manual pause control and SSMD pauses...")
    print("=" * 70)
    print("\nThis combines:")
    print("  • pause_mode='manual' - PyKokoro controls pauses precisely")
    print("  • Explicit pause control using SSMD breaks (...c, ...s, ...p)")
    print("  • Automatic handling of long sentences")
    print("  • Natural pause variance for more human-like speech")
    print()

    print("Processing text...")
    print(f"Text length: {len(text)} characters")
    print()

    trace = Trace()
    doc = doc_parser.parse(text, cfg, trace)
    segments = doc.segments or []
    phoneme_segments = g2p.phonemize(segments, doc, cfg, trace)
    processed_segments = phoneme_processing.process(phoneme_segments, cfg, trace)

    _print_phoneme_segments(processed_segments, limit=5)
    _print_paragraph_pause_debug(processed_segments)

    generated_segments = audio_generation.generate(processed_segments, cfg, trace)
    trim_silence = generation.pause_mode == "manual"
    processed_audio_segments = kokoro.postprocess_audio_segments(generated_segments, trim_silence)
    samples = kokoro.concatenate_audio_segments(processed_audio_segments)
    sample_rate = SAMPLE_RATE

    output_file = "pauses_splitting_demo.wav"
    sf.write(output_file, samples, sample_rate)
    duration = len(samples) / sample_rate

    print("✓ Generation complete!")
    print()
    _plot_segments(samples, sample_rate, processed_audio_segments)
    print("=" * 70)
    print(f"Generated: {output_file}")
    print(f"Duration: {duration:.2f} seconds ({duration / 60:.1f} minutes)")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Total samples: {len(samples):,}")
    print("=" * 70)
    print()

    # Show SSMD pause markers used
    pause_counts = {
        "...c": text.count("...c"),
        "...s": text.count("...s"),
        "...p": text.count("...p"),
    }

    total_pause_time = (
        pause_counts["...c"] * 0.3 + pause_counts["...s"] * 0.6 + pause_counts["...p"] * 1.2
    )

    print("SSMD pause statistics:")
    clause_total = pause_counts["...c"] * 0.3
    sentence_total = pause_counts["...s"] * 0.6
    paragraph_total = pause_counts["...p"] * 1.2
    print(f"  Clause pauses (...c):     {pause_counts['...c']} × 0.3s = {clause_total:.1f}s")
    print(f"  Sentence pauses (...s):   {pause_counts['...s']} × 0.6s = {sentence_total:.1f}s")
    print(f"  Paragraph pauses (...p):  {pause_counts['...p']} × 1.2s = {paragraph_total:.1f}s")
    print(f"  Total pause time:         ~{total_pause_time:.1f}s")
    print(f"  Estimated speech time:    ~{duration - total_pause_time:.1f}s")
    print()

    print("Pause modes:")
    print("  • pause_mode='tts' (default) - TTS generates pauses naturally")
    print("  • pause_mode='manual' - PyKokoro controls pauses with precision")
    print()
    print("SSMD break markers:")
    print("  • ...n - No pause (0ms)")
    print("  • ...w - Weak pause (150ms)")
    print("  • ...c - Clause/comma pause (300ms)")
    print("  • ...s - Sentence pause (600ms)")
    print("  • ...p - Paragraph pause (1000ms)")
    print("  • ...500ms - Custom pause (500 milliseconds)")
    print("  • ...2s - Custom pause (2 seconds)")
    print()
    print("Pause variance options:")
    print("  • pause_variance=0.0 - No variance (exact pauses)")
    print("  • pause_variance=0.05 - Default (±100ms at 95% confidence)")
    print("  • pause_variance=0.1 - More variation (±200ms at 95% confidence)")
    print("  • random_seed=42 - Reproducible results")
    print("  • random_seed=None - Different pauses each time")
    print()


if __name__ == "__main__":
    main()
