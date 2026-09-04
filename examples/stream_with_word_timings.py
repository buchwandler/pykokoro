"""Render sentence units and select the source span at a playback sample cursor."""

from __future__ import annotations

from pykokoro import GenerationConfig, PipelineConfig, build_pipeline


def main() -> None:
    text = "Word timing lets an application highlight the word being spoken."
    with (
        build_pipeline(
            config=PipelineConfig(
                voice="af_sarah",
                generation=GenerationConfig(lang="en-us"),
            )
        ) as pipeline,
        pipeline.prepare_units(text, unit="sentence") as prepared,
    ):
        for result in prepared.render():
            if not result.word_timings:
                print("No word timings available for this model.")
            for word in result.word_timings:
                source_span = text[word.char_start : word.char_end]
                print(f"{source_span!r}: samples {word.start_sample}-{word.end_sample}")
                playback_sample = (word.start_sample + word.end_sample) // 2
                print(f"  active at sample {playback_sample}: {source_span!r}")
            result.release_audio()


if __name__ == "__main__":
    main()
