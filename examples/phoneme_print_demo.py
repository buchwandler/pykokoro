"""
Generate phonemes with the standard pipeline.

Usage:
    python examples/phoneme_print_demo.py
"""

from pykokoro import GenerationConfig, KokoroPipeline, PipelineConfig


def main() -> None:
    text = "Hello world. This example prints phoneme outputs from the pipeline."
    cfg = PipelineConfig(voice="af_sarah", generation=GenerationConfig(lang="en-us"))
    pipeline = KokoroPipeline(cfg)
    result = pipeline.run(text)

    print("Text:")
    print(text)
    print("\nPhoneme segments:")
    for segment in result.phoneme_segments:
        print(segment.format_readable())


if __name__ == "__main__":
    main()
