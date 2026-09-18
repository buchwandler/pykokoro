"""Export a PyKokoro render as an AudioCompose job."""

from __future__ import annotations

from pathlib import Path

from pykokoro import GenerationConfig, KokoroPipeline, PipelineConfig


def main() -> None:
    pipeline = KokoroPipeline(PipelineConfig(generation=GenerationConfig(lang="en-us")))
    job = pipeline.to_audio_job("Hello from an explicit audio job.")
    job.save(Path("hello.audiojob.json"))
    result = pipeline.run("Hello from an AudioResult.")
    result.save_wav("hello.wav")
    pipeline.close()


if __name__ == "__main__":
    main()
