#!/usr/bin/env python3
"""
Korean TTS example using pykokoro.

This example demonstrates text-to-speech synthesis in Korean
using the Kokoro model.

IMPORTANT NOTES:
- The Kokoro model was not explicitly trained on Korean
- There are no Korean-specific voices available
- espeak-ng's Korean phonemization support is limited
- The output may not accurately represent proper Korean pronunciation
- This example uses Japanese voices as they may handle Korean phonemes
  better than English voices due to linguistic similarities

This example is provided for experimental purposes and to demonstrate
the phonemization pipeline. For production-quality Korean TTS, consider
using models specifically trained on Korean language data.

Usage:
    python examples/korean.py

Output:
    korean_demo.wav - Generated Korean speech audio (experimental quality)
"""

import logging
import os

import soundfile as sf

try:
    from ._output import artifact_path
except ImportError:
    from _output import artifact_path

from pykokoro import KokoroPipeline, PipelineConfig
from pykokoro.generation_config import GenerationConfig
from pykokoro.voice_manager import VoiceBlend

# Enable phoneme debugging to see what phonemes are generated
os.environ["PYKOKORO_DEBUG_PHONEMES"] = "1"

# Configure logging to display phoneme debug information
logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")

# Korean text samples - Greetings, proverbs, and common phrases
# Note: espeak-ng's Korean support may be limited, so results may vary
TEXT = """
안녕하세요! 한국어 음성 합성 예제에 오신 것을 환영합니다.

한국어는 아름답고 독특한 언어입니다.
한글은 세종대왕이 창제한 과학적인 문자 체계입니다.

티끌 모아 태산이라는 속담이 있습니다.
작은 것들이 모여서 큰 것을 이룰 수 있다는 뜻입니다.

서울은 대한민국의 수도이며 매우 큰 도시입니다.
한국의 전통 음식으로는 김치, 불고기, 비빔밥이 유명합니다.

오늘은 날씨가 좋습니다. 하늘이 맑고 바람이 시원합니다.
공원에서 산책하며 커피를 마시고 싶습니다.

숫자를 세어 봅시다.
하나, 둘, 셋, 넷, 다섯, 여섯, 일곱, 여덟, 아홉, 열.

한국의 사계절은 모두 아름답습니다.
봄에는 꽃이 피고, 여름에는 푸르며, 가을에는 단풍이 들고, 겨울에는 눈이 옵니다.

공부는 평생 해야 하는 것입니다.
배움에는 끝이 없다는 말이 있습니다.

감사합니다. 좋은 하루 되세요!
안녕히 가세요!
"""

# Voice blend - using Asian voices for potentially better Korean pronunciation
# Japanese voices might handle Korean phonemes better due to linguistic similarities
BLEND = "jf_alpha:50,jf_gongitsune:50"  # Blend of Japanese female voices

LANG = "ko"  # Korean language code


def main():
    """Generate Korean speech using phonemization."""
    print("Initializing TTS engine...")
    pipe = KokoroPipeline(
        PipelineConfig(
            voice=VoiceBlend.parse(BLEND),
            generation=GenerationConfig(lang=LANG, speed=1.0),
        )
    )

    print("=" * 60)
    print("NOTE: Kokoro was NOT explicitly trained on Korean.")
    print("The model will attempt Korean phonemization via espeak-ng.")
    print("WARNING: espeak-ng's Korean support is limited.")
    print("Pronunciation may not be accurate or native-sounding.")
    print("Using Japanese voices for potentially better results.")
    print("=" * 60)
    print(f"\nVoice: {BLEND}")
    print(f"Language: {LANG}")

    print("\nGenerating audio...")
    res = pipe.run(TEXT)
    samples, sample_rate = res.audio, res.sample_rate

    output_file = artifact_path("korean_demo.wav")
    sf.write(output_file, samples, sample_rate)

    duration = len(samples) / sample_rate
    print(f"\nCreated {output_file}")
    print(f"Duration: {duration:.2f} seconds")


if __name__ == "__main__":
    main()
