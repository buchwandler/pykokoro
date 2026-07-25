"""Pytest configuration and fixtures for pykokoro tests."""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def pytest_configure(config):
    """Print diagnostic information at test session start."""
    if os.getenv("PYKOKORO_TEST_DIAGNOSTICS") != "1":
        return
    print("\n" + "=" * 70, file=sys.stderr)
    print("PYKOKORO TEST DIAGNOSTICS", file=sys.stderr)
    print("=" * 70, file=sys.stderr)
    print(f"Python version: {sys.version}", file=sys.stderr)

    try:
        import kokorog2p

        print(f"kokorog2p version: {kokorog2p.__version__}", file=sys.stderr)
    except ImportError as e:
        print(f"kokorog2p import failed: {e}", file=sys.stderr)

    try:
        import subprocess

        result = subprocess.run(
            ["espeak-ng", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        print(f"espeak-ng version: {result.stdout.strip()}", file=sys.stderr)
    except (OSError, subprocess.SubprocessError) as e:
        print(f"espeak-ng check failed: {e}", file=sys.stderr)

    try:
        from pykokoro.tokenizer import create_tokenizer

        tokenizer = create_tokenizer()
        result = tokenizer.phonemize("test", lang="en-us")
        print(
            f"Test phonemization: 'test' -> '{result}' (len={len(result)})",
            file=sys.stderr,
        )
    except (RuntimeError, OSError) as e:
        print(f"Test phonemization failed: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc(file=sys.stderr)

    print("=" * 70 + "\n", file=sys.stderr)
