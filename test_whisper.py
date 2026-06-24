#!/usr/bin/env python3
"""Tests for standalone whisper_cpp adapter.

Usage:
    python test_whisper.py              # run all 7 tests
    python test_whisper.py /path/to.wav # run tests + transcribe real audio

No deps. Tests use MockWhisper, never hit whisper-cli.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time

from adapter import MockWhisper, Provider, STTError, TranscribeResult


def test_provider_name_matches():
    p = Provider(config={"mock": MockWhisper()})
    assert p.name == "whisper_cpp", f"expected whisper_cpp, got {p.name}"
    print("  ✔ test_provider_name_matches")


def test_transcribe_returns_transcribe_result():
    async def run():
        mock = MockWhisper()
        p = Provider(config={"mock": mock})
        r = await p.transcribe(b"AUDIO", "audio/wav")
        assert isinstance(r, TranscribeResult), f"expected TranscribeResult, got {type(r)}"
        assert r.provider == "whisper_cpp", f"expected whisper_cpp, got {r.provider}"
        assert r.language == "en", f"expected en, got {r.language}"
        print("  ✔ test_transcribe_returns_transcribe_result")
    asyncio.run(run())


def test_transcribe_passes_audio_to_upstream():
    async def run():
        mock = MockWhisper()
        p = Provider(config={"mock": mock})
        await p.transcribe(b"x" * 1234, "audio/wav")
        assert len(mock.received_calls) > 0, "adapter must invoke upstream"
        assert mock.received_calls[-1]["audio_len"] == 1234, (
            f"expected 1234, got {mock.received_calls[-1]['audio_len']}"
        )
        print("  ✔ test_transcribe_passes_audio_to_upstream")
    asyncio.run(run())


def test_transcribe_records_duration_ms():
    async def run():
        mock = MockWhisper()
        mock.canned_duration_ms = 1337
        p = Provider(config={"mock": mock})
        r = await p.transcribe(b"AUDIO", "audio/wav")
        assert r.duration_ms == 1337, f"expected 1337, got {r.duration_ms}"
        print("  ✔ test_transcribe_records_duration_ms")
    asyncio.run(run())


def test_transcribe_propagates_upstream_error():
    async def run():
        mock = MockWhisper()
        mock.upstream_failure = (502, "boom")
        p = Provider(config={"mock": mock})
        try:
            await p.transcribe(b"AUDIO", "audio/wav")
            assert False, "expected STTError"
        except STTError as e:
            assert e.status == 502, f"expected 502, got {e.status}"
            print("  ✔ test_transcribe_propagates_upstream_error")
    asyncio.run(run())


def test_transcribe_handles_empty_audio():
    async def run():
        mock = MockWhisper()
        p = Provider(config={"mock": mock})
        r = await p.transcribe(b"", "audio/wav")
        assert isinstance(r, TranscribeResult), "empty audio must return TranscribeResult"
        print("  ✔ test_transcribe_handles_empty_audio")
    asyncio.run(run())


def test_channel_specific_behaviour_vad_skips_silent_input():
    async def run():
        mock = MockWhisper()
        p = Provider(config={"mock": mock})
        silent = b"\x00" * 16000
        r = await p.transcribe(silent, "audio/wav")
        assert r.text == "", f"expected empty text, got {r.text!r}"
        assert mock.subprocess_call_count == 0, (
            f"expected 0 subprocess calls, got {mock.subprocess_call_count}"
        )
        print("  ✔ test_channel_specific_behaviour_vad_skips_silent_input")
    asyncio.run(run())


ALL_TESTS = [
    test_provider_name_matches,
    test_transcribe_returns_transcribe_result,
    test_transcribe_passes_audio_to_upstream,
    test_transcribe_records_duration_ms,
    test_transcribe_propagates_upstream_error,
    test_transcribe_handles_empty_audio,
    test_channel_specific_behaviour_vad_skips_silent_input,
]


async def demo_transcribe(wav_path: str):
    """Transcribe a real WAV file using whisper-cli."""
    wav_path = os.path.normpath(wav_path)  # handle Windows backslashes
    print(f"\n=== Real audio demo: {wav_path} ===\n")

    with open(wav_path, "rb") as f:
        audio = f.read()

    print(f"  Audio size: {len(audio):,} bytes")
    print(f"  Audio duration: ~{len(audio) // 32000}s (approx at 16kHz)")

    p = Provider()
    start = time.perf_counter()
    try:
        result = await p.transcribe(audio, "audio/wav")
        elapsed = time.perf_counter() - start
        print(f"\n  Result:")
        print(f"    Text:      {result.text!r}")
        print(f"    Language:  {result.language}")
        print(f"    Duration:  {result.duration_ms}ms")
        print(f"    Time:      {elapsed:.2f}s")
        if elapsed > 1.0:
            print(f"     Above 1s target. Set WHISPER_MODEL=tiny and WHISPER_THREADS=4")
        else:
            print(f"     Under 1s target")
    except Exception as e:
        elapsed = time.perf_counter() - start
        print(f"\n  Error after {elapsed:.2f}s: {e}")
        print("  Make sure whisper-cli is on PATH and model is downloaded.")


def main():
    print("=== whisper_cpp standalone tests ===\n")

    passed = 0
    failed = 0
    for test in ALL_TESTS:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  ✘ {test.__name__}: {e}")
            failed += 1

    print(f"\n  {passed}/{len(ALL_TESTS)} passed", end="")
    if failed:
        print(f", {failed} failed")
        sys.exit(1)
    else:
        print()

    # real audio demo if arg provided
    if len(sys.argv) > 1:
        asyncio.run(demo_transcribe(sys.argv[1]))


if __name__ == "__main__":
    main()