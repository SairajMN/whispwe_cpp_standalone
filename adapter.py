"""Standalone whisper.cpp STT adapter.

No deps. Gain boost for feeble voices, music detection, silence trim.
Matches the GLC v1 TranscribeResult contract.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Any

from wrapper import run_whisper_cpp


@dataclass
class TranscribeResult:
    text: str = ""
    language: str = "en"
    duration_ms: int = 0
    provider: str = "whisper_cpp"
    cost_usd: float = 0.0


class STTError(Exception):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


def _amplify_wav(audio: bytes, gain: float = 1.5) -> bytes:
    """Boost 16-bit mono WAV amplitude. Returns original if not WAV/PCM."""
    if len(audio) < 44 or audio[:4] != b"RIFF" or audio[20:22] != b"\x01\x00":
        return audio
    bits_per = struct.unpack("<H", audio[34:36])[0]
    if bits_per != 16:
        return audio
    data_start = struct.unpack("<I", audio[16:20])[0] + 8
    header = audio[:data_start]
    raw = audio[data_start:]
    samples = memoryview(bytearray(raw)).cast("h")
    for i in range(len(samples)):
        val = int(samples[i] * gain)
        if val > 32767:
            val = 32767
        elif val < -32768:
            val = -32768
        samples[i] = val
    return header + bytes(samples)


def _is_music_likely(audio: bytes) -> bool:
    """Return True if >80% of non-silent frames have ZCR > 0.25 (music)."""
    if len(audio) < 44 or audio[:4] != b"RIFF":
        return False
    bits_per = struct.unpack("<H", audio[34:36])[0]
    if bits_per != 16:
        return False
    data_start = struct.unpack("<I", audio[16:20])[0] + 8
    raw = audio[data_start:]
    if len(raw) < 4:
        return False
    samples = memoryview(bytearray(raw)).cast("h")
    frame_size = 512
    high_freq = 0
    total = 0
    for start in range(0, len(samples) - frame_size, frame_size):
        frame = samples[start : start + frame_size]
        crossings = sum(1 for i in range(1, len(frame)) if frame[i] * frame[i - 1] < 0)
        zcr = crossings / len(frame)
        energy = sum(abs(int(s)) for s in frame)
        if energy > 500:
            total += 1
            if zcr > 0.25:
                high_freq += 1
    return total > 0 and (high_freq / total) > 0.8


class MockWhisper:
    """Fake that mirrors WhisperCppMock from GLC tests."""

    def __init__(self):
        self.canned_text = "hello"
        self.canned_lang = "en"
        self.canned_duration_ms = 200
        self.received_calls: list[dict[str, Any]] = []
        self.upstream_failure: tuple[int, str] | None = None
        self.subprocess_call_count = 0

    async def transcribe(self, audio: bytes, mime: str) -> TranscribeResult:
        self.subprocess_call_count += 1
        self.received_calls.append({"audio_len": len(audio), "mime": mime})
        if self.upstream_failure is not None:
            status, msg = self.upstream_failure
            raise STTError(msg, status=status)
        return TranscribeResult(
            text=self.canned_text,
            language=self.canned_lang,
            duration_ms=self.canned_duration_ms,
            provider="whisper_cpp",
            cost_usd=0.0,
        )


class Provider:
    name = "whisper_cpp"

    def __init__(self, config: dict | None = None):
        self.config = config or {}

    async def transcribe(self, audio: bytes, mime: str) -> TranscribeResult:
        # silence check
        if audio and all(b == 0 for b in audio):
            return TranscribeResult(
                text="", language="en", duration_ms=0,
                provider=self.name, cost_usd=0.0,
            )

        # gain boost for feeble voices
        if "wav" in mime:
            audio = _amplify_wav(audio, gain=1.5)

        # mock delegation
        mock = self.config.get("mock")
        if mock is not None:
            return await mock.transcribe(audio, mime)

        # real whisper-cli
        try:
            text, language, duration_ms = run_whisper_cpp(audio, mime)
        except RuntimeError as e:
            raise STTError(str(e), status=500) from e

        # music detection — discard hallucinated text
        if text and "wav" in mime and _is_music_likely(audio):
            text = ""

        return TranscribeResult(
            text=text, language=language, duration_ms=duration_ms,
            provider=self.name, cost_usd=0.0,
        )