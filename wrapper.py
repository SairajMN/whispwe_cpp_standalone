"""whisper-cli subprocess wrapper with VAD silence trim.

No deps. Runs whisper-cli as a subprocess, trims silence >3s before
feeding audio to save inference time.
"""

from __future__ import annotations

import json
import os
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path

# ponytail: env-var switches, no config file
MODEL_SIZE = os.getenv("WHISPER_MODEL", "tiny")  # tiny, base, small, medium, large
MODEL_DIR = Path(os.path.expanduser(f"~/.glc/models/whisper-{MODEL_SIZE}"))
MODEL_FILE = MODEL_DIR / f"ggml-{MODEL_SIZE}.bin"
THREADS = int(os.getenv("WHISPER_THREADS", "4"))


def _trim_silence_wav(audio: bytes, min_silence_ms: int = 3000) -> bytes:
    """Remove contiguous silence > `min_silence_ms` from 16-bit mono WAV.

    Keeps non-silent segments, re-stitches. Returns original if not a
    valid 16-bit mono PCM WAV or if no silence found.
    """
    if len(audio) < 44 or audio[:4] != b"RIFF":
        return audio
    bits_per = struct.unpack("<H", audio[34:36])[0]
    if bits_per != 16:
        return audio
    channels = struct.unpack("<H", audio[22:24])[0]
    if channels != 1:
        return audio  # ponytail: only mono for now
    sample_rate = struct.unpack("<I", audio[24:28])[0]
    data_start = struct.unpack("<I", audio[16:20])[0] + 8
    raw = audio[data_start:]
    if len(raw) < 4:
        return audio

    samples = memoryview(bytearray(raw)).cast("h")
    frame_ms = 30  # 30ms frames
    frame_len = int(sample_rate * frame_ms / 1000)
    silence_frames = int(min_silence_ms / frame_ms)

    is_silent = []
    for start in range(0, len(samples), frame_len):
        frame = samples[start : start + frame_len]
        energy = sum(abs(int(s)) for s in frame) / max(len(frame), 1)
        is_silent.append(energy < 500)

    # ponytail: collapse runs of silence > min_silence_ms
    keep_regions: list[tuple[int, int]] = []
    i = 0
    while i < len(is_silent):
        if is_silent[i]:
            run_start = i
            while i < len(is_silent) and is_silent[i]:
                i += 1
            run_len = i - run_start
            if run_len >= silence_frames:
                # skip this long silence run
                continue
            # keep short silence as-is (brief pauses in speech)
            keep_regions.append((run_start * frame_len, i * frame_len))
        else:
            speech_start = i
            while i < len(is_silent) and not is_silent[i]:
                i += 1
            keep_regions.append((speech_start * frame_len, i * frame_len))

    if len(keep_regions) == 1 and keep_regions[0] == (0, len(samples)):
        return audio  # no change

    header = audio[:data_start]
    out_samples: list[int] = []
    for lo, hi in keep_regions:
        out_samples.extend(samples[lo:hi])

    # rebuild WAV with updated data size
    data_bytes = struct.pack(f"<{len(out_samples)}h", *out_samples)
    data_size = len(data_bytes)
    file_size = data_start + data_size - 8
    new_header = bytearray(header)
    new_header[4:8] = struct.pack("<I", file_size)
    new_header[data_start - 4 : data_start] = struct.pack("<I", data_size)
    return bytes(new_header) + data_bytes


def run_whisper_cpp(audio: bytes, mime: str) -> tuple[str, str, int]:
    cli = shutil.which("whisper-cli") or shutil.which("whisper.cpp")
    if cli is None:
        raise RuntimeError(
            "whisper-cli not found on PATH. Install from https://github.com/ggerganov/whisper.cpp"
        )
    if not MODEL_FILE.exists():
        raise RuntimeError(
            f"model not found at {MODEL_FILE}. Set WHISPER_MODEL=tiny for fast download, "
            f"or run: daemon/install.sh --models"
        )

    # ponytail: trim silence >3s before feeding to whisper
    if "wav" in mime:
        audio = _trim_silence_wav(audio, min_silence_ms=3000)

    suffix = ".wav" if "wav" in mime else ".bin"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(audio)
        audio_path = Path(f.name)

    try:
        out = subprocess.run(
            [cli, "-m", str(MODEL_FILE), "-f", str(audio_path), "-oj", "-t", str(THREADS)],
            capture_output=True,
            text=True,
            check=True,
        )
    finally:
        audio_path.unlink(missing_ok=True)

    json_path = audio_path.with_suffix(audio_path.suffix + ".json")
    if json_path.exists():
        d = json.loads(json_path.read_text())
        json_path.unlink(missing_ok=True)
        segments = d.get("transcription") or d.get("segments") or []
        text = " ".join((s.get("text") or "").strip() for s in segments).strip()
        language = d.get("language") or "en"
        duration_ms = int(segments[-1].get("offsets", {}).get("to", 0)) if segments else 0
        return text, language, duration_ms

    return out.stdout.strip(), "en", 0