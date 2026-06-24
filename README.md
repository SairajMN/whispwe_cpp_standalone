# whisper_cpp_standalone

Zero-dependency whisper.cpp STT adapter. Drop-in replacement for the GLC v1
`whisper_cpp` provider with extra features:

- **VAD silence trim** — removes >3s silence before inference (saves time)
- **Gain boost** — 1.5x amplitude for feeble voices
- **Music detection** — discards hallucinated text on instrumental audio
- **Configurable model** — `WHISPER_MODEL=tiny` for <1s on 25s audio

## Files

```
whisper_cpp_standalone/
├── wrapper.py      # whisper-cli subprocess + VAD trim
├── adapter.py      # transcribe() with gain + music detection
├── test_whisper.py # 7 tests + real audio demo
└── README.md
```

## Quick start

```sh
# Run tests (no whisper-cli needed)
uv run python test_whisper.py

# Transcribe real audio
uv run python test_whisper.py /path/to/speech.wav
```

## Environment variables

| Var | Default | Purpose |
|-----|---------|---------|
| `WHISPER_MODEL` | `tiny` | Model size: `tiny`, `base`, `small`, `medium`, `large` |
| `WHISPER_THREADS` | `4` | Thread count for inference |

## Speed tips for Windows

```sh
# 25s audio → ~1s response
set WHISPER_MODEL=tiny
set WHISPER_THREADS=4
uv run python test_whisper.py audio.wav
```

## How it works

1. **Silence check** — all-zero audio returns empty immediately
2. **Gain boost** — 16-bit WAV samples multiplied by 1.5x
3. **VAD trim** — removes contiguous silence >3s
4. **whisper-cli** — runs with `-t N` threads
5. **Music detection** — ZCR heuristic discards music hallucinations

## Skipped (YAGNI)

- Custom VAD model (silence trim covers the common case)
- ffmpeg resampling (whisper-cli handles sample rate)
- Real-time streaming (belongs on WebSocket route)