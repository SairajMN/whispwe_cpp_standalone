#!/usr/bin/env python3
"""Add random pauses to an audio file.

Usage: python add_pauses.py input.mp3 output.wav [--target 30] [--min-pause 1] [--max-pause 3]
"""

from __future__ import annotations

import argparse
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path


def get_duration(filepath: str) -> float:
    """Get audio duration in seconds using ffprobe."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", filepath],
        capture_output=True, text=True, check=True,
    )
    return float(result.stdout.strip())


def extract_segment(input_path: str, start: float, duration: float, output_path: str):
    """Extract a segment from input audio."""
    subprocess.run(
        ["ffmpeg", "-y", "-i", input_path, "-ss", str(start), "-t", str(duration),
         "-ar", "16000", "-ac", "1", "-sample_fmt", "s16", output_path],
        capture_output=True, check=True,
    )


def create_silence(duration: float, output_path: str):
    """Create a silence WAV file."""
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", f"anullsrc=r=16000:cl=mono",
         "-t", str(duration), "-ar", "16000", "-ac", "1", "-sample_fmt", "s16", output_path],
        capture_output=True, check=True,
    )


def concat_files(file_list: list[str], output_path: str):
    """Concatenate multiple WAV files using ffmpeg."""
    # Create concat list with absolute paths
    concat_file = output_path + ".concat.txt"
    with open(concat_file, "w") as f:
        for filepath in file_list:
            abs_path = os.path.abspath(filepath)
            f.write(f"file '{abs_path}'\n")
    
    result = subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_file,
         "-ar", "16000", "-ac", "1", output_path],
        capture_output=True, text=True,
    )
    
    if result.returncode != 0:
        print(f"ffmpeg concat error:\n{result.stderr}")
        raise subprocess.CalledProcessError(result.returncode, result.args)
    
    os.unlink(concat_file)


def add_random_pauses(input_path: str, output_path: str, target_duration: float = 30.0,
                      min_pause: float = 1.0, max_pause: float = 3.0):
    """Extract audio and insert random pauses to reach target duration."""
    
    # Get input duration
    input_duration = get_duration(input_path)
    print(f"Input duration: {input_duration:.1f}s")
    
    # Extract first 15s of actual audio
    audio_duration = min(15.0, input_duration)
    temp_audio = output_path + ".audio.wav"
    extract_segment(input_path, 0, audio_duration, temp_audio)
    print(f"Extracted {audio_duration:.1f}s of audio")
    
    # Calculate how much silence we need
    current_duration = audio_duration
    needed_silence = target_duration - current_duration
    
    if needed_silence <= 0:
        print(f"Audio already {current_duration:.1f}s, no pauses needed")
        os.rename(temp_audio, output_path)
        return
    
    print(f"Need {needed_silence:.1f}s of pauses to reach {target_duration:.1f}s")
    
    # Generate random pause positions with minimum spacing
    num_pauses = random.randint(2, 4)
    min_spacing = audio_duration / (num_pauses + 1)
    pause_positions = []
    for i in range(num_pauses):
        pos = random.uniform(i * min_spacing + 0.5, (i + 1) * min_spacing - 0.5)
        pause_positions.append(pos)
    pause_positions.sort()
    
    # Calculate pause durations to hit target exactly
    needed_silence = target_duration - audio_duration
    base_pause = needed_silence / num_pauses
    # Adaptive bounds: if we need more silence than max_pause allows, bump max
    effective_max = max(max_pause, base_pause + 0.5)
    pause_durations = [base_pause + random.uniform(-0.3, 0.3) for _ in range(num_pauses)]
    pause_durations = [max(min_pause, min(effective_max, d)) for d in pause_durations]
    
    print(f"Inserting {num_pauses} pauses at positions: {[f'{p:.1f}s' for p in pause_positions]}")
    print(f"Pause durations: {[f'{d:.1f}s' for d in pause_durations]}")
    
    # Split audio at pause positions and interleave with silence
    segments = []
    prev_pos = 0.0
    
    for idx, pause_pos in enumerate(pause_positions):
        # Add audio segment before this pause
        if pause_pos > prev_pos:
            seg_duration = pause_pos - prev_pos
            seg_file = output_path + f".seg_{len(segments)}.wav"
            extract_segment(temp_audio, prev_pos, seg_duration, seg_file)
            segments.append(seg_file)
        
        # Add calculated silence
        pause_duration = pause_durations[idx]
        silence_file = output_path + f".silence_{len(segments)}.wav"
        create_silence(pause_duration, silence_file)
        segments.append(silence_file)
        
        prev_pos = pause_pos
    
    # Add remaining audio after last pause
    if prev_pos < audio_duration:
        seg_duration = audio_duration - prev_pos
        seg_file = output_path + f".seg_{len(segments)}.wav"
        extract_segment(temp_audio, prev_pos, seg_duration, seg_file)
        segments.append(seg_file)
    
    # Concatenate all segments
    concat_files(segments, output_path)
    
    # Cleanup temp files
    os.unlink(temp_audio)
    for seg in segments:
        if os.path.exists(seg):
            os.unlink(seg)
    
    # Verify output duration
    output_duration = get_duration(output_path)
    print(f"Output duration: {output_duration:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="Add random pauses to audio")
    parser.add_argument("input", help="Input audio file (MP3, WAV, etc.)")
    parser.add_argument("output", help="Output WAV file")
    parser.add_argument("--target", type=float, default=30.0, help="Target duration in seconds (default: 30)")
    parser.add_argument("--min-pause", type=float, default=1.0, help="Minimum pause duration (default: 1)")
    parser.add_argument("--max-pause", type=float, default=3.0, help="Maximum pause duration (default: 3)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: {args.input} not found")
        sys.exit(1)
    
    add_random_pauses(args.input, args.output, args.target, args.min_pause, args.max_pause)
    print(f"\n✅ Created {args.output}")


if __name__ == "__main__":
    main()