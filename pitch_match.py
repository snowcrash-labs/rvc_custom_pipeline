#!/usr/bin/env python3
"""Determine how far a singer's average pitch needs to be shifted to
fall within a target frequency range (e.g. the usable range output
by hnr_range.py).

Usage:
    python pitch_match.py song.wav --range-low 293.7 --range-high 698.5
    python pitch_match.py song.wav --range-low 293.7 --range-high 698.5 --json
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import librosa
import numpy as np

logger = logging.getLogger(__name__)

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def midi_from_hz(freq_hz: float) -> float:
    if freq_hz <= 0:
        return -1
    return 12 * np.log2(freq_hz / 440.0) + 69


def hz_from_midi(midi: float) -> float:
    return 440.0 * 2 ** ((midi - 69) / 12.0)


def note_from_hz(freq_hz: float) -> str:
    if freq_hz <= 0:
        return "?"
    midi = int(round(midi_from_hz(freq_hz)))
    octave = (midi // 12) - 1
    return f"{NOTE_NAMES[midi % 12]}{octave}"


def analyse_singer_pitch(
    audio_path: Path,
    sr: int = 16000,
    hop_length: int = 512,
    fmin: float = 65.0,
    fmax: float = 1500.0,
) -> dict:
    """Track pitch across an audio file and return statistics."""
    y, _ = librosa.load(str(audio_path), sr=sr, mono=True)

    f0, voiced_flag, _ = librosa.pyin(
        y, fmin=fmin, fmax=fmax, sr=sr, hop_length=hop_length,
        fill_na=0.0,
    )

    voiced_f0 = f0[voiced_flag & (f0 > 0)]

    if len(voiced_f0) == 0:
        return {"error": "No voiced frames detected"}

    midi_values = np.array([midi_from_hz(f) for f in voiced_f0])

    mean_midi = float(np.mean(midi_values))
    median_midi = float(np.median(midi_values))
    min_midi = float(np.min(midi_values))
    max_midi = float(np.max(midi_values))

    return {
        "mean_hz": round(hz_from_midi(mean_midi), 1),
        "mean_note": note_from_hz(hz_from_midi(mean_midi)),
        "mean_midi": round(mean_midi, 2),
        "median_hz": round(hz_from_midi(median_midi), 1),
        "median_note": note_from_hz(hz_from_midi(median_midi)),
        "median_midi": round(median_midi, 2),
        "low_hz": round(hz_from_midi(min_midi), 1),
        "low_note": note_from_hz(hz_from_midi(min_midi)),
        "high_hz": round(hz_from_midi(max_midi), 1),
        "high_note": note_from_hz(hz_from_midi(max_midi)),
        "range_semitones": round(max_midi - min_midi, 1),
        "n_voiced_frames": len(voiced_f0),
    }


def compute_pitch_shift(
    singer: dict,
    range_low_hz: float,
    range_high_hz: float,
) -> dict:
    """Compute the semitone/octave shift needed to place the singer's
    average pitch within the target range."""
    target_center_midi = (midi_from_hz(range_low_hz) + midi_from_hz(range_high_hz)) / 2
    target_center_hz = hz_from_midi(target_center_midi)

    singer_mean_midi = singer["mean_midi"]
    singer_median_midi = singer["median_midi"]

    shift_from_median = target_center_midi - singer_median_midi

    shift_octaves_raw = shift_from_median / 12.0
    shift_semitones = round(shift_octaves_raw) * 12

    shifted_median_hz = hz_from_midi(singer_median_midi + shift_semitones)
    shifted_low_hz = hz_from_midi(midi_from_hz(singer["low_hz"]) + shift_semitones)
    shifted_high_hz = hz_from_midi(midi_from_hz(singer["high_hz"]) + shift_semitones)

    in_range = range_low_hz <= shifted_median_hz <= range_high_hz

    return {
        "target_range": {
            "low_hz": range_low_hz,
            "low_note": note_from_hz(range_low_hz),
            "high_hz": range_high_hz,
            "high_note": note_from_hz(range_high_hz),
            "center_hz": round(target_center_hz, 1),
            "center_note": note_from_hz(target_center_hz),
        },
        "shift_semitones": shift_semitones,
        "shift_octaves": shift_semitones // 12,
        "shift_exact_semitones": round(shift_from_median, 2),
        "shifted_median_hz": round(shifted_median_hz, 1),
        "shifted_median_note": note_from_hz(shifted_median_hz),
        "shifted_low_hz": round(shifted_low_hz, 1),
        "shifted_high_hz": round(shifted_high_hz, 1),
        "median_in_target_range": in_range,
    }


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    p = argparse.ArgumentParser(
        description="Determine pitch shift needed to match a singer's average pitch "
                    "to a target frequency range (from hnr_range.py).",
    )
    p.add_argument("input", type=Path, help="Audio file to analyse")
    p.add_argument("--range-low", type=float, required=True,
                   help="Lower bound of target frequency range (Hz)")
    p.add_argument("--range-high", type=float, required=True,
                   help="Upper bound of target frequency range (Hz)")
    p.add_argument("--sr", type=int, default=16000, help="Analysis sample rate")
    p.add_argument("--fmin", type=float, default=65.0)
    p.add_argument("--fmax", type=float, default=1500.0)
    p.add_argument("--json", action="store_true", help="Output as JSON")
    args = p.parse_args()

    singer = analyse_singer_pitch(
        args.input, sr=args.sr, fmin=args.fmin, fmax=args.fmax,
    )

    if "error" in singer:
        print(f"Error: {singer['error']}")
        sys.exit(1)

    shift = compute_pitch_shift(singer, args.range_low, args.range_high)

    if args.json:
        import json
        print(json.dumps({"singer": singer, "pitch_shift": shift}, indent=2))
        return

    print(f"\n{'=' * 62}")
    print(f"  Pitch Match Analysis")
    print(f"{'=' * 62}\n")

    print(f"  Singer analysis:")
    print(f"    Mean pitch:    {singer['mean_note']:>5}  ({singer['mean_hz']} Hz)")
    print(f"    Median pitch:  {singer['median_note']:>5}  ({singer['median_hz']} Hz)")
    print(f"    Range:         {singer['low_note']:>5}  ({singer['low_hz']} Hz)"
          f" — {singer['high_note']} ({singer['high_hz']} Hz)")
    print(f"    Span:          {singer['range_semitones']} semitones"
          f" ({singer['range_semitones']/12:.1f} octaves)")

    t = shift["target_range"]
    print(f"\n  Target range (from hnr_range.py):")
    print(f"    Low:           {t['low_note']:>5}  ({t['low_hz']} Hz)")
    print(f"    High:          {t['high_note']:>5}  ({t['high_hz']} Hz)")
    print(f"    Center:        {t['center_note']:>5}  ({t['center_hz']} Hz)")

    direction = "up" if shift["shift_semitones"] > 0 else "down"
    if shift["shift_semitones"] == 0:
        direction = "none needed"

    print(f"\n  Recommended transpose (nearest octave):")
    print(f"    Semitones:     {shift['shift_semitones']:+d}  ({direction})")
    print(f"    Octaves:       {shift['shift_octaves']:+d}")
    print(f"    Exact offset:  {shift['shift_exact_semitones']:+.2f} semitones")

    print(f"\n  After shift:")
    print(f"    Median:        {shift['shifted_median_note']:>5}  ({shift['shifted_median_hz']} Hz)")
    print(f"    Range:         {note_from_hz(shift['shifted_low_hz']):>5}  ({shift['shifted_low_hz']} Hz)"
          f" — {note_from_hz(shift['shifted_high_hz'])} ({shift['shifted_high_hz']} Hz)")
    status = "YES" if shift["median_in_target_range"] else "NO"
    print(f"    In target:     {status}")
    print()


if __name__ == "__main__":
    main()
