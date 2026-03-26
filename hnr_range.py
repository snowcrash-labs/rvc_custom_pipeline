#!/usr/bin/env python3
"""Determine the usable pitch range of an RVC model by analysing the
Harmonic-to-Noise Ratio across a vocal scale recording.

Feed this script an RVC-converted audio file that originated from a
singer performing a chromatic or stepwise scale spanning roughly three
octaves.  The script estimates pitch and HNR per frame, aggregates by
musical note, and reports the contiguous note range where the model
produces clean output.

Three HNR estimation methods are provided:

  1. **autocorrelation** — classical Praat-style HNR from the
     normalised autocorrelation peak in the expected pitch-period range.
  2. **cepstral** — cepstral peak prominence: the ratio of the cepstral
     peak (at the pitch period) to the local cepstral floor.
  3. **spectral** — ratio of harmonic partial energy to total spectral
     energy, estimated by summing narrow bands around detected harmonics.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import librosa
import numpy as np

logger = logging.getLogger(__name__)

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def hz_to_note(freq_hz: float) -> str:
    if freq_hz <= 0:
        return "?"
    midi = 12 * np.log2(freq_hz / 440.0) + 69
    midi_round = int(round(midi))
    octave = (midi_round // 12) - 1
    note = NOTE_NAMES[midi_round % 12]
    return f"{note}{octave}"


def midi_from_hz(freq_hz: float) -> int:
    if freq_hz <= 0:
        return -1
    return int(round(12 * np.log2(freq_hz / 440.0) + 69))


def hz_from_midi(midi: int) -> float:
    return 440.0 * 2 ** ((midi - 69) / 12.0)


def note_from_midi(midi: int) -> str:
    octave = (midi // 12) - 1
    return f"{NOTE_NAMES[midi % 12]}{octave}"


# ── HNR method 1: autocorrelation ──────────────────────────────────────

def hnr_autocorrelation(
    frame: np.ndarray,
    sr: int,
    fmin: float = 65.0,
    fmax: float = 1500.0,
) -> float:
    min_lag = max(1, int(sr / fmax))
    max_lag = min(len(frame) - 1, int(sr / fmin))
    if min_lag >= max_lag:
        return -np.inf

    frame = frame - frame.mean()
    energy = np.dot(frame, frame)
    if energy < 1e-10:
        return -np.inf

    ac = np.correlate(frame, frame, mode="full")
    ac = ac[len(frame) - 1:]
    ac_norm = ac / (ac[0] + 1e-10)

    r_max = np.max(ac_norm[min_lag:max_lag + 1])
    r_max = np.clip(r_max, 1e-6, 1.0 - 1e-6)
    return 10.0 * np.log10(r_max / (1.0 - r_max))


# ── HNR method 2: cepstral peak prominence ─────────────────────────────

def hnr_cepstral(
    frame: np.ndarray,
    sr: int,
    fmin: float = 65.0,
    fmax: float = 1500.0,
) -> float:
    min_quefrency = max(1, int(sr / fmax))
    max_quefrency = min(len(frame) // 2 - 1, int(sr / fmin))
    if min_quefrency >= max_quefrency:
        return -np.inf

    windowed = frame * np.hanning(len(frame))
    spectrum = np.fft.rfft(windowed)
    log_spectrum = np.log(np.abs(spectrum) + 1e-10)
    cepstrum = np.fft.irfft(log_spectrum)

    search_region = cepstrum[min_quefrency:max_quefrency + 1]
    if len(search_region) == 0:
        return -np.inf

    peak_val = np.max(search_region)
    floor_val = np.median(np.abs(search_region))
    if floor_val < 1e-10:
        floor_val = 1e-10

    return 20.0 * np.log10(np.abs(peak_val) / floor_val + 1e-10)


# ── HNR method 3: spectral harmonic energy ratio ───────────────────────

def hnr_spectral(
    frame: np.ndarray,
    sr: int,
    f0: float,
    n_harmonics: int = 10,
    bandwidth_hz: float = 40.0,
) -> float:
    if f0 <= 0:
        return -np.inf

    windowed = frame * np.hanning(len(frame))
    spectrum = np.abs(np.fft.rfft(windowed)) ** 2
    freqs = np.fft.rfftfreq(len(frame), 1.0 / sr)

    total_energy = np.sum(spectrum)
    if total_energy < 1e-10:
        return -np.inf

    harmonic_energy = 0.0
    for h in range(1, n_harmonics + 1):
        center = f0 * h
        if center > sr / 2:
            break
        mask = np.abs(freqs - center) <= bandwidth_hz / 2
        harmonic_energy += np.sum(spectrum[mask])

    noise_energy = total_energy - harmonic_energy
    if noise_energy < 1e-10:
        noise_energy = 1e-10

    return 10.0 * np.log10(harmonic_energy / noise_energy)


# ── Pitch tracking ─────────────────────────────────────────────────────

def track_pitch(y: np.ndarray, sr: int, hop_length: int, fmin: float, fmax: float):
    """Track pitch using pyin. Returns (f0, voiced_flag) arrays."""
    f0, voiced_flag, _ = librosa.pyin(
        y, fmin=fmin, fmax=fmax, sr=sr, hop_length=hop_length,
        fill_na=0.0,
    )
    return f0, voiced_flag


# ── Main analysis ──────────────────────────────────────────────────────

def _frame_to_time(frame_idx: int, hop_length: int, sr: int) -> float:
    return round(frame_idx * hop_length / sr, 3)


def _fmt_time(seconds: float) -> str:
    m, s = divmod(seconds, 60)
    return f"{int(m)}:{s:05.2f}"


def analyse_hnr_by_note(
    audio_path: Path,
    method: str = "autocorrelation",
    sr: int = 16000,
    frame_length: int = 2048,
    hop_length: int = 512,
    fmin: float = 65.0,
    fmax: float = 1500.0,
    hnr_threshold_db: float = 5.0,
) -> dict:
    y, orig_sr = librosa.load(str(audio_path), sr=sr, mono=True)

    f0, voiced = track_pitch(y, sr, hop_length, fmin, fmax)

    frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length).T
    n_frames = min(len(frames), len(f0))

    note_hnr: dict[int, list[float]] = {}
    note_frames: dict[int, list[int]] = {}

    for i in range(n_frames):
        if not voiced[i] or f0[i] <= 0:
            continue

        frame = frames[i]
        midi = midi_from_hz(f0[i])
        if midi < 0:
            continue

        if method == "autocorrelation":
            hnr = hnr_autocorrelation(frame, sr, fmin, fmax)
        elif method == "cepstral":
            hnr = hnr_cepstral(frame, sr, fmin, fmax)
        elif method == "spectral":
            hnr = hnr_spectral(frame, sr, f0[i])
        else:
            raise ValueError(f"Unknown method: {method}")

        if np.isfinite(hnr):
            note_hnr.setdefault(midi, []).append(hnr)
            note_frames.setdefault(midi, []).append(i)

    if not note_hnr:
        return {"error": "No voiced frames detected"}

    note_stats = {}
    for midi in sorted(note_hnr):
        vals = note_hnr[midi]
        fi = note_frames[midi]
        t_first = _frame_to_time(fi[0], hop_length, sr)
        t_last = _frame_to_time(fi[-1], hop_length, sr)
        note_stats[midi] = {
            "note": note_from_midi(midi),
            "hz": round(hz_from_midi(midi), 1),
            "mean_hnr": round(float(np.mean(vals)), 1),
            "median_hnr": round(float(np.median(vals)), 1),
            "min_hnr": round(float(np.min(vals)), 1),
            "n_frames": len(vals),
            "first_time": t_first,
            "last_time": t_last,
        }

    all_midis = sorted(note_stats.keys())

    usable = [m for m in all_midis if note_stats[m]["median_hnr"] >= hnr_threshold_db]

    if usable:
        longest_run = _longest_contiguous_run(usable)
        low_midi, high_midi = longest_run[0], longest_run[-1]
    else:
        low_midi, high_midi = None, None

    usable_range = {
        "low": note_from_midi(low_midi) if low_midi else None,
        "high": note_from_midi(high_midi) if high_midi else None,
        "low_hz": round(hz_from_midi(low_midi), 1) if low_midi else None,
        "high_hz": round(hz_from_midi(high_midi), 1) if high_midi else None,
        "semitones": (high_midi - low_midi + 1) if low_midi else 0,
        "low_time": note_stats[low_midi]["first_time"] if low_midi else None,
        "high_time": note_stats[high_midi]["last_time"] if high_midi else None,
    }

    return {
        "method": method,
        "hnr_threshold_db": hnr_threshold_db,
        "per_note": note_stats,
        "usable_range": usable_range,
    }


def _longest_contiguous_run(sorted_midis: list[int]) -> list[int]:
    """Find the longest contiguous (step-1) subsequence in sorted midi notes."""
    best_run = [sorted_midis[0]]
    current_run = [sorted_midis[0]]

    for i in range(1, len(sorted_midis)):
        if sorted_midis[i] - sorted_midis[i - 1] <= 2:
            current_run.append(sorted_midis[i])
        else:
            if len(current_run) > len(best_run):
                best_run = current_run
            current_run = [sorted_midis[i]]

    if len(current_run) > len(best_run):
        best_run = current_run

    return best_run


# ── CLI ─────────────────────────────────────────────────────────────────

def _print_results(results: dict) -> None:
    method = results["method"]
    threshold = results["hnr_threshold_db"]
    per_note = results["per_note"]
    usable = results["usable_range"]

    print(f"\n{'=' * 78}")
    print(f"  HNR Range Analysis  —  method: {method}  threshold: {threshold} dB")
    print(f"{'=' * 78}\n")

    print(f"  {'Note':<6} {'Hz':>8} {'Median HNR':>12} {'Mean HNR':>10} {'Min HNR':>9}"
          f" {'Frames':>8} {'Time span':>16}  ")
    print(f"  {'─' * 5}  {'─' * 7}  {'─' * 11}  {'─' * 9}  {'─' * 8}"
          f"  {'─' * 7}  {'─' * 15}")

    for midi in sorted(per_note):
        s = per_note[midi]
        marker = ""
        if usable["low"] and usable["high"]:
            low_midi = midi_from_hz(usable["low_hz"])
            high_midi = midi_from_hz(usable["high_hz"])
            if low_midi <= midi <= high_midi:
                marker = "  ✓"
            elif s["median_hnr"] < threshold:
                marker = "  ✗"
        time_span = f"{_fmt_time(s['first_time'])}–{_fmt_time(s['last_time'])}"
        print(f"  {s['note']:<6} {s['hz']:>7.1f}  {s['median_hnr']:>10.1f}  "
              f"{s['mean_hnr']:>9.1f}  {s['min_hnr']:>8.1f}  {s['n_frames']:>6}"
              f"  {time_span:>15}{marker}")

    print()
    if usable["low"]:
        span = usable["semitones"]
        octaves = span / 12
        print(f"  Usable range: {usable['low']} ({usable['low_hz']} Hz) — "
              f"{usable['high']} ({usable['high_hz']} Hz)")
        print(f"  Span:         {span} semitones ({octaves:.1f} octaves)")
        print(f"  Timestamps:   {_fmt_time(usable['low_time'])} — "
              f"{_fmt_time(usable['high_time'])}")
    else:
        print("  No usable range detected above threshold.")
    print()


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    p = argparse.ArgumentParser(
        description="Analyse the usable pitch range of an RVC model by measuring "
                    "HNR across a vocal scale recording.",
    )
    p.add_argument("input", type=Path, help="RVC-converted audio file (vocal scale)")
    p.add_argument("--method", choices=["autocorrelation", "cepstral", "spectral", "all"],
                   default="all", help="HNR estimation method (default: all)")
    p.add_argument("--sr", type=int, default=16000, help="Analysis sample rate")
    p.add_argument("--frame-length", type=int, default=2048)
    p.add_argument("--hop-length", type=int, default=512)
    p.add_argument("--fmin", type=float, default=65.0, help="Minimum expected pitch (Hz)")
    p.add_argument("--fmax", type=float, default=1500.0, help="Maximum expected pitch (Hz)")
    p.add_argument("--threshold", type=float, default=5.0,
                   help="HNR threshold (dB) below which a note is considered unusable")
    p.add_argument("--json", action="store_true", help="Output as JSON instead of table")
    args = p.parse_args()

    methods = ["autocorrelation", "cepstral", "spectral"] if args.method == "all" else [args.method]

    all_results = {}
    for method in methods:
        results = analyse_hnr_by_note(
            args.input,
            method=method,
            sr=args.sr,
            frame_length=args.frame_length,
            hop_length=args.hop_length,
            fmin=args.fmin,
            fmax=args.fmax,
            hnr_threshold_db=args.threshold,
        )
        all_results[method] = results

        if not args.json:
            _print_results(results)

    if args.json:
        import json

        serialisable = {}
        for method, res in all_results.items():
            r = dict(res)
            r["per_note"] = {str(k): v for k, v in r["per_note"].items()}
            serialisable[method] = r
        print(json.dumps(serialisable, indent=2))


if __name__ == "__main__":
    main()
