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


# ── Pitch-stability range analysis ──────────────────────────────────────

def analyse_pitch_stability(
    audio_path: Path,
    sr: int = 16000,
    hop_length: int = 256,
    hnr_frame_length: int = 2048,
    fmin: float = 65.0,
    fmax: float = 1500.0,
    window_ms: int = 500,
    std_threshold_semitones: float = 3.0,
    voiced_ratio_threshold: float = 0.9,
    hnr_threshold_db: float = 6.0,
    gap_tolerance: int = 0,
    csv_path: Optional[Path] = None,
) -> dict:
    """Determine usable pitch range by analysing F0 stability and HNR.

    Extracts a frame-level F0 contour with pYIN and per-frame
    autocorrelation HNR, then aggregates both into *window_ms*-wide
    windows.  The longest contiguous run of stable windows determines
    the usable range.

    A window is **stable** when all three conditions hold:
      - at least *voiced_ratio_threshold* of its frames are voiced,
      - the standard deviation of voiced F0 values (in semitones) is
        below *std_threshold_semitones*, AND
      - the median autocorrelation HNR has not dropped by more than
        *hnr_threshold_db* from the trailing baseline (median of the
        previous 4 windows).  This relative criterion avoids penalising
        the low register where HNR is naturally lower.

    Up to *gap_tolerance* consecutive unstable windows are tolerated
    without breaking the run.

    Args:
        audio_path:       path to the RVC-converted sweep audio.
        hnr_frame_length: analysis frame size for HNR computation.
        hnr_threshold_db: maximum allowed HNR drop (dB) from the
                          trailing baseline before a window is flagged.
        csv_path:         if given, write per-frame F0 data here for
                          Sonic Visualiser.

    Returns:
        dict with ``usable_range`` plus ``per_window`` stats and the csv
        path if written.
    """
    y, _ = librosa.load(str(audio_path), sr=sr, mono=True)

    f0, voiced_flag, _ = librosa.pyin(
        y, fmin=fmin, fmax=fmax, sr=sr, hop_length=hop_length,
        fill_na=0.0,
    )
    f0_midi = np.where(f0 > 0, 12 * np.log2(f0 / 440.0 + 1e-10) + 69, 0.0)
    times = np.arange(len(f0)) * hop_length / sr

    # Per-frame HNR (autocorrelation) aligned with F0 frames
    hnr_frames = librosa.util.frame(
        y, frame_length=hnr_frame_length, hop_length=hop_length,
    ).T
    n_frames = min(len(hnr_frames), len(f0))
    per_frame_hnr = np.full(len(f0), -np.inf)
    for i in range(n_frames):
        if voiced_flag[i] and f0[i] > 0:
            per_frame_hnr[i] = hnr_autocorrelation(
                hnr_frames[i], sr, fmin, fmax,
            )

    # ── Write per-frame CSV for Sonic Visualiser ──────────────────────
    if csv_path is not None:
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w") as fh:
            fh.write("time,f0_hz,f0_midi,voiced,hnr_db\n")
            for i in range(len(f0)):
                hnr_val = per_frame_hnr[i] if i < len(per_frame_hnr) else -np.inf
                hnr_str = f"{hnr_val:.2f}" if np.isfinite(hnr_val) else ""
                fh.write(f"{times[i]:.6f},{f0[i]:.3f},{f0_midi[i]:.3f},"
                         f"{int(voiced_flag[i] and f0[i] > 0)},{hnr_str}\n")
        logger.info("Pitch CSV written: %s (%d frames)", csv_path, len(f0))

    # ── Per-window stability ──────────────────────────────────────────
    wf = max(1, int(window_ms / 1000.0 * sr / hop_length))
    n_windows = (len(f0) - wf) // wf + 1

    per_window = []
    for w in range(n_windows):
        start = w * wf
        end = start + wf
        chunk_f0 = f0_midi[start:end]
        chunk_v = voiced_flag[start:end] & (chunk_f0 > 0)

        t_start = times[start]
        t_end = times[min(end - 1, len(times) - 1)]
        n_voiced = int(np.sum(chunk_v))
        ratio = n_voiced / wf

        if n_voiced >= 3:
            voiced_vals = chunk_f0[chunk_v]
            std_semi = float(np.std(voiced_vals))
            mean_midi = float(np.mean(voiced_vals))
            mean_hz = float(440.0 * 2 ** ((mean_midi - 69) / 12.0))
            hnr_voiced = per_frame_hnr[start:end][chunk_v]
            finite_hnr = hnr_voiced[np.isfinite(hnr_voiced)]
            median_hnr = float(np.median(finite_hnr)) if len(finite_hnr) > 0 else -np.inf
        else:
            std_semi = float("inf")
            mean_midi = 0.0
            mean_hz = 0.0
            median_hnr = -np.inf

        per_window.append({
            "index": w,
            "t_start": round(t_start, 3),
            "t_end": round(t_end, 3),
            "voiced_ratio": round(ratio, 3),
            "std_semitones": round(std_semi, 3) if np.isfinite(std_semi) else None,
            "median_hnr_db": round(median_hnr, 1) if np.isfinite(median_hnr) else None,
            "mean_midi": round(mean_midi, 2),
            "mean_hz": round(mean_hz, 1),
            "note": hz_to_note(mean_hz) if mean_hz > 0 else "?",
            "stable": False,  # set below after HNR trend pass
        })

    # ── Stability: voiced + std + relative HNR drop ───────────────────
    # HNR is frequency-dependent (lower at low pitches), so an absolute
    # threshold would falsely reject the low register.  Instead, we
    # track a trailing HNR baseline and flag a window as noisy when HNR
    # drops by more than *hnr_threshold_db* dB from that baseline.
    _HNR_LOOKBACK = 4
    for w in range(n_windows):
        pw = per_window[w]
        ratio = pw["voiced_ratio"]
        std_semi = pw["std_semitones"] if pw["std_semitones"] is not None else float("inf")
        median_hnr = pw["median_hnr_db"] if pw["median_hnr_db"] is not None else -np.inf

        base_ok = ratio >= voiced_ratio_threshold and std_semi < std_threshold_semitones

        # Compute trailing HNR baseline from the preceding windows
        lookback_hnr = [
            per_window[j]["median_hnr_db"]
            for j in range(max(0, w - _HNR_LOOKBACK), w)
            if per_window[j]["median_hnr_db"] is not None
        ]
        if lookback_hnr:
            baseline_hnr = float(np.median(lookback_hnr))
            hnr_ok = median_hnr >= baseline_hnr - hnr_threshold_db
        else:
            hnr_ok = True

        pw["stable"] = base_ok and hnr_ok

    # ── Find contiguous stable range (with gap tolerance) ─────────────
    stable_mask = [w["stable"] for w in per_window]
    best_start, best_end = _find_stable_run(stable_mask, gap_tolerance)

    if best_start is not None:
        range_windows = per_window[best_start:best_end + 1]
        stable_windows = [w for w in range_windows if w["stable"]]
        all_midi = [w["mean_midi"] for w in stable_windows if w["mean_midi"] > 0]
        low_midi_val = min(all_midi) if all_midi else 0
        high_midi_val = max(all_midi) if all_midi else 0
        low_hz = 440.0 * 2 ** ((low_midi_val - 69) / 12.0) if low_midi_val else 0
        high_hz = 440.0 * 2 ** ((high_midi_val - 69) / 12.0) if high_midi_val else 0

        usable_range = {
            "low": hz_to_note(low_hz) if low_hz > 0 else None,
            "high": hz_to_note(high_hz) if high_hz > 0 else None,
            "low_hz": round(low_hz, 1) if low_hz > 0 else None,
            "high_hz": round(high_hz, 1) if high_hz > 0 else None,
            "low_time": per_window[best_start]["t_start"],
            "high_time": per_window[best_end]["t_end"],
            "semitones": round(high_midi_val - low_midi_val) if low_midi_val else 0,
        }
    else:
        usable_range = {
            "low": None, "high": None,
            "low_hz": None, "high_hz": None,
            "low_time": None, "high_time": None,
            "semitones": 0,
        }

    return {
        "method": "pitch_stability",
        "std_threshold_semitones": std_threshold_semitones,
        "voiced_ratio_threshold": voiced_ratio_threshold,
        "hnr_threshold_db": hnr_threshold_db,
        "window_ms": window_ms,
        "n_windows": n_windows,
        "n_stable": sum(stable_mask),
        "usable_range": usable_range,
        "per_window": per_window,
        "csv_path": str(csv_path) if csv_path else None,
    }


def _find_stable_run(
    stable_mask: list[bool],
    gap_tolerance: int = 1,
) -> tuple[int | None, int | None]:
    """Find the longest contiguous run of True values, allowing brief gaps.

    *gap_tolerance* consecutive False values are tolerated without
    breaking the run.  Returns (start_index, end_index) inclusive, or
    (None, None) if no stable window exists.
    """
    best_start = best_end = None
    best_len = 0
    run_start = None
    gap_count = 0

    for i, s in enumerate(stable_mask):
        if s:
            if run_start is None:
                run_start = i
            gap_count = 0
        else:
            if run_start is not None:
                gap_count += 1
                if gap_count > gap_tolerance:
                    run_end = i - gap_count
                    run_len = run_end - run_start + 1
                    if run_len > best_len:
                        best_start, best_end, best_len = run_start, run_end, run_len
                    run_start = None
                    gap_count = 0

    if run_start is not None:
        run_end = len(stable_mask) - 1
        while run_end >= run_start and not stable_mask[run_end]:
            run_end -= 1
        run_len = run_end - run_start + 1
        if run_len > best_len:
            best_start, best_end = run_start, run_end

    return best_start, best_end


# ── CLI ─────────────────────────────────────────────────────────────────

def _print_results(results: dict) -> None:
    method = results["method"]

    if method == "pitch_stability":
        _print_pitch_stability_results(results)
        return

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


def _print_pitch_stability_results(results: dict) -> None:
    usable = results["usable_range"]
    per_w = results["per_window"]

    print(f"\n{'=' * 100}")
    print(f"  Pitch Stability Analysis  —  window: {results['window_ms']}ms"
          f"  std<{results['std_threshold_semitones']} semi"
          f"  voiced>={results['voiced_ratio_threshold']:.0%}"
          f"  HNR drop<{results['hnr_threshold_db']} dB")
    print(f"{'=' * 100}\n")

    print(f"  {'Time':>10}  {'Note':<6} {'Hz':>7}  {'Voiced':>7}"
          f"  {'Std(semi)':>10}  {'HNR(dB)':>8}  {'Status':>8}")
    print(f"  {'─' * 9}   {'─' * 5}  {'─' * 7}  {'─' * 6}"
          f"   {'─' * 9}   {'─' * 7}   {'─' * 7}")

    for w in per_w:
        std_str = f"{w['std_semitones']:.3f}" if w["std_semitones"] is not None else "   –"
        hnr_str = f"{w['median_hnr_db']:.1f}" if w.get("median_hnr_db") is not None else "  –"
        marker = "stable" if w["stable"] else "NOISY"
        print(f"  {_fmt_time(w['t_start']):>5}–{_fmt_time(w['t_end']):<5}"
              f" {w['note']:<6} {w['mean_hz']:>7.1f}  {w['voiced_ratio']:>5.0%}"
              f"   {std_str:>9}   {hnr_str:>7}   {marker:>7}")

    print()
    print(f"  Windows: {results['n_windows']} total, {results['n_stable']} stable")
    if usable["low"]:
        span = usable["semitones"]
        octaves = span / 12
        print(f"  Usable range: {usable['low']} ({usable['low_hz']} Hz) — "
              f"{usable['high']} ({usable['high_hz']} Hz)")
        print(f"  Span:         {span} semitones ({octaves:.1f} octaves)")
        print(f"  Timestamps:   {_fmt_time(usable['low_time'])} — "
              f"{_fmt_time(usable['high_time'])}")
    else:
        print("  No usable range detected.")
    if results.get("csv_path"):
        print(f"  CSV:          {results['csv_path']}")
    print()


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    p = argparse.ArgumentParser(
        description="Analyse the usable pitch range of an RVC model from a "
                    "vocal scale recording.",
    )
    p.add_argument("input", type=Path, help="RVC-converted audio file (vocal scale)")
    p.add_argument(
        "--method",
        choices=["autocorrelation", "cepstral", "spectral", "pitch_stability", "all"],
        default="pitch_stability",
        help="Analysis method (default: pitch_stability)",
    )
    p.add_argument("--sr", type=int, default=16000, help="Analysis sample rate")
    p.add_argument("--frame-length", type=int, default=2048)
    p.add_argument("--hop-length", type=int, default=512)
    p.add_argument("--fmin", type=float, default=65.0, help="Minimum expected pitch (Hz)")
    p.add_argument("--fmax", type=float, default=1500.0, help="Maximum expected pitch (Hz)")
    p.add_argument("--threshold", type=float, default=5.0,
                   help="HNR threshold (dB) for HNR methods")
    p.add_argument("--std-threshold", type=float, default=3.0,
                   help="Pitch std threshold (semitones) for pitch_stability")
    p.add_argument("--voiced-threshold", type=float, default=0.9,
                   help="Min voiced frame ratio per window for pitch_stability")
    p.add_argument("--hnr-threshold", type=float, default=6.0,
                   help="Max HNR drop (dB) from trailing baseline for pitch_stability")
    p.add_argument("--window-ms", type=int, default=500,
                   help="Window size in ms for pitch_stability")
    p.add_argument("--csv", type=Path, default=None,
                   help="Write per-frame F0 CSV for Sonic Visualiser")
    p.add_argument("--json", action="store_true", help="Output as JSON instead of table")
    args = p.parse_args()

    if args.method == "pitch_stability":
        csv_out = args.csv
        if csv_out is None:
            csv_out = args.input.with_suffix(".f0.csv")
        results = analyse_pitch_stability(
            args.input,
            sr=args.sr,
            hop_length=args.hop_length,
            fmin=args.fmin,
            fmax=args.fmax,
            window_ms=args.window_ms,
            std_threshold_semitones=args.std_threshold,
            voiced_ratio_threshold=args.voiced_threshold,
            hnr_threshold_db=args.hnr_threshold,
            csv_path=csv_out,
        )
        if args.json:
            import json
            print(json.dumps(results, indent=2, default=str))
        else:
            _print_results(results)
        return

    methods = (["autocorrelation", "cepstral", "spectral"]
               if args.method == "all" else [args.method])

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
