#!/usr/bin/env python3
"""Evaluate noise level in vocal output using HNR and CREPE confidence.

Two complementary approaches are provided:

1. **Harmonics-to-Noise Ratio (HNR)** — autocorrelation-based estimate of
   periodic vs aperiodic energy.  Higher HNR = cleaner signal.

2. **CREPE pitch confidence** — the neural pitch tracker outputs a confidence
   in [0, 1] for each frame.  Low mean confidence across voiced frames is a
   proxy for high noise content (the model is uncertain because noise masks
   the harmonic structure).

Either metric can be thresholded to flag noisy outputs for exclusion from
the SVDD training set.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import librosa
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class NoiseEvalResult:
    hnr_db: float
    crepe_mean_confidence: Optional[float]
    crepe_voiced_confidence: Optional[float]
    is_clean: bool
    details: dict


# ---------------------------------------------------------------------------
# HNR (autocorrelation method, similar to Praat)
# ---------------------------------------------------------------------------

def compute_hnr(
    y: np.ndarray,
    sr: int,
    frame_length: int = 2048,
    hop_length: int = 512,
    fmin: float = 75.0,
    fmax: float = 600.0,
) -> float:
    """Compute mean Harmonics-to-Noise Ratio in dB over all frames.

    Uses the autocorrelation method: for each frame the peak of the
    normalised autocorrelation in the plausible pitch-period range
    gives r_max.  HNR = 10 * log10(r_max / (1 - r_max)).
    """
    min_lag = int(sr / fmax)
    max_lag = int(sr / fmin)

    frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length).T
    hnr_values = []

    for frame in frames:
        frame = frame - frame.mean()
        energy = np.dot(frame, frame)
        if energy < 1e-10:
            continue

        autocorr = np.correlate(frame, frame, mode="full")
        autocorr = autocorr[len(frame) - 1:]  # keep non-negative lags
        autocorr_norm = autocorr / (autocorr[0] + 1e-10)

        search_start = min(min_lag, len(autocorr_norm) - 1)
        search_end = min(max_lag + 1, len(autocorr_norm))
        if search_start >= search_end:
            continue

        r_max = np.max(autocorr_norm[search_start:search_end])
        r_max = np.clip(r_max, 1e-6, 1.0 - 1e-6)
        hnr_frame = 10.0 * np.log10(r_max / (1.0 - r_max))
        hnr_values.append(hnr_frame)

    if not hnr_values:
        return -np.inf
    return float(np.mean(hnr_values))


# ---------------------------------------------------------------------------
# CREPE confidence
# ---------------------------------------------------------------------------

def compute_crepe_confidence(
    y: np.ndarray,
    sr: int,
    model_capacity: str = "tiny",
    voiced_threshold: float = 0.5,
    step_size: int = 10,
) -> tuple[float, float]:
    """Compute mean CREPE confidence over all frames and voiced-only frames.

    Args:
        y: mono audio signal.
        sr: sample rate.
        model_capacity: CREPE model size (tiny/small/medium/large/full).
        voiced_threshold: confidence threshold to consider a frame voiced.
        step_size: CREPE step size in ms.

    Returns:
        (mean_confidence_all, mean_confidence_voiced)
    """
    import crepe

    _, _, confidence, _ = crepe.predict(
        y, sr,
        model_capacity=model_capacity,
        step_size=step_size,
        viterbi=False,
        verbose=0,
    )

    mean_all = float(np.mean(confidence))
    voiced_mask = confidence >= voiced_threshold
    mean_voiced = float(np.mean(confidence[voiced_mask])) if voiced_mask.any() else 0.0

    return mean_all, mean_voiced


# ---------------------------------------------------------------------------
# Combined evaluation
# ---------------------------------------------------------------------------

def evaluate_noise(
    audio_path: Path,
    hnr_threshold_db: float = 10.0,
    crepe_threshold: float = 0.6,
    use_crepe: bool = True,
    crepe_model_capacity: str = "tiny",
    sr: int = 16000,
) -> NoiseEvalResult:
    """Run noise evaluation on an audio file.

    The file is considered clean if *both* thresholds are met (when CREPE
    is enabled) or if only the HNR threshold is met (when CREPE is
    disabled).

    Args:
        audio_path: path to wav file.
        hnr_threshold_db: minimum HNR in dB to consider clean.
        crepe_threshold: minimum mean CREPE confidence (voiced frames)
                         to consider clean.
        use_crepe: whether to run CREPE analysis.
        crepe_model_capacity: CREPE model size.
        sr: sample rate for analysis (audio is resampled).

    Returns:
        NoiseEvalResult with metrics and clean/noisy verdict.
    """
    y, _ = librosa.load(str(audio_path), sr=sr, mono=True)

    hnr = compute_hnr(y, sr)
    hnr_clean = hnr >= hnr_threshold_db

    crepe_all = None
    crepe_voiced = None
    crepe_clean = True

    if use_crepe:
        crepe_all, crepe_voiced = compute_crepe_confidence(
            y, sr, model_capacity=crepe_model_capacity,
        )
        crepe_clean = crepe_voiced >= crepe_threshold

    is_clean = hnr_clean and crepe_clean

    details = {
        "hnr_db": hnr,
        "hnr_threshold_db": hnr_threshold_db,
        "hnr_pass": hnr_clean,
    }
    if use_crepe:
        details.update({
            "crepe_mean_all": crepe_all,
            "crepe_mean_voiced": crepe_voiced,
            "crepe_threshold": crepe_threshold,
            "crepe_pass": crepe_clean,
        })

    result = NoiseEvalResult(
        hnr_db=hnr,
        crepe_mean_confidence=crepe_all,
        crepe_voiced_confidence=crepe_voiced,
        is_clean=is_clean,
        details=details,
    )
    logger.info(
        "[%s] HNR=%.1f dB  CREPE_voiced=%.3f  clean=%s",
        audio_path.name,
        hnr,
        crepe_voiced if crepe_voiced is not None else -1,
        is_clean,
    )
    return result


def evaluate_batch(
    audio_paths: list[Path],
    **kwargs,
) -> list[NoiseEvalResult]:
    """Evaluate multiple files, returning results in the same order."""
    return [evaluate_noise(p, **kwargs) for p in audio_paths]


if __name__ == "__main__":
    import argparse
    import json

    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser(description="Evaluate vocal noise level (HNR + CREPE)")
    p.add_argument("inputs", nargs="+", type=Path, help="Audio file(s) to evaluate")
    p.add_argument("--hnr-threshold", type=float, default=10.0, help="Min HNR (dB) for clean")
    p.add_argument("--crepe-threshold", type=float, default=0.6, help="Min CREPE voiced confidence for clean")
    p.add_argument("--no-crepe", action="store_true", help="Skip CREPE analysis (HNR only)")
    p.add_argument("--crepe-model", default="tiny", choices=["tiny", "small", "medium", "large", "full"])
    p.add_argument("--sr", type=int, default=16000)
    p.add_argument("--json", action="store_true", help="Output results as JSON")
    args = p.parse_args()

    all_results = []
    for path in args.inputs:
        result = evaluate_noise(
            path,
            hnr_threshold_db=args.hnr_threshold,
            crepe_threshold=args.crepe_threshold,
            use_crepe=not args.no_crepe,
            crepe_model_capacity=args.crepe_model,
            sr=args.sr,
        )
        all_results.append({"file": str(path), **result.details, "is_clean": result.is_clean})

    if args.json:
        print(json.dumps(all_results, indent=2, default=str))
    else:
        for r in all_results:
            status = "CLEAN" if r["is_clean"] else "NOISY"
            print(f"[{status}] {r['file']}  HNR={r['hnr_db']:.1f} dB", end="")
            if "crepe_mean_voiced" in r:
                print(f"  CREPE_voiced={r['crepe_mean_voiced']:.3f}", end="")
            print()
