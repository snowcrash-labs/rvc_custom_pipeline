#!/usr/bin/env python3
"""Dereverberation using UVR5 VR-DeEchoDeReverb, applied stochastically.

In the SVDD pipeline, dereverberation is applied 53% of the time so
the detection model sees a mixture of reverberant and dry vocals.  The
implementation mirrors the APCleaner from audiotools/ml/cleaning/rvc.py.
"""
from __future__ import annotations

import logging
import os
import random
import time
from pathlib import Path
from typing import Optional, Tuple

import librosa
import numpy as np
import soundfile as sf
from pedalboard import Pedalboard, NoiseGate

logger = logging.getLogger(__name__)

DEREVERB_PROBABILITY = 0.53

SCRIPTDIR = os.path.dirname(os.path.realpath(__file__))


class Dereverberation:
    """UVR5 DeEcho/DeReverb wrapper with noise-gate post-processing."""

    def __init__(self, model_path: str, params_path: str, device: str = "cuda", agg: int = 4):
        self.model_path = model_path
        self.params_path = params_path
        self.device = device
        self.agg = agg
        self.model = None
        self.board = Pedalboard([NoiseGate(threshold_db=-40)])

    def load(self):
        from audiotools.ml.cleaning.ext.rvc.infer.modules.uvr5.preprocess import AudioPreDeEcho
        self.model = AudioPreDeEcho(
            agg=self.agg,
            model_path=self.model_path,
            device=self.device,
            is_half=True,
            tta=False,
            params_path=self.params_path,
        )
        logger.info("Loaded DeReverb model from %s", self.model_path)

    def process(self, wav: np.ndarray, sr: int) -> Tuple[np.ndarray, int]:
        """Dereverberate audio.

        Args:
            wav: (samples,) or (samples, channels) array.
            sr: sample rate (will be resampled to 44100 internally).

        Returns:
            (dereverberated_wav, sr) with noise-gate applied.
        """
        if self.model is None:
            self.load()

        input_len = wav.shape[0] if wav.ndim == 1 else wav.shape[0]
        sr_orig = sr
        sr_trg = 44100

        if wav.ndim == 1:
            wav = wav[:, np.newaxis]
            wav = np.repeat(wav, 2, axis=1)

        if sr_orig != sr_trg:
            wav = librosa.resample(wav.T, orig_sr=sr_orig, target_sr=sr_trg, res_type="soxr_hq").T
            sr = sr_trg

        nb_retry = 3
        for attempt in range(nb_retry):
            try:
                wav_dereverb, _, sr = self.model._path_audio_(wav.T, sr)
                if np.isnan(wav_dereverb).any():
                    logger.warning("NaN values in dereverberated audio, retrying...")
                    raise ValueError("NaN in output")
                break
            except Exception as e:
                logger.warning("Dereverb attempt %d failed: %s", attempt + 1, e)
                if attempt == nb_retry - 1:
                    raise
                time.sleep(1.0)

        if sr_orig != sr_trg:
            wav_dereverb = librosa.resample(
                wav_dereverb.T, orig_sr=sr_trg, target_sr=sr_orig, res_type="soxr_hq"
            ).T
            sr = sr_orig

        if wav_dereverb.shape[0] != input_len:
            if wav_dereverb.shape[0] > input_len:
                wav_dereverb = wav_dereverb[:input_len]
            else:
                wav_dereverb = np.pad(
                    wav_dereverb,
                    ((0, input_len - wav_dereverb.shape[0]), (0, 0)),
                )

        wav_out = self.board(wav_dereverb, sample_rate=sr)
        return wav_out, sr


def maybe_dereverb(
    vocals_path: Path,
    output_path: Path,
    dereverb_model: Optional[Dereverberation] = None,
    probability: float = DEREVERB_PROBABILITY,
    force: Optional[bool] = None,
) -> Tuple[Path, bool]:
    """Conditionally apply dereverberation.

    Args:
        vocals_path: input vocal wav.
        output_path: where to write (possibly dereverberated) output.
        dereverb_model: pre-loaded Dereverberation instance (lazy-loaded if None
                        and dereverberation is triggered).
        probability: chance of applying dereverb (default 0.53).
        force: if True always apply, if False never apply, None = stochastic.

    Returns:
        (output_path, was_applied).
    """
    apply = force if force is not None else (random.random() < probability)

    if not apply:
        if vocals_path != output_path:
            import shutil
            shutil.copy2(vocals_path, output_path)
        logger.info("Dereverberation skipped (p=%.2f) for %s", probability, vocals_path.name)
        return output_path, False

    if dereverb_model is None:
        raise RuntimeError(
            "Dereverberation was selected but no model instance was provided. "
            "Pass a Dereverberation object or set force=False."
        )

    wav, sr = librosa.load(str(vocals_path), sr=None, mono=False)
    if wav.ndim == 2:
        wav = wav.T  # (channels, samples) -> (samples, channels)

    wav_out, sr_out = dereverb_model.process(wav, sr)

    if wav_out.ndim == 2 and wav_out.shape[1] == 1:
        wav_out = wav_out.squeeze(1)

    sf.write(str(output_path), wav_out, sr_out, subtype="PCM_16")
    logger.info("Dereverberation applied to %s -> %s", vocals_path.name, output_path.name)
    return output_path, True


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser(description="Apply dereverberation to vocals")
    p.add_argument("input", type=Path)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--model-path", required=True, help="Path to VR-DeEchoDeReverb.pth")
    p.add_argument(
        "--params-path", required=True,
        help="Path to 4band_v3.json model params",
    )
    p.add_argument("--force", action="store_true", help="Always apply (ignore probability)")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    out = args.output or args.input.parent / f"{args.input.stem}_dereverb.wav"
    model = Dereverberation(args.model_path, args.params_path, device=args.device)
    model.load()
    maybe_dereverb(out, out, dereverb_model=model, force=args.force or None)
