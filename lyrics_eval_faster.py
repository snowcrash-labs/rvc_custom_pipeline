#!/usr/bin/env python3
"""Faster-whisper-backed WER/CER evaluation.

Mirrors :mod:`lyrics_eval` but uses *faster-whisper*'s
:class:`BatchedInferencePipeline` so long clips are chunked and the chunks
are batched through the encoder in a single GPU pass. Designed for long-lived
worker processes that load the model once and reuse it across many clips.

Usage::

    from lyrics_eval_faster import load_pipeline, evaluate_pair
    pipeline = load_pipeline(device_index=0)
    result = evaluate_pair(pipeline, vocals_path, acapella_path)
    print(result.wer, result.cer)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional

# Reuse normalisation + result dataclass from the reference module so the
# output shape of both backends is identical.
from lyrics_eval import LyricsEvalResult, _normalize_text

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "turbo"


# ---------------------------------------------------------------------------
# VAD-segment splicing (slice a clip to the union of speech segments and
# concatenate, so we can disable faster-whisper's own VAD and still avoid
# wasting inference on long silent/instrumental stretches)
# ---------------------------------------------------------------------------


def slice_and_concat(
    audio_path: Path,
    segments_ms: Iterable[tuple[int, int]],
    out_path: Path,
) -> Path:
    """Concatenate the given (start_ms, end_ms) slices of *audio_path* into
    *out_path* at the source sample rate. Segments are clipped to file bounds
    and silently skipped if empty. Returns *out_path*.
    """
    import numpy as np
    import soundfile as sf

    info = sf.info(str(audio_path))
    sr = info.samplerate
    total = info.frames

    chunks: list[np.ndarray] = []
    for start_ms, end_ms in segments_ms:
        s = max(0, int(start_ms * sr / 1000))
        e = min(total, int(end_ms * sr / 1000))
        if e <= s:
            continue
        data, _ = sf.read(str(audio_path), start=s, stop=e, always_2d=False)
        chunks.append(data)

    if not chunks:
        # Degenerate: no segments overlapped the file. Write a 100 ms silence
        # stub so downstream transcription returns an empty string rather than
        # failing.
        chunks = [np.zeros(int(0.1 * sr), dtype=np.float32)]

    out = np.concatenate(chunks, axis=0)
    sf.write(str(out_path), out, sr)
    return out_path


def load_pipeline(
    *,
    model_name: str = DEFAULT_MODEL,
    device: str = "cuda",
    device_index: int = 0,
    compute_type: str = "float16",
):
    """Load a faster-whisper model and wrap it in a BatchedInferencePipeline.

    Returns the pipeline. The underlying model is held by ``pipeline.model``.
    """
    from faster_whisper import WhisperModel, BatchedInferencePipeline

    logger.info(
        "Loading faster-whisper model=%s device=%s idx=%d compute=%s",
        model_name, device, device_index, compute_type,
    )
    model = WhisperModel(
        model_name,
        device=device,
        device_index=device_index,
        compute_type=compute_type,
    )
    return BatchedInferencePipeline(model=model)


def _transcribe(
    pipeline,
    audio_path: Path,
    *,
    batch_size: int = 8,
    vad_filter: bool = True,
    clip_timestamps: Optional[list[dict]] = None,
) -> dict:
    """Transcribe with the batched pipeline.

    If *clip_timestamps* (list of ``{"start": s, "end": s}`` in seconds) is
    provided, uses those as the speech regions to batch across and skips VAD;
    otherwise falls back to *vad_filter* for region selection.
    """
    kwargs = {"batch_size": batch_size}
    if clip_timestamps is not None:
        kwargs["vad_filter"] = False
        kwargs["clip_timestamps"] = clip_timestamps
    else:
        kwargs["vad_filter"] = vad_filter
    segments, info = pipeline.transcribe(str(audio_path), **kwargs)
    # BatchedInferencePipeline.transcribe returns a generator of segments —
    # realise it so we can concatenate the text.
    segs = list(segments)
    text = " ".join(s.text.strip() for s in segs).strip()
    return {
        "text": text,
        "language": info.language or "",
        "segments": [
            {"start": s.start, "end": s.end, "text": s.text} for s in segs
        ],
    }


def evaluate_pair(
    pipeline,
    source_audio: Path,
    converted_audio: Path,
    *,
    batch_size: int = 8,
    vad_filter: bool = True,
    clip_timestamps: Optional[list[dict]] = None,
) -> LyricsEvalResult:
    """Transcribe source + converted with a shared pipeline and compute WER/CER.

    If *clip_timestamps* is provided (list of ``{"start": s, "end": s}`` in
    seconds), both clips are transcribed over those exact regions with VAD
    disabled — giving an apples-to-apples comparison when the caller knows
    where speech is (e.g. from a shared ``vad_segments.csv``). Otherwise
    faster-whisper's own VAD is used per clip (regions can differ between
    source and converted).
    """
    import jiwer

    source = _transcribe(
        pipeline, source_audio,
        batch_size=batch_size, vad_filter=vad_filter, clip_timestamps=clip_timestamps,
    )
    converted = _transcribe(
        pipeline, converted_audio,
        batch_size=batch_size, vad_filter=vad_filter, clip_timestamps=clip_timestamps,
    )

    src_lang = source["language"]
    cvt_lang = converted["language"]
    if src_lang != cvt_lang:
        logger.warning(
            "Language mismatch: source=%s, converted=%s — WER may be inflated",
            src_lang, cvt_lang,
        )

    raw_source = source["text"]
    raw_converted = converted["text"]
    norm_source = _normalize_text(raw_source)
    norm_converted = _normalize_text(raw_converted)

    if not norm_source and not norm_converted:
        return LyricsEvalResult(
            source_text=raw_source, converted_text=raw_converted,
            source_language=src_lang, converted_language=cvt_lang,
            wer=0.0, cer=0.0,
            details={"note": "both transcriptions empty"},
        )
    if not norm_source or not norm_converted:
        return LyricsEvalResult(
            source_text=raw_source, converted_text=raw_converted,
            source_language=src_lang, converted_language=cvt_lang,
            wer=1.0, cer=1.0,
            details={"note": "one transcription empty"},
        )

    word_out = jiwer.process_words(norm_source, norm_converted)
    cer_val = jiwer.cer(norm_source, norm_converted)
    return LyricsEvalResult(
        source_text=raw_source, converted_text=raw_converted,
        source_language=src_lang, converted_language=cvt_lang,
        wer=word_out.wer, cer=cer_val,
        details={
            "substitutions": word_out.substitutions,
            "deletions": word_out.deletions,
            "insertions": word_out.insertions,
            "hits": word_out.hits,
            "mer": word_out.mer,
            "wil": word_out.wil,
            "wip": word_out.wip,
        },
    )
