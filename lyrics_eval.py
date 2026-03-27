#!/usr/bin/env python3
"""Evaluate voice conversion quality by comparing lyrics transcriptions.

Uses OpenAI Whisper (``turbo`` model, matching the lyrics_service in
sc-hiro-backend) to transcribe both the source vocals and the converted
output, then computes Word Error Rate (WER) and Character Error Rate
(CER) using *jiwer*.

A low WER indicates the conversion preserved lyrical intelligibility.

Library usage::

    from lyrics_eval import evaluate_lyrics_similarity

    result = evaluate_lyrics_similarity(source_path, converted_path)
    print(result.wer, result.cer)

Standalone CLI::

    python lyrics_eval.py source.wav converted.wav
    python lyrics_eval.py source.wav converted.wav --json

Dependencies: ``openai-whisper``, ``jiwer``
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Whisper model management (lazy-loaded singleton)
# ---------------------------------------------------------------------------

_whisper_model = None
_whisper_model_name: Optional[str] = None


def _get_whisper_model(model_name: str = "turbo"):
    """Return a cached Whisper model, loading it on first call."""
    global _whisper_model, _whisper_model_name
    if _whisper_model is None or _whisper_model_name != model_name:
        import whisper

        logger.info("Loading Whisper model: %s", model_name)
        _whisper_model = whisper.load_model(model_name)
        _whisper_model_name = model_name
    return _whisper_model


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------


def transcribe(audio_path: Path, model_name: str = "turbo") -> dict:
    """Transcribe an audio file with Whisper.

    Returns a dict with ``text``, ``language``, and raw ``segments``.
    """
    model = _get_whisper_model(model_name)
    logger.info("Transcribing %s ...", audio_path.name)
    result = model.transcribe(str(audio_path))
    text = result["text"].strip()
    language = result.get("language", "")
    logger.info(
        "Transcription (%s, %s): %s",
        audio_path.name,
        language,
        text[:120] + ("..." if len(text) > 120 else ""),
    )
    return {
        "text": text,
        "language": language,
        "segments": result.get("segments", []),
    }


# ---------------------------------------------------------------------------
# Text normalisation (for fair WER/CER comparison)
# ---------------------------------------------------------------------------

_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_MULTI_SPACE_RE = re.compile(r"\s+")


def _normalize_text(text: str) -> str:
    """Lower-case, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = _PUNCT_RE.sub("", text)
    text = _MULTI_SPACE_RE.sub(" ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Evaluation result
# ---------------------------------------------------------------------------


@dataclass
class LyricsEvalResult:
    source_text: str
    converted_text: str
    source_language: str
    converted_language: str
    wer: float
    cer: float
    details: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------


def _save_eval_json(result: LyricsEvalResult, output_dir: Path) -> Path:
    """Write the full evaluation result to a JSON file."""
    import json

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "lyrics_eval.json"
    data = {
        "source_text": result.source_text,
        "converted_text": result.converted_text,
        "source_language": result.source_language,
        "converted_language": result.converted_language,
        "wer": result.wer,
        "cer": result.cer,
        **result.details,
    }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Lyrics eval JSON saved: %s", path)
    return path


def evaluate_lyrics_similarity(
    source_audio: Path,
    converted_audio: Path,
    *,
    model_name: str = "turbo",
    output_dir: Optional[Path] = None,
) -> LyricsEvalResult:
    """Transcribe *source_audio* and *converted_audio*, then compare via WER/CER.

    Args:
        source_audio: path to the original (separated) vocals.
        converted_audio: path to the voice-converted vocals.
        model_name: Whisper model size (default ``turbo``).
        output_dir: if provided, save evaluation results as
            ``lyrics_eval.json`` in this directory.

    Returns:
        :class:`LyricsEvalResult` with metrics.
    """
    import jiwer

    source = transcribe(source_audio, model_name)
    converted = transcribe(converted_audio, model_name)

    raw_source = source["text"]
    raw_converted = converted["text"]
    src_lang = source["language"]
    cvt_lang = converted["language"]

    if src_lang != cvt_lang:
        logger.warning(
            "Language mismatch: source=%s, converted=%s — WER may be inflated",
            src_lang,
            cvt_lang,
        )

    norm_source = _normalize_text(raw_source)
    norm_converted = _normalize_text(raw_converted)

    if not norm_source and not norm_converted:
        result = LyricsEvalResult(
            source_text=raw_source,
            converted_text=raw_converted,
            source_language=src_lang,
            converted_language=cvt_lang,
            wer=0.0,
            cer=0.0,
            details={"note": "both transcriptions empty"},
        )
        if output_dir is not None:
            _save_eval_json(result, output_dir)
        return result

    if not norm_source or not norm_converted:
        result = LyricsEvalResult(
            source_text=raw_source,
            converted_text=raw_converted,
            source_language=src_lang,
            converted_language=cvt_lang,
            wer=1.0,
            cer=1.0,
            details={"note": "one transcription empty"},
        )
        if output_dir is not None:
            _save_eval_json(result, output_dir)
        return result

    word_out = jiwer.process_words(norm_source, norm_converted)
    cer_val = jiwer.cer(norm_source, norm_converted)

    result = LyricsEvalResult(
        source_text=raw_source,
        converted_text=raw_converted,
        source_language=src_lang,
        converted_language=cvt_lang,
        wer=word_out.wer,
        cer=cer_val,
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

    if output_dir is not None:
        _save_eval_json(result, output_dir)

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    p = argparse.ArgumentParser(
        description="Compare lyrics transcriptions of source and converted vocals (WER/CER)",
    )
    p.add_argument("source", type=Path, help="Source vocals audio file (pre-conversion)")
    p.add_argument("converted", type=Path, help="Converted vocals audio file (post-conversion)")
    p.add_argument("--model", default="turbo",
                   help="Whisper model name (default: turbo)")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Directory to save lyrics_eval.json (default: same dir as source)")
    p.add_argument("--json", action="store_true",
                   help="Output results as JSON to stdout")
    args = p.parse_args()

    out_dir = args.output_dir or args.source.parent

    result = evaluate_lyrics_similarity(
        args.source, args.converted,
        model_name=args.model,
        output_dir=out_dir,
    )

    if args.json:
        output = {
            "source_file": str(args.source),
            "converted_file": str(args.converted),
            "source_text": result.source_text,
            "converted_text": result.converted_text,
            "source_language": result.source_language,
            "converted_language": result.converted_language,
            "wer": result.wer,
            "cer": result.cer,
            **result.details,
        }
        print(json.dumps(output, indent=2))
    else:
        print(f"\nWER={result.wer:.3f}  CER={result.cer:.3f}")
        print(f"  Source lang:    {result.source_language}")
        print(f"  Converted lang: {result.converted_language}")
        print(f"  Source:    {result.source_text[:150]}")
        print(f"  Converted: {result.converted_text[:150]}")
        if result.details.get("note"):
            print(f"  Note: {result.details['note']}")
        else:
            d = result.details
            print(f"  Hits={d['hits']}  Sub={d['substitutions']}  "
                  f"Del={d['deletions']}  Ins={d['insertions']}")
