#!/usr/bin/env python3
"""Split vocals on silence, tracking segment timestamps.

Unlike the original desilence_split.py which discards timing info,
this version records the start/end of every retained vocal segment
in a CSV co-located with the audio output directory.  The timestamps
are essential for later re-inserting converted vocals at the correct
positions in the instrumental track.
"""
from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from pydub import AudioSegment
from pydub.silence import detect_nonsilent

logger = logging.getLogger(__name__)


@dataclass
class VocalSegment:
    index: int
    start_ms: int
    end_ms: int
    duration_ms: int
    filename: str


def detect_vocal_segments(
    audio_path: Path,
    min_silence_len: int = 2000,
    silence_thresh: int = -40,
    keep_silence: int = 100,
    min_segment_len: int = 3000,
) -> Tuple[AudioSegment, List[VocalSegment]]:
    """Detect non-silent regions and return (audio, segment_list).

    Each segment carries its original position (start/end in ms) in the
    full track so we can reconstitute the timeline after RVC conversion.
    """
    audio = AudioSegment.from_file(str(audio_path))
    ranges = detect_nonsilent(
        audio,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh,
    )

    segments: List[VocalSegment] = []
    idx = 0
    for start_ms, end_ms in ranges:
        duration = end_ms - start_ms
        if duration < min_segment_len:
            continue
        idx += 1
        seg_start = max(0, start_ms - keep_silence)
        seg_end = min(len(audio), end_ms + keep_silence)
        segments.append(VocalSegment(
            index=idx,
            start_ms=seg_start,
            end_ms=seg_end,
            duration_ms=seg_end - seg_start,
            filename=f"{idx:05d}.wav",
        ))

    logger.info(
        "[%s] %d vocal segments >= %d ms",
        audio_path.name, len(segments), min_segment_len,
    )
    return audio, segments


def export_segments(
    audio: AudioSegment,
    segments: List[VocalSegment],
    output_dir: Path,
) -> List[Path]:
    """Export each segment to its own wav file under output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    for seg in segments:
        chunk = audio[seg.start_ms:seg.end_ms]
        out_path = output_dir / seg.filename
        chunk.export(str(out_path), format="wav")
        paths.append(out_path)
    return paths


def write_timestamps_csv(
    segments: List[VocalSegment],
    audio_path: Path,
    output_dir: Path,
) -> Path:
    """Write vocal segment timestamps to CSV.

    The CSV is named ``{stem}_vad_seg_ts.csv`` and contains start/end
    times in seconds, suitable for use as annotation data in tools like
    Sonic Visualiser or Audacity.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{audio_path.stem}_vad_seg_ts.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["start_time", "end_time"])
        for seg in segments:
            writer.writerow([
                round(seg.start_ms / 1000.0, 6),
                round(seg.end_ms / 1000.0, 6),
            ])
    logger.info("Wrote %d segment timestamps to %s", len(segments), csv_path)
    return csv_path


def desilence_and_track(
    audio_path: Path,
    output_dir: Path,
    min_silence_len: int = 2000,
    silence_thresh: int = -40,
    keep_silence: int = 100,
    min_segment_len: int = 3000,
) -> Tuple[List[Path], List[VocalSegment], Path]:
    """Full desilencing pipeline: detect, export segments, write CSV.

    The CSV is saved inside *output_dir* as
    ``{audio_stem}_vad_seg_ts.csv`` with ``start_time`` / ``end_time``
    columns in seconds.

    Returns:
        (segment_wav_paths, segments, csv_path)
    """
    audio, segments = detect_vocal_segments(
        audio_path,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh,
        keep_silence=keep_silence,
        min_segment_len=min_segment_len,
    )
    if not segments:
        logger.warning("No vocal segments found in %s", audio_path.name)
        csv_path = output_dir / f"{audio_path.stem}_vad_seg_ts.csv"
        return [], [], csv_path

    wav_paths = export_segments(audio, segments, output_dir)
    csv_path = write_timestamps_csv(segments, audio_path, output_dir)
    return wav_paths, segments, csv_path


def reassemble_from_segments(
    converted_dir: Path,
    segments: List[VocalSegment],
    total_duration_ms: int,
    sample_rate: int = 44100,
    channels: int = 2,
) -> AudioSegment:
    """Reassemble converted vocal segments back into a full-length track.

    Silence is inserted between segments at the correct positions so the
    result can be mixed 1:1 with the instrumental.
    """
    silence_frame = AudioSegment.silent(duration=total_duration_ms, frame_rate=sample_rate)
    if channels == 1:
        silence_frame = silence_frame.set_channels(1)

    for seg in segments:
        seg_path = converted_dir / seg.filename
        if not seg_path.exists():
            logger.warning("Missing converted segment: %s", seg_path)
            continue
        chunk = AudioSegment.from_file(str(seg_path))
        silence_frame = silence_frame.overlay(chunk, position=seg.start_ms)

    return silence_frame


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser(description="Desilence vocals with timestamp tracking")
    p.add_argument("input", type=Path, help="Input vocal wav")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--export-chunks", action="store_true",
                   help="Also export individual segment wav files")
    p.add_argument("--min-silence-len", type=int, default=2000)
    p.add_argument("--silence-thresh", type=int, default=-40)
    p.add_argument("--keep-silence", type=int, default=100)
    p.add_argument("--min-segment-len", type=int, default=3000)
    args = p.parse_args()

    out_dir = args.output_dir or args.input.parent
    audio, segs = detect_vocal_segments(
        args.input,
        min_silence_len=args.min_silence_len,
        silence_thresh=args.silence_thresh,
        keep_silence=args.keep_silence,
        min_segment_len=args.min_segment_len,
    )
    csv_p = write_timestamps_csv(segs, args.input, out_dir)
    print(f"Found {len(segs)} vocal segments, timestamps at {csv_p}")

    if args.export_chunks:
        paths = export_segments(audio, segs, out_dir)
        print(f"Exported {len(paths)} segment wav files to {out_dir}")
