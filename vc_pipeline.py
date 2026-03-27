#!/usr/bin/env python3
"""End-to-end SVDD dataset generation pipeline.

For each input audio track this pipeline:

  1. **Separates** vocals and instrumentals using a randomly chosen backend
     (Demucs or Roformer) to introduce diverse separation artefacts.
  2. **Dereverberation** is applied to the separated vocals with 53%
     probability (configurable) so the SVDD model sees both reverberant and
     dry examples.
  3. **Desilencing** splits the vocals on silence, recording segment
     timestamps in a CSV file co-located with the audio.
  4. **RVC conversion** runs each vocal segment through the target voice
     checkpoint.
  5. **Reassembly** places converted segments at their original timestamps
     and **mixes** with the instrumental track.

All outputs for a given track are placed in a single directory:
    {output_root}/source_{audio_md5}_rvcmodel_{rvc_md5}/
containing the final wav, timestamps CSV, and per-track metadata JSON.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional

from pydub import AudioSegment

from separation import SeparationBackend, separate
from dereverberation import (
    DEREVERB_PROBABILITY,
    Dereverberation,
    DereverbMelBandRoformer,
    maybe_dereverb,
)

_DEREVERB_CLASSES = {"vrnet": Dereverberation, "mbr": DereverbMelBandRoformer}
from desilence import desilence_and_track, reassemble_from_segments, VocalSegment
from rvc_convert import (
    convert_segments,
    find_checkpoint_files,
    md5_of_file,
    md5_of_string,
)
try:
    import boto3
    _s3 = boto3.client("s3")
except ImportError:
    _s3 = None

logger = logging.getLogger(__name__)


def _md5_of_audio(audio_path: Path) -> str:
    return md5_of_file(audio_path)


def _output_stem(source_md5: str, rvc_md5: str) -> str:
    return f"source_{source_md5}_rvcmodel_{rvc_md5}"


def process_track(
    input_audio: Path,
    checkpoint_dir: Optional[Path],
    output_root: Path,
    *,
    rvc_md5: Optional[str] = None,
    sep_backend: Optional[SeparationBackend] = None,
    dereverb_model: Optional[Dereverberation] = None,
    dereverb_probability: float = DEREVERB_PROBABILITY,
    f0_up_key: int = 0,
    f0_method: str = "rmvpe",
    device: Optional[str] = None,
    min_silence_len: int = 2000,
    silence_thresh: int = -40,
    keep_silence: int = 100,
    min_segment_len: int = 3000,
    do_separation: bool = True,
    do_desilence: bool = True,
    do_rvc: bool = True,
    do_reassembly: bool = True,
    force: bool = False,
) -> Optional[dict]:
    """Run the full pipeline on a single audio track.

    Individual stages can be disabled via the ``do_*`` flags.  When a
    stage is disabled, the audio passes through unchanged:

    - ``do_separation=False``: input audio is used directly (no
      vocal/instrumental split).
    - ``do_desilence=False``: the whole track is treated as one segment.
    - ``do_rvc=False``: segments are kept as-is (no voice conversion).
      ``checkpoint_dir`` may be ``None`` when RVC is disabled.
    - ``do_reassembly=False``: output is the processed audio without
      mixing back with an instrumental stem.

    Returns a metadata dict on success, None on failure.
    """
    source_md5 = _md5_of_audio(input_audio)

    pth_candidates, best_index = None, None
    if do_rvc:
        if checkpoint_dir is None:
            logger.error("--checkpoint-dir is required when RVC is enabled")
            return None
        pth_candidates, best_index = find_checkpoint_files(checkpoint_dir)
        if not pth_candidates:
            logger.error("No .pth checkpoint found in %s", checkpoint_dir)
            return None
        if rvc_md5 is None:
            rvc_md5 = md5_of_file(pth_candidates[0])

    if rvc_md5 is None:
        rvc_md5 = "norvc"

    stem = _output_stem(source_md5, rvc_md5)
    output_dir = output_root / stem
    final_wav = output_dir / f"{stem}.wav"

    if final_wav.exists() and not force:
        logger.info("Already processed, skipping: %s (use --force to regenerate)", final_wav.name)
        return {"output": str(final_wav), "source_md5": source_md5, "rvc_md5": rvc_md5, "skipped": True}
    if final_wav.exists() and force:
        logger.info("Force mode: removing existing output %s", final_wav.name)
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    backend_used = None
    dereverb_applied = False
    num_segments = 0
    num_converted = 0

    with tempfile.TemporaryDirectory(prefix="svdd_") as tmpdir:
        tmp = Path(tmpdir)

        # --- 1. Separation ---
        vocals_raw = tmp / "vocals_raw.wav"
        instrumental = tmp / "instrumental.wav"
        if do_separation:
            try:
                _, _, backend_used = separate(
                    input_audio, vocals_raw, instrumental,
                    backend=sep_backend, device=device,
                )
            except Exception as e:
                logger.error("Separation failed for %s: %s", input_audio.name, e)
                return None
        else:
            logger.info("Separation skipped — using input audio directly")
            shutil.copy2(input_audio, vocals_raw)

        # --- 2. Dereverberation ---
        vocals_processed = tmp / "vocals_processed.wav"
        if dereverb_model is not None:
            try:
                _, dereverb_applied = maybe_dereverb(
                    vocals_raw, vocals_processed,
                    dereverb_model=dereverb_model,
                    probability=dereverb_probability,
                )
            except Exception as e:
                logger.warning("Dereverberation failed, using raw audio: %s", e)
                shutil.copy2(vocals_raw, vocals_processed)
        else:
            shutil.copy2(vocals_raw, vocals_processed)

        # --- 3. Desilence + timestamp tracking ---
        segments_dir = tmp / "segments"
        output_csv = None

        if do_desilence:
            seg_paths, segments, csv_path = desilence_and_track(
                vocals_processed, segments_dir,
                min_silence_len=min_silence_len,
                silence_thresh=silence_thresh,
                keep_silence=keep_silence,
                min_segment_len=min_segment_len,
            )
            if not seg_paths:
                logger.warning("No segments found for %s", input_audio.name)
                return None
            output_csv = output_dir / f"{vocals_processed.stem}_vad_seg_ts.csv"
            shutil.copy2(csv_path, output_csv)
        else:
            logger.info("Desilencing skipped — treating whole track as one segment")
            segments_dir.mkdir(parents=True, exist_ok=True)
            single_seg = segments_dir / "00001.wav"
            shutil.copy2(vocals_processed, single_seg)
            audio_dur = AudioSegment.from_file(str(vocals_processed))
            segments = [VocalSegment(index=1, start_ms=0, end_ms=len(audio_dur),
                                    duration_ms=len(audio_dur), filename="00001.wav")]
            seg_paths = [single_seg]

        num_segments = len(segments)

        # --- 4. RVC conversion ---
        if do_rvc:
            converted_dir = tmp / "converted"
            converted = convert_segments(
                seg_paths, converted_dir,
                pth_path=pth_candidates[0],
                index_path=best_index,
                f0_up_key=f0_up_key,
                f0_method=f0_method,
            )
            if not converted:
                logger.error("RVC conversion produced no output for %s", input_audio.name)
                return None
            num_converted = len(converted)
            final_segments_dir = converted_dir
        else:
            logger.info("RVC skipped — keeping segments as-is")
            num_converted = len(seg_paths)
            final_segments_dir = segments_dir

        # --- 5. Reassemble + mix with instrumental ---
        if do_reassembly and do_separation and instrumental.exists():
            instrumental_audio = AudioSegment.from_file(str(instrumental))
            total_duration_ms = len(instrumental_audio)

            reassembled_vocals = reassemble_from_segments(
                final_segments_dir, segments, total_duration_ms,
            )

            mixed = instrumental_audio.overlay(reassembled_vocals)
            mixed.export(str(final_wav), format="wav")
            logger.info("Final mix exported: %s", final_wav.name)
        else:
            if do_desilence:
                audio_dur_obj = AudioSegment.from_file(str(vocals_processed))
                reassembled = reassemble_from_segments(
                    final_segments_dir, segments, len(audio_dur_obj),
                )
                reassembled.export(str(final_wav), format="wav")
            else:
                single_out = final_segments_dir / "00001.wav"
                shutil.copy2(single_out, final_wav)
            if not do_reassembly:
                logger.info("Reassembly skipped — output is processed audio only")
            elif not do_separation:
                logger.info("No instrumental track — output is processed audio only")
            logger.info("Output exported: %s", final_wav.name)

    # --- Write per-track metadata ---
    metadata = {
        "source_audio": str(input_audio),
        "source_md5": source_md5,
        "rvc_checkpoint_dir": str(checkpoint_dir) if checkpoint_dir else None,
        "rvc_md5": rvc_md5,
        "stages": {
            "separation": do_separation,
            "dereverberation": dereverb_applied,
            "desilencing": do_desilence,
            "rvc": do_rvc,
            "reassembly": do_reassembly and do_separation,
        },
        "separation_backend": backend_used.value if backend_used else None,
        "dereverb_probability": dereverb_probability,
        "f0_up_key": f0_up_key,
        "f0_method": f0_method,
        "num_segments": num_segments,
        "num_converted": num_converted,
        "output_wav": str(final_wav),
    }
    if output_csv:
        metadata["timestamps_csv"] = str(output_csv)

    meta_path = output_dir / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2))
    logger.info("Metadata written to %s", meta_path)

    metadata["_local_output_dir"] = str(output_dir)
    return metadata


def upload_to_s3(local_output_root: Path, stem: str, s3_dest: str) -> List[str]:
    """Upload all files in the ``{stem}/`` directory to S3.

    Args:
        local_output_root: parent directory containing the ``{stem}/`` folder.
        stem: the ``source_{md5}_rvcmodel_{md5}`` name.
        s3_dest: S3 URI prefix, e.g. ``s3://bucket/prefix/``.

    Returns:
        list of uploaded S3 URIs.
    """
    if _s3 is None:
        raise RuntimeError("boto3 is not installed; cannot upload to S3")

    if not s3_dest.startswith("s3://"):
        raise ValueError(f"Expected s3:// URI, got: {s3_dest}")

    parts = s3_dest[5:].split("/", 1)
    bucket = parts[0]
    prefix = parts[1].rstrip("/") if len(parts) > 1 else ""

    uploaded: List[str] = []

    subdir = local_output_root / stem
    if subdir.is_dir():
        for f in subdir.iterdir():
            if not f.is_file():
                continue
            key = f"{prefix}/{stem}/{f.name}" if prefix else f"{stem}/{f.name}"
            _s3.upload_file(str(f), bucket, key)
            uri = f"s3://{bucket}/{key}"
            uploaded.append(uri)
            logger.info("Uploaded %s -> %s", f.name, uri)

    return uploaded


def _make_dereverb(backend: Optional[str], device: str = "cuda"):
    """Create and load a dereverberation model, or return None to skip."""
    cls = _DEREVERB_CLASSES.get(backend)
    if cls is None:
        return None
    m = cls(device=device)
    m.load()
    return m


def process_batch(
    input_dir: Path,
    checkpoint_dir: Optional[Path],
    output_root: Path,
    *,
    glob_pattern: str = "*.wav",
    dereverb_backend: Optional[str] = None,
    dereverb_device: str = "cuda",
    **kwargs,
) -> List[dict]:
    """Process all audio files in a directory."""
    audio_files = sorted(input_dir.glob(glob_pattern))
    if not audio_files:
        logger.warning("No audio files found matching %s in %s", glob_pattern, input_dir)
        return []

    output_root.mkdir(parents=True, exist_ok=True)
    logger.info("Processing %d tracks from %s", len(audio_files), input_dir)

    dereverb = _make_dereverb(dereverb_backend, dereverb_device)

    results = []
    for audio_path in audio_files:
        try:
            result = process_track(
                audio_path, checkpoint_dir, output_root,
                dereverb_model=dereverb,
                **kwargs,
            )
            if result:
                results.append(result)
        except Exception as e:
            logger.error("Failed to process %s: %s", audio_path.name, e, exc_info=True)

    logger.info("Batch complete: %d / %d succeeded", len(results), len(audio_files))
    return results


def _is_s3_uri(path_str: str) -> bool:
    return path_str.startswith("s3://")


def _download_s3_dir(s3_uri: str, local_dir: Path, extensions: Optional[list] = None) -> Path:
    """Download files from an S3 prefix to a local directory.

    Args:
        s3_uri: S3 URI prefix (e.g. ``s3://bucket/prefix/``).
        local_dir: local directory to download into.
        extensions: if set, only download files with these extensions (e.g. [".pth", ".index"]).

    Returns:
        local_dir.
    """
    import subprocess
    local_dir.mkdir(parents=True, exist_ok=True)

    if extensions:
        for ext in extensions:
            subprocess.run(
                ["aws", "s3", "cp", s3_uri, str(local_dir) + "/",
                 "--recursive", "--exclude", "*", "--include", f"*{ext}"],
                check=True,
            )
    else:
        subprocess.run(
            ["aws", "s3", "cp", s3_uri, str(local_dir) + "/", "--recursive"],
            check=True,
        )
    logger.info("Downloaded S3 prefix %s -> %s", s3_uri, local_dir)
    return local_dir


def _upload_s3_dir(local_dir: Path, s3_uri: str) -> List[str]:
    """Upload all files in local_dir to an S3 prefix."""
    import subprocess
    s3_uri = s3_uri.rstrip("/") + "/"
    subprocess.run(
        ["aws", "s3", "cp", str(local_dir) + "/", s3_uri, "--recursive"],
        check=True,
    )
    uploaded = []
    for f in local_dir.rglob("*"):
        if f.is_file():
            rel = f.relative_to(local_dir)
            uploaded.append(f"{s3_uri}{rel}")
    logger.info("Uploaded %d files to %s", len(uploaded), s3_uri)
    return uploaded


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    p = argparse.ArgumentParser(
        description="SVDD dataset generation pipeline: separate -> dereverb -> desilence -> RVC -> remix",
    )
    input_group = p.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input-dir", type=str, help="Directory of source audio tracks")
    input_group.add_argument("--input-file", type=str, help="Single source audio track")
    p.add_argument("--checkpoint-dir", type=str, default=None,
                   help="RVC checkpoint directory (.pth + .index), local path or s3:// URI. "
                        "Required unless --no-rvc is set.")
    p.add_argument("--output-dir", type=str, required=True,
                   help="Output root directory, local path or s3:// URI")
    p.add_argument("--glob", default="*.wav", help="Glob pattern for input files (default *.wav)")
    p.add_argument("--s3-dest", default=None, help="S3 URI prefix to upload outputs (e.g. s3://bucket/prefix/)")

    p.add_argument("--force", action="store_true",
                   help="Regenerate output even if it already exists")

    stage_group = p.add_argument_group("stage toggles")
    stage_group.add_argument("--no-separation", action="store_true",
                             help="Skip source separation — input audio is used directly")
    stage_group.add_argument("--no-desilence", action="store_true",
                             help="Skip desilencing — treat the whole track as one segment")
    stage_group.add_argument("--no-rvc", action="store_true",
                             help="Skip RVC voice conversion (--checkpoint-dir not required)")
    stage_group.add_argument("--no-reassembly", action="store_true",
                             help="Skip reassembly — output processed audio without mixing "
                                  "back with the instrumental stem")

    sep_group = p.add_argument_group("separation")
    sep_group.add_argument("--sep-backend", choices=["demucs", "roformer", "mdxchain", "uvr5"],
                           default=None, help="Force separation backend (default: random)")
    sep_group.add_argument("--device", default=None, help="Torch device (default: auto)")

    dereverb_group = p.add_argument_group("dereverberation")
    dereverb_group.add_argument("--dereverb-backend", choices=["vrnet", "mbr"], default=None,
                                help="Dereverb backend: vrnet (UVR5 VR-Net) or mbr "
                                     "(MelBandRoformer). Omit to skip dereverberation.")
    dereverb_group.add_argument("--dereverb-device", default="cuda")
    dereverb_group.add_argument("--dereverb-probability", type=float, default=DEREVERB_PROBABILITY)

    silence_group = p.add_argument_group("desilencing")
    silence_group.add_argument("--min-silence-len", type=int, default=2000)
    silence_group.add_argument("--silence-thresh", type=int, default=-40)
    silence_group.add_argument("--keep-silence", type=int, default=100)
    silence_group.add_argument("--min-segment-len", type=int, default=3000)

    rvc_group = p.add_argument_group("rvc")
    rvc_group.add_argument("--f0-up-key", type=int, default=0, help="Pitch shift in semitones")
    rvc_group.add_argument("--f0-method", default="rmvpe")

    args = p.parse_args()

    do_rvc = not args.no_rvc
    do_separation = not args.no_separation
    do_desilence = not args.no_desilence
    do_reassembly = not args.no_reassembly

    if do_rvc and args.checkpoint_dir is None:
        p.error("--checkpoint-dir is required unless --no-rvc is set")

    # --- Resolve S3 paths ---
    _cleanup_temps = []

    checkpoint_dir = None
    if args.checkpoint_dir:
        checkpoint_dir_str = args.checkpoint_dir
        if _is_s3_uri(checkpoint_dir_str):
            ckpt_tmp = Path(tempfile.mkdtemp(prefix="svdd_ckpt_"))
            _cleanup_temps.append(ckpt_tmp)
            logger.info("Downloading checkpoint from S3: %s", checkpoint_dir_str)
            _download_s3_dir(checkpoint_dir_str, ckpt_tmp, extensions=[".pth", ".index"])
            checkpoint_dir = ckpt_tmp
        else:
            checkpoint_dir = Path(checkpoint_dir_str)

    output_dir_str = args.output_dir
    s3_output_dest = None
    if _is_s3_uri(output_dir_str):
        s3_output_dest = output_dir_str
        output_dir = Path(tempfile.mkdtemp(prefix="svdd_out_"))
        _cleanup_temps.append(output_dir)
        logger.info("Output will be written locally then uploaded to S3: %s", s3_output_dest)
    else:
        output_dir = Path(output_dir_str)

    s3_dest = args.s3_dest or s3_output_dest

    input_file = Path(args.input_file) if args.input_file else None
    input_dir = Path(args.input_dir) if args.input_dir else None

    sep_backend = SeparationBackend(args.sep_backend) if args.sep_backend else None
    common_kwargs = dict(
        sep_backend=sep_backend,
        dereverb_probability=args.dereverb_probability,
        device=args.device,
        f0_up_key=args.f0_up_key,
        f0_method=args.f0_method,
        min_silence_len=args.min_silence_len,
        silence_thresh=args.silence_thresh,
        keep_silence=args.keep_silence,
        min_segment_len=args.min_segment_len,
        do_separation=do_separation,
        do_desilence=do_desilence,
        do_rvc=do_rvc,
        do_reassembly=do_reassembly,
        force=args.force,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    if input_file:
        dereverb = _make_dereverb(args.dereverb_backend, args.dereverb_device)
        result = process_track(
            input_file, checkpoint_dir, output_dir,
            dereverb_model=dereverb,
            **common_kwargs,
        )
        results = [result] if result else []
    else:
        results = process_batch(
            input_dir=input_dir,
            checkpoint_dir=checkpoint_dir,
            output_root=output_dir,
            glob_pattern=args.glob,
            dereverb_backend=args.dereverb_backend,
            dereverb_device=args.dereverb_device,
            **common_kwargs,
        )

    if s3_dest and results:
        for r in results:
            if r.get("skipped"):
                continue
            stem = _output_stem(r["source_md5"], r["rvc_md5"])
            try:
                s3_uris = upload_to_s3(output_dir, stem, s3_dest)
                r["s3_uris"] = s3_uris
            except Exception as e:
                logger.error("S3 upload failed for %s: %s", stem, e)

    summary_path = output_dir / "pipeline_summary.json"
    summary_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nDone. {len(results)} tracks processed. Summary: {summary_path}")

    if s3_output_dest:
        try:
            _upload_s3_dir(output_dir, s3_output_dest)
            print(f"Outputs uploaded to {s3_output_dest}")
        except Exception as e:
            logger.error("S3 upload of output directory failed: %s", e)
            print(f"WARNING: S3 upload failed, local outputs preserved at {output_dir}")
            _cleanup_temps.remove(output_dir)
