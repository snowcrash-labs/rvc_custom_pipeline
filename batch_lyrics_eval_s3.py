#!/usr/bin/env python3
"""Iterate over subprefixes in an S3 prefix and augment each ``metadata.json``
with WER/CER metrics comparing ``vocals.wav`` (original) against
``acapella.wav`` (voice-converted).

Multi-GPU: one worker process per visible GPU, each loading faster-whisper
turbo once. Inside each worker, S3 downloads are pipelined ahead of GPU
inference via a small thread pool so the GPU never waits on I/O.

Usage::

    # Use all visible GPUs, first 20 subprefixes
    python batch_lyrics_eval_s3.py \
        --s3-prefix s3://rvc-data-for-riaa/voice-model-rvc-pipeline/ \
        --limit 20

    # Dry-run a single, specific subprefix
    python batch_lyrics_eval_s3.py \
        --s3-prefix s3://rvc-data-for-riaa/voice-model-rvc-pipeline/ \
        --only source_7e5f...xxx \
        --dry-run
"""
from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import shutil
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Empty
from typing import Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

VOCALS_KEY = "vocals.wav"
ACAPELLA_KEY = "acapella.wav"
METADATA_KEY = "metadata.json"
VAD_CSV_KEY = "vad_segments.csv"

SENTINEL = None  # queue stop signal


def _load_vad_segments_csv(path: Path) -> list[tuple[int, int]]:
    """Parse vad_segments.csv → list of (start_ms, end_ms) tuples."""
    import csv
    segs: list[tuple[int, int]] = []
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            segs.append((int(row["start_ms"]), int(row["end_ms"])))
    return segs


# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------

def _parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"Not an s3:// URI: {uri!r}")
    prefix = parsed.path.lstrip("/")
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    return parsed.netloc, prefix


def _iter_subprefixes(s3, bucket: str, prefix: str):
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter="/"):
        for cp in page.get("CommonPrefixes") or []:
            yield cp["Prefix"]


def _object_exists(s3, bucket: str, key: str) -> bool:
    from botocore.exceptions import ClientError
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as exc:
        if exc.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound"):
            return False
        raise


def _build_lyrics_block(result) -> dict:
    block = {
        "wer": result.wer,
        "cer": result.cer,
        "source_text": result.source_text,
        "converted_text": result.converted_text,
        "source_language": result.source_language,
        "converted_language": result.converted_language,
    }
    block.update(result.details)
    return block


# ---------------------------------------------------------------------------
# Download a single subprefix's required files into a fresh tempdir
# ---------------------------------------------------------------------------

def _download_subprefix(s3, bucket: str, subprefix: str) -> Optional[dict]:
    """Returns dict with workdir/metadata/paths, or None if something's missing."""
    name = subprefix.rstrip("/").rsplit("/", 1)[-1]
    for k in (VOCALS_KEY, ACAPELLA_KEY, METADATA_KEY):
        if not _object_exists(s3, bucket, subprefix + k):
            logger.warning("[%s] missing %s — skipping", name, k)
            return None

    workdir = Path(tempfile.mkdtemp(prefix="lyrics_eval_"))
    try:
        meta_local = workdir / METADATA_KEY
        s3.download_file(bucket, subprefix + METADATA_KEY, str(meta_local))
        metadata = json.loads(meta_local.read_text(encoding="utf-8"))
        vocals_local = workdir / VOCALS_KEY
        acapella_local = workdir / ACAPELLA_KEY
        s3.download_file(bucket, subprefix + VOCALS_KEY, str(vocals_local))
        s3.download_file(bucket, subprefix + ACAPELLA_KEY, str(acapella_local))

        vad_csv_local: Optional[Path] = None
        if _object_exists(s3, bucket, subprefix + VAD_CSV_KEY):
            vad_csv_local = workdir / VAD_CSV_KEY
            s3.download_file(bucket, subprefix + VAD_CSV_KEY, str(vad_csv_local))

        return {
            "name": name,
            "subprefix": subprefix,
            "workdir": workdir,
            "metadata": metadata,
            "meta_local": meta_local,
            "vocals": vocals_local,
            "acapella": acapella_local,
            "vad_csv": vad_csv_local,
        }
    except Exception:
        shutil.rmtree(workdir, ignore_errors=True)
        raise


# ---------------------------------------------------------------------------
# GPU worker — long-lived process, one per GPU
# ---------------------------------------------------------------------------

def _worker(
    device_index: int,
    in_queue: "mp.Queue",
    out_queue: "mp.Queue",
    bucket: str,
    *,
    model_name: str,
    compute_type: str,
    batch_size: int,
    prefetch: int,
    overwrite: bool,
    dry_run: bool,
) -> None:
    # Silence boto/httpx chatter in workers.
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s %(levelname)s W{device_index} %(message)s",
    )
    logging.getLogger("botocore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

    import boto3
    from lyrics_eval_faster import load_pipeline, evaluate_pair

    s3 = boto3.Session().client("s3")
    pipeline = load_pipeline(
        model_name=model_name,
        device="cuda",
        device_index=device_index,
        compute_type=compute_type,
    )
    logger.info("worker ready on cuda:%d", device_index)

    dl_pool = ThreadPoolExecutor(max_workers=max(2, prefetch))
    in_flight: list = []  # (subprefix, future)

    def _submit_next() -> bool:
        try:
            sp = in_queue.get(timeout=0.5)
        except Empty:
            return True  # keep trying
        if sp is SENTINEL:
            return False  # shutdown
        fut = dl_pool.submit(_download_subprefix, s3, bucket, sp)
        in_flight.append((sp, fut))
        return True

    # Seed prefetch
    alive = True
    while alive and len(in_flight) < prefetch:
        alive = _submit_next()

    while in_flight:
        sp, fut = in_flight.pop(0)
        # Top up prefetch
        if alive and len(in_flight) < prefetch:
            alive = _submit_next()

        name = sp.rstrip("/").rsplit("/", 1)[-1]
        try:
            job = fut.result()
        except Exception as e:
            logger.exception("[%s] download failed: %s", name, e)
            out_queue.put({"subprefix": sp, "status": "download_error", "error": repr(e)})
            continue
        if job is None:
            out_queue.put({"subprefix": sp, "status": "skipped_missing"})
            continue

        try:
            if not overwrite and "lyrics_eval" in job["metadata"]:
                shutil.rmtree(job["workdir"], ignore_errors=True)
                out_queue.put({"subprefix": sp, "status": "skipped_already_done"})
                continue

            t0 = time.time()
            vad_source = "faster_whisper_builtin"
            clip_timestamps: Optional[list[dict]] = None
            if job.get("vad_csv") is not None:
                try:
                    segs_ms = _load_vad_segments_csv(job["vad_csv"])
                except Exception as e:
                    logger.warning(
                        "[%s] failed to parse vad_segments.csv (%s) — falling back to built-in VAD",
                        name, e,
                    )
                    segs_ms = []
                if segs_ms:
                    clip_timestamps = [
                        {"start": s / 1000.0, "end": e / 1000.0} for s, e in segs_ms
                    ]
                    vad_source = f"vad_segments_csv({len(segs_ms)})"

            result = evaluate_pair(
                pipeline, job["vocals"], job["acapella"],
                batch_size=batch_size, clip_timestamps=clip_timestamps,
            )
            dt = time.time() - t0
            block = _build_lyrics_block(result)
            block["vad_source"] = vad_source
            job["metadata"]["lyrics_eval"] = block
            job["meta_local"].write_text(
                json.dumps(job["metadata"], indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            if dry_run:
                logger.info(
                    "[%s] WER=%.3f CER=%.3f infer=%.1fs (dry-run)",
                    name, result.wer, result.cer, dt,
                )
            else:
                s3.upload_file(
                    str(job["meta_local"]), bucket, sp + METADATA_KEY,
                    ExtraArgs={"ContentType": "application/json"},
                )
                logger.info(
                    "[%s] WER=%.3f CER=%.3f infer=%.1fs uploaded",
                    name, result.wer, result.cer, dt,
                )
            out_queue.put({
                "subprefix": sp, "status": "ok",
                "wer": result.wer, "cer": result.cer, "infer_s": dt,
                "source_language": result.source_language,
                "converted_language": result.converted_language,
            })
        except Exception as e:
            logger.exception("[%s] inference failed: %s", name, e)
            out_queue.put({"subprefix": sp, "status": "inference_error", "error": repr(e)})
        finally:
            shutil.rmtree(job["workdir"], ignore_errors=True)

        # If queue has been drained while we were inferring, keep topping up
        while alive and len(in_flight) < prefetch:
            alive = _submit_next()

    dl_pool.shutdown(wait=True)
    logger.info("worker shutting down")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _detect_gpu_count() -> int:
    # Prefer pynvml / nvidia-smi but fall back to env hints.
    try:
        import ctranslate2
        n = ctranslate2.get_cuda_device_count()
        if n > 0:
            return n
    except Exception:
        pass
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    if vis:
        return len([x for x in vis.split(",") if x.strip() != ""])
    return 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--s3-prefix", required=True)
    p.add_argument("--only", default=None,
                   help="Process only this single subprefix (dir name, not full URI)")
    p.add_argument("--limit", type=int, default=None,
                   help="Max subprefixes to process")
    p.add_argument("--model", default="turbo", help="faster-whisper model (default: turbo)")
    p.add_argument("--compute-type", default="float16",
                   help="CTranslate2 compute type (float16, int8_float16, int8)")
    p.add_argument("--batch-size", type=int, default=8,
                   help="BatchedInferencePipeline chunk batch size (default: 8)")
    p.add_argument("--num-gpus", type=int, default=None,
                   help="Number of GPU workers (default: auto-detect)")
    p.add_argument("--prefetch", type=int, default=2,
                   help="Subprefixes to prefetch per worker (default: 2)")
    p.add_argument("--overwrite", action="store_true",
                   help="Recompute even if lyrics_eval already present")
    p.add_argument("--dry-run", action="store_true",
                   help="Compute metrics but do not upload updated metadata.json")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s M %(message)s",
    )
    logging.getLogger("botocore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    import boto3
    s3 = boto3.client("s3")
    bucket, prefix = _parse_s3_uri(args.s3_prefix)

    num_gpus = args.num_gpus or _detect_gpu_count()
    logger.info("using %d GPU worker(s)", num_gpus)

    ctx = mp.get_context("spawn")
    in_queue: mp.Queue = ctx.Queue(maxsize=num_gpus * args.prefetch * 2)
    out_queue: mp.Queue = ctx.Queue()

    workers = []
    for gpu in range(num_gpus):
        w = ctx.Process(
            target=_worker,
            args=(gpu, in_queue, out_queue, bucket),
            kwargs=dict(
                model_name=args.model,
                compute_type=args.compute_type,
                batch_size=args.batch_size,
                prefetch=args.prefetch,
                overwrite=args.overwrite,
                dry_run=args.dry_run,
            ),
            daemon=False,
        )
        w.start()
        workers.append(w)

    # Enumerate subprefixes and feed the queue
    if args.only:
        only = args.only.strip("/")
        iterator = iter([f"{prefix}{only}/"])
    else:
        iterator = _iter_subprefixes(s3, bucket, prefix)

    submitted = 0
    for sp in iterator:
        if args.limit is not None and submitted >= args.limit:
            break
        in_queue.put(sp)
        submitted += 1
    logger.info("queued %d subprefix(es)", submitted)
    for _ in range(num_gpus):
        in_queue.put(SENTINEL)

    # Drain results as they come in
    t0 = time.time()
    results = []
    while len(results) < submitted:
        try:
            r = out_queue.get(timeout=1.0)
        except Empty:
            if not any(w.is_alive() for w in workers):
                break
            continue
        results.append(r)

    for w in workers:
        w.join(timeout=30)
        if w.is_alive():
            logger.warning("worker %s still alive, terminating", w.pid)
            w.terminate()

    dt = time.time() - t0
    ok = sum(1 for r in results if r.get("status") == "ok")
    skipped = sum(1 for r in results if r.get("status", "").startswith("skipped"))
    errors = sum(1 for r in results if r.get("status", "").endswith("error"))
    logger.info(
        "Done. submitted=%d ok=%d skipped=%d errors=%d elapsed=%.1fs (%.2f subpref/s)",
        submitted, ok, skipped, errors, dt, (ok / dt) if dt > 0 else 0.0,
    )
    # Emit per-item summary for inspection in tests
    print(json.dumps({"summary": {
        "submitted": submitted, "ok": ok, "skipped": skipped, "errors": errors,
        "elapsed_s": dt,
    }, "results": results}, indent=2, default=str))
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
