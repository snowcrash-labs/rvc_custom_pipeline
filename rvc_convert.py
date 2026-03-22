#!/usr/bin/env python3
"""RVC voice conversion for individual vocal segments.

Mirrors the RVC inference logic from aws_batch_interface/jobs/rvc/worker.py
but operates on local files rather than S3.  Supports both the ``rvc`` CLI
and direct Python API.
"""
from __future__ import annotations

import hashlib
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


def _find_rvc_binary() -> str:
    """Locate the ``rvc`` CLI binary, preferring the same env as the running interpreter."""
    exe_dir = Path(sys.executable).parent
    candidate = exe_dir / "rvc"
    if candidate.exists():
        return str(candidate)
    found = shutil.which("rvc")
    if found:
        return found
    return "rvc"


def _default_rvc_assets_root() -> Path:
    env = os.environ.get("RVC_ASSETS_ROOT")
    if env:
        return Path(env)
    return Path(tempfile.gettempdir()) / "rvc_assets"


RVC_ASSETS_ROOT = _default_rvc_assets_root()


def md5_of_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def md5_of_string(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def run_rvc_cli(
    pth_path: Path,
    input_wav: Path,
    output_wav: Path,
    index_path: Optional[Path] = None,
    f0_up_key: int = 0,
    f0_method: str = "rmvpe",
    timeout: int = 600,
) -> bool:
    """Run the ``rvc infer`` CLI subprocess.

    The RVC library strips directory info from Path inputs (it calls
    ``.name`` on Path objects), so we must pass only the filename and
    set ``cwd`` to the file's parent directory.
    """
    rvc_bin = _find_rvc_binary()
    cmd = [
        rvc_bin, "infer",
        "-m", str(pth_path),
        "-i", input_wav.name,
        "-o", str(output_wav),
    ]
    if f0_up_key != 0:
        cmd.extend(["-fu", str(f0_up_key)])
    if index_path and index_path.exists():
        cmd.extend(["-if", str(index_path)])

    root = RVC_ASSETS_ROOT
    env = os.environ.copy()
    env.update({
        "index_root": str(root / "logs"),
        "weight_root": str(root / "weights"),
        "hubert_path": str(root / "assets" / "hubert" / "hubert_base.pt"),
        "rmvpe_path": str(root / "assets" / "rmvpe" / "rmvpe.pt"),
        "rmvpe_root": str(root / "assets" / "rmvpe"),
    })

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        env=env,
        cwd=str(input_wav.parent),
    )
    if result.returncode != 0:
        for line in (result.stderr or "").strip().splitlines()[-10:]:
            logger.warning("RVC stderr: %s", line)
        return False
    if not output_wav.exists() or output_wav.stat().st_size < 1000:
        logger.warning("RVC output missing or too small: %s", output_wav)
        return False
    return True


def convert_segments(
    segment_paths: List[Path],
    output_dir: Path,
    pth_path: Path,
    index_path: Optional[Path] = None,
    f0_up_key: int = 0,
    f0_method: str = "rmvpe",
) -> List[Path]:
    """Convert a list of vocal segment wavs through RVC.

    Each output file keeps the same filename as its input so that
    reassembly via timestamps is straightforward.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    converted: List[Path] = []

    for seg_path in segment_paths:
        out_path = output_dir / seg_path.name
        ok = run_rvc_cli(
            pth_path, seg_path, out_path,
            index_path=index_path,
            f0_up_key=f0_up_key,
            f0_method=f0_method,
        )
        if ok:
            converted.append(out_path)
            logger.info("Converted segment: %s", seg_path.name)
        else:
            logger.warning("Failed to convert segment: %s", seg_path.name)

    logger.info("Converted %d / %d segments", len(converted), len(segment_paths))
    return converted


def find_checkpoint_files(
    checkpoint_dir: Path,
) -> tuple[List[Path], Optional[Path]]:
    """Find .pth and .index files in a local checkpoint directory.

    Returns (pth_candidates, best_index_path).
    """
    pth_files = sorted(checkpoint_dir.glob("*.pth"), key=lambda p: p.name)
    excluded = ("._D_", "._G_", "_D_", "_G_", "D_", "G_")
    inference_pth = [
        p for p in pth_files
        if not any(p.name.startswith(prefix) for prefix in excluded)
    ]
    if not inference_pth:
        inference_pth = [p for p in pth_files if "D_" not in p.name]

    index_files = sorted(checkpoint_dir.glob("*.index"), key=lambda p: p.name)
    preferred = [p for p in index_files if p.name.startswith("added")]
    best_index = preferred[0] if preferred else (index_files[0] if index_files else None)

    return inference_pth, best_index


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser(description="RVC convert vocal segments")
    p.add_argument("--segments-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--checkpoint-dir", type=Path, required=True)
    p.add_argument("--f0-up-key", type=int, default=0)
    p.add_argument("--f0-method", default="rmvpe")
    args = p.parse_args()

    pth_candidates, best_index = find_checkpoint_files(args.checkpoint_dir)
    if not pth_candidates:
        raise FileNotFoundError(f"No .pth files found in {args.checkpoint_dir}")

    seg_wavs = sorted(args.segments_dir.glob("*.wav"))
    convert_segments(
        seg_wavs, args.output_dir,
        pth_path=pth_candidates[0],
        index_path=best_index,
        f0_up_key=args.f0_up_key,
        f0_method=args.f0_method,
    )
