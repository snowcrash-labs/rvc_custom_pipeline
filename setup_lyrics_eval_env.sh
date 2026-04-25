#!/usr/bin/env bash
# Create a minimal venv that can run batch_lyrics_eval_s3.py on a
# g4dn.12xlarge (or any NVIDIA box with a working CUDA driver).
#
# Idempotent: re-running only re-syncs dependencies.
#
# Usage:
#   ./setup_lyrics_eval_env.sh
#   source .lyrics_eval_venv/bin/activate
#   python batch_lyrics_eval_s3.py --s3-prefix s3://.../ --limit 100 --dry-run

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-${SCRIPT_DIR}/.lyrics_eval_venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

echo ">> Python: $("${PYTHON_BIN}" --version) ($(command -v "${PYTHON_BIN}"))"

if [[ ! -d "${VENV_DIR}" ]]; then
    echo ">> Creating venv at ${VENV_DIR}"
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
else
    echo ">> Reusing existing venv at ${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

echo ">> Upgrading pip/wheel"
python -m pip install --upgrade pip wheel setuptools

echo ">> Installing dependencies"
# faster-whisper's ctranslate2 needs cuBLAS 12 + cuDNN 9 at runtime. They
# are NOT bundled in the ctranslate2 wheel, so we install them as pip
# packages (which ship the .so files under <venv>/.../nvidia/.../lib/).
# An activate hook below adds those dirs to LD_LIBRARY_PATH.
python -m pip install \
    "faster-whisper==1.2.1" \
    "nvidia-cublas-cu12" \
    "nvidia-cudnn-cu12" \
    "boto3>=1.34" \
    "jiwer>=3.0" \
    "soundfile>=0.12" \
    "numpy<2"

echo ">> Writing LD_LIBRARY_PATH activate hook"
PY_SITE="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
ACTIVATE="${VENV_DIR}/bin/activate"
HOOK_MARK="# >>> lyrics_eval CUDA libs hook >>>"
if ! grep -qF "${HOOK_MARK}" "${ACTIVATE}"; then
    cat >> "${ACTIVATE}" <<EOF

${HOOK_MARK}
_LYRICS_EVAL_CUDA_LIB_DIRS="${PY_SITE}/nvidia/cublas/lib:${PY_SITE}/nvidia/cudnn/lib:${PY_SITE}/nvidia/cuda_nvrtc/lib"
export LD_LIBRARY_PATH="\${_LYRICS_EVAL_CUDA_LIB_DIRS}\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}"
# <<< lyrics_eval CUDA libs hook <<<
EOF
fi
# Re-source so the rest of this script (verify + pre-cache) also sees it.
# shellcheck disable=SC1090
source "${ACTIVATE}"

echo
echo ">> Verifying install"
python - <<'PY'
import ctranslate2, faster_whisper, boto3, soundfile, numpy
from importlib.metadata import version as _v
print(f"  python            : {__import__('sys').version.split()[0]}")
print(f"  faster-whisper    : {faster_whisper.__version__}")
print(f"  ctranslate2       : {ctranslate2.__version__}")
print(f"  CUDA devices seen : {ctranslate2.get_cuda_device_count()}")
print(f"  boto3             : {boto3.__version__}")
print(f"  jiwer             : {_v('jiwer')}")
print(f"  soundfile         : {soundfile.__version__}")
print(f"  numpy             : {numpy.__version__}")
# Import jiwer last to prove it loads.
import jiwer  # noqa: F401
PY

echo
echo ">> Pre-caching faster-whisper turbo weights (one-time HF download)"
python - <<'PY'
# Load and discard, so the weights land in ~/.cache/huggingface and the
# per-worker load at runtime is a no-op download-wise.
from faster_whisper import WhisperModel
WhisperModel("turbo", device="cpu", compute_type="int8")
print("  turbo weights cached")
PY

cat <<EOF

Done.

To use:
  source "${VENV_DIR}/bin/activate"
  cd "${SCRIPT_DIR}"
  python batch_lyrics_eval_s3.py \\
      --s3-prefix s3://rvc-data-for-riaa/voice-model-rvc-pipeline/ \\
      --limit 100 \\
      --dry-run

The script auto-detects GPU count. On g4dn.12xlarge it will use all 4 T4s
with one worker process per GPU. Pass --num-gpus N to override.
EOF
