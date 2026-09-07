#!/usr/bin/env bash
# Build the `miburi` conda env: torch (CUDA 13, for Blackwell/sm_120) + the
# upstream clone installed editable.  Safe to re-run -- pip resumes from
# ~/.cache/pip and skips anything already satisfied.
#
# Run detached so it survives a dead shell:
#     tmux new -d -s miburi-install '"MIBURI User/setup_env.sh"'
#     tmux attach -t miburi-install          # watch
#     tail -f logs/setup_env.log             # or just read the log

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLONE="${REPO_ROOT}/miburi"
ENV_BIN="${MIBURI_ENV_BIN:-$HOME/miniforge3/envs/miburi/bin}"
LOG_DIR="${REPO_ROOT}/logs"
mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/setup_env.log"

exec > >(tee -a "${LOG}") 2>&1
echo "=== setup_env.sh started $(date -Is) ==="

export PATH="${ENV_BIN}:${PATH}"
cd "${CLONE}"

# The CUDA wheels are 100-350 MB each and pypi.nvidia.com drops connections.
# A first attempt died on a DNS blip at 305/348 MB of cudnn, so: generous
# in-pip retries, plus an outer loop, since pip resumes partial downloads.
PIP_RETRY_FLAGS=(--retries 10 --timeout 60 --resume-retries 20)

retry_pip () {
    local label="$1"; shift
    local attempt
    for attempt in 1 2 3 4 5; do
        echo "--- ${label}: attempt ${attempt}/5 ---"
        if pip install "${PIP_RETRY_FLAGS[@]}" "$@"; then
            echo "--- ${label} OK ---"
            return 0
        fi
        echo "--- ${label}: attempt ${attempt} failed, retrying in 15s ---"
        sleep 15
    done
    echo "!!! ${label} FAILED after 5 attempts"
    return 1
}

echo "--- [1/3] torch 2.9.0 + cu130 ---"
retry_pip "torch" torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 \
    --index-url https://download.pytorch.org/whl/cu130 || exit 1

echo "--- [2/3] miburi package (editable) ---"
retry_pip "miburi" -e . || exit 1

echo "--- [3/3] verification ---"
python - <<'PY'
import torch
print("torch          ", torch.__version__)
print("cuda available ", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device         ", torch.cuda.get_device_name(0))
    cap = torch.cuda.get_device_capability(0)
    print("capability      sm_%d%d" % cap)
    free, total = torch.cuda.mem_get_info()
    print("vram            %.1f GB free / %.1f GB total" % (free / 1e9, total / 1e9))
    # Blackwell needs a kernel that actually compiled for sm_120.
    x = torch.randn(1024, 1024, device="cuda")
    print("matmul check   ", bool(torch.isfinite(x @ x).all()))
for mod in ("smplx", "sentencepiece", "sphn", "safetensors", "trimesh", "librosa"):
    try:
        __import__(mod)
        print(f"import {mod:15s} ok")
    except Exception as exc:
        print(f"import {mod:15s} FAILED: {exc}")
PY

echo "=== setup_env.sh finished $(date -Is) ==="
