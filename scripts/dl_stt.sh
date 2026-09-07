#!/usr/bin/env bash
# Resumable download of the local STT model bundled with MoshiRAG.
set -uo pipefail
REPO=kyutai/stt-1b-en_fr-candle
HF="${HF_CLI:-huggingface-cli}"
# MIBURI's loaders hardcode cache_dir="./assets_dep/kyutai_cache", so a download
# into the default HF cache is invisible to the server and it silently refetches.
PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CACHE="${PROJECT}/miburi/assets_dep/kyutai_cache"
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DOWNLOAD_TIMEOUT=30
attempt=0
while true; do
    attempt=$((attempt+1)); echo "=== attempt $attempt $(date -Is)"
    "$HF" download "$REPO" config.json model.safetensors mimi-pytorch-e351c8d8@125.safetensors \
        --cache-dir "$CACHE" && { echo STT_DOWNLOAD_DONE; exit 0; }
    echo "retrying in 15s" >&2; sleep 15
done
