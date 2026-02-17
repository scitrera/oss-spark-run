#!/bin/bash
set -uo pipefail
echo "Checking GGUF model cache for {repo_id} (quant: {quant})..."
SAFE_NAME=$(echo "{repo_id}" | tr '/' '--')
CACHE_PATH="{cache}/hub/models--$SAFE_NAME"

# Check if GGUF file matching quant already exists
if [ -d "$CACHE_PATH/snapshots" ]; then
    MATCH=$(find "$CACHE_PATH/snapshots" -name "*{quant}*.gguf" -print -quit 2>/dev/null)
    if [ -n "$MATCH" ]; then
        echo "GGUF model already cached: $MATCH"
        exit 0
    fi
fi

echo "Downloading GGUF model: {repo_id} (quant: {quant})..."
if command -v huggingface-cli &>/dev/null; then
    huggingface-cli download "{repo_id}" --include "*{quant}*" {revision_flag}--cache-dir "{cache}/hub"
else
    echo "ERROR: huggingface-cli not available on this host" >&2
    exit 1
fi
