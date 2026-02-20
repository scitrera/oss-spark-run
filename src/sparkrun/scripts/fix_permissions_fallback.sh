#!/bin/bash
# Fix cache directory permissions without sudo (runs as root via run_remote_sudo_script).
# Params: {user}, {cache_dir} (empty string = auto-detect via getent)
set -euo pipefail

if [ -n "{cache_dir}" ]; then
    CACHE_DIR="{cache_dir}"
else
    TARGET_HOME=$(getent passwd "{user}" | cut -d: -f6)
    if [ -z "$TARGET_HOME" ]; then echo "ERROR: cannot resolve home for {user}"; exit 1; fi
    CACHE_DIR="$TARGET_HOME/.cache/huggingface"
fi

[ -d "$CACHE_DIR" ] || {{ echo "SKIP: $CACHE_DIR does not exist"; exit 0; }}
chown -R {user} "$CACHE_DIR"
echo "OK: fixed permissions on $CACHE_DIR for {user}"
