#!/bin/bash
# Install a scoped sudoers entry for passwordless chown on the HuggingFace cache.
# Params: {user}, {cache_dir} (empty string = auto-detect via getent)
set -euo pipefail

if [ -n "{cache_dir}" ]; then
    CACHE_DIR="{cache_dir}"
else
    TARGET_HOME=$(getent passwd "{user}" | cut -d: -f6)
    if [ -z "$TARGET_HOME" ]; then echo "ERROR: cannot resolve home for {user}"; exit 1; fi
    CACHE_DIR="$TARGET_HOME/.cache/huggingface"
fi

CHOWN_PATH="/usr/bin/chown"
SUDOERS_FILE="/etc/sudoers.d/sparkrun-chown-{user}"

cat > "$SUDOERS_FILE" << SUDOERS_EOF
# Installed by: sparkrun setup fix-permissions --save-sudo
{user} ALL=(root) NOPASSWD: $CHOWN_PATH -R {user} $CACHE_DIR
SUDOERS_EOF

if visudo -cf "$SUDOERS_FILE" >/dev/null 2>&1; then
    chmod 0440 "$SUDOERS_FILE"
    echo "OK: installed sudoers entry in $SUDOERS_FILE"
else
    rm -f "$SUDOERS_FILE"
    echo "ERROR: sudoers validation failed"
    exit 1
fi
