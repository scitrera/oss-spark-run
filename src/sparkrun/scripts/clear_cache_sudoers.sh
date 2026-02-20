#!/bin/bash
# Install a scoped sudoers entry for passwordless page cache clearing.
# Params: {user}
set -euo pipefail

SUDOERS_FILE="/etc/sudoers.d/sparkrun-dropcaches-{user}"

cat > "$SUDOERS_FILE" << SUDOERS_EOF
# Installed by: sparkrun setup clear-cache --save-sudo
{user} ALL=(root) NOPASSWD: /usr/bin/tee /proc/sys/vm/drop_caches
SUDOERS_EOF

if visudo -cf "$SUDOERS_FILE" >/dev/null 2>&1; then
    chmod 0440 "$SUDOERS_FILE"
    echo "OK: installed sudoers entry in $SUDOERS_FILE"
else
    rm -f "$SUDOERS_FILE"
    echo "ERROR: sudoers validation failed"
    exit 1
fi
