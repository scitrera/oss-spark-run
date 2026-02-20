#!/bin/bash
# Drop the Linux page cache without sudo (runs as root via run_remote_sudo_script).
set -euo pipefail

sync
echo 3 > /proc/sys/vm/drop_caches
echo "OK: page cache cleared"
