#!/bin/bash
# Drop the Linux page cache using sudo -n (non-interactive).
set -euo pipefail

sync
echo 3 | sudo -n /usr/bin/tee /proc/sys/vm/drop_caches > /dev/null 2>&1
echo "OK: page cache cleared"
