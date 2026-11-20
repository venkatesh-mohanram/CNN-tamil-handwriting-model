#!/usr/bin/env bash
set -euo pipefail
HANDOFF_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${TAMIL_PYTHON:-python3}" "$HANDOFF_ROOT/start.py" "$@"
