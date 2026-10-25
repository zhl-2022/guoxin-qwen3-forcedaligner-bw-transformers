#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/env.sh"
exec "$FORCEDALIGNER_PYTHON" -B "$ROOT/submission_entry008.py" run "$@"
