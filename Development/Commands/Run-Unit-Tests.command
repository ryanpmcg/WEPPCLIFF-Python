#!/bin/bash
# Run unit tests only (config, parsing, etc.).
set -uo pipefail
source "$(dirname "$0")/_common.sh"

echo "WEPPCLIFF — Run Unit Tests"
echo "Project: $ROOT"
echo ""

"$PYTHON" -m pytest Development/tests/test_units.py -v
pause_on_exit $?