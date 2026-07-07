#!/bin/bash
# Run the full pytest suite (unit, CLI validation, autotest structure).
set -uo pipefail
source "$(dirname "$0")/_common.sh"

echo "WEPPCLIFF — Run All Pytest Tests"
echo "Project: $ROOT"
echo "Python:  $PYTHON"
echo ""

"$PYTHON" -m pytest Development/tests/ -v
pause_on_exit $?