#!/bin/bash
# Run CLI breakpoint validation tests.
set -uo pipefail
source "$(dirname "$0")/_common.sh"

echo "WEPPCLIFF — Run CLI Validation Tests"
echo "Project: $ROOT"
echo ""

"$PYTHON" -m pytest Development/tests/test_cli_validate.py -v
pause_on_exit $?