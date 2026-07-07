#!/bin/bash
# Run AutoTest.1 and verify breakpoint structural rules (rules 1–5).
set -uo pipefail
source "$(dirname "$0")/_common.sh"

echo "WEPPCLIFF — AutoTest.1 Structure Test"
echo "Project: $ROOT"
echo "This runs weppcliff on ASOS_BPT_KMQE.csv and validates the .cli output."
echo ""

"$PYTHON" -m pytest Development/tests/test_autotest.py::test_autotest1_cli_structure -v
pause_on_exit $?