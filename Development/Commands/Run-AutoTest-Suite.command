#!/bin/bash
# Full installation verification: 8 autotests, MD5 checks, CLI structure gate.
set -uo pipefail
source "$(dirname "$0")/_common.sh"

echo "WEPPCLIFF — Run Full AutoTest Suite"
echo "Project: $ROOT"
echo "This may take several minutes."
echo ""

sh "$ROOT/Autotest/AutoTest.sh"
pause_on_exit $?