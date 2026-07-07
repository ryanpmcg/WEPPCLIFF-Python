#!/bin/bash
# Run the automated WEPPCLIFF tutorial (16 example runs).
set -uo pipefail
source "$(dirname "$0")/_common.sh"

echo "WEPPCLIFF — Run AutoTutorial"
echo "Project: $ROOT"
echo "This may take several minutes."
echo ""

bash "$ROOT/Tutorial/AutoTutorial.sh"
pause_on_exit $?