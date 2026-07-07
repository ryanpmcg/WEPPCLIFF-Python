#!/bin/bash
# Validate breakpoint structure on an existing AutoTest.1.cli file.
set -uo pipefail
source "$(dirname "$0")/_common.sh"

CLI="$ROOT/Autotest/Output/AutoTest.1.cli"

echo "WEPPCLIFF — Validate AutoTest.1.cli"
echo "File: $CLI"
echo ""

if [ ! -f "$CLI" ]; then
  echo "AutoTest.1.cli not found."
  echo "Run 'Run-AutoTest-Suite.command' or 'Run-AutoTest1-Structure-Test.command' first."
  pause_on_exit 1
fi

"$PYTHON" "$ROOT/Development/scripts/validate_cli_output.py" "$CLI"
pause_on_exit $?