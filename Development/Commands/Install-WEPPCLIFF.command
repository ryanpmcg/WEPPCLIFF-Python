#!/bin/bash
# Create venv, install dependencies, and run first-run setup.
set -uo pipefail
source "$(dirname "$0")/_common.sh"

echo "WEPPCLIFF — Install"
echo "Project: $ROOT"
echo ""

sh "$ROOT/Installer/Install.sh"
pause_on_exit $?