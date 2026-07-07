#!/bin/bash
# Compare R vs Python AutoTest.1 breakpoints and write a diff report.
set -uo pipefail
source "$(dirname "$0")/_common.sh"

echo "WEPPCLIFF — Analyze AutoTest.1 Breakpoints"
echo "Project: $ROOT"
echo ""

"$PYTHON" "$ROOT/Development/scripts/analyze_test1_breakpoints.py"
echo ""
echo "Report: $ROOT/Development/reports/AutoTest.1_Breakpoint_Diff_Report.txt"
pause_on_exit $?