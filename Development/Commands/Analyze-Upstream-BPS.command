#!/bin/bash
# Compare upstream BPS data between R .rds and Python .pkl (AutoTest.1).
set -uo pipefail
source "$(dirname "$0")/_common.sh"

echo "WEPPCLIFF — Analyze Upstream BPS"
echo "Project: $ROOT"
echo "Requires Rscript and AutoTest.1.rds / AutoTest.1.pkl."
echo ""

"$PYTHON" "$ROOT/Development/scripts/analyze_upstream_bps.py"
echo ""
echo "Report: $ROOT/Development/reports/AutoTest.1_Upstream_BPS_Report.txt"
pause_on_exit $?