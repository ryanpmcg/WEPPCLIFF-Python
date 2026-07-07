#!/bin/sh

DIRECTORY=$(cd "$(dirname "$0")" && pwd)
cd "$DIRECTORY/.." || exit 1

PYTHON="python3"
if [ -x ".venv/bin/python" ]; then
  PYTHON=".venv/bin/python"
fi

TEST="
  Verifying WEPPCLIFF installation (i.e. if you have not completed installation
  this test will fail). This test will only take a couple minutes to finish and
  will begin in a few seconds. Please check back momentarily for the results."

CHECKSUM="
  Running checksum tests. Almost done..."

SUCCESS="
  CONGRATULATIONS! Your WEPPCLIFF installation was successfully verified by 8
  automated test runs of 8 highly unique file outputs. Now you are ready to
  use WEPPCLIFF with peace of mind, knowing everything is working as intended."

FAILURE="
  OOPS! Something is not right. Your WEPPCLIFF installation failed at least one
  of 8 automated test runs. Please remove and reinstall WEPPCLIFF then try this
  test again. If the problem persists and you believe the problem lies within
  WEPPCLIFF, please create an issue at: https://github.com/ryanpmcg/WEPPCLIFF

  Thank you! Hopefully the WEPPCLIFF community can help you get up and running
  soon."

clear 2>/dev/null || true
printf "%s" "$TEST"
sleep 10

mkdir -p "$DIRECTORY/Output"

$PYTHON weppcliff.py -o "$DIRECTORY/Output" -e "$DIRECTORY/Output" -fn AutoTest.1 -f ASOS_BPT_KMQE.csv -la y -verb t -ed 2
$PYTHON weppcliff.py -o "$DIRECTORY/Output" -e "$DIRECTORY/Output" -fn AutoTest.2 -f AH537_BPT_CP.csv -cp t -u m PRECIP -sm 2 -la y -verb t -ei t -ee ARS -ipb t -pd t -ed 2
$PYTHON weppcliff.py -o "$DIRECTORY/Output" -e "$DIRECTORY/Output" -fn AutoTest.3 -f AH537_BPT_IP.csv -cp f -u m PRECIP -sm 2 -la y -verb t -ei t -ee ARS -ipb t -pd t -ed 2
$PYTHON weppcliff.py -o "$DIRECTORY/Output" -e "$DIRECTORY/Output" -fn AutoTest.4 -f AH537_1MIN_UP.csv -pi 1 -cp f -u m PRECIP -sm 2 -la y -verb t -ei t -ee ARS -ipb t -pd t -ed 2
$PYTHON weppcliff.py -o "$DIRECTORY/Output" -e "$DIRECTORY/Output" -fn AutoTest.5 -f AH537_1MIN.csv -pi 1 -cp f -u m PRECIP -sm 2 -la y -verb t -ei t -ee ARS -ipb t -pd t -ed 2
$PYTHON weppcliff.py -o "$DIRECTORY/Output" -e "$DIRECTORY/Output" -fn AutoTest.6 -f AH537_5MIN.csv -pi 5 -cp f -u m PRECIP -sm 2 -la y -verb t -ei t -ee ARS -ipb t -pd t -ed 2
$PYTHON weppcliff.py -o "$DIRECTORY/Output" -e "$DIRECTORY/Output" -fn AutoTest.7 -f AH537_10MIN.csv -pi 10 -cp f -u m PRECIP -sm 2 -la y -verb t -ei t -ee ARS -ipb t -pd t -ed 2
$PYTHON weppcliff.py -o "$DIRECTORY/Output" -e "$DIRECTORY/Output" -fn AutoTest.8 -f AH537_15MIN.csv -pi 15 -cp f -u m PRECIP -sm 2 -la y -verb t -ei t -ee ARS -ipb t -pd t -ed 2

printf "\n  Validating breakpoint structure on AutoTest.1.cli...\n"
STRUCT_FAIL=0
$PYTHON "$DIRECTORY/../Development/scripts/validate_cli_output.py" "$DIRECTORY/Output/AutoTest.1.cli" || STRUCT_FAIL=1

clear 2>/dev/null || true
printf "%s\n" "$CHECKSUM"
FAIL=0
if [ $STRUCT_FAIL -ne 0 ]; then
  FAIL=1
fi

R_REF="/Users/home/Software/Antigravity/WEPPCLIFF/AUTOTEST/OUTPUT"

for ITER in 1 2 3 4 5 6 7 8
do
  TEMP="$DIRECTORY/Output/AutoTest.$ITER.cli"
  REF="$DIRECTORY/Benchmarks/AutoTest.$ITER.cli"
  if [ -f "$R_REF/AutoTest.$ITER.cli" ]; then
    REF="$R_REF/AutoTest.$ITER.cli"
  fi
  BENCH=$(md5 -q "$REF")
  TEMPMD5=$(md5 -q "$TEMP")

  if [ "$BENCH" = "$TEMPMD5" ]; then
    printf "\n  Passed CLI Test $ITER"
  else
    printf "\n  Failed CLI Test $ITER"
    FAIL=1
  fi
  printf "\n"
done

if [ $FAIL -eq 0 ]; then
  printf "\n$SUCCESS"
else
  printf "\n$FAILURE"
fi

printf "\n\n"