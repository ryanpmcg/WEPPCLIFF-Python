#!/bin/bash

DIRECTORY=$(cd "$(dirname "$0")" && pwd)
cd "$DIRECTORY/.." || exit 1

PYTHON="python3"
if [ -x ".venv/bin/python" ]; then
 PYTHON=".venv/bin/python"
fi

$PYTHON weppcliff.py -f ASOS_BPT_KMQE.csv -la y
$PYTHON weppcliff.py -f ASOS_BPT_KMQE.csv -fn bpt -la y -verb
$PYTHON weppcliff.py -f ASOS_BPT_KMQE.csv -la y -verb -o JUNK -pd -sdt "2000-01-01 00:00:00" -edt "2004-12-31 24:00:00"
$PYTHON weppcliff.py -f ASOS_BPT_KMQE_Standalone.csv -la y -verb -cv f -pd -fn KMQE-Standalone -ei -ee ARS -ed 1
$PYTHON weppcliff.py -f AH537_BPT_CP.csv -cp -u m PRECIP -sm 2 -la y -ipb
$PYTHON weppcliff.py -f AH537_BPT_CP.csv -cp -u m PRECIP -sm 2 -la y -verb -ei -ee AH537 -ipb -pd -ed 3
$PYTHON weppcliff.py -f AH537_BPT_IP.csv -u m PRECIP -sm 2 -la y -verb -ei -ee AH537 -ipb -pd -ed 3
$PYTHON weppcliff.py -f AH537_1MIN_UP.csv -pi 1 -u m PRECIP -sm 2 -la y -verb -ei -ee AH537 -ipb -pd -ed 3
$PYTHON weppcliff.py -f AH537_1MIN.csv -pi 1 -u m PRECIP -sm 2 -la y -verb -ei -ee AH537 -ipb -pd -ed 3
$PYTHON weppcliff.py -f AH537_5MIN.csv -pi 5 -u m PRECIP -sm 2 -la y -verb -ei -ee AH537 -ipb -pd -ed 3
$PYTHON weppcliff.py -f AH537_10MIN.csv -pi 10 -u m PRECIP -sm 2 -la y -verb -ei -ee AH537 -ipb -pd -ed 3
$PYTHON weppcliff.py -f AH537_15MIN.csv -pi 15 -u m PRECIP -sm 2 -la y -verb -ei -ee AH537 -ipb -pd -ed 3
$PYTHON weppcliff.py -f AH537_15MIN_UP.csv -pi 15 -u m PRECIP -sm 2 -la y -verb -ei -ee AH537 -ipb -pd -ed 3
$PYTHON weppcliff.py -f AH537_15MIN_HT.csv -pi 15 -u m PRECIP -sm 2 -la y -verb -ei -ee AH537 -ipb -pd -ed 3
$PYTHON weppcliff.py -f ASOS_BPT_KMQE.csv -la y -verb -qc -pd
$PYTHON weppcliff.py -f ASOS_BPT_KMQE.csv -la y -verb -qc -pd -id -qi -pm -st -ed 1
$PYTHON weppcliff.py -f ASOS_BPT_KMQE.csv -la y -verb -qc -pd -id -iv -pm -st -ed 1