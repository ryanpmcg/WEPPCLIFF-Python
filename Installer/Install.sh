#!/bin/sh
DIRECTORY=$(cd "$(dirname "$0")" && pwd)
cd "$DIRECTORY/.." || exit 1

python3 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install "pymice[ml] @ git+https://github.com/ryanpmcg/PyMICE.git"
pip install -r requirements.txt
pip install -e .
python weppcliff.py -fr t -la y