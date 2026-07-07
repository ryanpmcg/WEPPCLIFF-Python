# Shared setup for WEPPCLIFF Commands/*.command launchers.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT" || exit 1

PYTHON="python3"
if [ -x "$ROOT/.venv/bin/python" ]; then
  PYTHON="$ROOT/.venv/bin/python"
fi

pause_on_exit() {
  local code=$1
  echo ""
  if [ "$code" -eq 0 ]; then
    echo "Done (exit 0)."
  else
    echo "Finished with errors (exit $code)."
  fi
  read -r -p "Press Return to close this window..."
  exit "$code"
}