#!/usr/bin/env bash
# Launch the prepared Python environment without making the user activate it.
# Called by Run Demo.cmd; also usable directly from a Linux/WSL terminal.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$project_dir"
demo_python="${PLANT_DISEASE_PYTHON:-$HOME/.venvs/cecs551-plant-disease/bin/python}"
if [[ ! -x "$demo_python" ]]; then
    echo "The project Python environment was not found: $demo_python" >&2
    echo "Follow docs/setup.md first, or set PLANT_DISEASE_PYTHON to its Python executable." >&2
    exit 1
fi
exec "$demo_python" -m plant_disease.demo --open-browser "$@"
