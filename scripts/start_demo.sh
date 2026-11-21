#!/usr/bin/env bash
# Find a usable native Linux/macOS/WSL environment; activation is not required.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$project_dir"
if [[ -n "${PLANT_DISEASE_PYTHON:-}" ]]; then
    candidates=("$PLANT_DISEASE_PYTHON")
else
    candidates=("$project_dir/.venv/bin/python" "$project_dir/venv/bin/python")
    [[ -z "${VIRTUAL_ENV:-}" ]] || candidates+=("$VIRTUAL_ENV/bin/python")
    candidates+=("$HOME/.venvs/cecs551-plant-disease/bin/python" python3 python)
fi
for candidate in "${candidates[@]}"; do
    demo_python="$(command -v -- "$candidate" || true)"
    [[ -n "$demo_python" && -x "$demo_python" ]] || continue
    if reason="$("$demo_python" scripts/launch_demo.py --check "$@" 2>&1)"; then
        exec "$demo_python" scripts/launch_demo.py "$@"
    fi
    printf '%s\n%s\n' "Not ready: $demo_python" "$reason" >&2
done
printf '%s\n' 'No ready Python environment found. See docs/presentation/demo-guide.md.' >&2
exit 1
