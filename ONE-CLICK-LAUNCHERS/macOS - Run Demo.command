#!/usr/bin/env bash
# macOS Finder opens executable .command files in Terminal.
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
bash "$project_dir/scripts/start_demo.sh" "$@"
result=$?
if [[ $result -ne 0 ]]; then
    read -r -p "Demo could not start. Press Enter to close. "
fi
exit "$result"
