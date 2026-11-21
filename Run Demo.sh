#!/usr/bin/env bash
# Run from the repository folder or any other working directory.
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$project_dir/scripts/start_demo.sh" "$@"
