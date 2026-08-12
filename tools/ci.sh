#!/usr/bin/env bash
# The gate CI runs, and the one to run before pushing. Everything here works
# on a bare machine: no NIC, no licence, no SDK (SPEC §spec:testing).
set -euo pipefail
cd "$(dirname "$0")/.."

bold() { printf '\033[1m== %s ==\033[0m\n' "$1"; }
ok() { printf '  \033[32mok\033[0m   %s\n' "$1"; }

failed=0
run() {
  local name="$1"; shift
  if "$@"; then ok "$name"; else printf '  \033[31mFAIL\033[0m %s\n' "$name"; failed=1; fi
}

bold "lint"
run "ruff check" uv run ruff check
run "ruff format --check" uv run ruff format --check

bold "types"
run "mypy" uv run mypy

bold "tests"
run "pytest" uv run pytest -q

bold "summary"
if [ "$failed" -ne 0 ]; then printf '\033[31mFAILED\033[0m\n'; exit 1; fi
printf '\033[32mall gates passed\033[0m\n'
