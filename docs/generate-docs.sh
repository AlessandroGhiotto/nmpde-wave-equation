#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if ! command -v doxygen >/dev/null 2>&1; then
  echo "Error: doxygen not found in PATH." >&2
  exit 1
fi

doxygen Doxyfile
