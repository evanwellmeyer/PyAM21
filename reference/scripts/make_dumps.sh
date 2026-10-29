#!/bin/bash
# Run a case with the dump hook on. Usage: make_dumps.sh CASE [STEPS] [TAG]
#   STEPS: comma list of steps to dump (default: all); TAG: dump directory name (default CASE)
set -euo pipefail
REF=$(cd "$(dirname "$0")/.." && pwd)
CASE=$1; STEPS=${2:-}; TAG=${3:-$1}
rm -rf "$REF/dumps/$TAG"; mkdir -p "$REF/dumps/$TAG"
REF_DUMP_DIR="$REF/dumps/$TAG" REF_DUMP_STEPS="$STEPS" bash "$REF/scripts/run_case.sh" "$CASE" strict "$REF/runs/dump/$TAG" | tail -1
cp "$REF/runs/dump/$TAG/RESTART/mg_drag.res.nc" "$REF/dumps/$TAG/" 2>/dev/null || true
echo "$(ls "$REF/dumps/$TAG" | wc -l) files in dumps/$TAG ($(du -sh "$REF/dumps/$TAG" | cut -f1))"
