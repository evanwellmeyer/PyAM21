#!/bin/bash
# Run one SCM case with the reference executable.
# Usage: bash reference/scripts/run_case.sh CASE [VARIANT] [RUNDIR]
#   CASE: arm astex gcss_arm gcss_astex gcss_atex gcss_bomex gcss_dycoms2_rf01_a
#         gcss_dycoms2_rf01_b gcss_dycoms2_rf02 gcss_rico_short mpace_a mpace_b
# Extra environment: NML_PATCH=file  (a python snippet applied to input.nml, see patch_nml.py)
set -euo pipefail
REF=$(cd "$(dirname "$0")/.." && pwd)
CASE=$1
VARIANT=${2:-strict}
RUN=${3:-$REF/runs/$VARIANT/$CASE}
DATA=$REF/input_data
export CONDA_REF=${CONDA_REF:-$HOME/miniconda3/envs/eamxx-ref}
rm -rf "$RUN"; mkdir -p "$RUN/INPUT" "$RUN/RESTART"
cd "$RUN"
cp "$REF/input/$CASE/input.nml" input.nml
cp "$REF/input/$CASE/diag_table" diag_table
cp "$REF/input/$CASE/field_table" field_table
cp "$REF/input/$CASE/data_table" data_table
if [ -n "${NML_PATCH:-}" ]; then "$CONDA_REF/bin/python" "$REF/scripts/patch_nml.py" input.nml "$NML_PATCH"; fi
cd INPUT
ln -s "$DATA/albedo.data/albedo.data.nc" .
ln -s "$DATA/navy_topography.data/navy_topography.data.nc" .
ln -s "$DATA/navy_pctwater.data/navy_pctwater.data.nc" .
for f in "$DATA"/lad_data/* "$DATA"/sea_esf/*; do ln -s "$f" .; done
if [ "$CASE" = mpace_a ] || [ "$CASE" = mpace_b ]; then
  for f in "$DATA"/hurrell_sst/*; do ln -s "$f" .; done
else
  for f in "$DATA"/sst_data/*; do ln -s "$f" .; done
fi
for f in "$REF/input/$CASE"/*; do
  b=$(basename "$f")
  case $b in input.nml|diag_table|field_table|data_table) ;; *) cp "$f" . ;; esac
done
cd "$RUN"
ulimit -s hard 2>/dev/null || true
start=$(date +%s)
"$REF/build/$VARIANT/scm.x" > stdout.log 2> stderr.log || { echo "RUN FAILED: $RUN"; tail -20 stdout.log stderr.log; exit 1; }
echo "RUN OK ($(( $(date +%s) - start )) s): $RUN"
