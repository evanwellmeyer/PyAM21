#!/bin/bash
# Unpack GFDL's Fortran source (reference/src) from the SCM public release, downloading
# the release tarball first if it is not present.
# Usage: bash reference/scripts/get_source.sh
set -euo pipefail
REF=$(cd "$(dirname "$0")/.." && pwd)
TAR=$REF/SCM_public_release.tar.gz
URL=ftp://ftp.gfdl.noaa.gov/perm/GFDL_pubrelease/SCM/SCM_public_release.tar.gz
if [ ! -f "$TAR" ]; then curl -fL -o "$TAR" "$URL"; fi
tar xzf "$TAR" -C "$REF" src
echo "OK: $REF/src"
