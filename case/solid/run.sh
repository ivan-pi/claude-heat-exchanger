#!/usr/bin/env bash
# Solid participant (Code B). Usage: ./run.sh [path-to-solid-fdm]
set -e -u
cd "$(dirname "$0")"
EXE=${1:-../../build/solvers/solid-fdm/solid-fdm}
exec "$EXE" ../precice-config.xml ../params.txt output
