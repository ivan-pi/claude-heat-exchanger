#!/usr/bin/env bash
# Fluid participant (Code A). Usage: ./run.sh [path-to-fluid-lbm]
set -e -u
cd "$(dirname "$0")"
EXE=${1:-../../build/solvers/fluid-lbm/fluid-lbm}
exec "$EXE" ../precice-config.xml ../params.txt output
