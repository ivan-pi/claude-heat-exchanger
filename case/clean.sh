#!/usr/bin/env bash
cd "$(dirname "$0")"
rm -rf precice-run fluid/output solid/output fluid/*.log solid/*.log \
       fluid/precice-*.json solid/precice-*.json fluid/precice-profiling solid/precice-profiling \
       */precice-*-iterations.log */precice-*-convergence.log */precice-*-events*.json
