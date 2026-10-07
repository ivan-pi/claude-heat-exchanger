#!/usr/bin/env bash
# Run both participants on this machine; they connect via TCP sockets on the loopback
# interface (m2n:sockets in precice-config.xml). Logs: fluid/fluid.log, solid/solid.log
set -e -u
cd "$(dirname "$0")"
./clean.sh
( fluid/run.sh > fluid/fluid.log 2>&1 ) &
PID_F=$!
( solid/run.sh > solid/solid.log 2>&1 ) &
PID_S=$!
STATUS=0
wait $PID_F || STATUS=$?
wait $PID_S || STATUS=$?
tail -n 3 fluid/fluid.log solid/solid.log
exit $STATUS
