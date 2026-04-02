#!/bin/bash
# Sync RoboWheel-Experiment.md between sim/docs and robowheel-demo
# Usage: bash sync_docs.sh [to-sim|to-demo]

SIM_DOC="/mnt/users/yjy/sim/docs/RoboWheel-Experiment.md"
DEMO_DOC="/mnt/users/yjy/robowheel-demo/RoboWheel-Experiment.md"

case "${1:-to-sim}" in
  to-sim)
    cp "$DEMO_DOC" "$SIM_DOC"
    echo "Synced: robowheel-demo → sim/docs"
    ;;
  to-demo)
    cp "$SIM_DOC" "$DEMO_DOC"
    echo "Synced: sim/docs → robowheel-demo"
    ;;
  *)
    echo "Usage: $0 [to-sim|to-demo]"
    ;;
esac
