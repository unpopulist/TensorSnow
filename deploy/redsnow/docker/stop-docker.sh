#!/bin/bash
# Stop a pair started by start-docker.sh: start.sh's stop.sh first (SIGTERM to each rank's shim, whose trap runs
# docker stop; SIGKILL after STOP_TIMEOUT), then remove any container labelled redsnow.launcher=start-docker that is
# still there on either node (a KILLed shim leaves its container running). Touches no other container.
# DRY_RUN=1: say what would be stopped.
set -euo pipefail
source "$(dirname "$(readlink -f "$0")")/common.sh"
log() { printf '[stop-docker.sh] %s\n' "$*"; }

"$TREE_ROOT/deploy/redsnow/stop.sh" || log "stop.sh reported a problem (above); removing containers anyway"
q="docker ps -aq --filter label=$LABEL_KEY=start-docker"
mapfile -t here < <($q || true)
there=$(wssh "$q" 2>/dev/null) || { log "cannot reach $WORKER_SSH: check rank 1 there by hand ($q)"; there=""; }
mapfile -t there < <(printf '%s' "$there" | grep . || true)
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  log "would remove here: ${here[*]:-none}; on the worker: ${there[*]:-none}"; exit 0
fi
if (( ${#here[@]} )); then
  docker stop -t "${STOP_TIMEOUT:-30}" "${here[@]}" >/dev/null 2>&1 || true
  docker rm -f "${here[@]}" >/dev/null 2>&1 || true
  log "removed container(s) here: ${here[*]}"
fi
if (( ${#there[@]} )); then
  wssh "docker stop -t ${STOP_TIMEOUT:-30} ${there[*]} >/dev/null 2>&1; docker rm -f ${there[*]} >/dev/null 2>&1" || true
  log "removed container(s) on the worker: ${there[*]}"
fi
log "done"
