#!/bin/bash
# Stop the two TensorFold ranks that ./start.sh started (pids in $STATE_DIR/pids): SIGTERM to each rank's
# process group, SIGKILL after STOP_TIMEOUT seconds. A running start.sh notices and exits too.
# Touches only those pids, and only while they are still `tensorfold` processes. With the spill tier on (SPILL_GIB,
# recorded by start.sh) rank 0 stops first, while rank 1 still serves its shutdown flush, and STOP_TIMEOUT defaults to
# start.sh's.
# DRY_RUN=1: say what would be stopped.

set -euo pipefail
source "$(dirname "$(readlink -f "$0")")/env.sh"     # deploy/redsnow/.env, if any (the environment wins)

STATE_DIR="${STATE_DIR:-/tmp/tensorfold-redsnow}"
DRY_RUN="${DRY_RUN:-0}"
PIDS="$STATE_DIR/pids"

log() { printf '[stop.sh] %s\n' "$*"; }

[[ -f "$PIDS" ]] || { log "no $PIDS: nothing started by start.sh is recorded"; exit 0; }
rank0=$(sed -n 's/^rank0=//p' "$PIDS")
rank1=$(sed -n 's/^rank1=//p' "$PIDS")
worker=$(sed -n 's/^worker=//p' "$PIDS")
spill=$(sed -n 's/^spill=//p' "$PIDS")
recorded=$(sed -n 's/^stop_timeout=//p' "$PIDS")
[[ "$recorded" =~ ^[0-9]+$ ]] || recorded=30
STOP_TIMEOUT="${STOP_TIMEOUT:-$recorded}"
WORKER_SSH="${WORKER_SSH:-$worker}"               # the worker start.sh recorded, unless set
wssh() {
    [[ -n "$WORKER_SSH" ]] || { log "no worker recorded in $PIDS: set WORKER_SSH"; return 255; }
    ssh -o BatchMode=yes -o ConnectTimeout=10 "$WORKER_SSH" "$@"
}

if [[ "$DRY_RUN" == 1 ]]; then
    log "would stop rank 0 pid ${rank0:-none} here and rank 1 pid ${rank1:-none} on $WORKER_SSH"
    exit 0
fi

left=0
stop_rank1() {
    [[ "$rank1" =~ ^[0-9]+$ ]] || return 0
    # the remote side checks the pid is still TensorFold, then TERM, waits, KILL
    wssh "grep -q tensorfold /proc/$rank1/cmdline 2>/dev/null || exit 3
             kill -TERM -- -$rank1 2>/dev/null || kill -TERM $rank1
             for i in \$(seq $STOP_TIMEOUT); do kill -0 $rank1 2>/dev/null || exit 0; sleep 1; done
             kill -KILL -- -$rank1 2>/dev/null || kill -KILL $rank1" &
    wpid=$!
}
[[ "$spill" == 1 ]] || stop_rank1         # spill tier on: rank 1 only once rank 0 has flushed (below)
if [[ "$rank0" =~ ^[0-9]+$ ]] && grep -q tensorfold "/proc/$rank0/cmdline" 2>/dev/null; then
    kill -TERM -- "-$rank0" 2>/dev/null || kill -TERM "$rank0" 2>/dev/null || true
    for (( i = 0; i < STOP_TIMEOUT; i++ )); do
        kill -0 "$rank0" 2>/dev/null || break
        sleep 1
    done
    if kill -0 "$rank0" 2>/dev/null; then
        kill -KILL -- "-$rank0" 2>/dev/null || kill -KILL "$rank0" 2>/dev/null || true
    fi
    log "rank 0 (pid $rank0) stopped"
else
    log "rank 0 (pid ${rank0:-none}) is not running"
fi
[[ "$spill" != 1 ]] || stop_rank1
if [[ -n "${wpid:-}" ]]; then
    code=0; wait "$wpid" || code=$?
    case "$code" in
        0) log "rank 1 (pid $rank1 on $WORKER_SSH) stopped" ;;
        3) log "rank 1 (pid $rank1 on $WORKER_SSH) is not running" ;;
        *) log "could not reach or stop rank 1 on $WORKER_SSH (ssh status $code): check it by hand"; left=1 ;;
    esac
fi
(( left )) || rm -f "$PIDS"
