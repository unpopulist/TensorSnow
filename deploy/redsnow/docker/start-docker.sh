#!/bin/bash
# Serve RED-SNOW from the redsnow-tensorfold image instead of the venv: the container mode of deploy/redsnow/start.sh.
#
# start.sh does the work. This wrapper checks what only the container mode needs, installs rank-shim.sh on both Sparks
# as $SHIM_DIR/bin/{tensorfold,python} (same path on each, the image baked into shim.conf) and runs start.sh with
# VENV=$SHIM_DIR, so every refusal (port 8000 / rendezvous port taken, MemAvailable under 110 GiB, GIDs, ping, model
# folders, drafter, argument parse), the rank-1-first launch, the readiness wait, the smoke test and the watchdog are
# start.sh's own. Each rank is one container (redsnow-rank0 here, redsnow-rank1 on the worker); see rank-shim.sh.
#
#   IMAGE=redsnow-tensorfold:<tag> ./start-docker.sh          # default: the tag build.sh gives HEAD (BRANCH)
#   DRY_RUN=1 ./start-docker.sh                                # all checks, print both rank commands, start nothing
#   ./start-docker.sh --reasoning-effort high                  # extra arguments go to tensorfold serve (start.sh)
# Every start.sh setting (CONTEXT, COMM, DRAFTER, MODEL_RANK0/1, PORT, ...) passes through the environment or
# deploy/redsnow/.env (copy .env.example). Required: HEAD_ADDR, WORKER_ADDR, WORKER_SSH.
# Container-only settings: IMAGE, SHIM_DIR, CACHE_ROOT (per-image triton cache and prefix snapshots on each node).
# Stop with ./stop-docker.sh (start.sh's stop.sh plus removal of any container of ours left behind).
set -euo pipefail
source "$(dirname "$(readlink -f "$0")")/common.sh"

START_SH="${START_SH:-$TREE_ROOT/deploy/redsnow/start.sh}"
IMAGE="${IMAGE:-$(image_tag "$BRANCH")}"
SHIM_DIR="${SHIM_DIR:-$HOME/.local/share/redsnow-docker}"   # same absolute path on both nodes, outside any git tree
# the spill tier (SPILL_GIB, start.sh): a clean stop's flush needs SPILL_FLUSH_S, so the containers get its time as well
if [[ "${SPILL_GIB:-0}" =~ ^0*([.]0*)?$ ]]; then
  STOP_TIMEOUT="${STOP_TIMEOUT:-30}"
else
  _flush="${SPILL_FLUSH_S:-60}"
  STOP_TIMEOUT="${STOP_TIMEOUT:-$(( ${_flush%.*} + 30 ))}"
fi
DRY_RUN="${DRY_RUN:-0}"
MODEL_RANK0="${MODEL_RANK0:-${MODEL_DIR:-/var/tmp/models/RED-SNOW-5.3-FLASH-EXL3-SAGE-4.91bpw}}"   # same defaults
MODEL_RANK1="${MODEL_RANK1:-${RANK1_MODEL_DIR:-/var/tmp/models/redsnow-rank1}}"                     # as start.sh
DRAFTER="${DRAFTER:-${DRAFTER_DIR:-/var/tmp/models/GLM-5.3-Flash-DFlash2}}"

log()    { printf '[start-docker.sh] %s\n' "$*"; }
refuse() { printf '[start-docker.sh] refusing: %s\n' "$*" >&2; exit 1; }

[[ -x "$START_SH" ]] || refuse "no $START_SH"
command -v docker >/dev/null || refuse "docker is not installed here"
for _v in HEAD_ADDR WORKER_ADDR WORKER_SSH; do
  [[ -n "${!_v:-}" && "${!_v}" != *[\<\>]* ]] || refuse "set $_v in deploy/redsnow/.env (copy .env.example) or in the environment"
done
wssh true 2>/dev/null || refuse "cannot ssh to $WORKER_SSH without a password"
[[ "$(wssh 'echo $HOME')" == "$HOME" ]] || refuse "the worker's HOME differs from $HOME: set SHIM_DIR to a path valid on both"

# ---------------------------------------------------------------- image on both nodes, same content
docker image inspect "$IMAGE" >/dev/null 2>&1 || refuse "image $IMAGE missing here (deploy/redsnow/docker/build.sh)"
here_id=$(image_ident "$IMAGE"); there_id=$(worker_image_ident "$IMAGE")
[[ "$there_id" != missing ]] || refuse "image $IMAGE missing on the worker (build.sh --to-worker)"
[[ "$here_id" == "$there_id" ]] || refuse "the worker's $IMAGE differs from this one (build.sh --to-worker)"
commit=$(docker image inspect -f '{{index .Config.Labels "redsnow.tf.commit"}}' "$IMAGE")
log "image $IMAGE (TensorFold commit ${commit:0:12}, identical on both nodes)"
for node in here worker; do
  cmd='docker info 2>/dev/null | grep -qi nvidia && test -e /dev/infiniband/uverbs0'
  if [[ $node == here ]]; then bash -c "$cmd"; else wssh "$cmd"; fi ||
    refuse "$node: docker has no nvidia runtime, or /dev/infiniband/uverbs0 is missing"
done
# a running container of ours means a pair is up (or half up): stop-docker.sh first
for node in here worker; do
  q="docker ps -q --filter label=$LABEL_KEY=start-docker"
  up=$(if [[ $node == here ]]; then $q; else wssh "$q"; fi) || refuse "cannot list containers ($node)"
  [[ -z "$up" ]] || refuse "a $CONTAINER_PREFIX* container is running ($node): ./stop-docker.sh first"
done

# ---------------------------------------------------------------- the shim on both nodes
tag=${IMAGE##*:}; [[ "$IMAGE" == *:* ]] || tag=latest
CACHE_ROOT="${CACHE_ROOT:-$HOME/.cache/redsnow-docker/$tag}"
conf=$(printf 'IMAGE=%q\nCACHE_ROOT=%q\nCT_STOP=%q\nLABEL=%q\nPREFIX=%q\n' \
  "$IMAGE" "$CACHE_ROOT" "$(( STOP_TIMEOUT > 10 ? STOP_TIMEOUT - 5 : 5 ))" "$LABEL_KEY=start-docker" "$CONTAINER_PREFIX")
install_shim='set -e; d=$1; mkdir -p "$d/bin" "$d/torch_ext/roce"; cat > "$d/bin/.shim.new"
  chmod 755 "$d/bin/.shim.new"; cp "$d/bin/.shim.new" "$d/bin/python"; mv "$d/bin/.shim.new" "$d/bin/tensorfold"
  printf "%s\n" "$2" > "$d/shim.conf"; cat "$d/bin/tensorfold" "$d/bin/python" "$d/shim.conf" | sha256sum | cut -c1-16'
sum_here=$(bash -c "$install_shim" _ "$SHIM_DIR" "$conf" < "$DOCKER_DIR/rank-shim.sh")
sum_there=$(wssh "bash -c $(printf '%q' "$install_shim") _ $(printf '%q' "$SHIM_DIR") $(printf '%q' "$conf")" < "$DOCKER_DIR/rank-shim.sh")
[[ "$sum_here" == "$sum_there" ]] || refuse "the shim differs between the nodes ($sum_here / $sum_there)"
log "shim $sum_here in $SHIM_DIR on both nodes; caches in $CACHE_ROOT"

# the served chat template (start.sh's default is $TREE/deploy/redsnow/chat_template.jinja, TREE=SHIM_DIR below):
# this tree's copy on both nodes; the shim mounts the --chat-template file into each container
tpl_src="$TREE_ROOT/deploy/redsnow/chat_template.jinja"
if [[ -z "${CHAT_TEMPLATE:-}" && -f "$tpl_src" ]]; then
  mkdir -p "$SHIM_DIR/deploy/redsnow" && cp "$tpl_src" "$SHIM_DIR/deploy/redsnow/chat_template.jinja"
  wssh "mkdir -p $(printf '%q' "$SHIM_DIR/deploy/redsnow") && cat > $(printf '%q' "$SHIM_DIR/deploy/redsnow/chat_template.jinja")" < "$tpl_src" ||
    refuse "cannot copy the chat template to the worker"
fi

# mount plan per rank, checked where the rank runs (paths must exist; links must stay inside the mounted folder)
tf="$SHIM_DIR/bin/tensorfold"
"$tf" --redsnow-mounts serve "$MODEL_RANK0" --rank 0 --drafter "$DRAFTER" | sed 's/^/  rank 0: /' ||
  refuse "rank 0's folders cannot be mounted (above)"
wssh "$(printf '%q ' "$tf" --redsnow-mounts serve "$MODEL_RANK1" --rank 1 --drafter "$DRAFTER")" | sed 's/^/  rank 1: /' ||
  refuse "rank 1's folders cannot be mounted (above)"

# ---------------------------------------------------------------- start.sh, with the shim as its "venv"
# TREE=SHIM_DIR: start.sh's tree fingerprint compares two non-git folders (equal); the real identity check is the image
# above. TORCH_EXT only satisfies start.sh's COMM=roce host check (the containers use the image's baked directory).
backstop() {   # any container of ours that start.sh's kills left behind (a KILLed shim leaves its container running)
  docker rm -f $(docker ps -aq --filter "label=$LABEL_KEY=start-docker") >/dev/null 2>&1 || true
  wssh "docker rm -f \$(docker ps -aq --filter label=$LABEL_KEY=start-docker) >/dev/null 2>&1 || true" || true
}
[[ "$DRY_RUN" == 1 ]] || trap backstop EXIT
export VENV="$SHIM_DIR" TREE="$SHIM_DIR" TORCH_EXT="$SHIM_DIR/torch_ext"
export MODEL_RANK0 MODEL_RANK1 DRAFTER STOP_TIMEOUT DRY_RUN WORKER_SSH IMAGE
code=0
"$START_SH" "$@" & child=$!
# a background job ignores SIGINT, so Ctrl-C reaches start.sh as SIGTERM (its trap stops both ranks either way)
trap 'kill -TERM "$child" 2>/dev/null || true' TERM HUP INT
while :; do
  wait "$child" && code=0 || code=$?
  kill -0 "$child" 2>/dev/null || break
done
exit "$code"
