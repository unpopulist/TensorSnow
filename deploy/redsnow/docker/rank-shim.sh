#!/bin/bash
# Container shim for deploy/redsnow/start.sh. start-docker.sh installs this file as $SHIM_DIR/bin/tensorfold and
# $SHIM_DIR/bin/python on both Sparks (same path) next to $SHIM_DIR/shim.conf (IMAGE, CACHE_ROOT, CT_STOP), and runs
# start.sh with VENV=$SHIM_DIR. start.sh then keeps all its checks, the rank-1-first launch, the readiness wait, the
# smoke test and the watchdog unchanged, while each "rank process" it starts is this shim running one container:
#
#   tensorfold serve <model> ... --rank N ...  ->  one container redsnow-rankN (docker run --rm --init, GPU, RDMA)
#   python -c ...                              ->  docker run --rm --network none ... IMAGE python -c ...  (no GPU)
#   tensorfold --redsnow-mounts serve ...      ->  print the mount plan and check it (start-docker.sh's preflight)
#
# The model and drafter folders are mounted read-only at their real paths. A Hugging Face snapshot (e.g. a model
# folder that links into the HF cache) is passed by its resolved path and its whole models--* repo folder is mounted, so the
# snapshot's ../../blobs links resolve inside the container. TORCH_EXTENSIONS_DIR stays the image's baked one.
# --snapshot-dir (start.sh passes it only with the spill tier on, SPILL_GIB): that folder is mounted read-write at its
# real path; the tier's files take the folder's owner.
# SIGTERM/SIGINT/SIGHUP: docker stop -t CT_STOP (tini forwards TERM to tensorfold), then exit with its code.
set -uo pipefail
here=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)
# shellcheck source=/dev/null
source "$here/shim.conf" || { echo "shim: no $here/shim.conf" >&2; exit 2; }
: "${IMAGE:?}" "${CACHE_ROOT:?}" "${CT_STOP:=25}" "${LABEL:=redsnow.launcher=start-docker}" "${PREFIX:=redsnow-rank}"

# the environment start.sh gives a rank (COMMON_ENV + NCCL_IB_GID_INDEX), forwarded by name
fwd=()
while IFS='=' read -r n _; do
  [[ "$n" =~ ^(TF_[A-Z0-9_]+|TENSORFOLD_[A-Z0-9_]+|NCCL_[A-Z0-9_]+|HF_HUB_OFFLINE|PYTHONUNBUFFERED|MALLOC_ARENA_MAX|MAX_JOBS|CUDA_VISIBLE_DEVICES|CUDA_LAUNCH_BLOCKING)$ ]] &&
    fwd+=(-e "$n")
done < <(env)

if [[ "$(basename "$0")" == python ]]; then
  exec docker run --rm --network none --entrypoint python "${fwd[@]}" "$IMAGE" "$@"
fi

plan_only=0
[[ "${1:-}" == --redsnow-mounts ]] && { plan_only=1; shift; }
[[ "${1:-}" == serve ]] || exec docker run --rm --network none --entrypoint tensorfold "${fwd[@]}" "$IMAGE" "$@"

args=("$@")
rank=0 port="" mounts=() seen=" "
# mount_path <index in args>: resolve args[i], mount what it needs read-only, put the resolved path back
mount_path() {
  local i=$1 p=${args[$1]} r m t
  [[ "$p" == none || "$p" != /* ]] && return 0
  r=$(readlink -e -- "$p") || { echo "shim: $p does not exist on $(hostname)" >&2; return 1; }
  if [[ "$r" =~ ^(.*/models--[^/]+)/snapshots/[^/]+$ ]]; then m=${BASH_REMATCH[1]}; else m=$r; fi
  # every link inside must land under the mounted folder
  while IFS= read -r -d '' t; do
    t=$(readlink -e -- "$t") || { echo "shim: broken link under $r" >&2; return 1; }
    [[ "$t" == "$m"/* ]] || { echo "shim: $r links outside $m ($t); not mountable as one folder" >&2; return 1; }
  done < <(find "$r" -maxdepth 2 -type l -print0)
  args[i]=$r
  [[ "$seen" == *" $m "* ]] || { mounts+=(-v "$m:$m:ro"); seen+="$m "; }
}
mount_path 1 || exit 2                                    # serve <model>
for (( i = 2; i < ${#args[@]}; i++ )); do
  case "${args[i]}" in
    --rank) rank=${args[i+1]:-0} ;;
    --port) port=${args[i+1]:-} ;;
    --drafter|--chat-template) mount_path $((i + 1)) || exit 2 ;;
    --snapshot-dir)                                       # the spill tier's folder: read-write, only when passed
      d=${args[i+1]:-}
      [[ "$d" == /* && -d "$d" ]] || { echo "shim: --snapshot-dir $d is not a directory on $(hostname)" >&2; exit 2; }
      [[ "$seen" == *" $d "* ]] || { mounts+=(-v "$d:$d"); seen+="$d "; } ;;
  esac
done
[[ "$rank" =~ ^[0-9]+$ ]] || { echo "shim: bad --rank $rank" >&2; exit 2; }
name="$PREFIX$rank"
health=()
[[ "$rank" == 0 && -n "$port" ]] && health=(-e "REDSNOW_HEALTH_URL=http://127.0.0.1:$port/v1/models")
cache="$CACHE_ROOT/rank$rank"

if (( plan_only )); then
  printf 'container %s on %s, image %s\n' "$name" "$(hostname)" "$IMAGE"
  for (( i = 1; i < ${#mounts[@]}; i += 2 )); do printf '  mount %s\n' "${mounts[i]}"; done
  printf '  cache %s -> /cache (triton), %s/state -> /root/.cache/tensorfold\n' "$cache" "$cache"
  printf '  args  %s\n' "${args[*]}"
  exit 0
fi

# one container per rank name; a stopped one of ours is removed, a running one refused
if state=$(docker inspect -f "{{.State.Running}} {{index .Config.Labels \"${LABEL%%=*}\"}}" "$name" 2>/dev/null); then
  [[ "${state#* }" == "${LABEL#*=}" ]] || { echo "shim: a container named $name exists and is not ours" >&2; exit 2; }
  [[ "${state%% *}" == false ]] || { echo "shim: $name is already running (stop-docker.sh)" >&2; exit 2; }
  docker rm -f "$name" >/dev/null
fi
mkdir -p "$cache/triton" "$cache/state"

stopping=0
stop() { stopping=1; docker stop -t "$CT_STOP" "$name" >/dev/null 2>&1 || true; }
trap stop TERM INT HUP                                    # before the launch: no window without the handler
docker run --rm --init --name "$name" --label "$LABEL" \
  --gpus all --network host --ipc host \
  --device /dev/infiniband --cap-add IPC_LOCK \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$cache:/cache" -v "$cache/state:/root/.cache/tensorfold" \
  "${mounts[@]}" "${fwd[@]}" "${health[@]}" \
  "$IMAGE" tensorfold "${args[@]}" </dev/null &
child=$!
(( stopping )) && stop                                    # a signal that came before docker run existed
while :; do
  wait "$child"; code=$?
  kill -0 "$child" 2>/dev/null || break                  # wait returned for a signal: keep waiting for docker run
done
# asked to stop, but the container outlived its CLI (a TERM that reached docker run before the container existed)
(( stopping )) && docker rm -f "$name" >/dev/null 2>&1
exit "$code"
