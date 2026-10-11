#!/bin/bash
# Build the RED-SNOW TensorFold image on the head Spark from one commit of the tree (default: HEAD, or BRANCH),
# tagged redsnow-tensorfold:<commit12>-<recipe8> (recipe = Dockerfile, prebuild.py, healthcheck.py and
# the build args; -pip appended for TORCH_SOURCE=pip). Only committed work goes in: the tree is a `git archive`,
# so the image is reproducible from its tag; uncommitted changes under src/ are reported and left out.
#
#   ./build.sh                          # build the checked-out commit (HEAD)
#   ./build.sh --commit 24bf820         # another commit
#   TORCH_SOURCE=pip ./build.sh         # torch 2.11.0 cu130 on nvcr.io/nvidia/cuda:13.0.2-devel (the tested stack)
#   ./build.sh --check                  # Dockerfile lint only (docker build --check); builds nothing
#   ./build.sh --to-worker              # build (or reuse), then copy the image to the worker:
#                                       #   docker save | [zstd] | ssh $WORKER_SSH docker load (WORKER_SSH: .env)
#   ./build.sh --dry-run                # print what would run
# Also: --no-cache, --tag-latest (also tag redsnow-tensorfold:latest).
# Needs ~45 GB free under Docker's root on each Spark the first time (the NGC base is ~11.3 GB compressed; layers
# already stored, e.g. from another image on the same base, are not pulled again).
set -euo pipefail
source "$(dirname "$(readlink -f "$0")")/common.sh"

COMMIT_REF="$BRANCH"; CHECK=0; TO_WORKER=0; DRY=0; NOCACHE=(); LATEST=0
MAX_JOBS="${MAX_JOBS:-8}"
FREE_GB="${FREE_GB:-45}"
while (( $# )); do
  case "$1" in
    --commit) COMMIT_REF=$2; shift ;;
    --check) CHECK=1 ;;
    --to-worker) TO_WORKER=1 ;;
    --dry-run) DRY=1 ;;
    --no-cache) NOCACHE=(--no-cache) ;;
    --tag-latest) LATEST=1 ;;
    -h|--help) sed -n '2,/^set -euo/p' "$0" | sed '$d; s/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done
log() { printf '[build.sh] %s\n' "$*"; }
die() { printf '[build.sh] ERROR: %s\n' "$*" >&2; exit 1; }
run() { if (( DRY )); then printf '[dry-run] %s\n' "$*"; else "$@"; fi; }

[[ "$TORCH_SOURCE" =~ ^(base|pip)$ ]] || die "TORCH_SOURCE is base or pip, not $TORCH_SOURCE"
command -v docker >/dev/null || die "docker is not installed"
docker buildx version >/dev/null 2>&1 || die "docker buildx is needed (named build contexts)"

if (( CHECK )); then
  empty=$(mktemp -d); trap 'rm -rf -- "$empty"' EXIT
  log "docker build --check (lint only)"
  docker build --check --build-context tree="$empty" -f "$DOCKER_DIR/Dockerfile" "$DOCKER_DIR"
  exit 0
fi

COMMIT=$(git -C "$TREE_ROOT" rev-parse --verify "$COMMIT_REF^{commit}") || die "no commit $COMMIT_REF"
TAG=$(image_tag "$COMMIT")
RECIPE=$(recipe_hash)
log "commit $COMMIT ($COMMIT_REF), recipe $RECIPE, torch $TORCH_SOURCE on $BASE_IMAGE -> $TAG"
if [[ "$(git -C "$TREE_ROOT" rev-parse HEAD)" == "$COMMIT" ]] && [[ -n "$(git -C "$TREE_ROOT" status --porcelain -- src)" ]]; then
  log "note: uncommitted changes under src/ are NOT in the image (commit them first):"
  git -C "$TREE_ROOT" status --porcelain -- src | sed 's/^/    /'
fi

if docker image inspect "$TAG" >/dev/null 2>&1 && (( ${#NOCACHE[@]} == 0 )); then
  log "$TAG already built"
else
  root=$(docker info -f '{{.DockerRootDir}}' 2>/dev/null || echo /var/lib/docker)
  free=$(df -BG --output=avail "$root" | tail -1 | tr -dc '0-9')
  (( free >= FREE_GB )) || die "only ${free} GB free under $root, ${FREE_GB} GB wanted (FREE_GB)"
  docker image inspect "$BASE_IMAGE" >/dev/null 2>&1 ||
    log "base $BASE_IMAGE is not tagged here: docker build pulls it (only layers not already stored are downloaded)"
  ctx=$(mktemp -d "${TMPDIR:-/var/tmp}/redsnow-docker.XXXXXX")
  trap 'rm -rf -- "$ctx"' EXIT
  # the tree at COMMIT, nothing else (no .git, no venv, no untracked files)
  run bash -c "git -C '$TREE_ROOT' archive --format=tar '$COMMIT' | tar -x -C '$ctx'"
  run docker build "${NOCACHE[@]}" \
    --build-context tree="$ctx" \
    -f "$DOCKER_DIR/Dockerfile" \
    --build-arg BASE_IMAGE="$BASE_IMAGE" \
    --build-arg TORCH_SOURCE="$TORCH_SOURCE" \
    --build-arg TORCH_PIP_SPEC="$TORCH_PIP_SPEC" \
    --build-arg TORCH_INDEX="$TORCH_INDEX" \
    --build-arg EXTRAS="$EXTRAS" \
    --build-arg CUDA_ARCH="$CUDA_ARCH" \
    --build-arg MAX_JOBS="$MAX_JOBS" \
    --build-arg TF_COMMIT="$COMMIT" \
    --build-arg RECIPE_HASH="$RECIPE" \
    -t "$TAG" "$DOCKER_DIR"
  (( DRY )) || {
    log "built $TAG ($(docker image inspect -f '{{.Size}}' "$TAG" | awk '{printf "%.1f GB", $1/1e9}'))"
    docker run --rm --entrypoint cat "$TAG" /opt/tensorfold-ext/MANIFEST.txt | sed 's/^/    /'
  }
fi
(( LATEST )) && run docker tag "$TAG" "$IMAGE_REPO:latest"

if (( TO_WORKER )); then
  # Copy to the worker (WORKER_SSH). docker save streams every layer (~37 GB uncompressed) even when the worker
  # already stores the base; docker load keeps only the missing ones. zstd on both ends cuts the stream to roughly a
  # third. A registry (docker run registry:2 on the head; push; pull there) would send only the missing layers.
  wssh true || die "cannot ssh to $WORKER_SSH without a password"
  if [[ "$(worker_image_ident "$TAG")" == "$(image_ident "$TAG")" ]]; then
    log "the worker already has $TAG"; exit 0
  fi
  wroot=$(wssh "docker info -f '{{.DockerRootDir}}'" 2>/dev/null || echo /var/lib/docker)
  wfree=$(wssh "df -BG --output=avail '$wroot' | tail -1 | tr -dc '0-9'")
  (( wfree >= FREE_GB )) || die "only ${wfree} GB free under the worker's $wroot, ${FREE_GB} GB wanted"
  if command -v zstd >/dev/null && wssh 'command -v zstd >/dev/null'; then
    log "docker save $TAG | zstd | ssh $WORKER_SSH 'zstd -d | docker load'"
    run bash -c "docker save '$TAG' | zstd -T0 -3 -q | ssh -o BatchMode=yes '$WORKER_SSH' 'zstd -d -q | docker load'"
  else
    log "docker save $TAG | ssh $WORKER_SSH docker load"
    run bash -c "docker save '$TAG' | ssh -o BatchMode=yes '$WORKER_SSH' docker load"
  fi
  (( DRY )) || [[ "$(worker_image_ident "$TAG")" == "$(image_ident "$TAG")" ]] || die "the worker's $TAG differs from the head's"
  (( LATEST )) && run wssh "docker tag '$TAG' '$IMAGE_REPO:latest'"
  log "the worker has $TAG"
fi
log "image: $TAG    (start: IMAGE=$TAG deploy/redsnow/docker/start-docker.sh)"
