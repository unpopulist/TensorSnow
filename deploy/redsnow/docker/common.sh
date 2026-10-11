# Shared by build.sh, start-docker.sh and stop-docker.sh (sourced). Every value can be set in the environment or in
# deploy/redsnow/.env (copy .env.example; the environment wins).
DOCKER_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$DOCKER_DIR/../env.sh"
TREE_ROOT=$(cd "$DOCKER_DIR/../../.." && pwd)                 # this TensorFold checkout
IMAGE_REPO="${IMAGE_REPO:-redsnow-tensorfold}"
BRANCH="${BRANCH:-HEAD}"                                       # the commit build.sh builds by default
BASE_IMAGE="${BASE_IMAGE:-nvcr.io/nvidia/pytorch:26.07-py3}"
TORCH_SOURCE="${TORCH_SOURCE:-base}"                          # base: the NGC image's torch; pip: torch 2.11.0 cu130
if [[ "$TORCH_SOURCE" == pip && "$BASE_IMAGE" == nvcr.io/nvidia/pytorch:* ]]; then
  BASE_IMAGE="nvcr.io/nvidia/cuda:13.0.2-devel-ubuntu24.04"   # pip torch goes on a CUDA base without torch
fi
TORCH_PIP_SPEC="${TORCH_PIP_SPEC:-torch==2.11.0}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu130}"
EXTRAS="${EXTRAS:-av==18.1.0 xgrammar>=0.2.8,<0.3}"
CUDA_ARCH="${CUDA_ARCH:-12.1}"                                # GB10
WORKER_SSH="${WORKER_SSH:-}"                                 # required for the worker: user@host, key-based ssh
LABEL_KEY="redsnow.launcher"                                  # containers start-docker.sh owns carry this label
CONTAINER_PREFIX="${CONTAINER_PREFIX:-redsnow-rank}"           # redsnow-rank0 (head), redsnow-rank1 (worker)

# recipe_hash: what besides the commit goes into the image (Dockerfile, prebuild/health scripts, build args)
recipe_hash() {
  { cat "$DOCKER_DIR/Dockerfile" "$DOCKER_DIR/prebuild.py" "$DOCKER_DIR/healthcheck.py"
    printf '%s\n' "$BASE_IMAGE" "$TORCH_SOURCE" "$TORCH_PIP_SPEC" "$TORCH_INDEX" "$EXTRAS" "$CUDA_ARCH"; } |
    sha256sum | cut -c1-8
}
# image_tag <commit-ish>: repo:<commit12>-<recipe8>[-pip]
image_tag() {
  local c; c=$(git -C "$TREE_ROOT" rev-parse --verify "$1^{commit}") || return 1
  echo "$IMAGE_REPO:${c:0:12}-$(recipe_hash)$([[ "$TORCH_SOURCE" == pip ]] && echo -pip)"
}
# An image's identity by content (layers + runtime config), equal across Docker's image stores (Mia's issue #8)
IMAGE_IDENT='{{.RootFS.Layers}}{{.Config.Env}}{{.Config.Entrypoint}}{{.Config.Cmd}}{{.Config.WorkingDir}}'
image_ident() { local s; s=$(docker image inspect -f "$IMAGE_IDENT" "$1" 2>/dev/null) && sha256sum <<<"$s" | cut -c1-64 || echo missing; }
wssh() {
  [[ -n "$WORKER_SSH" && "$WORKER_SSH" != *[\<\>]* ]] ||
    { echo "set WORKER_SSH in deploy/redsnow/.env (copy .env.example) or in the environment" >&2; return 255; }
  ssh -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=15 "$WORKER_SSH" "$@"
}
worker_image_ident() {
  local s; s=$(wssh "docker image inspect -f '$IMAGE_IDENT' '$1'" 2>/dev/null) && sha256sum <<<"$s" | cut -c1-64 || echo missing
}
