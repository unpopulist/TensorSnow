# syntax=docker/dockerfile:1
# RED-SNOW on this TensorFold tree, one image for both Sparks (aarch64, GB10 sm_121).
# Build with ./build.sh, not by hand: it exports the tree at one commit (git archive) into the named build context
# "tree", and tags the image <commit>-<recipe hash>.
#
#   TORCH_SOURCE=base (default): NVIDIA's PyTorch container as is (torch 2.13.0a0 nv26.07, CUDA 13.3, NCCL 2.30.7,
#                    triton 3.7.1). TensorFold upstream and MiaAI-Lab's GLM image use this torch; nothing replaces it.
#   TORCH_SOURCE=pip: a CUDA devel base without torch (BASE_IMAGE=nvcr.io/nvidia/cuda:13.0.2-devel-ubuntu24.04) plus
#                    torch==2.11.0 from the cu130 index in /opt/venv (the bare-metal venv stack; less tested).
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:26.07-py3
FROM ${BASE_IMAGE}

ARG TORCH_SOURCE=base
ARG TORCH_PIP_SPEC="torch==2.11.0"
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu130
ARG EXTRAS="av==18.1.0 xgrammar>=0.2.8,<0.3"
ARG MAX_JOBS=8

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PATH=/opt/venv/bin:${PATH}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Toolchain for the RoCE proxy (gcc + libibverbs headers; NGC images ship both) and, for TORCH_SOURCE=pip, Python
# plus torch in a venv. apt runs only when something is missing.
# hadolint ignore=DL3008,DL3013
RUN set -eux; \
    need=(); \
    command -v gcc >/dev/null || need+=(gcc g++); \
    test -r /usr/include/infiniband/verbs.h || need+=(libibverbs-dev); \
    test -e /usr/lib/aarch64-linux-gnu/libmlx5.so.1 || need+=(ibverbs-providers); \
    if [ "$TORCH_SOURCE" = pip ]; then need+=(python3 python3-venv python3-dev git); fi; \
    if [ "${#need[@]}" -gt 0 ]; then \
      apt-get update; apt-get install -y --no-install-recommends "${need[@]}"; rm -rf /var/lib/apt/lists/*; \
    fi; \
    if [ "$TORCH_SOURCE" = pip ]; then \
      python3 -m venv /opt/venv; \
      pip install --upgrade pip; \
      pip install "$TORCH_PIP_SPEC" --index-url "$TORCH_INDEX"; \
      pip install ninja; \
    elif [ "$TORCH_SOURCE" != base ]; then echo "TORCH_SOURCE is base or pip, not $TORCH_SOURCE" >&2; exit 1; fi; \
    python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"; \
    command -v ninja; command -v nvcc || test -x /usr/local/cuda/bin/nvcc

# The tree at one commit (build.sh: git archive of HEAD or BRANCH). Installed as a regular package into
# site-packages; torch and triton are pinned to what is installed, so no dependency can replace NVIDIA's torch.
# ("tree" is a named build context, not a stage)
# hadolint ignore=DL3022
COPY --from=tree . /opt/tensorfold-src
# hadolint ignore=DL3013
RUN set -eux; \
    pip list --format=freeze | grep -iE '^(torch|triton)==' > /tmp/keep-torch.txt; \
    cat /tmp/keep-torch.txt; \
    pip install -c /tmp/keep-torch.txt "/opt/tensorfold-src" ${EXTRAS}; \
    diff <(pip list --format=freeze | grep -iE '^(torch|triton)==') /tmp/keep-torch.txt; \
    tensorfold --version; \
    python -c "import tensorfold.cuda.server, tensorfold.families.glm5_next.cuda.engine, xgrammar, av"

# Every CUDA extension the GLM path loads, plus the RoCE proxy, compiled now for sm_121 into a baked directory, so a
# container only runs ninja's up-to-date check (no nvcc at start). docker build has no GPU: prebuild.py answers
# torch's capability query with ${CUDA_ARCH}. The directory stays writable: torch takes a lock file in it.
ARG CUDA_ARCH=12.1
ENV TORCH_EXTENSIONS_DIR=/opt/tensorfold-ext \
    TRITON_CACHE_DIR=/cache/triton \
    HF_HUB_OFFLINE=1 \
    TENSORFOLD_NO_UPDATE_CHECK=1 \
    PYTHONUNBUFFERED=1
COPY prebuild.py healthcheck.py /opt/redsnow-docker/
RUN set -eux; \
    MAX_JOBS="${MAX_JOBS}" python /opt/redsnow-docker/prebuild.py --arch "${CUDA_ARCH}" \
      --manifest /opt/tensorfold-ext/MANIFEST.txt; \
    chmod -R a+rwX /opt/tensorfold-ext

# rank 0 only (start-docker.sh's shim sets REDSNOW_HEALTH_URL there): the API answers /v1/models. Other ranks: healthy
# while running. Loading takes minutes, hence the long start period.
HEALTHCHECK --interval=30s --timeout=10s --start-period=3600s --retries=3 CMD ["python", "/opt/redsnow-docker/healthcheck.py"]

ARG TF_COMMIT=unknown
ARG RECIPE_HASH=unknown
LABEL org.opencontainers.image.title="redsnow-tensorfold" \
      org.opencontainers.image.revision="${TF_COMMIT}" \
      org.opencontainers.image.source="TensorFold (https://github.com/ashhart/TensorFold) with the RED-SNOW port" \
      redsnow.tf.commit="${TF_COMMIT}" \
      redsnow.recipe="${RECIPE_HASH}" \
      redsnow.torch="${TORCH_SOURCE}" \
      redsnow.cuda_arch="${CUDA_ARCH}"
WORKDIR /workspace
