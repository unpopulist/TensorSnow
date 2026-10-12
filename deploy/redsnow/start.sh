#!/bin/bash
# Serve RED-SNOW-5.3-FLASH-EXL3-SAGE-4.91bpw on :8000 with this TensorFold tree, two ranks over two DGX Sparks' RoCE
# link (tensor parallel, TP=2): rank 1 on the worker first, then rank 0 here (rank 0 serves the OpenAI API).
# Refuses while port 8000 (or the rendezvous port) is taken, while either Spark has under 110 GiB
# MemAvailable, or when a check below fails. It stops nothing that is already serving.
# Stays in the foreground: watches both ranks, stops both if either dies, and stops both on exit
# (Ctrl-C, SIGTERM, or an error). Run it under tmux or nohup. ./stop.sh stops a running pair.
# docker/start-docker.sh runs this script with each rank in a container (the recommended way; README.md).
#
# Site settings come from deploy/redsnow/.env (copy .env.example; REDSNOW_ENV names another file) or the
# environment, which wins over the file. Required: HEAD_ADDR, WORKER_ADDR, WORKER_SSH. Examples:
#   CONTEXT=131072 ./start.sh
#   MODEL_RANK1=/path/to/full/pack ./start.sh      (the full pack on the worker instead of the rank-1 folder)
#   ./start.sh --reasoning-effort high             # extra arguments go to `tensorfold serve` on both ranks
# DRY_RUN=1 runs every check and prints both commands, then exits without starting anything.

set -euo pipefail
HERE=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
source "$HERE/env.sh"

# ---------------------------------------------------------------- cluster (see README.md)
HEAD_ADDR="${HEAD_ADDR:-}"                        # required: rank 0 on the RoCE link (--master, NCCL bootstrap)
WORKER_ADDR="${WORKER_ADDR:-}"                    # required: rank 1 on the RoCE link (GID check, ping)
WORKER_SSH="${WORKER_SSH:-}"                      # required: user@host, key-based ssh from here to the worker
IFACE="${IFACE:-enp1s0f0np0}"                     # NCCL_SOCKET_IFNAME on both nodes (the netdev holding the addresses)
HCA="${HCA:-rocep1s0f0}"                          # NCCL_IB_HCA on both nodes (one rail)
HEAD_GID="${HEAD_GID:-}"                          # NCCL_IB_GID_INDEX per node: empty = the RoCE v2 entry of the
WORKER_GID="${WORKER_GID:-}"                      #   node's address, looked up in sysfs; a set value is verified
NCCL_CHANNELS="${NCCL_CHANNELS:-4}"               # NCCL_MIN/MAX_NCHANNELS (4 measured best on Sparks)
MASTER_PORT="${MASTER_PORT:-29551}"               # TensorFold's rendezvous (TCPStore on rank 0)
# The ranks' small all-gathers (decode partials, samplers): roce = one-shot RDMA writes over $HCA (b12x's transport,
# MiaAI-Lab's patch 0006: 11 us a 16 KiB gather against NCCL's 45), larger ones over NCCL; nccl: all.
# Same bits either way. The RoCE devices and GID are NCCL_IB_HCA's and NCCL_IB_GID_INDEX's below.
COMM="${COMM:-roce}"
ROCE_MAX_KB="${ROCE_MAX_KB:-512}"                 # TF_ROCE_MAX_KB: largest gather over RoCE
ROCE_WAIT_S="${ROCE_WAIT_S:-300}"                 # TF_ROCE_WAIT_S: a late peer costs a slow round, not both ranks

# ---------------------------------------------------------------- software and weights
TREE="${TREE:-$(cd "$HERE/../.." && pwd)}"       # this tree; the same path on both nodes
VENV="${VENV:-$TREE/.venv}"
TORCH_EXT="${TORCH_EXT:-$TREE/.torch_extensions}" # TORCH_EXTENSIONS_DIR (JIT kernels, per tree)
MODEL_RANK0="${MODEL_RANK0:-${MODEL_DIR:-/var/tmp/models/RED-SNOW-5.3-FLASH-EXL3-SAGE-4.91bpw}}"   # full pack (head)
MODEL_RANK1="${MODEL_RANK1:-${RANK1_MODEL_DIR:-/var/tmp/models/redsnow-rank1}}"   # split.py --rank 1 folder (worker)
DRAFTER="${DRAFTER:-${DRAFTER_DIR:-/var/tmp/models/GLM-5.3-Flash-DFlash2}}"      # same path on both; "none": MTP drafts

# ---------------------------------------------------------------- serving
SERVED_NAME="${SERVED_NAME:-RED-SNOW-5.3-FLASH}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
# Prompt + reply window. ~91.6 GiB of target weights per rank (MTP head left out) + ~0.6 GiB DFlash2;
# the engine's estimate for the rest is ~8.1 GiB at 65,536 tokens, ~9.2 at 131,072, ~11.4 at 262,144.
# An explicit window that does not fit is refused on both ranks at start. 0: the largest that fits.
CONTEXT="${CONTEXT:-327680}"
# TF_GLM_POOL_STREAMS: how many streams' full windows the shared cache pool holds. 1 (like Mia's recipe): one request
# can use the whole 320K window and the 4 streams share that pool (when it is full, the youngest stream is given back
# to the queue and replayed); 4: every stream keeps a full window reserved (then CONTEXT must fall to ~64K)
POOL_STREAMS="${POOL_STREAMS:-1}"
VISION="${VISION:-1}"             # 1: --vision, image and video input (rank 0 runs the tower: ~2 GiB of its window)
VISION_URLS="${VISION_URLS:-0}"   # 1: --vision-urls, public https image URLs too (default: data URLs)
KV_DTYPE="${KV_DTYPE:-fp8}"       # DSA latent cache: bf16 (exact) or fp8 (--kv-dtype fp8 = TF_GLM_KV=fp8: lossy, longer window)
PARALLEL="${PARALLEL:-4}"         # >1: concurrent requests share decode rounds (multi.py, 1 to 4 here); the pool holds
                                  # every stream's full CONTEXT unless POOL_STREAMS=1 (above), else start refuses
MAX_TOKENS="${MAX_TOKENS:-32768}" # reply budget when a request sets none
THINKING="${THINKING:-1}"
# Default reasoning effort when a request omits one. low/high render as
# Low/High, everything else (medium, xhigh, max) as Max. Empty: no default (the template's own). CLI --reasoning-effort wins.
REASONING_EFFORT="${REASONING_EFFORT:-high}"
# Chat template served instead of the checkpoint's (--chat-template): a clean, standard GLM-5.3-Flash template with
# the effort line, no embedded system prompt. The shared checkpoint on disk is never modified. "none": the checkpoint's own.
CHAT_TEMPLATE="${CHAT_TEMPLATE:-$TREE/deploy/redsnow/chat_template.jinja}"
# Memory TensorFold leaves free at start. The server peaks a few GiB past its own estimate (PARALLEL=4 takes
# ~3.7 GiB more); 14.5 is MiaAI-Lab's recipe value and leaves ~6-9 GiB free per node under load at 320K.
MEMORY_RESERVE_GIB="${MEMORY_RESERVE_GIB:-14.5}"
KEPT_GIB="${KEPT_GIB:-2}"         # TF_GLM_CACHE_GIB: other conversations' kept prompts (what is left, at most)
# MiaAI-Lab's recipe defaults for what this tree has ported; each can be set to 0 / another value:
SPLIT="${SPLIT:-1}"               # TF_GLM_HC_SPLIT=1 + TF_GLM_PREFILL_OVERLAP=2: prompt glue split by rows, overlapped
# TF_GLM_KDA_CHUNKED: chunked (WY) KDA prompt kernel (close to serial, not its bits), Mia's default 1. Prompt chunks and
# kept prompt states then sit on a 64-token grid, at any PARALLEL (each stream equals its PARALLEL=1 reply with it on).
KDA_CHUNKED="${KDA_CHUNKED:-1}"
MULTI_PREFILL="${MULTI_PREFILL:-1}"   # TF_GLM_MULTI_PREFILL: waiting prompts share prompt chunks
FILL_BUDGET_MS="${FILL_BUDGET_MS:-200}"   # TF_GLM_FILL_BUDGET_MS: prompt chunks in layer slices while others decode
FILL_DRAFTS="${FILL_DRAFTS:-1}"   # TF_GLM_FILL_DRAFTS: drafted decode rounds between those slices
# TF_GLM_FILL_ROWS (BadAd84's FILL_ROWS in Mia's recipe): a prompt chunk's rows while others decode. 2048 (TensorFold's
# own default is 1024) is the whole prompt chunk, whose buffers already exist: the same memory, the same replies, half
# the passes over the expert weights. At most the prompt chunk (the engine refuses more; this tree's is 2048,
# decode.PREFILL_ROWS); a smaller TF_GLM_PREFILL_ROWS lowers the default with it, as in Mia's config.sh. 1024: as before.
_fill_max="${TF_GLM_PREFILL_ROWS:-2048}"
FILL_ROWS="${FILL_ROWS:-${TF_GLM_FILL_ROWS:-$(( _fill_max < 2048 ? _fill_max : 2048 ))}}"
STREAM_SMOOTH="${STREAM_SMOOTH:-1}"   # TF_GLM_STREAM_SMOOTH (+ _MS 400): steadier token pacing for streamed replies
CACHE_ENTRIES="${CACHE_ENTRIES:-32}"  # TF_GLM_CACHE_ENTRIES: kept prompt states
MTP_HEAD="${MTP_HEAD:-auto}"      # TF_GLM_MTP: auto leaves the 1.8 GiB MTP head out beside DFlash2
# drafts (Mia's 0007/0013/0018/0032 recipe values; drafts only propose, so replies are the same under any of them):
COPY="${COPY:-1}"                 # TF_GLM_COPY_DRAFTS: prompt-lookup drafts ahead of DFlash2's
COPY_MAX="${COPY_MAX:-15}"        # TF_GLM_COPY_MAX: copy drafts a round (16-row verify windows)
DRAFT_POLICY="${DRAFT_POLICY:-fnc7:0.3}"   # TF_GLM_DFLASH_POLICY: up to 7 DFlash2 drafts, cut at noisy chance 0.3
COPY_CODE="${COPY_CODE:-1}"       # TF_GLM_WIDE_GRAPHS=16 + TF_GLM_COPY_REPLY_MATCH=16 (0: both off)
L2PF="${L2PF:-1}"                 # TF_GLM_L2PF: next weights into L2 during decode all-gathers (Mia's 0046)
# TF_GLM_EXL3_DEC_ORDER (Mia's 0090, by Lukasz Raczylo): the decode expert kernel's block launch order, the same bits
# for every value. 0: as launched (the old order); 1: the 128-column blocks of one k slice together, then the experts;
# 2: then the matrices, splits and member tiles. Default 2: on RED-SNOW (2026-10-12, six-prompt drafted decode, thinking
# on) 2 was 0.4-1.2% faster than 0 on every prompt, undrafted decode unchanged.
EXL3_DEC_ORDER="${EXL3_DEC_ORDER:-2}"
[[ "$EXL3_DEC_ORDER" =~ ^[012]$ ]] || { printf '[start.sh] refusing: EXL3_DEC_ORDER is 0, 1 or 2, not %s\n' "$EXL3_DEC_ORDER" >&2; exit 1; }
SHARED_PREFIX="${SHARED_PREFIX:-1}"   # TF_GLM_SHARED_PREFIX: keep the shared system prompt's state (Mia's 0015)
# TF_GLM_MULTI_LONE (Mia's 0035): a lone request moves to the pool's first rows for the one-stream graphs, evicting the
# kept prompts there; off, as in MiaAI-Lab's recipe
MULTI_LONE="${MULTI_LONE:-0}"
# Spill tier (Mia's 0088, off by default as in her recipe): kept prompt states go to local disk on each Spark and come
# back instead of a new prefill, also after a clean restart (--spill-gib, --snapshot-dir). SPILL_GIB: the cap per Spark
# (0: off). SPILL_DIR: the same absolute path on both Sparks, on local disk, owned by you (never a model or checkpoint
# folder); created on both at start. SPILL_HIGHWATER: past this fraction of the KV pool the kept prompts eviction would
# take next are written in the background (1.0: only when evicted). SPILL_MIN_TOKENS: shorter prompts are not written.
# SPILL_MIN_FREE_GIB: no write leaves less disk free. A clean stop writes what is kept within SPILL_FLUSH_S seconds
# (rank 0 first, rank 1 stays up for it); STOP_TIMEOUT then defaults to SPILL_FLUSH_S + 30.
SPILL_GIB="${SPILL_GIB:-0}"
SPILL_DIR="${SPILL_DIR:-$HOME/.cache/tensorfold-spill}"
SPILL_HIGHWATER="${SPILL_HIGHWATER:-0.70}"
SPILL_MIN_TOKENS="${SPILL_MIN_TOKENS:-8192}"
SPILL_MIN_FREE_GIB="${SPILL_MIN_FREE_GIB:-50}"
SPILL_FLUSH_S="${SPILL_FLUSH_S:-60}"
[[ "$SPILL_GIB" =~ ^[0-9]+([.][0-9]+)?$ ]] || { printf '[start.sh] refusing: SPILL_GIB is a number of GiB (0: off), not %s\n' "$SPILL_GIB" >&2; exit 1; }
SPILL=0
[[ "$SPILL_GIB" =~ ^0*([.]0*)?$ ]] || SPILL=1
if (( SPILL )); then
  [[ "$SPILL_FLUSH_S" =~ ^[0-9]+([.][0-9]+)?$ ]] || { printf '[start.sh] refusing: SPILL_FLUSH_S is seconds, not %s\n' "$SPILL_FLUSH_S" >&2; exit 1; }
  STOP_TIMEOUT="${STOP_TIMEOUT:-$(( ${SPILL_FLUSH_S%.*} + 30 ))}"
fi

# ---------------------------------------------------------------- launcher
MIN_GIB="${MIN_GIB:-110}"         # MemAvailable floor on each node
# The 320K window fits within ~0.5 GiB at a 14.5 GiB reserve: right after a large image load or file copy, free memory
# sits a little low for a while and the engine would refuse. Wait up to SETTLE_S seconds for both nodes to reach
# SETTLE_GIB MemAvailable first (then start anyway; the engine's own check decides).
SETTLE_GIB="${SETTLE_GIB:-115.7}"
SETTLE_S="${SETTLE_S:-180}"
WAIT_TIMEOUT="${WAIT_TIMEOUT:-3600}"   # seconds until the API must answer (first start compiles kernels)
STOP_TIMEOUT="${STOP_TIMEOUT:-30}"
SMOKE="${SMOKE:-1}"               # one short chat completion once the API answers
STATE_DIR="${STATE_DIR:-/tmp/tensorfold-redsnow}"
LOG0="${LOG0:-/tmp/tensorfold-redsnow-rank0.log}"
LOG1="${LOG1:-/tmp/tensorfold-redsnow-rank1.log}"   # on the worker
ALLOW_TREE_MISMATCH="${ALLOW_TREE_MISMATCH:-0}"
DRY_RUN="${DRY_RUN:-0}"

TF="$VENV/bin/tensorfold"
PY="$VENV/bin/python"
PIDS="$STATE_DIR/pids"
RANK0_PID=""
RANK1_PID=""

log()    { printf '[start.sh] %s\n' "$*"; }
refuse() { printf '[start.sh] refusing: %s\n' "$*" >&2; exit 1; }
wssh()   { ssh -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=15 "$WORKER_SSH" "$@"; }
for _v in HEAD_ADDR WORKER_ADDR WORKER_SSH; do
    [[ -n "${!_v}" && "${!_v}" != *[\<\>]* ]] ||
        refuse "set $_v in deploy/redsnow/.env (copy .env.example) or in the environment"
done

# ---------------------------------------------------------------- helpers also sent to the worker
mem_kib() { awk '/^MemAvailable:/ {print $2}' /proc/meminfo; }
# gid_ok <hca> <index> <ipv4>: the GID at <index> holds <ipv4> and is RoCE v2
gid_ok() {
    local want gid type
    # shellcheck disable=SC2046
    want=$(printf '0000:0000:0000:0000:0000:ffff:%02x%02x:%02x%02x' $(tr '.' ' ' <<<"$3"))
    gid=$(cat "/sys/class/infiniband/$1/ports/1/gids/$2" 2>/dev/null) || return 1
    type=$(cat "/sys/class/infiniband/$1/ports/1/gid_attrs/types/$2" 2>/dev/null) || return 1
    [[ "$gid" == "$want" && "$type" == *v2* ]]
}
# gid_find <hca> <ipv4>: the index of <hca>'s RoCE v2 GID that holds <ipv4>, on stdout
gid_find() {
    local f
    for f in "/sys/class/infiniband/$1/ports/1/gids/"*; do
        gid_ok "$1" "${f##*/}" "$2" && { echo "${f##*/}"; return 0; }
    done
    return 1
}
# model_kind <dir> <rank>: "split N" (a pre-split rank folder), "full N" (every shard of the index), or a
# reason it cannot serve <rank>, on stdout; status 1 for the latter
model_kind() {
    local dir=$1 rank=$2 other=$((1 - $2)) shards n missing=0 f
    [[ -f "$dir/config.json" ]] || { echo "no config.json in $dir"; return 1; }
    [[ -f "$dir/model.safetensors.index.json" ]] || { echo "no model.safetensors.index.json in $dir"; return 1; }
    shards=$(grep -o '"model-[0-9]*-of-[0-9]*\.safetensors"' "$dir/model.safetensors.index.json" | sort -u | tr -d '"')
    n=$(wc -l <<<"$shards")
    if compgen -G "$dir/*.rank$rank.safetensors" >/dev/null; then
        f=$(find "$dir" -maxdepth 1 -name "*.rank$rank.safetensors" | wc -l)
        [[ "$f" == "$n" ]] || { echo "$dir has $f of $n rank-$rank files (split incomplete?)"; return 1; }
        echo "split $f"; return 0
    fi
    if compgen -G "$dir/*.rank$other.safetensors" >/dev/null; then
        echo "$dir holds rank $other's split files only"; return 1
    fi
    for f in $shards; do [[ -e "$dir/$f" ]] || missing=$((missing + 1)); done
    (( missing == 0 )) || { echo "$dir lacks $missing of $n shards (sync the full pack, or write a rank-$rank folder: README)"; return 1; }
    echo "full $n"
}
# tree_id <tree>: commit + uncommitted diff + untracked files under src/, hashed
tree_id() {
    cd "$1" 2>/dev/null || { echo "no-tree"; return 0; }
    { git rev-parse HEAD 2>/dev/null || echo no-git; git diff HEAD -- src 2>/dev/null || true
      { git ls-files -o --exclude-standard -z src 2>/dev/null || true; } | xargs -0r sha256sum; } | sha256sum | cut -c1-16
}
# remote <function names...> -- <command>: run a command on the worker with these functions defined
remote() {
    local defs=""
    while [[ "$1" != -- ]]; do defs+="$(declare -f "$1"); "; shift; done
    shift
    wssh "bash -c $(printf '%q' "$defs$*")"
}

# ---------------------------------------------------------------- settings
[[ "$KV_DTYPE" =~ ^(bf16|fp8)$ ]] || refuse "KV_DTYPE is bf16 or fp8, not $KV_DTYPE"
[[ "$VISION" =~ ^[01]$ && "$VISION_URLS" =~ ^[01]$ ]] || refuse "VISION and VISION_URLS are 0 or 1"
[[ "$COMM" =~ ^(nccl|roce)$ ]] || refuse "COMM is nccl or roce, not $COMM"
[[ "$ROCE_MAX_KB" =~ ^[1-9][0-9]*$ ]] || refuse "ROCE_MAX_KB is a size in KiB, not $ROCE_MAX_KB"
[[ "$ROCE_WAIT_S" =~ ^[1-9][0-9]*$ ]] && (( ROCE_WAIT_S <= 3600 )) || refuse "ROCE_WAIT_S is 1 to 3600 seconds, not $ROCE_WAIT_S"
[[ "$PARALLEL" =~ ^[1-4]$ ]] || refuse "PARALLEL is 1 to 4 (this box tops out at 4 streams), not $PARALLEL"
[[ "$CONTEXT" =~ ^[0-9]+$ ]] || refuse "CONTEXT is a token count (0: the largest that fits), not $CONTEXT"
[[ "$MAX_TOKENS" =~ ^[1-9][0-9]*$ ]] || refuse "MAX_TOKENS is a token count, not $MAX_TOKENS"
[[ "$MTP_HEAD" =~ ^(0|1|auto)$ ]] || refuse "MTP_HEAD is 0, 1 or auto, not $MTP_HEAD"
[[ "$MEMORY_RESERVE_GIB" =~ ^[0-9]+([.][0-9]+)?$ ]] || refuse "MEMORY_RESERVE_GIB is a number of GiB, not $MEMORY_RESERVE_GIB"
[[ "$KEPT_GIB" =~ ^[0-9]+([.][0-9]+)?$ ]] || refuse "KEPT_GIB is a number of GiB, not $KEPT_GIB"
[[ -z "$REASONING_EFFORT" || "$REASONING_EFFORT" =~ ^(low|medium|high|xhigh|max)$ ]] ||
    refuse "REASONING_EFFORT is low/medium/high/xhigh/max (or empty), not $REASONING_EFFORT"

SERVE_ARGS=(--context "$CONTEXT" --parallel "$PARALLEL" --max-tokens "$MAX_TOKENS" --drafter "$DRAFTER")
[[ "$KV_DTYPE" == bf16 ]] || SERVE_ARGS+=(--kv-dtype "$KV_DTYPE")
[[ "$VISION" == 1 ]] && SERVE_ARGS+=(--vision)
[[ "$VISION" == 1 && "$VISION_URLS" == 1 ]] && SERVE_ARGS+=(--vision-urls)
if [[ "$THINKING" == 1 ]]; then SERVE_ARGS+=(--thinking); else SERVE_ARGS+=(--no-thinking); fi
[[ -z "$REASONING_EFFORT" ]] || SERVE_ARGS+=(--reasoning-effort "$REASONING_EFFORT")
if [[ "$CHAT_TEMPLATE" != none && -n "$CHAT_TEMPLATE" ]]; then SERVE_ARGS+=(--chat-template "$CHAT_TEMPLATE"); fi
if (( SPILL )); then                                # the spill tier (Mia's 0088): the same flags on both ranks
    [[ "$SPILL_DIR" == /* ]] || refuse "SPILL_DIR must be an absolute path (the same on both Sparks), not $SPILL_DIR"
    for _m in "$MODEL_RANK0" "$MODEL_RANK1" "$DRAFTER"; do
        [[ "$_m" == none ]] && continue
        case "$(readlink -m "$SPILL_DIR")/" in "$(readlink -m "$_m")"/*) refuse "SPILL_DIR $SPILL_DIR lies inside $_m" ;; esac
    done
    SERVE_ARGS+=(--spill-gib "$SPILL_GIB" --snapshot-dir "$SPILL_DIR" --spill-highwater "$SPILL_HIGHWATER"
                 --spill-min-tokens "$SPILL_MIN_TOKENS" --spill-min-free-gib "$SPILL_MIN_FREE_GIB")
fi
SERVE_ARGS+=("$@")                                  # last value of a flag wins (argparse)

# libnccl is bundled in the venv (nvidia/nccl) but comm.py does not search venv site-packages; point at it so
# both ranks find it (same relative path on both nodes). Overridable with TF_NCCL_LIB.
if [[ -z "${TF_NCCL_LIB:-}" ]]; then
    for _nccl in "$VENV"/lib/python*/site-packages/nvidia/nccl/lib/libnccl.so.2 "$VENV"/lib/python*/site-packages/torch/lib/libnccl.so*; do
        [[ -e "$_nccl" ]] && { TF_NCCL_LIB="$_nccl"; break; }
    done
fi
# (none found, as under start-docker.sh's shim: the image's own libnccl, which comm.py finds itself)
if [[ -n "${TF_NCCL_LIB:-}" ]]; then
    wssh "test -e $(printf %q "$TF_NCCL_LIB")" || refuse "libnccl missing on the worker at $TF_NCCL_LIB"
fi

# The environment both ranks get (same values: the engine refuses ranks started differently).
# Any TF_GLM_* or TENSORFOLD_* already set in this shell is passed on as well.
COMMON_ENV=(
    HF_HUB_OFFLINE=1
    TENSORFOLD_NO_UPDATE_CHECK=1
    TENSORFOLD_MEMORY_RESERVE_GIB="$MEMORY_RESERVE_GIB"
    TF_GLM_CACHE_GIB="$KEPT_GIB"
    TF_GLM_MTP="$MTP_HEAD"
    TENSORFOLD_NUCLEUS_UNION="${NUCLEUS_UNION:-1}"   # Mia's 0034: top_p checked over both ranks' candidates (same draws)
    TORCH_EXTENSIONS_DIR="$TORCH_EXT"
    MAX_JOBS="${MAX_JOBS:-8}"
    PYTHONUNBUFFERED=1
    MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"     # glibc: few malloc arenas (the loaders' threads left ~5 GiB in 64 MiB ones)
    NCCL_SOCKET_IFNAME="$IFACE"
    NCCL_IB_HCA="$HCA"
    NCCL_MIN_NCHANNELS="$NCCL_CHANNELS"
    NCCL_MAX_NCHANNELS="$NCCL_CHANNELS"
    TF_COMM_BACKEND="$COMM"
    ${TF_NCCL_LIB:+TF_NCCL_LIB="$TF_NCCL_LIB"}
    TF_ROCE_MAX_KB="$ROCE_MAX_KB"
    TF_ROCE_WAIT_S="$ROCE_WAIT_S"
    TF_GLM_HC_SPLIT="$SPLIT"
    TF_GLM_PREFILL_OVERLAP="$([[ "$SPLIT" == 1 ]] && echo 2 || echo 0)"
    TF_GLM_KDA_CHUNKED="$KDA_CHUNKED"
    TF_GLM_MULTI_PREFILL="$MULTI_PREFILL"
    TF_GLM_FILL_BUDGET_MS="$FILL_BUDGET_MS"
    TF_GLM_FILL_DRAFTS="$FILL_DRAFTS"
    TF_GLM_FILL_ROWS="$FILL_ROWS"
    TF_GLM_STREAM_SMOOTH="$STREAM_SMOOTH"
    TF_GLM_STREAM_SMOOTH_MS="${STREAM_SMOOTH_MS:-400}"
    TF_GLM_CACHE_ENTRIES="$CACHE_ENTRIES"
    TF_GLM_COPY_DRAFTS="$COPY"
    TF_GLM_COPY_MAX="$COPY_MAX"
    TF_GLM_DFLASH_POLICY="$DRAFT_POLICY"
    TF_GLM_WIDE_GRAPHS="$([[ "$COPY_CODE" == 1 ]] && echo 16 || echo 0)"
    TF_GLM_COPY_REPLY_MATCH="$([[ "$COPY_CODE" == 1 ]] && echo 16 || echo 0)"
    TF_GLM_L2PF="$L2PF"
    TF_GLM_EXL3_DEC_ORDER="$EXL3_DEC_ORDER"
    TF_GLM_SHARED_PREFIX="$SHARED_PREFIX"
    TF_GLM_MULTI_LONE="$MULTI_LONE"
    TF_GLM_POOL_STREAMS="$POOL_STREAMS"
)
(( SPILL )) && COMMON_ENV+=(TF_SPILL_FLUSH_S="$SPILL_FLUSH_S")
[[ -z "${NCCL_DEBUG:-}" ]] || COMMON_ENV+=(NCCL_DEBUG="$NCCL_DEBUG")
while IFS='=' read -r name _; do
    case "$name" in TENSORFOLD_MEMORY_RESERVE_GIB|TF_GLM_CACHE_GIB|TF_GLM_MTP|TENSORFOLD_NO_UPDATE_CHECK|TENSORFOLD_NUCLEUS_UNION) continue ;; esac
    case "$name" in TF_ROCE_MAX_KB|TF_ROCE_WAIT_S) continue ;; esac   # set above from ROCE_MAX_KB / ROCE_WAIT_S
    case "$name" in TF_GLM_HC_SPLIT|TF_GLM_PREFILL_OVERLAP|TF_GLM_KDA_CHUNKED|TF_GLM_MULTI_PREFILL|TF_GLM_FILL_BUDGET_MS|\
                    TF_GLM_FILL_DRAFTS|TF_GLM_FILL_ROWS|TF_GLM_STREAM_SMOOTH|TF_GLM_STREAM_SMOOTH_MS|TF_GLM_CACHE_ENTRIES) continue ;; esac
    case "$name" in TF_GLM_COPY_DRAFTS|TF_GLM_COPY_MAX|TF_GLM_DFLASH_POLICY|TF_GLM_WIDE_GRAPHS|TF_GLM_COPY_REPLY_MATCH|\
                    TF_GLM_L2PF|TF_GLM_SHARED_PREFIX|TF_GLM_MULTI_LONE|TF_GLM_POOL_STREAMS|TF_GLM_EXL3_DEC_ORDER) continue ;; esac   # set above from COPY* / DRAFT_POLICY / L2PF / ..
    # TF_ROCE_HCA / TF_ROCE_GID_INDEX name one node's devices: not passed to both (NCCL_IB_* per node instead)
    case "$name" in TF_ROCE_HCA|TF_ROCE_GID_INDEX) continue ;; esac
    case "$name" in TF_SPILL_FLUSH_S) continue ;; esac   # set above from SPILL_FLUSH_S (spill on)
    COMMON_ENV+=("$name=${!name}")
done < <(env | grep -E '^(TF_GLM|TF_ROCE|TF_SPILL|TENSORFOLD)_[A-Z0-9_]+=' || true)

RANK1_ARGS=(serve "$MODEL_RANK1" --tp 2 --rank 1 --master "$HEAD_ADDR" --master-port "$MASTER_PORT"
            "${SERVE_ARGS[@]}")
RANK0_ARGS=(serve "$MODEL_RANK0" --tp 2 --rank 0 --master "$HEAD_ADDR" --master-port "$MASTER_PORT"
            --name "$SERVED_NAME" --host "$HOST" --port "$PORT" "${SERVE_ARGS[@]}")

# ---------------------------------------------------------------- checks (nothing started yet)
mkdir -p "$STATE_DIR"
exec 9>"$STATE_DIR/start.lock"
flock -n 9 || refuse "another start.sh is running"

if [[ -f "$PIDS" ]]; then
    old0=$(sed -n 's/^rank0=//p' "$PIDS"); old1=$(sed -n 's/^rank1=//p' "$PIDS")
    if [[ -n "$old0" ]] && grep -q tensorfold "/proc/$old0/cmdline" 2>/dev/null; then
        refuse "rank 0 from an earlier start (pid $old0) is still running: ./stop.sh first"
    fi
    if [[ -n "$old1" ]] && wssh "grep -q tensorfold /proc/$old1/cmdline" 2>/dev/null; then
        refuse "rank 1 from an earlier start (pid $old1 on $WORKER_SSH) is still running: ./stop.sh first"
    fi
fi

port_busy() { ss -ltnH "sport = :$1" 2>/dev/null | grep -q .; }
port_busy "$PORT" && refuse "port $PORT is in use. This launcher will not bind it."
port_busy "$MASTER_PORT" && refuse "rendezvous port $MASTER_PORT is in use here"

settle_kib=$(awk -v g="$SETTLE_GIB" 'BEGIN {printf "%d", g * 1048576}')
for (( waited = 0; waited < SETTLE_S; waited += 5 )); do
    h=$(mem_kib); w=$(remote mem_kib -- mem_kib 2>/dev/null)
    [[ -n "$h" && -n "$w" && "$h" -ge "$settle_kib" && "$w" -ge "$settle_kib" ]] && break
    (( waited == 0 )) && log "waiting for free memory to settle (head ${h:-?} KiB, worker ${w:-?} KiB; want $SETTLE_GIB GiB)"
    sleep 5
done
head_mem=$(mem_kib)
min_kib=$((MIN_GIB * 1024 * 1024))
[[ -n "$head_mem" && "$head_mem" -ge "$min_kib" ]] ||
    refuse "head MemAvailable is ${head_mem:-missing} KiB, under $MIN_GIB GiB"
wssh true 2>/dev/null || refuse "cannot ssh to $WORKER_SSH without a password"
worker_mem=$(remote mem_kib -- mem_kib)
[[ -n "$worker_mem" && "$worker_mem" -ge "$min_kib" ]] ||
    refuse "worker MemAvailable is ${worker_mem:-missing} KiB, under $MIN_GIB GiB"

[[ -x "$TF" ]] || refuse "$TF missing here (README: set up the venv)"
wssh "test -x '$TF'" || refuse "$TF missing on the worker (README: worker setup)"

kind0=$(model_kind "$MODEL_RANK0" 0) || refuse "rank 0: $kind0"
kind1=$(remote model_kind -- model_kind "$(printf '%q' "$MODEL_RANK1")" 1) || refuse "rank 1: $kind1"
if [[ "$DRAFTER" != none ]]; then
    [[ -f "$DRAFTER/config.json" ]] || refuse "no DFlash2 drafter at $DRAFTER here"
    wssh "test -f '$DRAFTER/config.json'" || refuse "no DFlash2 drafter at $DRAFTER on the worker"
fi
if [[ "$CHAT_TEMPLATE" != none && -n "$CHAT_TEMPLATE" ]]; then
    # both ranks read --chat-template at parse time; the checkpoint snapshot is never touched
    [[ -f "$CHAT_TEMPLATE" ]] || refuse "chat template $CHAT_TEMPLATE missing here (CHAT_TEMPLATE=none for the checkpoint's)"
    wssh "test -f '$CHAT_TEMPLATE'" || refuse "chat template $CHAT_TEMPLATE missing on the worker (sync the branch)"
fi

if [[ -z "$HEAD_GID" ]]; then
    HEAD_GID=$(gid_find "$HCA" "$HEAD_ADDR") || refuse "$HCA here has no RoCE v2 GID holding $HEAD_ADDR (check HEAD_ADDR, HCA)"
fi
if [[ -z "$WORKER_GID" ]]; then
    WORKER_GID=$(remote gid_ok gid_find -- gid_find "$HCA" "$WORKER_ADDR") ||
        refuse "$HCA on the worker has no RoCE v2 GID holding $WORKER_ADDR (check WORKER_ADDR, HCA)"
fi
gid_ok "$HCA" "$HEAD_GID" "$HEAD_ADDR" ||
    refuse "GID $HEAD_GID of $HCA here is not the RoCE v2 entry of $HEAD_ADDR (set HEAD_GID)"
remote gid_ok -- gid_ok "$HCA" "$WORKER_GID" "$WORKER_ADDR" ||
    refuse "GID $WORKER_GID of $HCA on the worker is not the RoCE v2 entry of $WORKER_ADDR (set WORKER_GID)"
ping -c1 -W2 -I "$IFACE" "$WORKER_ADDR" >/dev/null 2>&1 || refuse "$WORKER_ADDR does not answer over $IFACE"
if [[ "$COMM" == roce ]]; then
    # the RoCE proxy is built at first start with gcc + libibverbs into $TORCH_EXT/roce, which must be writable
    # (b12x's read-only cache dir once made RoCE fail at start)
    roce_ok='test -r /usr/include/infiniband/verbs.h && command -v gcc >/dev/null && mkdir -p "$1/roce" && test -w "$1/roce"'
    bash -c "$roce_ok" _ "$TORCH_EXT" || refuse "COMM=roce needs gcc, libibverbs-dev and a writable $TORCH_EXT/roce here"
    wssh "bash -c $(printf '%q' "$roce_ok") _ $(printf '%q' "$TORCH_EXT")" ||
        refuse "COMM=roce needs gcc, libibverbs-dev and a writable $TORCH_EXT/roce on the worker"
fi

RANK1_CMD=(env "${COMMON_ENV[@]}" NCCL_IB_GID_INDEX="$WORKER_GID" "$TF" "${RANK1_ARGS[@]}")
RANK0_CMD=(env "${COMMON_ENV[@]}" NCCL_IB_GID_INDEX="$HEAD_GID" "$TF" "${RANK0_ARGS[@]}")

id0=$(tree_id "$TREE"); id1=$(remote tree_id -- tree_id "$(printf '%q' "$TREE")")
if [[ "$id0" != "$id1" ]]; then
    msg="the worker's $TREE ($id1) differs from this one ($id0): sync the branch (README)"
    [[ "$ALLOW_TREE_MISMATCH" == 1 ]] && log "warning: $msg" || refuse "$msg (ALLOW_TREE_MISMATCH=1 to go on)"
fi

# tensorfold's own parser, CPU only: a typo fails here
CUDA_VISIBLE_DEVICES="" "$PY" -c 'import sys; from tensorfold.cli import build_parser; build_parser().parse_args(sys.argv[1:])' \
    "${RANK0_ARGS[@]}" >/dev/null 2>"$STATE_DIR/args.err" ||
    { cat "$STATE_DIR/args.err" >&2; refuse "tensorfold serve rejects these arguments"; }

log "checks passed: head ${head_mem} KiB / worker ${worker_mem} KiB available; rank 0 $kind0, rank 1 $kind1; GID $HEAD_GID / $WORKER_GID; tree $id0"

remote_cmd="ulimit -s 65536 2>/dev/null; cd $(printf '%q' "$TREE") || exit 1; nohup setsid $(printf '%q ' "${RANK1_CMD[@]}")>$(printf '%q' "$LOG1") 2>&1 </dev/null & echo \$!"
if [[ "$DRY_RUN" == 1 ]]; then
    printf '[dry-run] rank 1 on %s:\n  %s\n' "$WORKER_SSH" "$remote_cmd"
    printf '[dry-run] rank 0 here:\n  %s\n' "$(printf '%q ' "${RANK0_CMD[@]}")"
    exit 0
fi

# ---------------------------------------------------------------- cleanup
stop_ranks() {
    local i
    if [[ -n "$RANK0_PID" ]]; then
        kill -TERM -- "-$RANK0_PID" 2>/dev/null || true
    fi
    if (( SPILL )); then        # rank 0's shutdown flush needs rank 1 (it writes its half too): rank 0 goes first
        for (( i = 0; i < STOP_TIMEOUT; i++ )); do
            { [[ -n "$RANK0_PID" ]] && kill -0 "$RANK0_PID" 2>/dev/null; } || break
            sleep 1
        done
    fi
    if [[ -n "$RANK1_PID" ]]; then
        wssh "kill -TERM -- -$RANK1_PID" 2>/dev/null || true
    fi
    for (( i = 0; i < STOP_TIMEOUT; i++ )); do
        { [[ -n "$RANK0_PID" ]] && kill -0 "$RANK0_PID" 2>/dev/null; } || break
        sleep 1
    done
    if [[ -n "$RANK0_PID" ]]; then
        kill -KILL -- "-$RANK0_PID" 2>/dev/null || true
    fi
    if [[ -n "$RANK1_PID" ]] &&
       ! wssh "for i in \$(seq $STOP_TIMEOUT); do kill -0 $RANK1_PID 2>/dev/null || exit 0; sleep 1; done; kill -KILL -- -$RANK1_PID" 2>/dev/null; then
        log "could not confirm rank 1 (pid $RANK1_PID on $WORKER_SSH) stopped: kept in $PIDS for ./stop.sh"
        return 0
    fi
    rm -f "$PIDS"
}
cleanup() {
    trap - EXIT INT TERM
    log "stopping both ranks"
    stop_ranks
}
trap cleanup EXIT
trap 'exit 130' INT TERM

# rank 0 alive (and still TensorFold); rank 1: 0 alive, 1 gone, 2 worker unreachable (unknown)
rank0_alive() { grep -q tensorfold "/proc/$RANK0_PID/cmdline" 2>/dev/null; }
rank1_state() {
    wssh true 2>/dev/null || return 2
    wssh "grep -q tensorfold /proc/$RANK1_PID/cmdline" 2>/dev/null && return 0
    return 1
}

# ---------------------------------------------------------------- launch: rank 1, then rank 0
if (( SPILL )); then           # the spill tier's directory on both Sparks (files take its owner: you)
    mkdir -p "$SPILL_DIR" && test -w "$SPILL_DIR" || refuse "SPILL_DIR $SPILL_DIR is not a writable directory here"
    wssh "mkdir -p $(printf '%q' "$SPILL_DIR") && test -w $(printf '%q' "$SPILL_DIR")" ||
        refuse "SPILL_DIR $SPILL_DIR is not a writable directory on the worker"
    log "spill tier on: up to $SPILL_GIB GiB a Spark in $SPILL_DIR (STOP_TIMEOUT ${STOP_TIMEOUT}s for the shutdown flush)"
fi
log "starting rank 1 on $WORKER_SSH (log $LOG1 there)"
RANK1_PID=$(wssh "$remote_cmd" | tail -n 1) || true
[[ "$RANK1_PID" =~ ^[0-9]+$ ]] || refuse "could not start rank 1 (got '$RANK1_PID')"
printf 'rank1=%s\nworker=%s\nspill=%s\nstop_timeout=%s\n' "$RANK1_PID" "$WORKER_SSH" "$SPILL" "$STOP_TIMEOUT" >"$PIDS"
log "rank 1 pid $RANK1_PID"

log "starting rank 0 (log $LOG0)"
ulimit -s 65536 2>/dev/null || true
nohup setsid "${RANK0_CMD[@]}" >"$LOG0" 2>&1 </dev/null 9>&- &
RANK0_PID=$!
printf 'rank0=%s\n' "$RANK0_PID" >>"$PIDS"
log "rank 0 pid $RANK0_PID"

# ---------------------------------------------------------------- wait for the API
URL="http://127.0.0.1:$PORT"
start=$SECONDS
next_remote=$SECONDS
until curl -sf --max-time 5 "$URL/v1/models" >/dev/null 2>&1; do
    rank0_alive || { tail -n 25 "$LOG0" >&2; refuse "rank 0 exited before the API answered (log $LOG0)"; }
    if (( SECONDS >= next_remote )); then
        next_remote=$((SECONDS + 30))
        st=0; rank1_state || st=$?
        if (( st == 1 )); then
            wssh "tail -n 25 $(printf '%q' "$LOG1")" >&2 || true
            refuse "rank 1 exited before the API answered (log $LOG1 on the worker)"
        fi
        log "loading... $((SECONDS - start))s ($(grep -o 'startup estimate [0-9.]* GiB' "$LOG0" | tail -1 || true))"
    fi
    (( SECONDS - start < WAIT_TIMEOUT )) || refuse "the API did not answer within ${WAIT_TIMEOUT}s"
    sleep 3
done
log "API answers after $((SECONDS - start))s"
if [[ "$COMM" == roce ]]; then
    grep -q 'RoCE all-gather is live' "$LOG0" || refuse "COMM=roce but rank 0's log has no 'RoCE all-gather is live'"
    log "$(grep -o 'RoCE all-gather is live.*' "$LOG0" | tail -1)"
fi

if [[ "$SMOKE" == 1 ]]; then
    reply=$(curl -s --max-time 300 "$URL/v1/chat/completions" -H 'Content-Type: application/json' \
        -d "{\"model\": \"$SERVED_NAME\", \"max_tokens\": 16, \"temperature\": 0, \"chat_template_kwargs\": {\"enable_thinking\": false}, \"messages\": [{\"role\": \"user\", \"content\": \"Reply with OK.\"}]}" |
        python3 -c 'import json,sys; c = json.load(sys.stdin)["choices"][0]["message"].get("content") or ""; assert c.strip(); print(repr(c.strip()[:40]))' 2>/dev/null) ||
        refuse "smoke test failed: no reply text (logs $LOG0, $LOG1 on the worker)"
    log "smoke test OK: $reply"
fi
log "$SERVED_NAME is serving on http://$HOST:$PORT/v1"

# ---------------------------------------------------------------- watch both ranks
while true; do
    sleep 30 & wait $! || true
    if ! rank0_alive; then
        log "rank 0 (pid $RANK0_PID) died; stopping rank 1"
        exit 1
    fi
    st=0; rank1_state || st=$?
    if (( st == 1 )); then
        log "rank 1 (pid $RANK1_PID) died; stopping rank 0"
        exit 1
    fi
done
