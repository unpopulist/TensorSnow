# RED-SNOW on two DGX Sparks (TensorFold, TP=2 over RoCE)

This directory serves
[vcruz305/RED-SNOW-5.3-FLASH-EXL3-SAGE-4.91bpw](https://huggingface.co/vcruz305/RED-SNOW-5.3-FLASH-EXL3-SAGE-4.91bpw)
(a mixed-width EXL3 pack of Blackfrost-AI's RED-SNOW-5.3-FLASH, itself a GLM-5.3-Flash finetune) on two NVIDIA DGX
Sparks. The weights are split by tensor parallelism (TP=2), and the two GPUs talk over the Sparks' QSFP link (RoCE).
[incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2) provides speculative drafts.
Rank 0 (the "head") serves an OpenAI-compatible API on port 8000. Rank 1 (the "worker") holds the other half of
every layer.

Model use: RED-SNOW's model card asks that it be used only by experienced operators, in environments they own or are
explicitly authorized to assess, with a human in control of scope, targets, credentials, tool execution and
destructive actions. See [the card](https://huggingface.co/Blackfrost-AI/RED-SNOW-5.3-FLASH-BF16). The DFlash2
drafter is licensed CC BY-NC-ND 4.0, which allows non-commercial use only. Set `DRAFTER=none` to serve without it
(slower: MTP drafts).

## Requirements

- Two DGX Sparks (GB10, 128 GB unified memory) on DGX OS, joined by a QSFP cable. Each end needs an IPv4 address on
  the link's RoCE netdev (default `enp1s0f0np0`, RDMA device `rocep1s0f0`).
- Nothing else on either GPU. Each rank holds ~92 GiB of weights, and the launcher refuses to start under 110 GiB
  `MemAvailable`. At the default 320K window it first waits for ~115.7 GiB available.
- Docker with the NVIDIA runtime and `docker buildx` on both nodes, plus `/dev/infiniband/uverbs0`. `zstd` makes the
  image copy faster.
- Key-based (passwordless) ssh from the head to the worker, and the same `$HOME` path on both (or set `SHIM_DIR`).
- Disk: ~45 GB free under Docker's root on each node for the image. The head needs ~200 GB for the full pack, plus
  ~100 GB while it writes the rank-1 folder. The worker needs ~100 GB for the rank-1 folder. Both need 2.2 GB for
  the drafter.
- Ports 8000 (API) and 29551 (rendezvous) free on the head.

## Setup

1. Settings: `cp deploy/redsnow/.env.example deploy/redsnow/.env` and fill in `HEAD_ADDR`, `WORKER_ADDR` (the two
   RoCE-link addresses) and `WORKER_SSH` (`user@host` of the worker). `.env` is ignored by git. Every other setting
   below can go there too, and a variable set in the environment wins over the file.
2. Image, built on the head and copied to the worker:

   ```bash
   deploy/redsnow/docker/build.sh --to-worker          # prints the tag: redsnow-tensorfold:<commit12>-<recipe8>
   ```

   The image is NVIDIA's PyTorch 26.07 container (torch 2.13, CUDA 13.3, NCCL 2.30.7) with this tree installed and
   every CUDA extension precompiled for sm_121. Only committed work goes in (`git archive` of `HEAD`, or of
   `--commit <ref>`).
3. Model files, at the default paths (any absolute path works; set `MODEL_RANK0`, `MODEL_RANK1`, `DRAFTER`):

   ```bash
   # head: the full pack, and the drafter
   hf download vcruz305/RED-SNOW-5.3-FLASH-EXL3-SAGE-4.91bpw --local-dir /var/tmp/models/RED-SNOW-5.3-FLASH-EXL3-SAGE-4.91bpw
   hf download incoai/GLM-5.3-Flash-DFlash2 --local-dir /var/tmp/models/GLM-5.3-Flash-DFlash2
   # head: write rank 1's half (CPU only, ~93 GiB), inside the image
   mkdir -p /var/tmp/models/redsnow-rank1
   docker run --rm --user "$(id -u):$(id -g)" -v /var/tmp/models:/var/tmp/models --entrypoint python \
     redsnow-tensorfold:<tag> -m tensorfold.families.glm5_next.cuda.split \
     /var/tmp/models/RED-SNOW-5.3-FLASH-EXL3-SAGE-4.91bpw --rank 1 /var/tmp/models/redsnow-rank1
   # worker: the rank-1 folder, in its own directory, and the drafter at the same path as on the head
   rsync -a --info=progress2 /var/tmp/models/redsnow-rank1/ <worker-ssh>:/var/tmp/models/redsnow-rank1/
   rsync -a --info=progress2 /var/tmp/models/GLM-5.3-Flash-DFlash2/ <worker-ssh>:/var/tmp/models/GLM-5.3-Flash-DFlash2/
   ```

   The split can be restarted: it skips files that already exist. Keep the rank-1 folder in its own directory. It
   carries its own `config.json` and index, which would overwrite a full pack's. A rank folder freezes the split
   rules of the tree that wrote it, so regenerate it if `split.py`'s rules change. Instead of the rank-1 folder, the
   worker can hold the full pack (`MODEL_RANK1=<path to it>`), which costs ~200 GB there.

## Start and stop

```bash
DRY_RUN=1 IMAGE=redsnow-tensorfold:<tag> deploy/redsnow/docker/start-docker.sh   # every check, starts nothing
tmux new -s redsnow "IMAGE=redsnow-tensorfold:<tag> deploy/redsnow/docker/start-docker.sh"
deploy/redsnow/docker/stop-docker.sh                                             # from any shell
```

`start-docker.sh` checks that both nodes hold the same image, installs a small shim on both (`rank-shim.sh`) and runs
`start.sh` with each rank as one container (`redsnow-rank0` on the head, `redsnow-rank1` on the worker). `start.sh`
waits for free memory to settle, then checks ports, memory, ssh, the model folders, the drafter, the RoCE GIDs and a
ping over the link, and tensorfold's own argument parser. It starts rank 1 on the worker and then rank 0 here. It
waits for `/v1/models` (about 2.5 minutes), sends one short chat completion, then stays in the foreground watching
both ranks and stops both if either dies. `stop-docker.sh` stops the pair it recorded and touches no other
container.

`start.sh` alone runs the same launch from a bare-metal venv (`$TREE/.venv` on both nodes, the same tree path on
each). The container is the tested path: under the pip torch 2.11 / NCCL 2.28 venv, four parallel streams stalled.

Smoke test:

```bash
curl -s http://<rank0-host>:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "RED-SNOW-5.3-FLASH", "max_tokens": 64,
  "messages": [{"role": "user", "content": "Reply with OK."}]}'
```

Logs: `/tmp/tensorfold-redsnow-rank0.log` on the head and `/tmp/tensorfold-redsnow-rank1.log` on the worker
(`LOG0`, `LOG1`). State: `/tmp/tensorfold-redsnow/pids` (`STATE_DIR`).

## start.sh refuses to start when

- a required setting (`HEAD_ADDR`, `WORKER_ADDR`, `WORKER_SSH`) is missing or still a `<placeholder>`;
- port 8000 or the rendezvous port 29551 is already taken on the head. The launcher never stops another server;
- `MemAvailable` is under 110 GiB on either Spark (`MIN_GIB`);
- the worker can't be reached by key-based ssh, or either node's `tensorfold` is missing;
- rank 0's folder isn't a complete pack, or rank 1's is neither a complete pack nor a complete rank-1 split;
- the drafter or the chat template is missing on either node;
- the RDMA device has no RoCE v2 GID holding the node's address (or a set `HEAD_GID` / `WORKER_GID` is not that
  entry), or the worker's address doesn't answer a ping over `IFACE`;
- the worker's tree or image differs from the head's (`ALLOW_TREE_MISMATCH=1` overrides the tree check);
- `tensorfold serve` rejects the arguments, or an earlier pair is still running.

The engine then makes its own check: a window that doesn't fit in memory is refused on both ranks, and the whole
launch fails. At the 320K default that margin is small (see Known limits).

## Settings

Set these in `deploy/redsnow/.env` or in the environment. Defaults are in `start.sh`.

| Variable | Default | Notes |
|---|---|---|
| `HEAD_ADDR`, `WORKER_ADDR` | required | The two nodes' IPv4 addresses on the RoCE link. Rank 0 binds the rendezvous on `HEAD_ADDR`. |
| `WORKER_SSH` | required | `user@host` the head uses to reach the worker (any interface). |
| `IFACE`, `HCA` | `enp1s0f0np0`, `rocep1s0f0` | NCCL socket interface and RDMA device on both nodes (`ibdev2netdev`). |
| `HEAD_GID`, `WORKER_GID` | looked up | NCCL_IB_GID_INDEX per node. Empty: the RoCE v2 GID holding the node's address, found in sysfs. |
| `MODEL_RANK0` (`MODEL_DIR`) | `/var/tmp/models/RED-SNOW-5.3-FLASH-EXL3-SAGE-4.91bpw` | The full pack on the head. |
| `MODEL_RANK1` (`RANK1_MODEL_DIR`) | `/var/tmp/models/redsnow-rank1` | The rank-1 folder (or a full pack) on the worker. |
| `DRAFTER` (`DRAFTER_DIR`) | `/var/tmp/models/GLM-5.3-Flash-DFlash2` | The same path on both nodes. `none`: no DFlash2 (MTP drafts; set `MTP_HEAD=1`). |
| `IMAGE` | the tag `build.sh` gives `HEAD` | Container mode only. Also `SHIM_DIR` (`~/.local/share/redsnow-docker`) and `CACHE_ROOT`. |
| `SERVED_NAME`, `HOST`, `PORT` | `RED-SNOW-5.3-FLASH`, `0.0.0.0`, 8000 | The API's model name and address. |
| `CONTEXT` | 327680 | `--context`, the prompt + reply window per request. `0`: the largest that fits. |
| `PARALLEL` | 4 | `--parallel`, 1 to 4 concurrent requests decoded in shared rounds. |
| `POOL_STREAMS` | 1 | `TF_GLM_POOL_STREAMS`. 1: the streams share one window's cache pool (when it is full, the youngest stream goes back to the queue and is replayed). 4: every stream keeps a full window reserved (then `CONTEXT` must fall to ~64K). |
| `KV_DTYPE` | fp8 | `fp8` (e4m3 latent cache, ~12.3 KB a token per rank, lossy) or `bf16` (exact, ~18.1 KB). |
| `MEMORY_RESERVE_GIB` | 14.5 | `TENSORFOLD_MEMORY_RESERVE_GIB`: memory left free at start. The server peaks a few GiB past its estimate under load. |
| `SETTLE_GIB`, `SETTLE_S` | 115.7, 180 | Wait up to `SETTLE_S` seconds for both nodes' `MemAvailable` to reach `SETTLE_GIB` before starting. |
| `KEPT_GIB`, `CACHE_ENTRIES` | 2, 32 | Memory and count for other conversations' kept prompt states. |
| `MAX_TOKENS` | 32768 | Reply budget when a request sets none. |
| `THINKING`, `REASONING_EFFORT` | 1, high | `--thinking` and the default effort (`low`, `high`, or `medium`/`xhigh`/`max`, all rendered as Max). |
| `CHAT_TEMPLATE` | `deploy/redsnow/chat_template.jinja` | Served instead of the checkpoint's template (standard GLM-5.3-Flash with the effort line). `none`: the checkpoint's own. |
| `VISION`, `VISION_URLS` | 1, 0 | Image and video input (rank 0 runs the vision tower, ~2 GiB); https image URLs. |
| `COMM`, `ROCE_MAX_KB`, `ROCE_WAIT_S` | roce, 512, 300 | Small all-gathers as one-shot RDMA writes (b12x's transport), larger ones over NCCL. `nccl`: all over NCCL. Same bits. |
| `COPY`, `COPY_MAX`, `DRAFT_POLICY`, `COPY_CODE` | 1, 15, `fnc7:0.3`, 1 | Prompt-lookup ("copy") drafts ahead of DFlash2's, up to 7 DFlash2 drafts cut at noisy chance 0.3, 16-row verify graphs. Drafts change speed, not replies. |
| `SPLIT`, `KDA_CHUNKED`, `MULTI_PREFILL`, `FILL_BUDGET_MS`, `FILL_DRAFTS`, `STREAM_SMOOTH` | 1, 1, 1, 200, 1, 1 | Prefill: glue split by rows and overlapped, chunked KDA, shared prompt chunks, prompt slices between decode rounds, steadier streamed pacing. |
| `L2PF`, `SHARED_PREFIX`, `MULTI_LONE` | 1, 1, 0 | L2 prefetch in decode; kept state at the end of a shared system prompt; lone-request graphs (off: they evict kept prompts). |
| `SPILL_GIB` (+ `SPILL_DIR`, ...) | 0 (off) | Kept prompt states on local disk. Off by default; see `start.sh` for its settings. |
| `MTP_HEAD`, `NCCL_CHANNELS`, `MASTER_PORT` | auto, 4, 29551 | |
| `WAIT_TIMEOUT`, `STOP_TIMEOUT`, `SMOKE`, `MIN_GIB` | 3600, 30, 1, 110 | |

Arguments after the script name go to `tensorfold serve` on both ranks, and the last value of a flag wins. Any
`TF_GLM_*`, `TF_ROCE_*`, `TF_SPILL_*` or `TENSORFOLD_*` variable set in the calling shell is passed to both ranks.

## Measured performance

Measured on two DGX Sparks (GB10) with the defaults above: RED-SNOW 4.91 bpw, FP8 KV, DFlash2 + copy drafts, 4
streams sharing one cache pool, a 320K-token window per request and a 14.5 GiB reserve.

| | |
|---|---|
| Decode, one request | prose ~37 tok/s, code ~60, JSON ~72, edit/quote ~95 |
| Decode, four requests at once | ~68 tok/s aggregate |
| Prefill | ~1,100–1,150 tok/s: 55K tokens in 46–49 s, 236K in 214 s, 306K in 282 s (needle found) |
| Lowest free memory under load | ~6–9 GiB per node |
| Startup | ~2.5 minutes to a ready API |

Decode speed depends on how well the drafts are accepted, so it varies with the kind of text.

## Known limits

- The startup memory margin is tight at 320K with a 14.5 GiB reserve (a few tenths of a GiB). Right after a large
  image load or file copy, free memory sits low for a while, and the engine refuses the window. The launcher waits
  for memory to settle (`SETTLE_GIB`); if it is still refused, wait a minute and start again, or lower `CONTEXT`.
- The 4 streams share one pool. A long request leaves less room for the others, and when the pool is full the
  youngest stream is requeued and replayed.
- DFlash2 is licensed for non-commercial use only (CC BY-NC-ND 4.0).
- The spill tier (`SPILL_GIB`) and the display-memory KV reservation are not enabled.

## Next steps

A 4-bit latent KV cache is planned. It would store symmetric int4 values with one fp16 scale per group of 32 (288
bytes per latent row, against 528 for FP8), and keep the indexer keys in FP8. The goal is a ~580K-token shared window
on two Sparks (against ~350K with FP8), for 256–512K-token contexts. The format was chosen by measuring real
RED-SNOW cache rows: attention-output error was ~0.75%, against ~0.26% for FP8 and ~1.1% for a per-row int4 scale.
The kernels are not built yet. Quality will be validated with needle and long-document tests before it becomes an
option.
