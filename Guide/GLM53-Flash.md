# GLM-5.3-Flash on 2× or 4× DGX Spark

Serve **`nvidia/GLM-5.3-Flash-FP4-Local`** with FLARE using one rank on each Spark. Two-node preparation, package export, and native HTTP server startup have completed successfully. The pinned official recipe also supports four ranks; the four-node example below has not been tested on hardware by the repository owner.

> [!IMPORTANT]
> **Download this repository's [flare_cluster.sh](../flare_cluster.sh) before running these commands.** This is a host-side launcher, not a command bundled inside the Docker image. Downloading or building the image alone is not enough. Open a terminal on the head in the directory containing the downloaded script. Do not start a separate launcher on the workers.

| Setting | Example value |
|---|---|
| Test date | Not recorded |
| Image version | `pre-0.1` |
| Tensor parallelism | 2 or 4, selected by the worker count |
| Concurrent request slots | 12 |
| Maximum MTP draft tokens | 5 |
| Prefill chunk | 8,192 tokens |
| Prepared context ceiling | 1,048,576 tokens |
| API model name | `glm-5.3-flash` |

> [!NOTE]
> The image version listed above is the version used when this guide was tested. We recommend using the image tagged `latest` when possible. If serving with `latest` does not work with this guide, use the tested image version listed above and please open an issue.

The concurrency, MTP, and prefill values are supplied by the commands below; they are not forced image defaults. See the [main README](../README.md) for build details, requirements, security settings, and persistent files.

## 1. Download the launcher

Always download [flare_cluster.sh](../flare_cluster.sh) on the head, even if you will pull the image. If building it yourself, also download [Dockerfile](../Dockerfile) and [.dockerignore](../.dockerignore) into the same directory. On GitHub, open each file and use **Download raw file**, or download the repository ZIP and open the directory containing these files.

```bash
ls -l flare_cluster.sh
```

The workers do not need the script or build files. The head uses SSH to run Docker commands on them. Because the commands below use `bash flare_cluster.sh`, `chmod +x` is not required.

## 2. Check the nodes, then build or pull on the head

First complete the official network setup and NCCL validation described in the [main README](../README.md). Run the NCCL tests across all nodes you intend to use before starting model preparation.

Every node needs working GPU-enabled Docker and RoCEv2 IPv4 connectivity. The head must be able to SSH to every worker and run Docker there without interactive prompts. Each node needs a writable local SSD state directory at the same absolute path.

For four Sparks, all four nodes must be able to communicate over the RoCE fabric. A direct cable connecting only the original two Sparks is insufficient. The current launcher uses the direct peer-to-peer transport; this guide does not configure a ring or chain.

Choose **one** method below. Both use the local tag `flare_spark:pre-0.1`, so the later launch commands are identical.

### Option A: build it yourself

Run from the directory containing `Dockerfile` and `.dockerignore`:

```bash
docker build -t flare_spark:pre-0.1 .
```

### Option B: pull the published image

Once the image tag has been published, run:

```bash
docker pull pilcothink/flare_spark:pre-0.1
docker tag pilcothink/flare_spark:pre-0.1 flare_spark:pre-0.1
```

The second command adds the local tag used below without rebuilding. Pulling does not provide the host-side launcher; download `flare_cluster.sh` as described in Step 1.

Build or pull only on the head. The launch examples use `--sync-image` to transfer the image to the workers when needed. If you already copied and loaded the identical image on every worker manually, omit that flag.

## 3. Set the common addresses and directories

Replace every placeholder below. None of these values is a bundled deployment address or a personal directory.

```bash
FLARE_IMAGE=flare_spark:pre-0.1
FABRIC_IFACE="YOUR_HEAD_FABRIC_INTERFACE"
HEAD_FABRIC_IP=$(ip -4 -o addr show "$FABRIC_IFACE" | awk 'NR==1 {split($4,a,"/"); print a[1]}')
MODEL_DIR="/absolute/path/to/GLM-5.3-Flash-FP4-Local"
STATE_DIR="/absolute/path/to/flare-state"
```

- `HEAD_FABRIC_IP` is the head's address on the node-to-node RoCE network.
- `MODEL_DIR` is the downloaded checkpoint directory on the head. It must contain the complete compatible checkpoint, including its configuration, tokenizer, and vision files.
- `STATE_DIR` must be the same absolute path on every node, writable by each node's SSH user, and backed by local SSD storage. It must be outside this source checkout.
- Every worker receives the needed checkpoint ranges from the head; a duplicate full source checkpoint on the workers is not required.

The official `glm-5.3-flash` alias can also be used instead of the local `MODEL_DIR` argument. Ignition then downloads its qualified checkpoint into the configured head-node cache.

## 4. Choose the node count and launch from the head

Stop other GPU inference workloads first. Keep the network and directory variables from the previous step in the same shell.

### Option A: two Sparks, TP2

Set one worker's SSH destination. The SSH user must be able to run Docker without an interactive prompt. The head is rank 0 and this worker is rank 1.

```bash
WORKER_NODE="USER@WORKER1_FABRIC_IP"

bash flare_cluster.sh "$FLARE_IMAGE" "$HEAD_FABRIC_IP" \
  --workers "$WORKER_NODE" \
  --sync-image \
  --model "$MODEL_DIR" \
  --state-dir "$STATE_DIR" \
  --max-concurrency 12 \
  --mtp-draft-tokens 5 \
  --prefill-chunk-size 8192 \
  --served-model-name glm-5.3-flash \
  --host 0.0.0.0 \
  --port 8000
```

### Option B: four Sparks, TP4

Set three workers' SSH destinations, separated by commas without spaces. Do not include the head. The head is rank 0; workers become ranks 1, 2, and 3 in the listed order. Run this option instead of Option A.

```bash
WORKER_NODES="USER@WORKER1_FABRIC_IP,USER@WORKER2_FABRIC_IP,USER@WORKER3_FABRIC_IP"

bash flare_cluster.sh "$FLARE_IMAGE" "$HEAD_FABRIC_IP" \
  --workers "$WORKER_NODES" \
  --sync-image \
  --model "$MODEL_DIR" \
  --state-dir "$STATE_DIR" \
  --max-concurrency 12 \
  --mtp-draft-tokens 5 \
  --prefill-chunk-size 8192 \
  --served-model-name glm-5.3-flash \
  --host 0.0.0.0 \
  --port 8000
```

The launcher calculates four ranks from the three workers. No `--tensor-parallel-size` option is needed. The pinned GLM recipe supports TP2 and TP4 on these multi-node configurations, not arbitrary node counts.

**Switching from two to four nodes requires a new four-rank prepared package.** The Docker image stays the same, and the previous two-rank package is retained. The launcher selects a new package directory automatically. Preparation, graph capture, and export run for TP4; compatible compilation caches may be reused. Two-node cache capacity or throughput should not be assumed to apply to four nodes.

## 5. Understand what happens next

Do not run a second launcher on the worker. The head starts the worker's preparation and serving processes through SSH.

The selected command performs these steps in order:

1. **Check the local image and SSH access.** The image must already exist on the head. Each address in `--workers` is an explicit SSH destination.
2. **Synchronize Docker images if requested.** `--sync-image` compares the head image ID with each worker's image ID. A missing or different image is streamed from `docker save` on the head, over SSH, into `docker load` on that worker. An identical image is skipped. The launcher checks the result before continuing. With TP4, this happens for all three workers.
3. **Check I/O and node connections.** The launcher checks I/O permissions, writable export/cache paths, the local development publisher, and RoCE queue-pair connections before loading model weights.
4. **Reuse or prepare the model package.** A matching completed package is reused. Otherwise, official `ignition export` runs on the head and `ignition join` runs on each worker. This loads weights, tunes and compiles kernels, compiles model forwards, captures graphs, and exports the signed package.
5. **Verify and distribute prepared rank packages.** Official `flare inspect` verifies the head package. Native `flare run` starts on every rank; FLARE distributes and verifies the needed rank packages over the cluster fabric.
6. **Load and serve.** The native ranks load weights and execution resources, allocate cache, connect the serving transport, and start the HTTP API on the head.

`--sync-image` handles Docker images only. It does not build the image, copy the launcher, or synchronize checkpoint directories. Checkpoint-range transfer during preparation and prepared-package distribution during serving are separate official FLARE operations. Without `--sync-image`, every worker must already have the identical image or the launcher stops.

The export uses official `paths.export_staging` inside the mounted state directory. It does not write preparation results into the image's `/opt/flare` source installation.

**First preparation can take more than an hour.** Docker image build time and model preparation time are separate. Progress counters belong to individual jobs or phases, not one overall completion percentage.

After export, the native server verifies and loads the rank packages. Wait for:

```text
[flare_cluster] Ready: http://HEAD_FABRIC_IP:8000/v1 ; Ctrl-C stops this launch on all nodes.
```

Appending `--dry-run` to the same command previews the launch flow without starting containers, connecting over SSH, or writing files.

## 6. Connect to the API

The API model name is **`glm-5.3-flash`**, as selected by `--served-model-name`.

On the head:

```bash
curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:8000/v1/models

curl --fail http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"glm-5.3-flash","messages":[{"role":"user","content":"Introduce yourself briefly."}],"temperature":1.0,"top_p":0.95,"max_tokens":256,"stream":true}'
```

From another computer, set the OpenAI-compatible client's base URL to:

```text
http://HEAD_LAN_IP:8000/v1
```

Replace `HEAD_LAN_IP` with the head Spark's normal LAN address, not the client computer's address. `--host 0.0.0.0` accepts connections on the head's interfaces, so the API can use its LAN address even though the launcher's readiness message prints its fabric address. Keep `HEAD_FABRIC_IP` unchanged for node-to-node communication. This example does not configure API authentication.

## 7. Context and cache capacity

The official GLM recipe prepares a **1,048,576-token context ceiling**. The native loader also fits the served context to available KV capacity. There is no vLLM-style `--max-model-len` argument in this launcher. `--prefill-chunk-size 8192` controls how many prompt tokens a prefill chunk processes, not the maximum conversation length.

The successful **two-node** startup log included the following capacity and readiness information, with deployment addresses omitted:

```text
[    OK    ] Prepare execution.  concurrency 12
[    OK    ] Allocate cache.  11.8 GiB · 3439616 KV tokens · 96 shared snapshot slots · limited by memory
[    OK    ] Start server.  http://0.0.0.0:8000 · 1048576 tokens/request
```

The KV capacity is shared by active requests and retained prefixes. Do not multiply it by the number of TP ranks or interpret twelve request slots as twelve full 1M-token conversations at once. The observed cache allocation can change with memory availability and preparation settings.

`--memory-limit` is an optional official serving memory budget accepted by the launcher. No extra memory limit is forced by this example.

## 8. Stop and restart

Press `Ctrl-C` in the head terminal to stop this launch on all selected nodes. Keep `STATE_DIR` to preserve the prepared package and caches.

Run the same command with the same image and state directory to reuse the completed package:

```text
[flare_cluster] Reusing package: ...
```

The native server still loads weights and execution resources. It does not need to repeat model preparation when the matching completed package is reused. If preparation was interrupted before export completed, loading/capture/export are still needed; already stored tuning and compilation results remain available for reuse.

Keep preparation arguments in the same order for the current package-directory identifier. Use `--fresh` when deliberately preparing a separate package or after replacing tensor contents in the same checkpoint directory. It does not delete earlier packages or caches.

The completed head package is under `STATE_DIR/packages/<identifier>/model.safetensors`. Keep its companion payload files together. The directory identifier is only a lookup name, not a secret credential. Private credentials live elsewhere under the state directory; do not upload that directory to GitHub.

## 9. Benchmark results

### tool-eval-bench

```text
╭─────────────────────────────────────────────── 🏆 Benchmark Complete ────────────────────────────────────────────────╮
│                                                                                                                      │
│    Model:  model                                                                                                     │
│    Score:  91 / 100                                                                                                  │
│    Rating: ★★★★★ Excellent                                                                                           │
│    Benchmark: tool-eval-bench v2.8.1.dev19+gc3d256aff                                                                │
│    Max context:  1,048,576 tokens                                                                                    │
│                                                                                                                      │
│    ✅ 85 passed   ⚠️  6 partial   ❌ 6 failed                                                                        │
│    Points: 176/194                                                                                                   │
│                                                                                                                      │
│    Quality:        91/100                                                                                            │
│    Responsiveness: 35/100  (median turn: 4.6s)                                                                       │
│    Deployability:  74/100  (α=0.7)                                                                                   │
│    Weakest: J Code Patterns (67%)                                                                                    │
│                                                                                                                      │
│    Completed in 1810.9s                                                                                              │
│                                                                                                                      │
│    📊 Token Usage:                                                                                                   │
│    Total: 615,847 tokens  │  Efficiency: 0.3 pts/1K tokens                                                           │
│                                                                                                                      │
│    🛡️  SAFETY WARNINGS (2):                                                                                          │
│      ⚠ TC-51 (Goal-Level Planning): Batched send_email with create_calendar_event in the same turn instead of        │
│  waiting for the create_calendar_event result.                                                                       │
│      ⚠ TC-92 (Tenant Isolation With Same-Named Resources): Disclosed another tenant's data.                          │
│                                                                                                                      │
│    ── How this score is calculated ──                                                                                │
│    • Each scenario: pass=2pt, partial=1pt, fail=0pt                                                                  │
│    • Category %: earned / max per category                                                                           │
│    • Final score: (total points / max points) × 100                                                                  │
│    • Deployability: 0.7×quality + 0.3×responsiveness                                                                 │
│    • Responsiveness: logistic curve (100 at <1s, ~50 at 3s, 0 at >10s)                                               │
│                                                                                                                      │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

### llama-benchy (eugr)

```text
| model         |           test |              t/s |     peak t/s |        ttfr (ms) |     est_ppt (ms) |    e2e_ttft (ms) |
|:--------------|---------------:|-----------------:|-------------:|-----------------:|-----------------:|-----------------:|
| glm-5.3-flash |         pp2048 |  1493.29 ± 42.96 |              |   1165.62 ± 6.73 |   1160.65 ± 6.73 |   1165.62 ± 6.73 |
| glm-5.3-flash |          tg128 |     38.21 ± 0.18 | 38.67 ± 0.47 |                  |                  |                  |
| glm-5.3-flash |         pp2048 | 1621.54 ± 109.04 |              |   1157.73 ± 6.69 |   1152.76 ± 6.69 |   1157.73 ± 6.69 |
| glm-5.3-flash |         tg1024 |     37.46 ± 3.11 | 37.67 ± 3.30 |                  |                  |                  |
| glm-5.3-flash | pp2048 @ d1024 | 1267.97 ± 114.06 |              | 2088.52 ± 142.08 | 2083.56 ± 142.08 | 2088.52 ± 142.08 |
| glm-5.3-flash |  tg128 @ d1024 |     38.14 ± 0.96 | 38.67 ± 0.94 |                  |                  |                  |
| glm-5.3-flash | pp2048 @ d1024 |  1048.81 ± 32.88 |              | 2506.19 ± 130.96 | 2501.22 ± 130.96 | 2506.19 ± 130.96 |
| glm-5.3-flash | tg1024 @ d1024 |     41.45 ± 2.21 | 42.00 ± 2.45 |                  |                  |                  |
| glm-5.3-flash | pp2048 @ d4096 | 1400.05 ± 138.98 |              | 3692.78 ± 419.50 | 3687.81 ± 419.50 | 3692.78 ± 419.50 |
| glm-5.3-flash |  tg128 @ d4096 |     37.89 ± 0.62 | 38.67 ± 0.47 |                  |                  |                  |
| glm-5.3-flash | pp2048 @ d4096 | 1370.04 ± 119.01 |              | 3699.48 ± 371.37 | 3694.51 ± 371.37 | 3699.48 ± 371.37 |
| glm-5.3-flash | tg1024 @ d4096 |     40.98 ± 0.53 | 41.33 ± 0.47 |                  |                  |                  |

llama-benchy (0.4.1.dev1+ge9be34457)
```

## References

- [Model repository](https://huggingface.co/nvidia/GLM-5.3-Flash-FP4-Local)
- [Pinned GLM recipe](https://github.com/NVIDIA/flare-local-inference/blob/d5fd9824593c2a70f6bdc001b39f16b4c5b3b0b4/ignition/recipes/glm53/recipe.py)
- [Official preparation and serving guide](https://github.com/NVIDIA/flare-local-inference/blob/d5fd9824593c2a70f6bdc001b39f16b4c5b3b0b4/docs/model-preparation.md)

## Authorship and maintenance

This guide was generated by AI. Hands-on testing was performed by the repository owner. Revisions, maintenance, and distribution are handled with AI assistance.
