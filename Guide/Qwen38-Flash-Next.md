# Qwen3.8 Flash Next on 1× or 2× DGX Spark

Serve [NVIDIA Qwen3.8 Flash Next FP4 Local](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local) with FLARE. The commands below use disk-backed N-gram embedding tables.

> [!IMPORTANT]
> **Download this repository's [flare_cluster.sh](../flare_cluster.sh) before running the two-Spark command.** The launcher runs on the head and is not included in the Docker image.

| Setting | Example value |
|---|---|
| Tensor parallelism | 1 or 2 |
| Concurrent request slots | 2 |
| Maximum MTP draft tokens | 4 |
| Prefill chunk | 4,096 tokens |
| Recipe context ceiling | 262,144 tokens per request |
| API model name | `qwen3.8-flash-next` |
| Test date | 2026-10-31 |
| Image version | `pre-0.2` |

## 1. Download the launcher

For two Sparks, download [flare_cluster.sh](../flare_cluster.sh) onto the head. To build the image, also download [Dockerfile](../Dockerfile) and [.dockerignore](../.dockerignore) to the same directory.

## 2. Check the nodes, then build or pull on the head

For two Sparks, complete the official multi-node setup and NCCL checks in the [main README](../README.md).

Choose one option on the Spark used for serving. For two nodes, use the head node.

```bash
docker build -t flare_spark:pre-0.2 .
```

Or, after the image has been published:

```bash
docker pull pilcothink/flare_spark:pre-0.2
docker tag pilcothink/flare_spark:pre-0.2 flare_spark:pre-0.2
```

## 3. Set the common addresses and directories

Set the checkpoint and writable local SSD state directory to your own paths:

```bash
MODEL_DIR="/absolute/path/to/Qwen3.8-Flash-Next-FP4-Local"
STATE_DIR="/absolute/path/to/flare-qwen-state"
```

## 4. Choose the node count and launch from the head

### Option A: one Spark

```bash
mkdir -p "$STATE_DIR/cache" "$STATE_DIR/config"

docker run --rm -it \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --security-opt seccomp=unconfined \
  -p 8000:8000 \
  -v "$MODEL_DIR:/models/qwen:ro" \
  -v "$STATE_DIR/cache:/root/.cache" \
  -v "$STATE_DIR/config:/root/.config/flare" \
  --entrypoint /bin/bash \
  flare_spark:pre-0.2
```

Inside the container, prepare the package:

```bash
ignition export /models/qwen \
  --gpus 0 \
  --max-concurrency 2 \
  --mtp-draft-tokens 4 \
  --prefill-chunk-size 4096
```

After export, replace the placeholder with the path printed after `Exported cache entry:` and run:

```bash
flare run /path/from/export/model.safetensors \
  --gpus 0 \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name qwen3.8-flash-next \
  --embedding-table-storage disk
```

### Option B: two Sparks, TP2

Complete the official multi-node setup and NCCL checks in the [main README](../README.md). The head must be able to SSH to the worker and run Docker without an interactive prompt. `STATE_DIR` must be the same absolute path on both nodes, writable on each node, and located on each node's local SSD.

Set the fabric interface and worker SSH address for your own setup:

```bash
FLARE_IMAGE=flare_spark:pre-0.2
FABRIC_IFACE="YOUR_HEAD_FABRIC_INTERFACE"
HEAD_FABRIC_IP=$(ip -4 -o addr show "$FABRIC_IFACE" | awk 'NR==1 {split($4,a,"/"); print a[1]}')
WORKER_NODE="USER@WORKER_FABRIC_IP"
```

Run on the head from the directory containing `flare_cluster.sh`:

```bash
bash flare_cluster.sh "$FLARE_IMAGE" "$HEAD_FABRIC_IP" \
  --workers "$WORKER_NODE" \
  --sync-image \
  --model "$MODEL_DIR" \
  --state-dir "$STATE_DIR" \
  --max-concurrency 2 \
  --mtp-draft-tokens 4 \
  --prefill-chunk-size 4096 \
  --served-model-name qwen3.8-flash-next \
  --host 0.0.0.0 \
  --port 8000
```

`--sync-image` copies the image to the worker when its image differs or is missing. Omit it when the same image is already loaded on both nodes. The launcher uses FLARE's default disk-backed embedding storage.

## 5. Understand what happens next

Preparation exports a package; FLARE then verifies it, loads the model and starts the API. The first preparation includes tuning and compilation and can take a long time. In the two-node command the launcher starts and stops the worker from the head.

## 6. Connect to the API

Once the server is ready, check the API model name:

```bash
curl --fail http://127.0.0.1:8000/v1/models
```

Use `qwen3.8-flash-next` as the model name. The example binds to all interfaces and does not configure API authentication.

## 7. Context and cache capacity

The recipe context ceiling is 262,144 tokens per request. The actual KV cache capacity depends on available memory and is shared by active requests. The prefill chunk size is not the maximum context length.

## 8. Stop and restart

Press `Ctrl-C` to stop serving. Keep the state directory to retain preparation caches and packages. For two Sparks, the launcher can reuse a matching package when the image, model and preparation settings are unchanged.

## 9. Benchmark results

### Two Sparks

```text
| model              |           test |              t/s |     peak t/s |        ttfr (ms) |     est_ppt (ms) |    e2e_ttft (ms) |
|:-------------------|---------------:|-----------------:|-------------:|-----------------:|-----------------:|-----------------:|
| qwen3.8-flash-next |         pp2048 |   782.32 ± 54.56 |              | 2251.03 ± 167.18 | 2246.16 ± 167.18 | 2251.03 ± 167.18 |
| qwen3.8-flash-next |          tg128 |     53.94 ± 0.54 | 54.33 ± 0.47 |                  |                  |                  |
| qwen3.8-flash-next |         pp2048 | 1137.54 ± 208.64 |              | 1621.61 ± 306.40 | 1616.73 ± 306.40 | 1621.61 ± 306.40 |
| qwen3.8-flash-next |         tg1024 |     58.19 ± 1.51 | 58.67 ± 1.25 |                  |                  |                  |
| qwen3.8-flash-next | pp2048 @ d1024 |   873.53 ± 31.83 |              |  2978.01 ± 11.44 |  2973.13 ± 11.44 |  2978.01 ± 11.44 |
| qwen3.8-flash-next |  tg128 @ d1024 |     45.70 ± 1.96 | 46.33 ± 2.05 |                  |                  |                  |
| qwen3.8-flash-next | pp2048 @ d1024 |  1237.58 ± 89.49 |              | 2163.47 ± 150.05 | 2158.59 ± 150.05 | 2163.47 ± 150.05 |
| qwen3.8-flash-next | tg1024 @ d1024 |     58.12 ± 2.21 | 58.67 ± 2.05 |                  |                  |                  |
| qwen3.8-flash-next | pp2048 @ d4096 |  1421.84 ± 18.77 |              |  3654.12 ± 76.46 |  3649.25 ± 76.46 |  3654.12 ± 76.46 |
| qwen3.8-flash-next |  tg128 @ d4096 |     51.92 ± 0.87 | 52.67 ± 0.94 |                  |                  |                  |
| qwen3.8-flash-next | pp2048 @ d4096 |  1474.31 ± 69.40 |              | 3524.24 ± 136.49 | 3519.36 ± 136.49 | 3524.24 ± 136.49 |
| qwen3.8-flash-next | tg1024 @ d4096 |     59.32 ± 5.01 | 60.00 ± 5.10 |                  |                  |                  |
```

### One Spark

```text
| model              |           test |              t/s |     peak t/s |        ttfr (ms) |     est_ppt (ms) |    e2e_ttft (ms) |
|:-------------------|---------------:|-----------------:|-------------:|-----------------:|-----------------:|-----------------:|
| qwen3.8-flash-next |         pp2048 |  1031.47 ± 48.40 |              |  1685.61 ± 23.59 |  1680.41 ± 23.59 |  1685.61 ± 23.59 |
| qwen3.8-flash-next |          tg128 |     36.44 ± 1.06 | 36.67 ± 1.25 |                  |                  |                  |
| qwen3.8-flash-next |         pp2048 |  1004.85 ± 34.41 |              |  1825.21 ± 71.63 |  1820.01 ± 71.63 |  1825.21 ± 71.63 |
| qwen3.8-flash-next |         tg1024 |     39.58 ± 1.42 | 40.00 ± 1.41 |                  |                  |                  |
| qwen3.8-flash-next | pp2048 @ d1024 | 1004.55 ± 130.51 |              | 2688.86 ± 393.45 | 2683.66 ± 393.45 | 2688.86 ± 393.45 |
| qwen3.8-flash-next |  tg128 @ d1024 |     34.02 ± 3.03 | 34.33 ± 3.30 |                  |                  |                  |
| qwen3.8-flash-next | pp2048 @ d1024 |  1104.70 ± 70.69 |              | 2424.24 ± 166.54 | 2419.04 ± 166.54 | 2424.24 ± 166.54 |
| qwen3.8-flash-next | tg1024 @ d1024 |     36.79 ± 1.92 | 37.33 ± 2.05 |                  |                  |                  |
| qwen3.8-flash-next | pp2048 @ d4096 | 1480.38 ± 283.99 |              | 3651.84 ± 533.82 | 3646.64 ± 533.82 | 3651.84 ± 533.82 |
| qwen3.8-flash-next |  tg128 @ d4096 |     31.50 ± 1.02 | 32.00 ± 0.82 |                  |                  |                  |
| qwen3.8-flash-next | pp2048 @ d4096 | 1304.81 ± 113.44 |              | 4116.06 ± 440.02 | 4110.86 ± 440.02 | 4116.06 ± 440.02 |
| qwen3.8-flash-next | tg1024 @ d4096 |     35.60 ± 1.97 | 36.00 ± 2.16 |                  |                  |                  |
```

## References

- [Model repository](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local)
- [NVIDIA FLARE](https://github.com/NVIDIA/spark-local-inference)

## Authorship and maintenance

This guide was generated by AI. Hands-on testing was performed by the repository owner. Revisions, maintenance, and distribution are handled with AI assistance.
