# FLARE on DGX Spark

**English** | [한국어](README.ko.md)

Build or pull one Docker image for model preparation and native serving, then launch a multi-node DGX Spark cluster from the head node.

This recipe uses [NVIDIA FLARE](https://github.com/NVIDIA/flare-local-inference). Ignition prepares the model with PyTorch and FlashInfer/B12X; the native Rust Flare server loads the resulting package. The separate `flare_cluster.sh` launcher handles Docker and SSH orchestration.

## Contributions welcome

This repository publishes guides for models that have been tested in practice. Any additional configuration that has not been tested on hardware is explicitly marked.

**If you have successfully run another model using this project, please open a pull request to add your tested guide under `Guide/`.** Include the model, hardware setup, launch command, and any important notes from your experience. Remove personal paths, deployment addresses, and credentials before submitting.

**Pull requests are always welcome**, including new model guides, fixes, improvements, and documentation updates.

**Model-specific launch commands, settings, and test results are in [Guide/](Guide/).**

> [!IMPORTANT]
> **Download this repository's [flare_cluster.sh](flare_cluster.sh) before running the launch commands.** It runs on the head host and is not installed into the Docker image. Downloading the image or NVIDIA's upstream repository alone does not provide this launcher. Run the commands from the directory containing the downloaded script.

## Required: official cluster setup and NCCL validation

> [!IMPORTANT]
> **Complete the official NVIDIA DGX Spark network setup and pass the NCCL communication tests before running any multi-node guide in this repository.** Successful ping or SSH alone is not sufficient validation of GPU collective communication.

Follow the official playbooks in order, using the instructions for your node count and network topology:

1. [Connect Multiple DGX Sparks](https://build.nvidia.com/spark/connect-multiple-sparks) — configure the physical links, network interfaces, and passwordless SSH. If NVIDIA Sync Cluster Assistant has already configured the cluster successfully, do not repeat the manual network setup.
2. [NCCL setup and validation](https://build.nvidia.com/spark/nccl/stacked-sparks) — build and run the prescribed NCCL tests across all nodes you intend to use. Select the [two-node direct-link test](https://build.nvidia.com/spark/nccl/two-nodes) or [four-node switch test](https://build.nvidia.com/spark/nccl/four-nodes) as appropriate.

Resolve communication errors before starting model preparation. Once the tests pass, return here and follow the image and model-guide steps. The launcher's own RoCE checks still run afterward.

## Repository layout

```text
.
├── Dockerfile
├── flare_cluster.sh
├── README.md
├── README.ko.md
└── Guide/
```

The hidden `.dockerignore` excludes local data from the Docker build context. `.gitignore` excludes common runtime, credential, model, and image archive files from Git.

## Requirements

- Linux DGX Spark nodes with working NVIDIA drivers and NVIDIA Container Toolkit configured for Docker. The selected node count must be supported by the model recipe.
- Active RoCEv2 IPv4 connectivity between all selected nodes and access to `/dev/infiniband`.
- Key-based SSH from the head to every worker, without an interactive password prompt.
- Permission to run Docker on every node.
- `bash`, `docker`, `ssh`, `python3`, `curl`, `ss`, `flock`, `realpath`, and `sha256sum` on the head. The guide also uses `ip` to obtain the head's fabric address.
- A writable state directory at the **same absolute path on every node**, backed by each node's local SSD.
- The source checkpoint available on the head, or a supported official model alias that Ignition can download.

Each worker receives its checkpoint ranges through the official preparation protocol. It does not need a full copy of the source checkpoint. Stop other GPU inference workloads before preparation.

## 1. Download the files on the head

Always download [flare_cluster.sh](flare_cluster.sh) on the head, including when using a pulled image. If you choose to build the image yourself, also download [Dockerfile](Dockerfile) and [.dockerignore](.dockerignore) into the same directory. On GitHub, open each file and use **Download raw file**, or download the repository ZIP and open the directory containing this README.

Check that the files are present before continuing:

```bash
ls -l flare_cluster.sh
```

The workers do not need a copy of the launcher. The head controls them through SSH. Commands use `bash flare_cluster.sh`, so no executable-bit change is required.

## 2. Build or pull the image on the head

Choose **one** of the following methods. Both produce the local tag `flare_spark:pre-0.1` used by the launch examples.

### Option A: build it yourself

Run this from the directory containing `Dockerfile` and `.dockerignore` on the head Spark:

```bash
docker build -t flare_spark:pre-0.1 .
```

### Option B: pull the published image

Once the image tag has been published, run this on the head:

```bash
docker pull pilcothink/flare_spark:pre-0.1
docker tag pilcothink/flare_spark:pre-0.1 flare_spark:pre-0.1
```

The second command gives the pulled image the same local tag used by the build method; it does not rebuild the image. The host-side `flare_cluster.sh` is still required. The launcher can then distribute this image to the workers with `--sync-image`.

`pre-0.1` is this recipe's image tag, not an upstream FLARE release number.

| Component | Build selection |
|---|---|
| FLARE source | `d5fd9824593c2a70f6bdc001b39f16b4c5b3b0b4` |
| Base image | `nvidia/cuda:13.2.0-devel-ubuntu24.04` |
| Python | 3.12 |
| PyTorch | Upstream-locked `2.14.1+cu132` |
| Rust | 1.94.1 |
| Preparation dependencies | The pinned upstream `uv.lock` |
| Native build dependencies | The pinned upstream Cargo lockfiles |

The image includes both Ignition and the native Flare executable. There is no separate preparation image to build or distribute. UV and Cargo use persistent BuildKit caches.

The Dockerfile corrects a known ARM64 packaging issue in the locked cuSPARSELt 0.8.1 wheel: its internal platform tag says `sbsa` although the library is AArch64. It checks the architecture, corrects that metadata and its record, and keeps the library bytes and version unchanged. Dependency checks remain enabled. No model-computation source patch is applied.

## 3. Choose image transfer on launch

After building or pulling the image on the head, choose either transfer method. Obtaining the image and transferring it are separate steps; `--sync-image` belongs to the **launcher command**, not `docker build` or `docker pull`.

**Automatic transfer:** add `--sync-image` after `--workers` in the launcher command, as shown in the guide. When you run that command:

1. The launcher checks the head's local image and connects to each listed worker through SSH.
2. It compares each worker's image ID with the head's image ID.
3. If the image is missing or different, it streams `docker save` from the head through SSH into `docker load` on that worker.
4. It verifies the image IDs again, then continues to model preparation or package reuse.

An identical worker image is skipped. Every listed worker is checked. This option does not build an image, download a checkpoint, or copy the host-side launcher. Prepared model packages are distributed later by FLARE itself.

**Manual transfer:** save the image on the head, copy the archive yourself, and load it on every worker:

```bash
# Head
docker save -o flare_spark_pre-0.1.tar flare_spark:pre-0.1

# Each worker, after copying the archive
docker load -i flare_spark_pre-0.1.tar
```

Without `--sync-image`, the launcher stops if image IDs differ. It does not automatically pull an image or discover worker nodes. `--workers` supplies the SSH destination explicitly.

## 4. Choose a model guide and launch from the head

Select a guide from [Guide/](Guide/) and use its complete launch command. Model selection, supported rank counts, context capacity, concurrency, MTP settings, and measured results belong to the individual guides.

Only the head runs `flare_cluster.sh`; the launcher starts and stops the worker containers over SSH. List workers only in `--workers`, separated by commas without spaces. The launcher calculates the total rank count as one head plus the listed workers; the model recipe must support that count.

The sequence is:

1. Check SSH and image identity; transfer mismatched or missing worker images only if `--sync-image` was supplied.
2. Check I/O permissions, writable export/cache paths, the local publisher, and RoCE queue-pair connections on all selected nodes.
3. Reuse a matching completed package, or run official `ignition export` on the head and `ignition join` on every worker. New preparation loads weights, tunes and compiles kernels, compiles model forwards, captures graphs, and exports a signed runtime package.
4. Verify the package with official `flare inspect`, then start official `flare run` on all ranks. FLARE distributes and verifies each worker's rank package before GPU loading.
5. Load weights and execution resources, allocate cache, connect the serving transport, and wait for HTTP health readiness on the head.

**First preparation can take more than an hour.** It is separate from building the Docker image. A completed package avoids repeating that preparation on later identical launches; serving still needs to load its weights and execution resources.

Changing the rank count requires a new package. The launcher selects a different package directory automatically; the image does not need rebuilding, and existing packages are retained. Model loading, graph capture, and export run again for the new rank count; compatible compilation caches may be reused.

## Persistent files and reuse

Keep the model and state directories **outside this source checkout**. The state directory can contain authentication tokens and private signing credentials; do not commit it.

| Path under `--state-dir` | Purpose |
|---|---|
| `preparation/` | Official configuration, preparation token, and the head's development signing credentials |
| `preparation/export/` | Writable temporary export staging |
| `cache/kernels/` | Kernel, tuning, and compiled-forward caches |
| `cache/packages/` | Reusable package data from preparation |
| `packages/<identifier>/` | The head's `model.safetensors` and companion payload files |
| `public/` | Public publisher verification profile |
| `runtime/` | Serving trust store and received rank-package cache |
| `logs/<launch-id>/` | Container logs preserved when a phase finishes or the launch stops |

The package directory identifier is derived from the image ID, model path and metadata, rank count, and preparation arguments. **It is a cache identifier, not a password or signing key.** Preparation-argument order currently affects it; keep the same launch command for reuse.

If export did not finish, a later launch still needs model loading, capture, and export. Existing tuning and compilation caches remain available for reuse. If checkpoint tensors are changed in place without changing metadata, use `--fresh`: the launcher does not hash every weight byte on each launch. `--fresh` creates a separate package without deleting earlier packages or caches.

Pressing `Ctrl-C` stops and removes only the containers created by that launch. State, caches, packages, and logs remain on the SSD. Files added only to a container's unmounted filesystem disappear when that container is removed.

## Upstream behavior and launcher behavior

FLARE/B12X owns model computation, precision selection, tuning, compilation, package format, signatures, cache management, and serving. Model settings are supplied through launch arguments; the Dockerfile does not force concurrency, MTP depth, prefill size, or a memory limit.

This repository's launcher adds SSH orchestration, optional image transfer, directory layout, preflight checks, package reuse selection, and lifecycle/log handling. **The launcher is not an NVIDIA-provided script.** Its checks use the pinned upstream interfaces, including the B12X native proxy for the small queue-pair check.

During preparation, the launcher finds active RoCEv2 IPv4 entries matching the fabric address and passes the supported B12X HCA/GID settings per node. It does not assume GID index 3. Native serving retains FLARE's own fabric discovery; the Python B12X environment settings do not pin the native server's fabric selection.

The containers use `seccomp=unconfined` so Linux `io_uring` is available. This disables the container's seccomp filter; it is a security tradeoff, not a setting with zero security impact. The launcher does not use `--privileged` or change host IP addresses, routes, firewall rules, or sysctl values. Private signing keys remain on the head and are not mounted into serving containers. Package signatures are verified before the exact local development key and namespace are accepted.

The example binds the API to all host interfaces and does not configure API authentication. Use an appropriate trusted network and choose the listener address deliberately.

## Upstream references

- [NVIDIA FLARE](https://github.com/NVIDIA/flare-local-inference)
- [Pinned model preparation and serving guide](https://github.com/NVIDIA/flare-local-inference/blob/d5fd9824593c2a70f6bdc001b39f16b4c5b3b0b4/docs/model-preparation.md)
- [Pinned multi-host guide](https://github.com/NVIDIA/flare-local-inference/blob/d5fd9824593c2a70f6bdc001b39f16b4c5b3b0b4/docs/multi-host.md)
- [Pinned CLI and configuration reference](https://github.com/NVIDIA/flare-local-inference/blob/d5fd9824593c2a70f6bdc001b39f16b4c5b3b0b4/docs/cli-design.md)
- [Upstream LICENSE](https://github.com/NVIDIA/flare-local-inference/blob/d5fd9824593c2a70f6bdc001b39f16b4c5b3b0b4/LICENSE) and [NOTICE](https://github.com/NVIDIA/flare-local-inference/blob/d5fd9824593c2a70f6bdc001b39f16b4c5b3b0b4/NOTICE)

## Authorship and maintenance

This documentation was generated by AI. Hands-on testing was performed by the repository owner. Revisions, maintenance, and distribution are handled with AI assistance.
