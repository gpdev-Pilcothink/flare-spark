# syntax=docker/dockerfile:1.7
# ============================================================================
# FLARE: one image for Ignition preparation AND native Rust serving
#
#   docker build -t flare_spark:pre-0.2 .
#
# Building this image installs software, without loading any model or GPU.
# flare_cluster.sh runs preparation peers and serving ranks from the head.
# One tag is installed on every node; no second runtime image is required.
# Resource settings belong to launch arguments, not this image.
#
# CUDA 13.2 is deliberate: this upstream revision pins PyTorch 2.14.1+cu132.
# Do not substitute the older vLLM image's Python/CUDA dependency set.
# ============================================================================

FROM astral/uv:0.13.5 AS uv-bin
FROM nvidia/cuda:13.2.0-devel-ubuntu24.04 AS prepare
ARG TARGETARCH

# ---- 1 / System dependencies; independent of Flare source changes -----------
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential ca-certificates cmake curl git nasm ninja-build pkg-config \
       protobuf-compiler libssl-dev liburing-dev libibverbs-dev ibverbs-providers \
       python3.12 python3.12-dev python3.12-venv \
    && rm -rf /var/lib/apt/lists/*
COPY --from=uv-bin /uv /uvx /usr/local/bin/

# Compile the serving executable in this SAME Ubuntu image. Ignition and
# Flare then use a matching native build and libc throughout the workflow.
ENV CARGO_HOME=/opt/cargo \
    RUSTUP_HOME=/opt/rustup \
    CUDA_HOME=/usr/local/cuda \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PATH=/opt/venv/bin:/opt/cargo/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ARG RUST_VERSION=1.94.1
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs -o /tmp/rustup-init.sh \
    && sh /tmp/rustup-init.sh -y --profile minimal --default-toolchain "${RUST_VERSION}" \
    && rm /tmp/rustup-init.sh

# ---- 2 / Pinned official source; no model-computation patch ------------------
ARG FLARE_REF=eb71702b2e710cb407702ed0cde2970c8e3bfdad
WORKDIR /opt/flare
RUN git init . \
    && git remote add origin https://github.com/NVIDIA/spark-local-inference.git \
    && git fetch --depth 1 origin "${FLARE_REF}" \
    && git checkout --detach FETCH_HEAD \
    && test "$(git rev-parse HEAD)" = "${FLARE_REF}"

# ---- 3 / Locked upstream preparation dependencies and local Rust extension --
# Keep upstream's editable checkout install: source_identity() needs ignition,
# helios/src, and heliopause/src under the SAME source root on both hosts.
# This checkout stays in the preparation image; it is not a temporary uv cache.
# Each native build below has persistent, architecture-specific BuildKit caches.
RUN --mount=type=cache,id=flare-prepare-uv-${TARGETARCH},target=/root/.cache/uv,sharing=locked \
    --mount=type=cache,id=flare-prepare-registry,target=/opt/cargo/registry,sharing=locked \
    --mount=type=cache,id=flare-prepare-git,target=/opt/cargo/git,sharing=locked \
    --mount=type=cache,id=flare-prepare-target-${TARGETARCH},target=/tmp/flare-cargo-target,sharing=locked \
    CARGO_TARGET_DIR=/tmp/flare-cargo-target \
      uv sync --locked --python /usr/bin/python3.12 --extra prepare --extra test

# The pinned upstream release includes three exact-context Torch/Triton fixes.
# Apply them while the image is built so non-root preparation containers can
# reuse the already-correct environment without writing to /opt/venv at startup.
# This is the upstream preparation mechanism, not a model or precision patch.
RUN /opt/venv/bin/python3 -m ignition.patches

# ---- 4 / Native helper required by multi-host preparation -------------------
RUN --mount=type=cache,id=flare-prepare-registry,target=/opt/cargo/registry,sharing=locked \
    --mount=type=cache,id=flare-prepare-git,target=/opt/cargo/git,sharing=locked \
    --mount=type=cache,id=flare-prepare-target-${TARGETARCH},target=/tmp/flare-cargo-target,sharing=locked \
    cargo build --locked --release --manifest-path flare/Cargo.toml --bin flare \
      --target-dir /tmp/flare-cargo-target \
    && install -Dm755 /tmp/flare-cargo-target/release/flare /usr/local/bin/flare \
    && strip /usr/local/bin/flare \
    && flare --version \
    && ignition export glm-5.3-flash --help >/dev/null \
    && ignition join --help >/dev/null \
    && python3 -c 'from ignition.preparation.source_identity import source_identity; assert source_identity()'

# ---- 5 / Packaging correction and strict checks; after cached native builds -
# NVIDIA's locked 0.8.1 ARM64 wheel has the right AArch64 library, but its
# internal WHEEL says "manylinux2014_sbsa" instead of the filename's "aarch64".
# Verified vendor wheel: https://pypi.org/project/nvidia-cusparselt-cu13/0.8.1/
# Related upstream report: https://github.com/astral-sh/uv/issues/17711
# Correct ONLY that exact installed metadata tag after checking ELF e_machine.
# Keep version 0.8.1, the lockfile, and every library byte unchanged. Update the
# WHEEL entry in RECORD too; uv pip check remains mandatory for all packages.
# UV_LINK_MODE=copy above keeps this correction out of the shared wheel cache.
RUN <<'FLARE_PACKAGE_CHECK'
set -eu
python3 - <<'CUSPARSELT_CHECK'
import base64
import csv
import ctypes
import hashlib
import io
import platform
import sys
from importlib.metadata import distribution
from pathlib import Path


def correct_cusparselt_tag():
    if sys.platform != "linux" or platform.machine() != "aarch64":
        return None
    dist = distribution("nvidia-cusparselt-cu13")
    if dist.version != "0.8.1":
        raise RuntimeError(f"Unexpected locked cuSPARSELt version: {dist.version}")
    info = Path("nvidia_cusparselt_cu13-0.8.1.dist-info")
    wheel = Path(dist.locate_file(info / "WHEEL"))
    record = Path(dist.locate_file(info / "RECORD"))
    library = Path(dist.locate_file("nvidia/cusparselt/lib/libcusparseLt.so.0"))
    with library.open("rb") as stream:
        header = stream.read(20)
    if (len(header) != 20 or header[:7] != b"\x7fELF\x02\x01\x01"
            or int.from_bytes(header[18:20], "little") != 183):
        raise RuntimeError("cuSPARSELt must contain a 64-bit little-endian AArch64 ELF")
    before = wheel.read_bytes()
    tags = [line for line in before.splitlines() if line.startswith(b"Tag:")]
    old_tag = b"Tag: py3-none-manylinux2014_sbsa"
    new_tag = b"Tag: py3-none-manylinux2014_aarch64"
    if tags == [new_tag]:
        print("FLARE_CUSPARSELT_AARCH64_TAG_ALREADY_CORRECT")
        return library
    if tags != [old_tag]:
        raise RuntimeError(f"Unexpected cuSPARSELt WHEEL tags: {tags!r}")
    rows = list(csv.reader(io.StringIO(record.read_text(encoding="utf-8"))))
    matches = [row for row in rows if row and row[0] == (info / "WHEEL").as_posix()]
    digest = lambda data: "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode("ascii")
    if (len(matches) != 1 or len(matches[0]) != 3
            or matches[0][1:] != [digest(before), str(len(before))]):
        raise RuntimeError("cuSPARSELt WHEEL does not match its installed RECORD")
    after = before.replace(old_tag, new_tag, 1)
    matches[0][1:] = [digest(after), str(len(after))]
    output = io.StringIO(newline="")
    csv.writer(output, lineterminator="\n").writerows(rows)
    wheel.write_bytes(after)
    record.write_text(output.getvalue(), encoding="utf-8")
    print("FLARE_CUSPARSELT_AARCH64_TAG_FIXED: sbsa -> aarch64; library unchanged")
    return library


library = correct_cusparselt_tag()
import torch
import flare_heliopause

assert torch.__version__.split("+")[0] == "2.14.1", torch.__version__
assert torch.version.cuda == "13.2", torch.version.cuda
if library is not None:
    ctypes.CDLL(str(library))
print("FLARE_TORCH_HELIOPAUSE_IMPORT_PASS (GPU execution not tested)")
CUSPARSELT_CHECK
uv pip check --python /opt/venv/bin/python3
FLARE_PACKAGE_CHECK

LABEL org.opencontainers.image.source="https://github.com/NVIDIA/spark-local-inference" \
      org.opencontainers.image.revision="${FLARE_REF}"
ENTRYPOINT ["/usr/local/bin/flare"]
CMD ["--help"]
