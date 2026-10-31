#!/usr/bin/env bash
# Head-driven Flare launcher. One image, one command on the head, SSH peers.
# The official Ignition and Flare programs own all model computation.
# This file only orchestrates their pinned CLIs, container paths and diagnostics.
set -euo pipefail

# ── 01 / Arguments and validation ────────────────────────────────────────────
usage() {
  cat <<'HELP'
Usage: bash flare_cluster.sh IMAGE HEAD_IP --workers SSH_HOST[,SSH_HOST...] \
  --model CHECKPOINT --state-dir ABSOLUTE_PATH [options]

IMAGE example: flare_spark:pre-0.1
CHECKPOINT: local checkpoint directory on the head, or a Flare recipe alias.
The state directory must be accessible at the same absolute path on all nodes.

Cluster options:
  --sync-image                 Copy the head image to peers when their ID differs.
  --fresh                      Export a new package, keeping existing packages.
  --prepare-port N             Preparation rendezvous (default 29630).
  --master-port N              Serving rendezvous (default 29800).
  --host ADDRESS --port N      HTTP listener (defaults 0.0.0.0, 8000).
  --served-model-name NAME     Optional API alias.
  --memory-limit LIMIT         Optional native serving memory limit.
  --dry-run                    Print the launch flow; no SSH, Docker or file writes.

Official preparation options, forwarded only when supplied:
  --max-concurrency N --mtp-draft-tokens N --prefill-chunk-size N
  --image-concurrency N --compile-workers N --autotune-effort LEVEL
  --speculative-token-capacity N --no-mtp --no-mtp-adaptive-verification
  --no-context-parallel --no-stream-overlap --language-model-only

First run: export on the head + join on peers, then serve on all ranks.
Later runs: reuse the matching signed package. Ctrl-C stops this launch's ranks.
Changing checkpoint tensor contents in place requires --fresh; the reuse key
checks metadata and arguments, not every weight byte.
HELP
}
fail() { printf 'Error: %s\n' "$*" >&2; exit 1; }
say() { printf '[flare_cluster] %s\n' "$*"; }
value() { [[ $# -ge 2 && -n "$2" ]] || fail "$1 needs a value"; }
port_ok() { [[ "$1" =~ ^[0-9]+$ ]] && ((10#$1 > 0 && 10#$1 <= 65535)); }
shell_join() { local a; for a in "$@"; do printf '%q ' "$a"; done; }
show() { printf '  '; shell_join "$@"; printf '\n'; }

if [[ ${1:-} == --help || ${1:-} == -h ]]; then usage; exit 0; fi
[[ $# -ge 2 ]] || { usage; exit 1; }
IMAGE=$1 HEAD_IP=$2; shift 2
WORKERS='' MODEL='' DATA='' ALIAS='' MEMORY=''
PREP_PORT=29630 MASTER_PORT=29800 HTTP_HOST=0.0.0.0 HTTP_PORT=8000
SYNC=false FRESH=false DRY=false
PREP_ARGS=() SERVE_ARGS=()
while (($#)); do
  if [[ "$1" == --*=* ]]; then set -- "${1%%=*}" "${1#*=}" "${@:2}"; fi
  case "$1" in
    --workers) value "$@"; WORKERS=$2; shift 2 ;;
    --model) value "$@"; MODEL=$2; shift 2 ;;
    --state-dir) value "$@"; DATA=$2; shift 2 ;;
    --prepare-port) value "$@"; PREP_PORT=$2; shift 2 ;;
    --master-port) value "$@"; MASTER_PORT=$2; shift 2 ;;
    --host) value "$@"; HTTP_HOST=$2; shift 2 ;;
    --port) value "$@"; HTTP_PORT=$2; shift 2 ;;
    --served-model-name) value "$@"; ALIAS=$2; shift 2 ;;
    --memory-limit) value "$@"; MEMORY=$2; shift 2 ;;
    --max-concurrency|--mtp-draft-tokens)
      value "$@"; PREP_ARGS+=("$1" "$2"); SERVE_ARGS+=("$1" "$2"); shift 2 ;;
    --prefill-chunk-size|--image-concurrency|--compile-workers|--autotune-effort|--speculative-token-capacity)
      value "$@"; PREP_ARGS+=("$1" "$2"); shift 2 ;;
    --no-mtp) PREP_ARGS+=("$1"); shift ;;
    --no-mtp-adaptive-verification) PREP_ARGS+=("$1"); SERVE_ARGS+=("$1"); shift ;;
    --no-context-parallel|--no-stream-overlap|--language-model-only)
      PREP_ARGS+=("$1"); shift ;;
    --sync-image) SYNC=true; shift ;;
    --fresh) FRESH=true; shift ;;
    --dry-run) DRY=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) fail "Unknown option: $1" ;;
  esac
done
[[ -n "$WORKERS" && -n "$MODEL" && -n "$DATA" ]] || fail '--workers, --model and --state-dir are required'
[[ "$HEAD_IP" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]] || fail 'HEAD_IP must be a reachable fabric IPv4 address'
[[ "$DATA" == /* && "$DATA" != / && "$DATA" != *','* && "$DATA" != *$'\n'* ]] || fail '--state-dir must be an absolute path without commas/newlines'
for p in "$PREP_PORT" "$MASTER_PORT" "$HTTP_PORT"; do port_ok "$p" || fail "Invalid port: $p"; done
[[ "$PREP_PORT" != "$MASTER_PORT" && "$PREP_PORT" != "$HTTP_PORT" && "$MASTER_PORT" != "$HTTP_PORT" ]] || fail 'The three ports must differ'
IFS=',' read -r -a PEERS <<< "$WORKERS"
for peer in "${PEERS[@]}"; do
  [[ "$peer" =~ ^([A-Za-z0-9._-]+@)?[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || fail "Invalid SSH peer: $peer"
  [[ "${peer##*@}" != "$HEAD_IP" ]] || fail 'The head cannot also be a worker'
done
[[ $(printf '%s\n' "${PEERS[@]}" | sort -u | wc -l) -eq ${#PEERS[@]} ]] || fail 'Duplicate worker'
RANKS=$((${#PEERS[@]} + 1))
((RANKS <= 16)) || fail 'The upstream transport accepts at most 16 ranks'
if [[ -n "$MEMORY" ]]; then SERVE_ARGS+=(--memory-limit "$MEMORY"); fi

# B12X checkpoint reads and Helios disk I/O require io_uring. Docker's default
# seccomp profile blocks its setup/enter/register syscalls. Disable that filter
# for these trusted Flare containers; retain the explicit user/device/cap limits.
# This is a container launch setting, not an image or model-computation change.
CONTAINER_SECURITY=(--security-opt seccomp=unconfined)

# The preview is deliberately side-effect free and needs no local model or GPU.
if $DRY; then
  say "One image: $IMAGE; $RANKS ranks; commands are issued only on the head."
  say 'If no matching package exists, prepare it:'
  show docker run "${CONTAINER_SECURITY[@]}" --entrypoint /opt/venv/bin/ignition "$IMAGE" export "$MODEL" \
    --ranks "$RANKS" --rendezvous "$HEAD_IP:$PREP_PORT" --gpus 0 "${PREP_ARGS[@]}"
  for i in "${!PEERS[@]}"; do
    show ssh "${PEERS[$i]}" docker run "${CONTAINER_SECURITY[@]}" --entrypoint /opt/venv/bin/ignition "$IMAGE" \
      join "$HEAD_IP:$PREP_PORT" --rank "$((i+1))" --ranks "$RANKS" --gpu 0
  done
  say 'Then verify the locally signed package and start its serving ranks:'
  show docker run "${CONTAINER_SECURITY[@]}" --entrypoint /usr/local/bin/flare "$IMAGE" run '<prepared-package>' \
    --rank 0 --rendezvous "$HEAD_IP:$MASTER_PORT" "${SERVE_ARGS[@]}" --host "$HTTP_HOST" --port "$HTTP_PORT"
  for i in "${!PEERS[@]}"; do
    show ssh "${PEERS[$i]}" docker run "${CONTAINER_SECURITY[@]}" --entrypoint /usr/local/bin/flare "$IMAGE" run \
      --rank "$((i+1))" --rendezvous "$HEAD_IP:$MASTER_PORT" "${SERVE_ARGS[@]}"
  done
  exit 0
fi

# ── 02 / State, SSH and identical images ──────────────────────────────────────
for tool in docker ssh python3 curl ss flock realpath sha256sum; do command -v "$tool" >/dev/null || fail "Missing command: $tool"; done
DATA=$(realpath -m -- "$DATA")
[[ "$DATA" != *','* ]] || fail 'The resolved state directory contains a comma'
MODEL_MOUNT=()
MODEL_INPUT=$MODEL
if [[ -d "$MODEL" ]]; then
  MODEL=$(realpath -e -- "$MODEL")
  [[ -f "$MODEL/config.json" ]] || fail "Missing checkpoint config: $MODEL/config.json"
  [[ "$MODEL" != *','* && "$MODEL" != *$'\n'* ]] || fail 'Checkpoint mount path contains a comma/newline'
  MODEL_INPUT=/model
  MODEL_MOUNT=(--mount "type=bind,src=$MODEL,dst=/model,readonly")
elif [[ "$MODEL" == /* || "$MODEL" == ./* || "$MODEL" == ../* ]]; then
  fail "Local checkpoint directory does not exist: $MODEL"
fi
mkdir -p "$DATA"/{preparation,public,cache,packages,runtime}
exec 9>"$DATA/.launcher.lock"
flock -n 9 || fail "Another launcher holds $DATA"
SSH_OPTIONS=(-o BatchMode=yes -o ConnectTimeout=10)
remote() { local peer=$1; shift; ssh -n "${SSH_OPTIONS[@]}" "$peer" "$(shell_join "$@")"; }
node() { local n=$1; shift; if ((n == 0)); then "$@"; else remote "${PEERS[$((n-1))]}" "$@"; fi; }
upload() {
  local peer=$1 src=$2 dst=$3
  ssh "${SSH_OPTIONS[@]}" "$peer" "$(shell_join bash -c 'umask 077; cat > "$1.tmp"; mv -f -- "$1.tmp" "$1"' -- "$dst")" < "$src"
}

# Preflight before starting any GPU process. No automatic pull or tag overwrite.
IMAGE_ID=$(docker image inspect --format '{{.Id}}' "$IMAGE") || fail "Build $IMAGE first"
# Detect the installed CLI, not the image tag: older exports accept --output;
# newer exports publish a cache entry and no longer accept that option.
EXPORT_HELP=$(docker run --rm --pull never --entrypoint /opt/venv/bin/ignition "$IMAGE" export --help) \
  || fail 'Cannot inspect ignition export options in the selected image'
if printf '%s\n' "$EXPORT_HELP" | grep -qE -- '(^|[[:space:]])--output([[:space:]=,]|$)'; then
  EXPORT_MODE=output
else
  EXPORT_MODE=cache
fi
say "Ignition export mode: $EXPORT_MODE"
USERS=("$(id -u):$(id -g)")
for peer in "${PEERS[@]}"; do
  say "Checking SSH and image on $peer"
  remote "$peer" true || fail "Passwordless SSH failed: $peer"
  peer_id=$(remote "$peer" docker image inspect --format '{{.Id}}' "$IMAGE" 2>/dev/null) || peer_id=''
  if [[ "$peer_id" != "$IMAGE_ID" ]]; then
    $SYNC || fail "Image on $peer differs or is missing; use --sync-image"
    say "Sending $IMAGE to $peer"
    docker save "$IMAGE" | ssh "${SSH_OPTIONS[@]}" "$peer" docker load
    [[ $(remote "$peer" docker image inspect --format '{{.Id}}' "$IMAGE") == "$IMAGE_ID" ]] || fail "Image ID mismatch on $peer"
  fi
  USERS+=("$(remote "$peer" bash -c 'printf "%s:%s" "$(id -u)" "$(id -g)"')")
  remote "$peer" mkdir -p -- "$DATA/preparation" "$DATA/public" "$DATA/cache" "$DATA/packages" "$DATA/runtime"
done
for p in "$PREP_PORT" "$MASTER_PORT" "$HTTP_PORT"; do
  [[ -z $(ss -H -ltn "sport = :$p") ]] || fail "Head port $p is already in use"
done

# ── 03 / Kernel I/O permission and RoCE device discovery ──────────────────────
# Check the actual kernel/container policy on every host before loading weights.
# No GPU is opened, no sysctl is changed, and no model file is read by this probe.
URING_PROBE=$(cat <<'PY'
import ctypes
import os
from pathlib import Path

lib = ctypes.CDLL("liburing.so.2")
lib.io_uring_queue_init.argtypes = [ctypes.c_uint, ctypes.c_void_p, ctypes.c_uint]
lib.io_uring_queue_init.restype = ctypes.c_int
lib.io_uring_queue_exit.argtypes = [ctypes.c_void_p]
lib.io_uring_queue_exit.restype = None
# Opaque, zeroed, generously sized storage for liburing's small control struct.
ring = ctypes.create_string_buffer(4096)
rc = lib.io_uring_queue_init(8, ctypes.byref(ring), 0)
if rc < 0:
    setting = Path("/proc/sys/kernel/io_uring_disabled")
    disabled = setting.read_text().strip() if setting.exists() else "unavailable"
    raise SystemExit(f"io_uring preflight failed: {os.strerror(-rc)} (errno={-rc}, kernel.io_uring_disabled={disabled}); check host kernel policy")
lib.io_uring_queue_exit(ctypes.byref(ring))
print("FLARE_IO_URING_PREFLIGHT_PASS", flush=True)
PY
)
for ((n=0;n<RANKS;n++)); do
  say "Checking io_uring policy on rank $n before model loading"
  node "$n" docker run --rm --pull never --user "${USERS[$n]}" \
    "${CONTAINER_SECURITY[@]}" --cap-add IPC_LOCK --ulimit memlock=-1:-1 \
    --entrypoint /opt/venv/bin/python3 "$IMAGE" -c "$URING_PROBE" \
    || fail "io_uring preflight failed on rank $n; no model preparation was started"
done

# FLARE_RDMA_DISCOVERY: select populated IPv4 RoCEv2 GIDs for the actual fabric
# address, rather than assuming that every machine's GID index is 3. These are
# official B12X environment options; no source or global network setting changes.
RDMA_DISCOVERY=$(cat <<'PY'
import ipaddress
import json
import socket
import sys
from pathlib import Path

target, is_head = sys.argv[1:]
if is_head == "1":
    address = ipaddress.IPv4Address(target)
else:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as route:
        route.connect((target, 9))  # Routing lookup only; sends no datagram.
        address = ipaddress.IPv4Address(route.getsockname()[0])
root = Path("/sys/class/infiniband")
groups, observed = {}, []
for hca in sorted(root.glob("*")):
    port = hca / "ports/1"
    try:
        if "ACTIVE" not in (port / "state").read_text():
            continue
        for gid_file in sorted((port / "gids").iterdir(), key=lambda p: int(p.name)):
            index = int(gid_file.name)
            try:
                gid = ipaddress.IPv6Address(gid_file.read_text().strip())
                kind = (port / "gid_attrs/types" / str(index)).read_text().strip()
                netdev = (port / "gid_attrs/ndevs" / str(index)).read_text().strip()
            except (OSError, ValueError):
                continue
            if gid.ipv4_mapped is not None:
                observed.append(f"{hca.name}:gid{index}:{kind}:{gid.ipv4_mapped}:{netdev}")
            if kind.casefold() == "roce v2" and gid.ipv4_mapped == address and netdev:
                groups.setdefault(index, []).append((hca.name, netdev))
    except (OSError, ValueError):
        continue
if not groups:
    raise SystemExit(f"No active IPv4 RoCEv2 GID matches fabric source {address}; observed={observed}")
# Retain up to four matching functions, as B12X supports. Prefer more usable
# functions; index 3 is only a tie-breaker after validating its type and address.
index = min(groups, key=lambda i: (-len(groups[i]), i != 3, i))
devices = groups[index][:4]
print(json.dumps(dict(hcas=",".join(x[0] for x in devices), gid=index,
                     netdev=devices[0][1], ipv4=str(address))))
PY
)
RDMA_HCAS=() RDMA_GIDS=() RDMA_NETDEVS=()
for ((n=0;n<RANKS;n++)); do
  is_head=0; ((n != 0)) || is_head=1
  selection=$(node "$n" docker run --rm --pull never --network host \
    --user "${USERS[$n]}" --device /dev/infiniband --entrypoint /opt/venv/bin/python3 \
    "$IMAGE" -c "$RDMA_DISCOVERY" "$HEAD_IP" "$is_head") \
    || fail "Cannot select an IPv4 RoCEv2 fabric on rank $n; no model was loaded"
  fields=$(printf '%s' "$selection" | python3 -c 'import json,sys; x=json.load(sys.stdin); print(x["hcas"],x["gid"],x["netdev"],x["ipv4"])') \
    || fail "Invalid RDMA discovery result from rank $n"
  read -r hcas gid netdev address <<< "$fields"
  RDMA_HCAS+=("$hcas"); RDMA_GIDS+=("$gid"); RDMA_NETDEVS+=("$netdev")
  say "RDMA rank $n: HCA=$hcas GID=$gid RoCEv2 IPv4=$address netdev=$netdev"
done

# ── 04 / Existing package identity and official configuration ─────────────────
# Only small metadata is read for package reuse; never scan all model weights.
KEY=$({
  printf '%s\0' "$IMAGE_ID" "$MODEL" "$RANKS" "${PREP_ARGS[@]}"
  if ((${#MODEL_MOUNT[@]})); then
    for f in config.json model.safetensors.index.json quantization_config.json tokenizer_config.json; do
      if [[ -f "$MODEL/$f" ]]; then sha256sum -- "$MODEL/$f"; fi
    done
  fi
  if $FRESH; then printf '%s\0' "$(date +%s%N)-$$"; fi
} | sha256sum | cut -c1-24)
PACKAGE_DIR="$DATA/packages/$KEY"
PACKAGE_REFERENCE="$PACKAGE_DIR/cache-entry"
PACKAGE_CACHE="$DATA/cache/packages"
PACKAGE=''
PACKAGE_ARG=''
mkdir -p "$PACKAGE_DIR" "$PACKAGE_CACHE"
cat > "$DATA/preparation/config.toml" <<'TOML'
[preparation]
checkpoint_token_file = "/state/preparation/agent.token"
[paths]
hub_cache = "/state/cache/huggingface/hub"
kernel_cache = "/state/cache/kernels"
package_cache = "/state/cache/packages"
export_staging = "/state/preparation/export"
cuda = "/usr/local/cuda"
TOML
if [[ ! -f "$DATA/preparation/agent.token" ]]; then
  python3 -c 'import os,secrets,sys; fd=os.open(sys.argv[1],os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600); os.write(fd,secrets.token_bytes(32)); os.close(fd)' "$DATA/preparation/agent.token"
fi
chmod 600 "$DATA/preparation/agent.token"
for peer in "${PEERS[@]}"; do
  upload "$peer" "$DATA/preparation/config.toml" "$DATA/preparation/config.toml"
  upload "$peer" "$DATA/preparation/agent.token" "$DATA/preparation/agent.token"
done

# ── 05 / Export/cache permissions and official development publisher ──────────
# Validate the exact native export constructor before an expensive GPU prepare.
# A peer with no export_staging otherwise stages relative to the image WORKDIR
# (/opt/flare), which is intentionally not writable by the host UID.
EXPORT_PATH_PROBE=$(cat <<'PY'
from pathlib import Path
import secrets
import sys
import tempfile
import torch
import flare_heliopause
from ignition.cli.config import Configuration

configuration = Configuration.load(Path("/state/preparation/config.toml"))
staging = configuration.directory("export_staging")
if staging is None or not staging.is_absolute():
    raise SystemExit("A writable absolute paths.export_staging is required")
paths = [staging, configuration.directory("kernel_cache"), configuration.directory("package_cache"),
         Path("/state/packages"), Path("/state/public"), Path("/state/runtime"), Path.home()]
for path in paths:
    if path is None:
        raise SystemExit("A preparation/cache directory is missing from configuration")
    path.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".flare-write-check-", dir=path) as directory:
        (Path(directory) / "probe").write_bytes(b"ok")
stage = flare_heliopause.NativeExportStage(staging / f"probe-{secrets.token_hex(8)}.json", staging)
try:
    if not Path(stage.path).is_dir():
        raise RuntimeError("NativeExportStage did not create a directory")
finally:
    stage.close()
# This is the same official publisher helper used by Ignition export. Create or
# validate only the head's existing key; private keys are never sent to peers.
if sys.argv[1] == "0":
    from ignition.development import publisher
    signer = publisher(configuration)
    if not signer.development or signer.namespace != "local-development":
        raise RuntimeError("Expected the official local development publisher")
    print(f"FLARE_PUBLISHER_PREFLIGHT_PASS fingerprint={signer.fingerprint.hex()}", flush=True)
print(f"FLARE_EXPORT_PATH_PREFLIGHT_PASS staging={staging}", flush=True)
PY
)
for ((n=0;n<RANKS;n++)); do
  say "Checking writable export/cache paths on rank $n before model loading"
  node "$n" docker run --rm --pull never --user "${USERS[$n]}" \
    --env HOME=/state/home --workdir /state/preparation \
    --mount "type=bind,src=$DATA,dst=/state" --entrypoint /opt/venv/bin/python3 \
    "$IMAGE" -c "$EXPORT_PATH_PROBE" "$n" \
    || fail "Native export path preflight failed on rank $n; no model was loaded"
done

# ── 06 / Container lifecycle; shared by checks, preparation and serving ───────
RUN_ID="flare-$(date +%s)-$$"
RUN_LOGS="$DATA/logs/$RUN_ID"
(umask 077; mkdir -p "$RUN_LOGS")
NAMES=() ACTIVE=() LOG_PIDS=()
save_log() {
  local n=$1 name=$2
  if ! (umask 077; node "$n" docker logs "$name" > "$RUN_LOGS/$name.log" 2>&1); then
    say "Warning: could not save the container log for $name" >&2
  fi
}
cleanup() {
  local status=$? n name owner
  trap - EXIT INT TERM HUP
  set +e
  for pid in "${LOG_PIDS[@]}"; do kill "$pid" 2>/dev/null; done
  for n in "${ACTIVE[@]}"; do
    name=${NAMES[$n]}
    owner=$(node "$n" docker inspect --format '{{index .Config.Labels "flare.cluster.launch"}}' "$name" 2>/dev/null)
    if [[ "$owner" == "$RUN_ID" ]]; then
      save_log "$n" "$name"
      say "Stopping rank $n: $name"
      node "$n" docker stop --time 60 "$name" >/dev/null 2>&1
      node "$n" docker rm "$name" >/dev/null 2>&1
    fi
  done
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

start() {
  local n=$1 phase=$2 executable=$3; shift 3
  local name="$RUN_ID-$phase-r$n"
  local cmd=(docker run --pull never -d --name "$name" --label "flare.cluster.launch=$RUN_ID"
    --user "${USERS[$n]}" --network host --ipc host "${CONTAINER_SECURITY[@]}"
    --device /dev/infiniband --cap-add IPC_LOCK --ulimit memlock=-1:-1
    --stop-timeout 60 --entrypoint "$executable")
  case "$phase" in
    prepare|roce-check)
      cmd+=(--env HOME=/state/home --workdir /state/preparation
        --mount "type=bind,src=$DATA,dst=/state"
        --env "B12X_ROCE_HCA=${RDMA_HCAS[$n]}" --env "B12X_ROCE_GID_INDEX=${RDMA_GIDS[$n]}"
        --env "GLOO_SOCKET_IFNAME=${RDMA_NETDEVS[$n]}")
      if [[ "$phase" == prepare ]]; then
        cmd+=(--gpus all --ulimit stack=67108864)
        if ((n == 0)); then cmd+=("${MODEL_MOUNT[@]}"); fi
      fi ;;
    serve)
      # Native Flare uses its official fabric discovery, not the Python B12X env.
      # Mount public keys and runtime state only; never preparation private keys.
      cmd+=(--gpus all --ulimit stack=67108864 --env HOME=/runtime --workdir /runtime
        --mount "type=bind,src=$DATA/public,dst=/public,readonly"
        --mount "type=bind,src=$DATA/runtime,dst=/runtime")
      if ((n == 0)); then
        cmd+=(--mount "type=bind,src=$PACKAGE_CACHE,dst=/state/cache/packages,readonly"
          --mount "type=bind,src=$PACKAGE_DIR,dst=/package,readonly")
      fi ;;
    *) fail "Unknown container phase: $phase" ;;
  esac
  NAMES[$n]=$name; ACTIVE+=("$n")
  cmd+=("$IMAGE" "$@")
  say "Starting $phase rank $n"
  node "$n" "${cmd[@]}" >/dev/null
}
running() { [[ $(node "$1" docker inspect --format '{{.State.Running}}' "${NAMES[$1]}") == true ]]; }
logs() { node "$1" docker logs --tail 60 "${NAMES[$1]}" >&2; }
follow() { node "$1" docker logs --follow "${NAMES[$1]}" & LOG_PIDS+=("$!"); }

wait_for_head_port() {
  local port=$1 message=$2
  until [[ -n $(ss -H -ltn "sport = :$port") ]]; do
    running 0 || { logs 0; fail "$message"; }
    sleep 1
  done
}

finish_phase() {
  local label=$1 interval=$2 suffix=${3:-} n code live pid
  while :; do
    live=false
    for n in "${ACTIVE[@]}"; do
      if running "$n"; then live=true
      else
        code=$(node "$n" docker inspect --format '{{.State.ExitCode}}' "${NAMES[$n]}")
        [[ "$code" == 0 ]] || { logs "$n"; fail "$label rank $n exited: $code$suffix"; }
      fi
    done
    $live || break
    sleep "$interval"
  done
  for pid in "${LOG_PIDS[@]}"; do wait "$pid" || true; done
  LOG_PIDS=()
  for n in "${ACTIVE[@]}"; do
    save_log "$n" "${NAMES[$n]}"
    node "$n" docker rm "${NAMES[$n]}" >/dev/null
  done
  ACTIVE=(); NAMES=()
}

# ── 07 / Real B12X QP connection before loading model weights ─────────────────
# Use the installed B12X proxy and route planner to connect real queue pairs
# before the expensive checkpoint read. No CUDA tensors or model allocations.
RDMA_PROBE=$(cat <<'PY'
import argparse
import ctypes
import mmap
import os
from datetime import timedelta
import torch.distributed as dist
from b12x.comm.roce._proxy import Layout, Proxy
from b12x.comm.roce._routes import MAX_RAILS, local_endpoints, plan_routes

parser = argparse.ArgumentParser()
parser.add_argument("--rendezvous", required=True)
parser.add_argument("--world-size", type=int, required=True)
parser.add_argument("--rank", type=int, required=True)
args = parser.parse_args()
names = tuple(os.environ["B12X_ROCE_HCA"].split(","))
gid = int(os.environ["B12X_ROCE_GID_INDEX"])
dist.init_process_group("gloo", init_method="tcp://" + args.rendezvous,
                        world_size=args.world_size, rank=args.rank, timeout=timedelta(seconds=60))
proxy = region = None
try:
    error = None
    blob, endpoints = b"", ()
    try:
        layout = Layout(args.world_size, 4096)
        region = mmap.mmap(-1, layout.total_bytes)
        pointer = ctypes.addressof(ctypes.c_char.from_buffer(region))
        proxy = Proxy(world_size=args.world_size, rank=args.rank, hca_names=names,
                      gid_index=gid, region_ptr=pointer, region_bytes=layout.total_bytes, slot_bytes=4096)
        blob, endpoints = proxy.local_blob(), local_endpoints(names, gid)
        print(f"RDMA probe rank={args.rank} hcas={names} gid={gid} endpoints={endpoints}", flush=True)
    except Exception as exc:
        error = str(exc)
    statuses = [None] * args.world_size
    dist.all_gather_object(statuses, (error, blob, endpoints))
    failures = [f"rank {n}: {s[0]}" for n, s in enumerate(statuses) if s[0]]
    if failures:
        raise RuntimeError("RDMA setup preflight: " + "; ".join(failures))
    error = None
    try:
        all_endpoints = [s[2] for s in statuses]
        rails = min(MAX_RAILS, *(len(e) for e in all_endpoints))
        routes = plan_routes(all_endpoints, args.rank, rails)
        proxy.connect([s[1] for s in statuses], routes)
    except Exception as exc:
        error = str(exc)
    verdicts = [None] * args.world_size
    dist.all_gather_object(verdicts, error)
    failures = [f"rank {n}: {e}" for n, e in enumerate(verdicts) if e]
    if failures:
        raise RuntimeError("RDMA QP preflight: " + "; ".join(failures))
    print(f"FLARE_RDMA_QP_PREFLIGHT_PASS rank={args.rank} rails={rails}", flush=True)
finally:
    if proxy is not None:
        proxy.close()
    if region is not None:
        region.close()
    dist.destroy_process_group()
PY
)
start 0 roce-check /opt/venv/bin/python3 -c "$RDMA_PROBE" \
  --rendezvous "$HEAD_IP:$PREP_PORT" --world-size "$RANKS" --rank 0
follow 0
wait_for_head_port "$PREP_PORT" 'RDMA preflight head exited; no model was loaded'
for ((n=1;n<RANKS;n++)); do
  start "$n" roce-check /opt/venv/bin/python3 -c "$RDMA_PROBE" \
    --rendezvous "$HEAD_IP:$PREP_PORT" --world-size "$RANKS" --rank "$n"
  follow "$n"
done
finish_phase 'RDMA QP preflight' 1 '; no model was loaded'

# ── 08 / Official multi-host export, or reuse an existing package ─────────────
if [[ "$EXPORT_MODE" == output ]]; then
  if [[ -f "$PACKAGE_DIR/model.safetensors" ]]; then
    PACKAGE="$PACKAGE_DIR/model.safetensors"
    PACKAGE_ARG=/package/model.safetensors
  fi
else
  if [[ -f "$PACKAGE_REFERENCE" ]]; then
    IFS= read -r PACKAGE_ARG < "$PACKAGE_REFERENCE" || true
  fi
  if [[ -n "$PACKAGE_ARG" && "$PACKAGE_ARG" == /state/cache/packages/*/model.safetensors ]]; then
    PACKAGE="$PACKAGE_CACHE/${PACKAGE_ARG#/state/cache/packages/}"
    if [[ ! -f "$PACKAGE" ]]; then PACKAGE=''; PACKAGE_ARG=''; fi
  else
    PACKAGE_ARG=''
  fi
fi
if [[ -z "$PACKAGE" ]]; then
  say 'Preparing a new multi-host package; existing tuning/compiler caches are retained.'
  EXPORT_ARGS=()
  if [[ "$EXPORT_MODE" == output ]]; then
    EXPORT_ARGS+=(--output "/state/packages/$KEY/model.safetensors")
  fi
  start 0 prepare /opt/venv/bin/ignition export "$MODEL_INPUT" \
    --config /state/preparation/config.toml --ranks "$RANKS" --rendezvous "$HEAD_IP:$PREP_PORT" --gpus 0 \
    "${PREP_ARGS[@]}" "${EXPORT_ARGS[@]}" --verbose
  follow 0
  wait_for_head_port "$PREP_PORT" 'Preparation head exited before rendezvous opened'
  for ((n=1;n<RANKS;n++)); do
    start "$n" prepare /opt/venv/bin/ignition join "$HEAD_IP:$PREP_PORT" \
      --config /state/preparation/config.toml --rank "$n" --ranks "$RANKS" --gpu 0
    follow "$n"
  done
  finish_phase Preparation 2
  if [[ "$EXPORT_MODE" == output ]]; then
    PACKAGE="$PACKAGE_DIR/model.safetensors"
    PACKAGE_ARG=/package/model.safetensors
  else
    export_log="$RUN_LOGS/$RUN_ID-prepare-r0.log"
    PACKAGE_ENTRY=$(sed -n 's/^Exported cache entry: //p' "$export_log" | tail -n 1)
    [[ "$PACKAGE_ENTRY" == /state/cache/packages/* ]] || fail 'Export did not report a package entry within the configured cache'
    PACKAGE_ARG="$PACKAGE_ENTRY/model.safetensors"
    PACKAGE="$PACKAGE_CACHE/${PACKAGE_ARG#/state/cache/packages/}"
  fi
  [[ -f "$PACKAGE" ]] || fail "Exported package is missing: $PACKAGE"
else
  say "Reusing package: $PACKAGE"
fi

# ── 09 / Official signature verification and native serving ──────────────────
[[ -f "$PACKAGE" && -f "$DATA/preparation/development-publisher.json" ]] || fail 'The package or its local publisher profile is missing'
cp -- "$DATA/preparation/development-publisher.json" "$DATA/public/development-publisher.json"
for peer in "${PEERS[@]}"; do upload "$peer" "$DATA/public/development-publisher.json" "$DATA/public/development-publisher.json"; done

# Signature verification is retained. Approval is restricted to this local export's
# exact verified key and local-development namespace, never a blanket trust bypass.
INSPECTION=$(docker run --rm --pull never --user "${USERS[0]}" --env HOME=/tmp \
  --mount "type=bind,src=$PACKAGE_CACHE,dst=/state/cache/packages,readonly" \
  --mount "type=bind,src=$PACKAGE_DIR,dst=/package,readonly" \
  --mount "type=bind,src=$DATA/public,dst=/public,readonly" --entrypoint /usr/local/bin/flare \
  "$IMAGE" inspect "$PACKAGE_ARG" --publisher-keys /public/development-publisher.json --json)
FINGERPRINT=$(printf '%s' "$INSPECTION" | python3 -c 'import json,re,sys; p=json.load(sys.stdin)["package"]; s=p["publisher"]; assert p["signatures_verified"] is True and p["ranks"]==int(sys.argv[1]); assert s["development"] is True and s["namespace"]=="local-development"; f=s["fingerprint"]; assert re.fullmatch("[0-9a-f]{64}",f); print(f)' "$RANKS")
if [[ "$EXPORT_MODE" == cache ]]; then
  printf '%s\n' "$PACKAGE_ARG" > "$PACKAGE_REFERENCE.tmp"
  mv -f -- "$PACKAGE_REFERENCE.tmp" "$PACKAGE_REFERENCE"
fi
TRUST_ARGS=(--publisher-keys /public/development-publisher.json --trust-store /runtime/trust.toml
  --package-cache /runtime/packages --dangerously-approve-key "$FINGERPRINT" --dangerously-accept-publisher local-development)
HEAD_ARGS=(--host "$HTTP_HOST" --port "$HTTP_PORT")
if [[ -n "$ALIAS" ]]; then HEAD_ARGS+=(--served-model-name "$ALIAS"); fi
start 0 serve /usr/local/bin/flare run "$PACKAGE_ARG" "${TRUST_ARGS[@]}" \
  --rank 0 --rendezvous "$HEAD_IP:$MASTER_PORT" --gpus 0 "${SERVE_ARGS[@]}" "${HEAD_ARGS[@]}"
follow 0
wait_for_head_port "$MASTER_PORT" 'Serving head exited before rendezvous opened'
for ((n=1;n<RANKS;n++)); do
  start "$n" serve /usr/local/bin/flare run "${TRUST_ARGS[@]}" \
    --rank "$n" --rendezvous "$HEAD_IP:$MASTER_PORT" --gpus 0 "${SERVE_ARGS[@]}"
  follow "$n"
done
READY=false
HEALTH_HOST=127.0.0.1
[[ "$HTTP_HOST" == 0.0.0.0 ]] || HEALTH_HOST=$HTTP_HOST
while :; do
  for n in "${ACTIVE[@]}"; do running "$n" || { logs "$n"; fail "Serving rank $n exited"; }; done
  if ! $READY && curl --silent --fail --max-time 2 "http://$HEALTH_HOST:$HTTP_PORT/health" >/dev/null; then
    say "Ready: http://$HEAD_IP:$HTTP_PORT/v1 ; Ctrl-C stops this launch on all nodes."
    READY=true
  fi
  sleep 2
done
