#!/usr/bin/env bash
# HW1 box bootstrap + run order (runs ON the rented box, from the repo root copy).
#
#   bash research/hw1/bootstrap.sh setup     # venv with the fleet pins (phase0/requirements_vmoe.txt)
#   bash research/hw1/bootstrap.sh fetch     # M2.5 NVFP4 layers [24,36) + MiMo layers 0,1,5 (parallel)
#   bash research/hw1/bootstrap.sh run       # the pre-registered order; stops after A0 if K1 fails
#   bash research/hw1/bootstrap.sh collect   # validate + tar results/ for scp back
#
# Renting, labels, the dead-man switch and teardown stay on the workspace box (docs/M25_ENGINE.md OPS);
# nothing here talks to vast. HF_TOKEN, if needed, comes from the environment and is never written.
# Env: HW1_BOX_LABEL (required, e.g. hw1-5090-a), HW1_FAMILY_ONLY=1 (3090/4090 legs: B/D/C only).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV="${HW1_VENV:-/root/hw1venv}"
M25_DIR="${M25_DIR:-/root/m25}"
MIMO_DIR="${MIMO_DIR:-/root/mimo}"
MIMO_REPO="XiaomiMiMo/MiMo-V2.6-Flash-RL"
MIMO_REV="5711b268169967567844e1e560e8a3966da959b1"     # pinned commit (public)
OUT="${HW1_OUT:-$ROOT/results/hw1}"
PY="$VENV/bin/python"
export PYTHONPATH="$ROOT/engines/minimax_m25:$ROOT/engines/kimi_k3:$ROOT/phase0:$ROOT/shard:$ROOT"
export M25_DIR

log() { printf '[hw1 %s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }

setup() {
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install -q --upgrade pip
  "$VENV/bin/pip" install -q -r "$ROOT/phase0/requirements_vmoe.txt"
  "$PY" - <<'EOF'
import torch, vllm, flashinfer
want = {"torch": "2.11.0", "vllm": "0.23.0", "flashinfer": "0.6.12"}
got = {"torch": torch.__version__.split("+")[0], "vllm": vllm.__version__, "flashinfer": flashinfer.__version__}
bad = {k: (got[k], v) for k, v in want.items() if got[k] != v}
p = torch.cuda.get_device_properties(0)
print(f"pins {got}  device {p.name} sm_{p.major}{p.minor}  {p.total_memory / 2**30:.1f} GiB")
raise SystemExit(f"PIN MISMATCH {bad}" if bad else 0)
EOF
}

fetch() {
  mkdir -p "$M25_DIR" "$MIMO_DIR" "$OUT"
  if [ "${HW1_FAMILY_ONLY:-0}" != "1" ]; then
    "$PY" "$ROOT/engines/minimax_m25/m25_pull_range.py" --lo 24 --hi 36 --dir "$M25_DIR" \
      > "$OUT/fetch_m25.log" 2>&1 &
  fi
  for li in 0 1 5; do
    "$PY" -m shard.fetch ranged --repo "$MIMO_REPO" --revision "$MIMO_REV" --dir "$MIMO_DIR" \
      --lo "$li" --hi "$((li + 1))" > "$OUT/fetch_mimo_$li.log" 2>&1 || { log "MiMo layer $li fetch FAILED"; exit 1; }
  done
  wait
  grep -q SHARD_FETCH_DONE "$OUT"/fetch_mimo_5.log
  log "fetch done: $(du -sh "$M25_DIR" "$MIMO_DIR" 2>/dev/null | tr '\n' ' ')"
}

m25() {  # m25 <mode> <ctx> <batch> <maxlen> <out> [extra args...]
  local mode=$1 ctx=$2 b=$3 maxlen=$4 out=$5; shift 5
  M25_BATCH=$b M25_KV_MAXLEN=$maxlen M25_GRAPH_MAX=64 "$PY" -u "$ROOT/research/hw1/m25_curve.py" \
    --mode "$mode" --ctxs "$ctx" --out "$OUT/$out" "$@"
}

run() {
  : "${HW1_BOX_LABEL:?set HW1_BOX_LABEL}"
  mkdir -p "$OUT"
  local L=$HW1_BOX_LABEL
  if [ "${HW1_FAMILY_ONLY:-0}" != "1" ]; then
    log "A0 anchor (gate K1)"
    if ! M25_BATCH=8 M25_KV_MAXLEN=17000 "$PY" -u "$ROOT/research/hw1/m25_curve.py" --mode anchor \
        --out "$OUT/${L}_a0.json"; then
      log "K1 FAILED: the anchor did not reproduce within 20% -- STOP, debug box/env"; exit 2
    fi
  fi
  log "C system bench"
  "$PY" "$ROOT/research/hw1/sys_bench.py" --out "$OUT/${L}_c.json"
  log "B MiMo ops (+ D dump for the cross-family diff)"
  local inp=()
  [ -f "$ROOT/research/hw1/golden_t32_bf16.safetensors" ] && inp=(--inputs "$ROOT/research/hw1/golden_t32_bf16.safetensors")
  "$PY" -u "$ROOT/research/hw1/mimo_ops.py" --weights "$MIMO_DIR" "${inp[@]}" --out "$OUT/${L}_b.json"
  "$PY" -u "$ROOT/research/hw1/mimo_ops.py" --mode dump --weights "$MIMO_DIR" "${inp[@]}" \
    --dump "$OUT/${L}_dump.safetensors" --out "$OUT/${L}_dump.json"
  [ "${HW1_FAMILY_ONLY:-0}" = "1" ] && { log "family leg done"; return 0; }
  log "A2 tau (one stream)"
  for c in 1024 8192 32768; do m25 tau "$c" 2 $((c + 64)) "${L}_a2_${c}.json" --layers 8; done
  log "A1 service curve (one process per context)"
  m25 curve 1024 32 1100 "${L}_a1_1k.json" --layers 8
  m25 curve 8192 32 8300 "${L}_a1_8k.json" --layers 8
  M25_KV_FP8=1 m25 curve 32768 16 32900 "${L}_a1_32k.json" --layers 8 --bs 1,2,4,8,16
  log "A3 prefill + A4 KV"
  m25 prefill 0 2 33000 "${L}_a3.json" --layers 8
  m25 kv 0 2 33000 "${L}_a4.json" --layers 8
  log "run done"
}

collect() {
  "$PY" "$ROOT/research/hw1/collect.py" "$OUT" --out "$OUT/bundle.json"
  tar czf "$ROOT/results/hw1_${HW1_BOX_LABEL:-box}.tgz" -C "$OUT" .
  log "results: $ROOT/results/hw1_${HW1_BOX_LABEL:-box}.tgz"
}

case "${1:-}" in
  setup) setup ;;
  fetch) fetch ;;
  run) run ;;
  collect) collect ;;
  *) echo "usage: $0 setup|fetch|run|collect" >&2; exit 64 ;;
esac
