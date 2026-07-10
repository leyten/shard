"""Trustless capability probe — admission is a capability FUNCTION, not an allowlist.

The seam that turns docs/ADMISSION_SPEC.md into running code. On join a node doesn't
SAY what it can do — it is MEASURED doing it, and the measurements feed a pure,
GPU-model-independent function that emits a ROLE:

    {peak-VRAM, fast-kernel, layer_ms, uplink, RTT-to-pool, dialable}  ->  role

Boundary (docs/INTEGRATION.md): shard owns the PROBE + the PHYSICS (this module);
c0mpute owns the ROLE DECISION — what to admit, how each role is priced, where to
place. `python3 -m shard.probe` reads `{cap, model?, spec?}` JSON on stdin and prints
the role verdict on stdout, exactly like `shard.plan`, so the TS control plane drives
the same physics as the engine without porting it. The measurement modes run on the
boxes (`--measure` needs the GPU stack; `--serve`/`--net-only` are stdlib-only).

Trust model (v0) — HONESTLY: trust-then-punish, not can't-lie. The cap the role
function judges is whatever the caller feeds it; what makes lying unprofitable is
WHERE each number gets re-checked, not admission-time magic:
  * uplink is timed by the RECEIVER (`--serve` reports the Mbps it observed) — but
    max-over-peers only resists a SLOW receiver, not a COLLUDING fast one, so the
    probe peer list must be ASSIGNED by the control plane, never chosen by the
    candidate.
  * dialability is a real DIAL-BACK: a pool peer originates an inbound connection to
    the candidate's listener and the nonce must echo — CGNAT can't manufacture that.
  * RTT is measured against the CANDIDATE POOL and re-measured independently by the
    pool at ring formation (mesh_rtt), so a lied-about RTT dies at placement.
  * the GPU half (peak-VRAM, layer_ms, kernel) runs the REAL block locally; a lie is
    caught AFTER admission — overstate VRAM and the placement load OOMs you out,
    understate layer_ms and the signed per-stage receipts expose the stall — and
    shard/challenge.py can recompute the probe block on demand. The ring eats one bad
    formation before reputation ejects the liar; signed probe transcripts + pool-run
    GPU spot-probes are the hardening that closes that window.

THE NUMBERS ARE v0 (SPEC_V0 below == the LIVING spec's thresholds). Every threshold
is a derived estimate awaiting live falsification. The verdict carries `binding` —
WHICH constraint denied the next-higher role — precisely so ring telemetry can point
at the number to revise. When live data diverges, change SPEC_V0 *and*
docs/ADMISSION_SPEC.md.

Physics (reproduces both live receipts — 13-15 tok/s scatter, 32 tight):
    tok/s = g / T,  T = N·RTT_pool + C,  N = ceil(n_layers / layers_this_card_holds)
    layers = (VRAM_total − reserve − load_peak_extra) / (footprint + kv)   # PEAK-gated
"""
import json
import math
import os
import socket
import sys
import threading
import time

from .plan import M25_PROFILE

# Admission thresholds — docs/ADMISSION_SPEC.md v0 (LIVING: revise here AND there).
SPEC_V0 = {
    "bar_interactive_tok_s": 20.0,   # single-stream bar for the interactive-anchor tag
    "bar_batched_agg_tok_s": 20.0,   # aggregate bar for a batched ring
    "batch_B": 4,                    # proven batch width (the 155 tok/s agg receipt ran B=4)
    "g_interactive": 4.0,            # accepted tokens/traversal, EAGLE+n-gram (measured 3.3-4.5)
    "g_batched": 2.5,                # MEASURED 2026-07-10, re-derived same day after the batched-drafter
                                     # + batched-graph levers landed (receipt batched-levers-sweep):
                                     # the drafter-serialization tax is GONE; 2.5 now = the direct
                                     # fp8-wire content-mix measurement (2.48, band 1.9-3.5). bf16 wire
                                     # measures 3.55 (band 2.2-5.8) — g is WIRE-MODE-dependent (fp8
                                     # shifts greedy content); quote per mode, keep the operative at
                                     # the deployment default (fp8)
    "c_base_ms": 46.0,               # measured all-5090 total compute over 62 layers (mixed ~56)
    "uplink_interactive_mbps": 200.0,  # 16k prefill = 50 MB/hop; 15 Mbps residential -> 160 s TTFT
    "uplink_batched_mbps": 100.0,
    "uplink_seeder_mbps": 20.0,      # below this a seeder feeds the swarm slower than the mirror
    "seeder_min_disk_gb": 10.0,      # smallest useful verified range to re-serve
    "verifier_max_layer_ms": 30_000.0,  # a spot-check must beat the challenge timeout, not the ring
}

ROLES = ("interactive-anchor", "batched-filler", "verifier", "seeder", "reject")

# ADMISSION model defaults = plan.py's engine profile. The 12-vs-13 cap tension this
# module shipped with was RESOLVED BY MEASUREMENT (2026-07-09: live probe + warm-stage
# VRAM reads): the full-layer cutlass footprint is ~2330 MB (the old 1700 was ~35%
# light), so a 32 GB card holds 12 by arithmetic — and a 13-layer stage ran warm at
# 31.5/32.6 GiB, brim-riding, never a plan target. plan.py now carries the measured
# numbers (layer_vram_mb 2330, cap_layers 12); admission inherits them directly.
ADMISSION_MODEL_V0 = dict(M25_PROFILE)
_PROVEN_CAP_VRAM_MB = 32768.0    # the card size cap_layers was proven on; bigger cards
                                 # scale by density (a flat cap made 48 GB == 32 GB)


# ---------------------------------------------------------------------------
# The pure physics half — torch-free, what c0mpute's control plane calls.
# ---------------------------------------------------------------------------

def derive_layers(cap, model):
    """Layers this card holds, PEAK-gated — the admit-then-OOM fix.

    Uses the MEASURED per-layer footprint (arch/backend-specific: cutlass ~1.7 GB,
    marlin ~4.25 GB) and the MEASURED load-time transient above resident (the NVFP4
    swizzle spike that OOM'd a 32 GB 5090 at 15 layers while resident said fine) —
    never a free-VRAM read. Capped at the profile's proven warm ceiling (`cap_layers`
    is itself a v0 number: the probe running on bigger cards is what will raise it).
    """
    fp = cap.get("footprint_mb_per_layer")
    total = float(cap.get("total_vram_mb") or 0.0)
    if not fp or fp <= 0 or total <= 0:
        return 0
    usable = total - float(model["reserve_mb"]) - float(cap.get("load_peak_extra_mb") or 0.0)
    per_layer = fp + float(model["kv_mb_per_layer"])
    # cap_layers was proven on a 32 GB card; scale the proven DENSITY to the card size
    # (a flat cap collapsed 48 GB to the 32 GB verdict — the spec's core distinction).
    density_cap = int(int(model["cap_layers"]) * total / _PROVEN_CAP_VRAM_MB)
    return max(0, min(int(usable // per_layer), density_cap))


def _hop_budget(bar_tok_s, g, c_ms, rtt_ms):
    """N_max: hops a ring can afford and still clear `bar` — (g/bar − C)/RTT, in ms."""
    if rtt_ms <= 0:
        rtt_ms = 1.0                       # same-box sentinel; never divide by zero
    return (g / bar_tok_s * 1000.0 - c_ms) / rtt_ms


def derive_role(cap, model=None, spec=None):
    """The admission function: a MEASURED capability vector -> a role + the physics.

    cap (every field MEASURED — see the module docstring's trust notes):
      total_vram_mb          device total (0 / absent = no usable GPU)
      footprint_mb_per_layer measured resident VRAM of ONE real layer (arch-specific)
      load_peak_extra_mb     measured load/run transient ABOVE resident (swizzle+workspace)
      has_fast_kernel        native NVFP4/marlin kernel ran under a CUDA graph (binary)
      layer_ms               measured decode ms/layer on the probe block
      can_recompute_block    the block produced sane output at ANY speed (CPU counts —
                             the control plane may also grant this via a passed challenge)
      uplink_mbps            measured upload, RECEIVER-timed
      rtt_to_pool_ms         median measured RTT to the CANDIDATE POOL (not isolation)
      nat_dialable           a peer dialed BACK successfully (nonce echoed)
      disk_free_gb           free disk for a seeder range

    Returns the verdict: role, layers, n_single, hop budgets, predicted tok/s, and
    `binding` — per denied role, WHICH gates failed (the revise signal the LIVING
    spec asks for: when telemetry contradicts a gate, that number gets changed).
    """
    m = {**ADMISSION_MODEL_V0, **(model or {})}
    s = {**SPEC_V0, **(spec or {})}
    n_layers = int(m["n_layers"])
    # measure_net reports None for unreachable-pool — same meaning as absent (a common
    # real case: firewalled candidate, pool down); it must yield a role, not a crash.
    rtt = float(cap["rtt_to_pool_ms"]) if cap.get("rtt_to_pool_ms") is not None else 9000.0
    uplink = float(cap.get("uplink_mbps") or 0.0)
    dialable = cap.get("nat_dialable") is True
    fast = cap.get("has_fast_kernel") is True

    layers = derive_layers(cap, m)
    n_single = math.ceil(n_layers / layers) if layers else None

    # C: total compute per traversal, partition-independent — anchored to the MEASURED
    # receipt (c_base_ms), not re-derived. The candidate's own excess slowness over the
    # fleet base is counted on the layers IT would hold; the rest run at base.
    base_ms = s["c_base_ms"] / n_layers
    layer_ms = float(cap["layer_ms"]) if cap.get("layer_ms") else base_ms
    c_ms = s["c_base_ms"] + max(0.0, layer_ms - base_ms) * (layers or 0)

    n_max_i = _hop_budget(s["bar_interactive_tok_s"], s["g_interactive"], c_ms, rtt)
    # A batched ring amortizes the SAME traversal over B streams (agg = B·g/T), so the
    # effective per-stream bar is bar/B — that, not g, is what B relaxes.
    n_max_b = _hop_budget(s["bar_batched_agg_tok_s"] / s["batch_B"], s["g_batched"], c_ms, rtt)

    predicted = predicted_agg = None
    if n_single:
        t_ms = n_single * rtt + c_ms
        predicted = round(s["g_interactive"] / t_ms * 1000.0, 2)
        predicted_agg = round(s["batch_B"] * s["g_batched"] / t_ms * 1000.0, 2)

    # Gate tables per role, evaluated top role down; a denial's failed gates become
    # `binding` — the exact numbers live telemetry should argue with.
    gates = [
        ("interactive-anchor", [
            ("fast_kernel", fast),
            ("vram_layers", bool(layers)),
            ("hops_vs_rtt", n_single is not None and n_single <= n_max_i),
            ("uplink", uplink >= s["uplink_interactive_mbps"]),
            ("nat_dialable", dialable),
        ]),
        ("batched-filler", [
            ("fast_kernel", fast),
            ("vram_layers", bool(layers)),
            ("hops_vs_rtt", n_single is not None and n_single <= n_max_b),
            ("uplink", uplink >= s["uplink_batched_mbps"]),
            ("nat_dialable", dialable),
        ]),
        ("verifier", [
            ("recompute_block", cap.get("can_recompute_block") is True),
            ("layer_ms_timeout", layer_ms <= s["verifier_max_layer_ms"]),
        ]),
        ("seeder", [
            ("uplink", uplink >= s["uplink_seeder_mbps"]),
            ("disk", float(cap.get("disk_free_gb") or 0.0) >= s["seeder_min_disk_gb"]),
        ]),
    ]
    role, binding = "reject", []
    for name, checks in gates:
        failed = [g for g, ok in checks if not ok]
        if not failed:
            role = name
            break
        binding.append({"role": name, "failed": failed})

    return {
        "role": role,
        "layers": layers,
        "n_single": n_single,
        "n_max_interactive": round(n_max_i, 2),
        "n_max_batched": round(n_max_b, 2),
        "predicted_tok_s": predicted,
        "predicted_agg_tok_s": predicted_agg,
        "c_ms": round(c_ms, 2),
        "binding": binding,
        "spec": "v0",
    }


# ---------------------------------------------------------------------------
# The probe-peer endpoint (`--serve`) — stdlib-only, run by pool members.
# Wire: one JSON header line, then (for upload) raw payload. Receiver does the timing.
# ---------------------------------------------------------------------------

_MAX_UPLOAD = 256 * 1024 * 1024      # refuse absurd upload claims (open port on the internet)
_MAX_HDR = 4096
_HDR_DEADLINE_S = 30                 # wall-clock cap on reading one header line (slowloris)
_UPLOAD_DEADLINE_S = 300             # wall-clock cap on one upload (matches the client side)
_MAX_CONNS = 64                      # concurrent probers; over it, refuse (internet-facing port)

_conn_slots = threading.BoundedSemaphore(_MAX_CONNS)


def _readline(sock, limit=_MAX_HDR, deadline_s=_HDR_DEADLINE_S):
    """Read one \\n-terminated line under BOTH a size cap and a wall-clock deadline —
    a per-recv timeout alone lets a slowloris dribble 1 byte/59 s and hold the thread."""
    buf, t0 = b"", time.time()
    while not buf.endswith(b"\n") and len(buf) < limit:
        if time.time() - t0 > deadline_s:
            break
        c = sock.recv(1)
        if not c:
            break
        buf += c
    return buf


def _serve_conn(conn, addr):
    try:
        conn.settimeout(60)
        hdr = json.loads(_readline(conn).decode())
        op = hdr.get("op")
        if op == "ping":
            conn.sendall((json.dumps({"ok": True, "nonce": hdr.get("nonce")}) + "\n").encode())
        elif op == "upload":
            n = int(hdr["bytes"])
            if not 0 < n <= _MAX_UPLOAD:
                conn.sendall(b'{"ok": false, "error": "bad size"}\n')
                return
            got, t0 = 0, time.time()
            while got < n and time.time() - t0 < _UPLOAD_DEADLINE_S:
                chunk = conn.recv(min(1 << 20, n - got))
                if not chunk:
                    break
                got += len(chunk)
            secs = max(time.time() - t0, 1e-6)      # RECEIVER-timed — the sender can't inflate it
            conn.sendall((json.dumps({"ok": got == n, "secs": round(secs, 4),
                                      "mbps": round(got * 8 / secs / 1e6, 2)}) + "\n").encode())
        elif op == "dialback":
            nonce = str(hdr.get("nonce", ""))[:64]
            ok = False
            try:
                back = socket.create_connection((addr[0], int(hdr["port"])), timeout=8)
                back.sendall((nonce + "\n").encode())
                ok = _readline(back).decode().strip() == nonce   # candidate must echo OUR nonce
                back.close()
            except OSError:
                ok = False
            conn.sendall((json.dumps({"ok": ok}) + "\n").encode())
        else:
            conn.sendall(b'{"ok": false, "error": "bad op"}\n')
    except Exception:                                 # noqa: BLE001 — a broken prober must never
        pass                                          # take the peer endpoint down with it
    finally:
        conn.close()
        _conn_slots.release()


def serve(port, ready_evt=None):
    """Run the probe-peer endpoint (ping / receiver-timed upload / dial-back) forever.
    Run by POOL members on control-plane assignment — a candidate must never pick its
    own probe peers (one colluding fast receiver would inflate max-over-peers uplink)."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", port))
    srv.listen(16)
    if ready_evt is not None:
        ready_evt.set()
    while True:
        conn, addr = srv.accept()
        if not _conn_slots.acquire(blocking=False):   # full: refuse, don't queue threads
            conn.close()
            continue
        threading.Thread(target=_serve_conn, args=(conn, addr), daemon=True).start()


def _dialback_listener(port, ready_evt, stop_evt):
    """The candidate's side of the dial-back check: echo whatever nonce the peer sends.
    Lives until measure_net finishes (a slow multi-peer upload pass can take minutes; a
    listener that expires early mislabels a dialable slow-uplink node un-dialable)."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", port))
    srv.listen(4)
    srv.settimeout(1.0)
    ready_evt.set()
    try:
        while not stop_evt.is_set():
            try:
                conn, _ = srv.accept()
            except TimeoutError:
                continue
            try:
                conn.settimeout(10)
                conn.sendall(_readline(conn))
            except OSError:
                pass
            finally:
                conn.close()
    finally:
        srv.close()


def measure_net(peers, upload_mb=16, dialback_port=None, dialback_advertise=None):
    """The candidate-side network probe against pool peers ("host:port", running --serve).

    The peer list is ASSIGNED by the control plane (candidate-chosen peers let one
    colluding fast receiver inflate max-over-peers uplink). Malformed peer strings are
    skipped, not fatal (IPv4 host:port only for v0).

    RTT per peer = best-of-3 TCP connect time — deliberately the SAME quantity
    mesh_rtt/plan_ring feed shard.topology as the per-hop cost, so admission and
    placement argue in one unit (~1 link RTT; do NOT "fix" this to a half-RTT one-way,
    that would double every hop budget and over-admit).

    dialback_port is where the candidate's listener BINDS; dialback_advertise is the
    port peers are told to dial (defaults to the bind port). They differ exactly when
    the candidate sits behind a port-mapping NAT (a vast box maps container 29600 to a
    random host port) — advertising the bind port there would fail every honest node.

    Returns {rtt_ms (per-peer), rtt_to_pool_ms (median), uplink_mbps (max of the
    receiver-reported rates), nat_dialable (any peer's dial-back succeeded)}.
    """
    import statistics  # noqa: PLC0415

    rtts, ups, dial = [], [], False
    stop = threading.Event()
    if dialback_port:
        ready = threading.Event()
        threading.Thread(target=_dialback_listener, args=(dialback_port, ready, stop),
                         daemon=True).start()
        ready.wait(5)
    payload = os.urandom(upload_mb * 1024 * 1024)
    for hp in peers:
        try:
            host, _, port = hp.rpartition(":")
            port = int(port)
        except ValueError:
            rtts.append(None)
            continue
        best = None
        for _ in range(3):
            try:
                t0 = time.time()
                s = socket.create_connection((host, port), timeout=6)
                dt = (time.time() - t0) * 1000        # connect time ≈ 1 link RTT (see docstring)
                s.sendall(b'{"op": "ping", "nonce": 1}\n')
                _readline(s)                          # protocol check only, not timed
                best = dt if best is None else min(best, dt)
                s.close()
            except OSError:
                pass
        rtts.append(round(best, 2) if best is not None else None)
        if best is None:
            continue
        # dial-back BEFORE the upload: a slow-uplink upload must not eat the window in
        # which the candidate's listener waits for the peer's inbound connection.
        if dialback_port and not dial:
            try:
                s = socket.create_connection((host, port), timeout=6)
                s.settimeout(20)
                s.sendall((json.dumps({"op": "dialback",
                                       "port": dialback_advertise or dialback_port,
                                       "nonce": os.urandom(8).hex()}) + "\n").encode())
                dial = bool(json.loads(_readline(s).decode()).get("ok"))
                s.close()
            except (OSError, ValueError):
                pass
        try:
            s = socket.create_connection((host, port), timeout=6)
            s.settimeout(300)
            s.sendall((json.dumps({"op": "upload", "bytes": len(payload)}) + "\n").encode())
            s.sendall(payload)
            rep = json.loads(_readline(s, deadline_s=_UPLOAD_DEADLINE_S).decode())
            if rep.get("ok"):
                ups.append(float(rep["mbps"]))        # the RECEIVER's number, not ours
            s.close()
        except (OSError, ValueError):
            pass
    stop.set()
    live = [r for r in rtts if r is not None]
    return {
        "rtt_ms": rtts,
        "rtt_to_pool_ms": round(statistics.median(live), 2) if live else None,
        "uplink_mbps": max(ups) if ups else 0.0,      # max: a slow receiver only understates
        "nat_dialable": dial,
    }


# ---------------------------------------------------------------------------
# The GPU half (`--measure`) — lazy imports, runs the REAL block on the box.
# Same load path the serving stage uses (research/hetero_moe_probe.py lineage).
# ---------------------------------------------------------------------------

def measure_gpu(model_dir, layer=30, backend="auto", kv_tokens=1024):
    """Load ONE real decoder layer and measure what admission needs:

      footprint_mb_per_layer  resident VRAM of the full layer (attn bf16 + MoE quant)
      load_peak_extra_mb      transient ABOVE resident (NVFP4 swizzle + decode workspace)
      layer_ms                decode-shaped attn+MoE forward, CUDA-graph replayed if safe
      has_fast_kernel         native cutlass/marlin quant method AND graph capture+replay
                              produced output matching eager (cosine)
      can_recompute_block     the forward produced finite output at any speed

    backend="auto" mirrors M25_MOE_BACKEND=auto: cutlass on sm_120+, marlin below.
    """
    import torch  # noqa: PLC0415 — lazy: the role half must stay importable torch-free
    if not torch.cuda.is_available():
        return {"total_vram_mb": 0, "can_recompute_block": False, "has_fast_kernel": False,
                "error": "no CUDA device"}
    from safetensors import safe_open  # noqa: PLC0415

    dev = torch.cuda.get_device_properties(0)
    cap_sm = torch.cuda.get_device_capability(0)
    if backend == "auto":
        backend = "cutlass" if cap_sm >= (12, 0) else "marlin"
    out = {
        "total_vram_mb": round(dev.total_memory / 2**20, 1),
        "gpu_name": dev.name, "sm": f"{cap_sm[0]}{cap_sm[1]}", "backend": backend,
        "has_fast_kernel": False, "can_recompute_block": False,
    }

    cfg = json.load(open(f"{model_dir}/config.json"))
    H = cfg["hidden_size"]
    idx = json.load(open(f"{model_dir}/model.safetensors.index.json"))["weight_map"]
    handles = {}

    def raw(name):
        f = idx[name]
        if f not in handles:
            handles[f] = safe_open(f"{model_dir}/{f}", "pt", device="cpu")
        return handles[f].get_tensor(name)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()

    # --- the full layer's NON-expert weights (attn + norms + gate), raw bf16 copies ---
    pfx = f"model.layers.{layer}."
    plain = {}
    for name in idx:
        if name.startswith(pfx) and ".experts." not in name:
            plain[name[len(pfx):]] = raw(name).cuda()

    # --- the layer's MoE via the REAL vLLM FusedMoE + quant method (the swizzle path) ---
    from vllm.model_executor.layers.fused_moe.layer import FusedMoE  # noqa: PLC0415
    from vllm.forward_context import set_forward_context  # noqa: PLC0415
    from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4Config  # noqa: PLC0415
    from vllm.distributed import init_distributed_environment, initialize_model_parallel  # noqa: PLC0415
    from vllm.config import VllmConfig, set_current_vllm_config  # noqa: PLC0415
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29587")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    torch.cuda.set_device(0)
    init_distributed_environment(world_size=1, rank=0, local_rank=0,
                                 distributed_init_method="env://", backend="nccl")
    vcfg = VllmConfig()
    # keep the CM referenced for the whole run: a dropped @contextmanager object is
    # GC'd -> its generator closes -> the config context EXITS under us mid-probe
    vcfg_ctx = set_current_vllm_config(vcfg)
    vcfg_ctx.__enter__()
    initialize_model_parallel(1)
    try:
        from vllm.v1.worker.workspace import init_workspace_manager  # noqa: PLC0415
        init_workspace_manager(torch.device("cuda"))
    except ImportError:
        pass
    try:
        vcfg.kernel_config.moe_backend = backend
    except AttributeError:
        pass

    hfq_path = f"{model_dir}/hf_quant_config.json"
    src = (json.load(open(hfq_path))["quantization"] if os.path.exists(hfq_path)
           else cfg["quantization_config"])
    qcfg = ModelOptNvFp4Config.from_config(src)

    E = cfg.get("num_local_experts", cfg.get("num_experts"))
    I = cfg.get("moe_intermediate_size") or cfg.get("intermediate_size")
    pmoe, pexp = pfx + "block_sparse_moe.", pfx + "block_sparse_moe.experts."
    eb = raw(pmoe + "e_score_correction_bias").float().cuda() if pmoe + "e_score_correction_bias" in idx else None
    kw = dict(num_experts=E, top_k=cfg["num_experts_per_tok"], hidden_size=H,
              intermediate_size=I, params_dtype=torch.bfloat16,
              renormalize=cfg.get("norm_topk_prob", True), use_grouped_topk=False,
              scoring_func="sigmoid", routed_scaling_factor=cfg.get("routed_scaling_factor", 1.0),
              quant_config=qcfg, prefix=pexp[:-1])
    if eb is not None:
        kw["e_score_correction_bias"] = eb
    try:
        moe = FusedMoE(**kw).cuda()
    except TypeError:                               # older FusedMoE signatures
        for k in ("e_score_correction_bias", "routed_scaling_factor", "scoring_func"):
            kw.pop(k, None)
        moe = FusedMoE(**kw).cuda()
    params = dict(moe.named_parameters())
    suffixes = sorted({k.split(f"{pexp}0.w1.")[1] for k in idx if k.startswith(f"{pexp}0.w1.")})
    loaded = 0
    for e in range(E):
        for proj in ("w1", "w3", "w2"):
            grp = "w2" if proj == "w2" else "w13"
            for suf in suffixes:
                name, pname = f"{pexp}{e}.{proj}.{suf}", f"{grp}_{suf}"
                if name in idx and pname in params:
                    moe.weight_loader(params[pname], raw(name).cuda(), name, proj, e)
                    loaded += 1
    if loaded == 0:                      # a forward over random init is finite but MEANS nothing —
        out["error"] = "no expert tensors matched the index"   # never grant recompute on it
        return out
    moe.quant_method.process_weights_after_loading(moe)   # <- the swizzle/repack peak
    torch.cuda.synchronize()

    resident = torch.cuda.memory_allocated() - base
    load_peak = torch.cuda.max_memory_allocated() - base
    out["footprint_mb_per_layer"] = round(resident / 2**20, 1)
    out["kernel"] = type(moe.quant_method).__name__

    # --- decode-shaped forward: GQA attention on the real weights + the MoE ---
    # forward in bf16: the checkpoint stores e.g. gate.weight fp32 (footprint above
    # measured the REAL stored dtypes; the cast copies here land in the run peak)
    def bf16(w):
        return w.to(torch.bfloat16) if w is not None else None
    gate_w = bf16(plain.get("block_sparse_moe.gate.weight"))
    n_heads = cfg["num_attention_heads"]
    n_kv = cfg.get("num_key_value_heads", n_heads)
    hd = cfg.get("head_dim", H // n_heads)
    wq, wk = bf16(plain.get("self_attn.q_proj.weight")), bf16(plain.get("self_attn.k_proj.weight"))
    wv, wo = bf16(plain.get("self_attn.v_proj.weight")), bf16(plain.get("self_attn.o_proj.weight"))
    attn_ok = all(w is not None for w in (wq, wk, wv, wo))
    kc = torch.randn(1, n_kv, kv_tokens, hd, dtype=torch.bfloat16, device="cuda") * 0.1
    vc = torch.randn(1, n_kv, kv_tokens, hd, dtype=torch.bfloat16, device="cuda") * 0.1
    x = torch.randn(1, H, dtype=torch.bfloat16, device="cuda") * 0.1

    def fwd():
        h = x
        if attn_ok:                                   # timing-faithful GQA decode step
            q = (h @ wq.T).view(1, 1, n_heads, hd).transpose(1, 2)
            k = (h @ wk.T).view(1, 1, n_kv, hd).transpose(1, 2)
            v = (h @ wv.T).view(1, 1, n_kv, hd).transpose(1, 2)
            rep = n_heads // n_kv
            a = torch.nn.functional.scaled_dot_product_attention(
                q, torch.cat([kc, k], 2).repeat_interleave(rep, 1),
                torch.cat([vc, v], 2).repeat_interleave(rep, 1))
            h = a.transpose(1, 2).reshape(1, n_heads * hd) @ wo.T
            if h.shape[-1] != H:                      # o_proj may project back from n_heads*hd
                h = x                                 # shape surprise: fall back to MoE-only
        rl = torch.nn.functional.linear(h.view(1, H), gate_w) if gate_w is not None else \
            torch.zeros(1, E, dtype=torch.bfloat16, device="cuda")
        with torch.no_grad(), set_forward_context(None, vcfg):
            return moe(h.view(1, H), rl)

    y = fwd()
    torch.cuda.synchronize()
    if not torch.isfinite(y).all():
        out["error"] = "non-finite block output"
        return out
    out["can_recompute_block"] = True

    def timeit(fn, n=50, warm=10):
        for _ in range(warm):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / n * 1000

    out["eager_layer_ms"] = round(timeit(fwd), 3)

    # --- the binary fast-kernel gate: native quant method AND graph capture+replay ---
    native = ("Marlin" in out["kernel"]) or ("NvFp4" in out["kernel"] and "Emul" not in out["kernel"])
    graph_ok, graph_ms = False, None
    try:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                ref = fwd()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            static_y = fwd()
        g.replay()
        torch.cuda.synchronize()
        a, b = static_y.float().flatten(), ref.float().flatten()
        cos = float(torch.nn.functional.cosine_similarity(a, b, dim=0))
        graph_ok = torch.isfinite(static_y).all().item() and cos >= 0.99
        if graph_ok:
            graph_ms = timeit(lambda: g.replay())
        out["graph_cosine"] = round(cos, 5)
        # capability gate is cosine; the production VERIFY path holds graphs to diff==0.0
        # bit-equality (m25_stage GraphRunner) — record the diff so live data can tighten.
        out["graph_max_abs_diff"] = float((a - b).abs().max())
    except Exception as e:                            # noqa: BLE001 — capture failure IS the verdict
        out["graph_error"] = f"{type(e).__name__}: {str(e)[:120]}"
    out["has_fast_kernel"] = bool(native and graph_ok)
    out["layer_ms"] = round(graph_ms, 3) if graph_ms is not None else out["eager_layer_ms"]

    # Peaks reported SEPARATELY: the admission formula wants the LOAD (swizzle) peak the
    # spec derived from the 15-layer OOM; the run peak (timing loops + graph pool + probe
    # KV) is a different, probe-shaped transient — folding it in silently shrank every
    # card by workspace that reserve_mb already models. Both are surfaced for telemetry.
    run_peak = torch.cuda.max_memory_allocated() - base
    out["load_peak_extra_mb"] = round((load_peak - resident) / 2**20, 1)
    out["run_peak_extra_mb"] = round((run_peak - resident) / 2**20, 1)
    return out


# ---------------------------------------------------------------------------
# CLI — role mode on stdin JSON (the c0mpute seam) + the measurement modes.
# ---------------------------------------------------------------------------

def _main() -> int:
    import argparse  # noqa: PLC0415
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--measure", action="store_true", help="GPU 1-block probe (needs the model dir)")
    ap.add_argument("--net-only", action="store_true", help="network vector only (stdlib, no GPU)")
    ap.add_argument("--serve", action="store_true", help="run the probe-peer endpoint")
    ap.add_argument("--dir", default="/root/m25")
    ap.add_argument("--layer", type=int, default=30)
    ap.add_argument("--backend", default="auto")
    ap.add_argument("--peers", default="", help="comma-separated host:port of --serve peers")
    ap.add_argument("--port", type=int, default=29655, help="--serve port / dial-back listen port")
    ap.add_argument("--dialback-advertise", type=int, default=0,
                    help="port peers dial back (behind a port-mapping NAT it differs from --port)")
    ap.add_argument("--upload-mb", type=int, default=16)
    a = ap.parse_args()

    if a.serve:
        serve(a.port)
        return 0
    if a.measure or a.net_only:
        cap = {}
        if a.measure:
            cap.update(measure_gpu(a.dir, a.layer, a.backend))
        if a.peers:
            cap.update(measure_net([p.strip() for p in a.peers.split(",") if p.strip()],
                                   upload_mb=a.upload_mb, dialback_port=a.port,
                                   dialback_advertise=a.dialback_advertise or None))
        if "disk_free_gb" not in cap:                 # seeder gate input; CPU boxes need it too
            import shutil  # noqa: PLC0415
            root = a.dir if os.path.isdir(a.dir) else "/"
            cap["disk_free_gb"] = round(shutil.disk_usage(root).free / 2**30, 1)
        json.dump(cap, sys.stdout, indent=1)
        return 0

    # default: `{cap, model?, spec?}` JSON in -> the role verdict out (mirrors shard.plan)
    try:
        req = json.load(sys.stdin)
    except Exception as e:  # noqa: BLE001 — a malformed request is a caller error, report as JSON
        json.dump({"error": f"bad request json: {e}"}, sys.stdout)
        return 2
    try:
        verdict = derive_role(req["cap"], req.get("model"), req.get("spec"))
    except KeyError as e:
        json.dump({"error": f"missing field: {e}"}, sys.stdout)
        return 2
    except Exception as e:  # noqa: BLE001
        json.dump({"error": f"probe failed: {e}"}, sys.stdout)
        return 1
    json.dump(verdict, sys.stdout)
    return 0


if __name__ == "__main__":
    sys.exit(_main())
