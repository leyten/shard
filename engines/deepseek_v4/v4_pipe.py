"""DeepSeek-V4-Flash PIPELINED ring — the full-ring launcher/coordinator, the V4 analogue of
k3_pipe (which is itself the K3 analogue of m25_pipe + m25_scatter_pipe). Step 3 of the V4 engine:
step 1 built the CPU oracle (v4_ref_cpu), step 2 the Stage (v4_stage), and this is the thing that
makes a scattered ring out of them — a head that embeds, middles that forward, a tail that samples,
and a weightless coordinator that drives greedy generation over the return tunnel.

WHAT IS DIFFERENT FROM K3 — AND WHAT IS NOT
The topology, the sidecar wiring, the reset/step/receipt/stop op protocol, the receipt sweep and the
SHARD_JOB_* stdout contract are k3_pipe's, unchanged. Two things move, both in the PAYLOAD:

  1. ONE TENSOR, NOT A PAIR.  A K3 layer boundary carries (h, block_residual). A V4 boundary carries
     a single `h [b, s, hc_mult, dim]` — the hc_mult=4 Hyper-Connection streams the reference keeps
     instead of one residual (model.py Block.hc_pre/hc_post). Expanded once at the head
     (`h.unsqueeze(2).repeat(1,1,4,1)`), collapsed once at the tail (`hc_head`), so every hop in
     between moves exactly 4x the hidden state and nothing else.
  2. THE TOKEN IDS RIDE EVERY HOP.  V4's first `n_hash_layers` (3 in the shipped config) route their
     MoE by HASHING THE TOKEN ID, not by a router logit: the reference threads `input_ids` into every
     Block.forward, which passes it to MoE.forward. A stage that holds a hash layer therefore cannot
     compute its block from `h` alone. So the step frame carries `ids [b, s]` int64 alongside `h` —
     the ids are the chunk's own tokens, already public to the coordinator, and they are tiny
     (8 B/token against 4*dim*2 = 32 KiB/token at dim 4096).

RECEIPTS attest BOTH: a stage hashes (h || ids) at input and output, so the chain
(out_root == in_root across adjacent stages) proves the hyper-connection streams AND the routing ids
crossed intact. A hop that dropped or rewrote the ids — which would silently change which experts
the next hash layer picks — breaks the chain, where a hidden-state-only chain could not see it.

fp8 WIRE (V4_FP8_WIRE, opt-in, default OFF): only `h` is packed — e4m3, ONE SCALE PER (position,
stream) so that a position's bytes never depend on how many other positions shared its frame, which
is what keeps the drafted stream bit-identical to the greedy one; chain-preserving, since the sender
hashes the packed wire bytes and the receiver hashes the same received bytes. The ids ALWAYS ride
int64: they are exact indices into a hash function and a vocab table, a quantized token id is not a
token id, and they are 4000x smaller than h anyway.

LAZY MODEL IMPORT.  k3_pipe does `import k3_stage as K3` at module scope. This module imports
v4_stage (and, in the selftest, v4_ref_cpu) INSIDE the functions that need them. The frame codec,
the receipt chain, the dial/accept/keep-warm machinery, the tiling and the sidecar wiring are pure
protocol with no model in them, and they are what tests/test_v4_pipe.py pins — that suite must run
on a box with no checkpoint, and it must have run before v4_stage existed.

TOPOLOGY (direct-return, per m25_scatter_pipe / k3_pipe)
  coordinator (on the head box, weightless) --pipe--> head engine ENG_IN
  head --FWD_RING--> s1 --FWD_RING--> ... --> tail          (fire-forward; each sidecar -forward)
  tail --ret--> coordinator, tunneled by the HEAD sidecar's FWD_RET -forward to the tail's inbound
The tail holds TWO inbound streams on its one engine port: the predecessor (job frames) and the
coordinator-return (identified by a hello_return greeting). Only the TAIL answers the coordinator.

WHAT CROSSES THE RETURN TUNNEL: a TOKEN, not logits (V4's vocab is 129280 — a per-token logits
vector would dwarf the boundary payload for no gain). The tail runs hc_head + final norm + lm_head,
argmaxes, and returns the int id.

FOUR WAYS TO DECODE, ONE ACCEPT RULE. coordinate() is greedy (g=1). coordinate_spec() drafts on the
COORDINATOR (n-gram/repeat — any proposer) and coordinate_dspark() lets the TAIL draft with V4's own
trained MTP speculator, which is where that drafter has to run because `main_hidden` never leaves the
box. Both drafted paths verify a whole chunk in one traversal, commit the accepted prefix plus one
correction, and are LOSSLESS: the committed stream is bit-identical to greedy, because every logits
row the tail ever computes is taken at the same GEMM shape (_tail_logit_rows) and because a rejected
tail is rolled back by Stage._seek before the next chunk. The accept rule itself is ONE function,
v4_dspark_draft.plan_verify_round, run by both ends of every round and asserted to agree.
coordinate_dspark_pipelined() is the fourth: the SAME drafted round with the block streamed as
separate s=1 frames instead of sent as one chunk, so the D stages fill instead of 5/6 idling and no
stage replays a block position by position. Opt-in (V4_PIPELINED_SPEC / a job's `pipelined` flag),
epoch-fenced, and held to the same bit-identical bar. See docs/V4_PIPELINED_SPEC.md.

  self-test (CPU, no GPU, no spend):  python3 phase0/v4_pipe.py selftest
  self-test, multi-GPU tail box:      python3 phase0/v4_pipe.py selftest-relay
  one stage:   python3 phase0/v4_pipe.py stage --stage 0 --nstages 3 --lo 0 --hi 15 --port 29610 --next 127.0.0.1:29611
  coordinator: python3 phase0/v4_pipe.py coord --head 127.0.0.1:29610 --tail 127.0.0.1:29612 --dir /root/v4
"""
import argparse
import collections
import json
import os
import queue
import select
import socket
import struct
import sys
import threading
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

import torch  # noqa: E402

# The lever registry (phase0/v4_levers.py). Module scope and UNGUARDED for the same reason v4_stage
# imports it that way: a process that cannot audit its own levers produces numbers nobody can trust,
# so a deploy that forgot the file must die in the launch log, not serve.
import v4_levers  # noqa: E402

try:                                                    # flat box layout (files pushed to /root/) else package
    from transport import send_msg, recv_msg
except ImportError:
    from shard.transport import send_msg, recv_msg
try:
    from receipt import (ReceiptSigner, load_or_make_node_key, pub_b64, verify_coverage,
                         wire_receipt, ReceiptError)
except ImportError:
    from shard.receipt import (ReceiptSigner, load_or_make_node_key, pub_b64, verify_coverage,
                               wire_receipt, ReceiptError)

# The direct-return port map, identical to k3_pipe/m25_scatter_pipe so a V4 ring reuses the SAME
# sidecar wiring: the sidecar is payload-agnostic (proven on M2.5 and K3), so only the launched
# engine binary changes.
LIBP2P, ENG_IN, FWD_RING, FWD_RET = 29600, 29610, 29611, 29612
# Multi-GPU-per-box: one box runs G stage processes, each pinned to a distinct local GPU, chained
# over LOOPBACK. The box's FIRST local stage binds ENG_IN (so the sidecar -inbound and the head
# coordinator reach it); its 2nd..Gth bind ENG_LOCAL_BASE+idx, ports the sidecar never touches.
ENG_LOCAL_BASE = 29620
NODELAY = (socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
# Forward-dial retry window (K3 ring Bug 1, 2026-07-29): every stage dials its --next at boot, but
# all stages load their weights IN PARALLEL and a V4 stage's load is minutes on a real shard. So a
# front stage is up and dialing long before a downstream neighbour binds. The window must cover the
# SLOWEST stage's whole load; it derives from the stage `timeout` unless V4_DIAL_RETRY_S overrides,
# and each individual connect is capped so one black-holing WAN box cannot burn the window in a
# single blocked connect.
DIAL_RETRY_S = float(os.environ.get("V4_DIAL_RETRY_S", "0") or 0)          # 0 => derive from timeout
DIAL_CONNECT_TIMEOUT = float(os.environ.get("V4_DIAL_CONNECT_TIMEOUT", "5") or 5)
NODE_KEY_PATH = os.environ.get("SHARD_NODE_KEY", "/root/.shard_node_key")
# C2: per-swarm epoch token — env-only (never argv/log), rides every greeting so a stage never
# adopts a silent/foreign connection. Optional; a bare localhost ring runs token-less.
SWARM_TOKEN = os.environ.get("SHARD_SWARM_TOKEN") or None
RECEIPTS = os.environ.get("SHARD_RECEIPTS", "") not in ("", "0")

V4_MODEL_ID = "deepseek-ai/DeepSeek-V4-Flash-0731"
N_LAYERS = 43                                          # the shipped config's n_layers (not the DSpark stages)

# The two fields V4's shipped config.json OMITS. ModelArgs defaults them to 4096 / 4, and everything
# a stage allocates is sized off them: the freqs_cis table, every kv_cache (window + max_seq_len//ratio
# compressed slots), the Indexer's cache and the DSpark drafter's end-of-context guard. Left alone, an
# 8k benchmark rolls off the end of freqs_cis and the drafter stops drafting at ~4090, while a batch
# of 4 pays 4x the cache for a ring that serves one sequence. generate.py:84-85 patches the same two
# fields for its own interactive path; a Stage built straight from config.json does not.
V4_MAX_SEQ = int(os.environ.get("V4_MAX_SEQ", "8192") or 8192)
V4_MAX_BATCH = int(os.environ.get("V4_MAX_BATCH", "1") or 1)

# A dead forward leg is an EDGE fault, not a process death — see _fwd_open.
_LEG_ERRORS = (OSError, EOFError)

# ── stage-side step timing (V4_TIMING) ─────────────────────────────────────────────────────────────
# A ring hop costs recv-wait + on-box work, and only the on-box half is ours to shrink. Live, the two
# are indistinguishable from the coordinator: it sees one round-trip. This accumulates wall time per
# PHASE of the serve loop and prints the per-frame means at the receipt sweep — one line per job.
#
# OFF (the default) every call site is a method on a shared no-op singleton: no clock read, no
# allocation, and byte-identical behaviour. ON it costs two perf_counter reads per phase plus, on
# CUDA, ONE torch.cuda.synchronize() after the layers. That sync is the point: kernels are queued
# async, so without it `fwd` measures dispatch only and the GPU time lands in whichever later phase
# first touches the result (the D2H copy in _make_step_frame) — precisely the attribution question
# this exists to answer. It adds no wall time, because the frame already blocks on that same copy
# before it can send.
V4_TIMING = bool(int(os.environ.get("V4_TIMING", "0") or 0))
V4_TIMING_EVERY = int(os.environ.get("V4_TIMING_EVERY", "0") or 0)      # >0: also print every N frames


class _NoTimer:
    """The V4_TIMING=0 stand-in — every hook is a no-op, so the serve loop is unchanged."""
    __slots__ = ()

    def lap(self, phase, obj=None):
        pass

    def sync(self):
        pass

    def frame(self, s=1):
        pass

    def report(self):
        pass

    def start(self):
        pass


_NO_TIMER = _NoTimer()


def _tensor_bytes(obj):
    """Bytes of the tensor blobs inside a decoded message — what actually crossed the wire, give or
    take the ~150 B JSON header. Scalars count as ZERO: they ride the header, and counting an int's
    VALUE as a byte count is the obvious trap here — the head's `ids` arrive as a plain list of token
    ids (the coordinator sends no tensor), so summing them reported a 1.5 KB frame as 189 KB."""
    if torch.is_tensor(obj):
        return obj.numel() * obj.element_size()
    if isinstance(obj, dict):
        return sum(_tensor_bytes(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(_tensor_bytes(v) for v in obj)
    return 0


def _frame_bytes(obj):
    """Byte count for a timed phase: an int is send_msg's own exact on-the-wire count and is taken
    as-is; anything else is a decoded message to size."""
    return obj if isinstance(obj, int) else _tensor_bytes(obj)


class _StepTimer:
    """Per-phase wall-clock accumulator for one stage's serve loop.

    `lap(phase, obj)` closes the phase that began at the previous lap (or at `start()`), so the
    phases tile the frame exactly: `recv` covers the block on the predecessor AND the frame decode,
    the rest is this box's own work. `obj` is optional and only carries BYTES — a message to size or
    a count from send_msg — so `recv`/`send` report an effective wire rate next to their wall time.
    That is the whole point on a consumer uplink: a drafted round's frame is `block_size+1` times
    fatter than a greedy one, and only bytes-over-seconds separates "the layers got slower" from
    "the uplink did". `start()` re-bases at a job reset; `report()` prints the per-frame means."""
    __slots__ = ("tag", "phases", "cuda", "every", "acc", "by", "n", "npos", "t")

    def __init__(self, tag, phases, device=None, every=0):
        self.tag, self.phases = tag, phases
        self.cuda = str(device or "").startswith("cuda")
        self.every = every
        self.start()

    def start(self):
        self.acc = dict.fromkeys(self.phases, 0.0)
        self.by = dict.fromkeys(self.phases, 0)
        self.n = self.npos = 0
        self.t = time.perf_counter()

    def lap(self, phase, obj=None):
        now = time.perf_counter()
        self.acc[phase] += now - self.t
        self.t = now
        if obj is not None:
            self.by[phase] += _frame_bytes(obj)

    def sync(self):
        if self.cuda:
            torch.cuda.synchronize()

    def frame(self, s=1):
        self.n += 1
        self.npos += int(s)
        if self.every and self.n % self.every == 0:
            self.report()

    def _phase(self, p):
        ms = 1000.0 * self.acc[p] / self.n
        if not self.by[p]:
            return f"{p}={ms:.2f}"
        kb = self.by[p] / 1024.0 / self.n
        mbps = (self.by[p] / 1e6) / self.acc[p] if self.acc[p] > 0 else float("inf")
        return f"{p}={ms:.2f}[{kb:.0f}KB {mbps:.1f}MB/s]"

    def report(self):
        if not self.n:
            return
        on_box = sum(1000.0 * self.acc[p] / self.n for p in self.phases if p != "recv")
        print(f"{self.tag} timing n={self.n} s={self.npos / self.n:.1f} "
              + " ".join(self._phase(p) for p in self.phases)
              + f" on_box={on_box:.2f} ms/frame", flush=True)


def _timer(tag, phases, device=None):
    """The serve loop's timer: a real one under V4_TIMING, else the shared no-op."""
    if not V4_TIMING:
        return _NO_TIMER
    return _StepTimer(tag, phases, device, V4_TIMING_EVERY)


def _v4():
    """phase0/v4_stage, imported on first use (see the module docstring's LAZY MODEL IMPORT note).

    Everything above this line is protocol; everything that calls this needs weights."""
    import v4_stage
    return v4_stage


def _dspark():
    """phase0/v4_dspark_draft, imported on first use. Costs `import torch` and nothing else.

    The coordinator needs exactly ONE thing from it — `plan_verify_round`, the accept rule — and the
    tail needs the drafter. Two implementations of "longest matching prefix" that ever disagreed
    would desynchronise the drafter from the committed stream while both halves still looked
    plausible, so there is one function and both ends import it. v4_dspark_draft resolves v4_stage
    lazily for precisely this reason: the protocol layer must keep running on a box with no
    checkpoint loader."""
    import v4_dspark_draft
    return v4_dspark_draft


# Building a Stage is not thread-safe, and that is a property of the REFERENCE, not of v4_stage:
# model.py keeps world_size/default_dtype/scale_fmt/scale_dtype as MODULE globals and its set_dtype()
# context manager drives torch's PROCESS-WIDE default dtype (v4_stage mirrors generate.py's
# construction environment exactly, including the nested set_dtype(float32) the hyper-connection
# parameters need). Two stages constructing CONCURRENTLY interleave those context managers, and the
# loser silently gets bf16 hyper-connection weights — which surfaces a whole ring later as
# "expected m1 and m2 to have the same dtype" mid-forward, or worse, as wrong numbers.
# A real deployment runs ONE stage per process and never notices. An in-process ring (the selftest,
# tests/test_v4_pipe.py, any co-located launcher that threads) does, so construction is serialized
# here. It costs nothing: one stage per process takes the lock once, uncontended.
_BUILD_LOCK = threading.Lock()


# ── receipt payload digest: the hyper-connection streams AND the routing ids ────────────────────────

def _tbytes(t):
    return t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


def _ids_tensor(v):
    """The chunk's token ids as a contiguous int64 tensor [b, s]. The coordinator sends a nested LIST
    (json header, no blob — it is a handful of ints); every hop after the head carries the tensor."""
    t = v if torch.is_tensor(v) else torch.as_tensor(v, dtype=torch.int64)
    return t.to(torch.int64).contiguous()


def _payload_bytes(h, ids):
    """Deterministic bytes of a V4 boundary payload (hyper-connections || token ids), for the receipt
    hash-chain. Hashing the IDS too is what makes the chain attest the hash-routed MoE input: V4's
    first n_hash_layers pick experts from the token id, so a hop that dropped or rewrote the ids would
    change which experts run downstream while `h` still looked plausible. Identical bytes on both
    sides of a lossless hop, so adjacent stages' out_root/in_root match by construction."""
    return _tbytes(h) + _tbytes(ids)


# ── fp8 activation wire (V4_FP8_WIRE): halve the inter-stage bytes/hop ──────────────────────────────
# V4's boundary is 4 hyper-connection streams x dim x 2 B = 32 KiB/token at dim 4096, and a DSpark
# chunk carries block_size+1 positions of it. Transporting h as fp8 (e4m3) halves those bytes, and
# the framing does NOT eat it: measured through the real transport codec at the shipped config, the
# s=1 pipelined decode frame is 32,968 B bf16 against 16,672 B fp8 (1.977x), of which the JSON
# header and length prefixes are 192 B — 0.58%. phase0/v4_wire_bench.py re-derives the table.
#
# CHAIN-PRESERVING: the receipt hashes the EXACT fp8 wire bytes on BOTH sides (sender's out_root over
# the packed tensor, receiver's in_root over the same received bytes), so out_root == in_root still
# holds and the receipts stay CHAINED even though the activation is lossy — no check_chain relaxation
# needed. The bf16 path is byte-identical when off. The IDS are never packed: they are exact indices
# into a hash function and a vocab table.
V4_FP8_WIRE = bool(int(os.environ.get("V4_FP8_WIRE", "0") or 0))


def _pack_t(t):
    """fp8 (e4m3) quantize one boundary activation for transport, ONE SCALE PER (position, stream):
    the amax reduces over `dim` ONLY, which is the same axis the reference's own act_quant scales
    along (kernel.py act_quant, block_size on the last dim). Returns (fp8 [b,s,hc,dim], bf16 [b,s,hc]).

    THE REDUCTION AXIS IS THE WHOLE DESIGN, AND IT IS A CORRECTNESS PROPERTY, NOT AN ACCURACY ONE.
    e4m3 is a FLOAT — it carries its own 4-bit exponent, 2^14.8 of normal dynamic range — so a
    coarser scale buys almost nothing in error: measured over the boundary tensors of a real V4
    stack, a per-TENSOR scale is 1.03x worse than a per-128-block one even when the chunk spans
    2^-20 of dynamic range, and block scales cost 4-32x more sidecar bytes. Granularity is not where
    the accuracy is.

    Where it does decide something is SPECULATION. A scale that reduces over the SEQUENCE axis makes
    a position's transmitted bytes depend on which OTHER positions happened to share its frame, and
    the DSpark path deliberately sends the same position inside chunks of different lengths (the
    bare first round is s=1, a drafted round is s=dspark_block_size+1). Under a per-tensor scale the
    drafted stream therefore stops being bit-identical to the greedy stream — the engine's headline
    property, and the thing coordinate_dspark's accept rule is built on — for a reason that has
    nothing to do with the model. Reducing over `dim` alone makes the packing of position p a
    function of position p's own values, so spec == greedy survives the lossy wire. MEASURED on the
    CPU ring with a drafter proposing the ring's own greedy continuation (so a self-consistent wire
    must accept everything): bf16 g=4.00, this codec g=4.00, a per-tensor scale g=1.71. Granularity
    is worth 2.3x of acceptance here, and nothing at all in error.

    The amax is taken in fp32, NOT in the activation's own dtype: `clamp(min=1e-8)` coerces its
    bound to the tensor dtype, and in float16 1e-8 underflows to 0.0, so the guard silently vanishes
    and an all-zero row divides by zero. The clamp on the quotient is the vendored kernel's own
    contract (act_quant: `(z / s).clamp(-FP8_MAX, FP8_MAX)`); without it a division that rounds a
    hair above 448 lands in e4m3fn's saturating band, and far enough above it is a silent NaN.

    The scale rides bf16 and the sender divides by that SAME rounded value, so both ends use bit-
    identical scales. That is 2 B per (position, stream) — 8 B against 32 KiB of payload per token,
    0.024% — and the residual the rounding adds is ~2^-9 against e4m3's own 2^-4 step.

    The upper clamp is unreachable for any finite activation (it binds only past 448 * bf16-max) and
    exists solely so ONE inf cannot poison its neighbours: an infinite amax would make the scale
    infinite, and then every FINITE value in the row divides to 0 and the inf itself to NaN. The
    bf16 wire carries a bad value in exactly one slot; without this the fp8 wire would turn it into
    a row of 4096 NaN. With it, the inf saturates to 448 and the rest of the row is untouched."""
    f = t.detach().float()
    scale = (f.abs().amax(-1, keepdim=True) / 448.0).clamp(
        min=1e-8, max=torch.finfo(torch.bfloat16).max).to(torch.bfloat16)
    q = (f / scale.float()).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).contiguous()
    return q, scale.squeeze(-1).contiguous()


def _unpack_t(q, scale):
    """fp8 [b,s,hc,dim] + bf16 [b,s,hc] -> bf16, in bf16 throughout. Deliberately NOT via fp32: the
    product is exact either way (an e4m3 value is 4 significant bits, a bf16 holds 8), and a fp32
    intermediate would transiently double the biggest tensor on the ring — a 8192-token prefill's h
    is 268 MB bf16 against 537 MB fp32, on a stage that is already holding weights."""
    return q.to(torch.bfloat16) * scale.unsqueeze(-1)


def _wire_bytes(h, ids, hs):
    """Deterministic bytes of a payload AS TRANSMITTED on the fp8 wire: the fp8 hidden bytes, the
    (unpacked, int64) ids, and the scales. Sender hashes what it packs, receiver hashes what it got
    -> out_root == in_root exactly."""
    return _tbytes(h) + _tbytes(ids) + _tbytes(hs)


def _recv_hids(msg, signer):
    """Take (h, ids) off a received step frame, upcasting h from fp8 when the wire carried it.
    Returns (h, ids, in_bytes) where in_bytes is the receipt in_root digest over the EXACT received
    wire bytes (None when not signing).

    The frame, not the local V4_FP8_WIRE, decides — a stage decodes whatever its predecessor sent,
    so the two do not have to agree. That is what makes a mixed ring merely inefficient instead of
    wrong, and it is also why `h8` and `h.dtype` MUST be cross-checked here: an fp8 `h` whose scale
    sidecar went missing would otherwise sail through `st.forward`'s `h.to(self.dtype)` as a tensor
    scaled by ~1/scale, with no exception, no NaN, and an intact receipt chain (both sides hash the
    same bytes). ValueError is in _STRAY, so a malformed frame resets the edge like any other."""
    h, ids = msg["h"], _ids_tensor(msg["ids"])
    packed = torch.is_tensor(h) and h.dtype == torch.float8_e4m3fn
    if packed != ("h8" in msg):
        raise ValueError(f"step frame: h is {getattr(h, 'dtype', type(h))} but h8 is "
                         f"{'present' if 'h8' in msg else 'absent'}")
    if packed:
        hs = msg["h8"]
        if not torch.is_tensor(hs) or hs.shape != h.shape[:-1]:
            raise ValueError(f"step frame: h8 {getattr(hs, 'shape', type(hs))} does not scale "
                             f"h {tuple(h.shape)} per (position, stream)")
        in_b = _wire_bytes(h, ids, hs) if signer is not None else None
        return _unpack_t(h, hs), ids, in_b
    return h, ids, (_payload_bytes(h, ids) if signer is not None else None)


def _make_step_frame(h, ids, start_pos, signer):
    """Build the forwarded step frame + its receipt out_root digest. fp8-packs h when V4_FP8_WIRE,
    carrying the per-tensor scale as the h8 sidecar; the out digest hashes the packed wire bytes so
    the next stage's in_root matches losslessly. The ids ride int64 in both modes.

    ONE device->host copy, shared. On a GPU stage `h` comes out of the layers on cuda and two
    separate things want it as host bytes: the receipt digest (`_tbytes` -> .detach().cpu()) and the
    wire (transport's `_pack_parts` -> .detach().cpu()). Leaving both to find their own way copied
    the same payload down twice per frame per hop. Landing it once here and handing the SAME CPU
    tensor to both is byte-identical by construction — they were always hashing and sending the same
    values, and now they hash and send the same buffer."""
    ids = _ids_tensor(ids)
    if V4_FP8_WIRE:
        qh, sh = _pack_t(h)
        qh = qh.detach().cpu().contiguous()
        frame = {"op": "step", "h": qh, "h8": sh, "ids": ids, "start_pos": start_pos}
        return frame, (_wire_bytes(qh, ids, sh) if signer is not None else None)
    hc = h.detach().cpu().contiguous()
    frame = {"op": "step", "h": hc, "ids": ids, "start_pos": start_pos}
    return frame, (_payload_bytes(hc, ids) if signer is not None else None)


# ── the epoch fence: recognising frames from a CANCELLED speculation ────────────────────────────────
# Pipelined speculation (coordinate_dspark_pipelined) streams s=1 frames without waiting for their
# replies, so when a reply rejects a draft there are up to W frames already in the ring that belong to
# a future the coordinator has just discarded. Every step frame therefore carries an `epoch` — a
# generation counter the coordinator bumps on every cancel — and every reply echoes it.
#
# WHO THE FENCE IS FOR, precisely, because it is easy to over-claim:
#   * THE COORDINATOR'S RECEIVER is where it is load-bearing. Replies for the discarded frames are
#     still on their way back, and without the epoch it would read them as judgments of the frames it
#     has just streamed into the NEW epoch — committing the wrong tokens at the right positions, which
#     is exactly the silent corruption this whole path has to be free of.
#   * THE SENDER drops its own queued frames of a fenced epoch before they reach the wire, so a cancel
#     stops the ring computing a future nobody wants. (Correctness never rests on winning that race:
#     a frame that slips out is fenced by the receiver anyway.)
#   * THE STAGES honour `fenced`/a stale epoch by passing the frame on WITHOUT computing it — no
#     forward, no checkpoint, no state touched, and no receipt observation, so the hash-chain stays
#     aligned across stages (the marker propagates, so every stage makes the same decision). This is a
#     GUARD, not the bubble-shrinking optimisation PipeInfer describes: on a single in-band FIFO ring a
#     cancel can never overtake the frames it cancels, so in normal operation no stage ever sees one.
#     Skipping queued work needs an out-of-band control path to every stage, which this topology (one
#     forward pipe from the head, one return from the tail) does not have. What the guard buys is that
#     a replaying relay, a re-dialled leg or a coordinator bug cannot rewind a stage with a frame from
#     a discarded future — it drops loudly instead of _seek()ing on stale state.
# The rollback itself needs none of this: the correction frame's `start_pos` IS the rewind command and
# Stage._seek is W-deep (see v4_stage._snapshot). The fence is about who is allowed to BELIEVE a reply.
# Header fields every hop must carry unchanged. `dnxt`/`dprev` are the lazy-draft hints and only the
# TAIL reads them, which is exactly why they belong here: a forwarding stage rebuilds the frame around
# its own activations (`_make_step_frame`), so anything the coordinator says to the tail has to be
# named as header or it is dropped at the first hop and the tail simply never sees the hint.
_PASSTHRU = ("epoch", "cpos", "dnxt", "dprev")


def _fenced(msg, epoch):
    """Is this step frame from a speculation that has been cancelled? An upstream stage's `fenced`
    marker is authoritative (so one decision propagates down the ring); otherwise an epoch older than
    the newest this stage has seen means the coordinator has moved on without it."""
    return bool(msg.get("fenced")) or int(msg.get("epoch", 0)) < epoch


# ── cwnd keep-warm (V4_KEEPWARM): defeat TCP slow-start-after-idle on the forward legs ──────────────
# Serial decode idles every ring leg for a full traversal (~seconds) between frames. With
# tcp_slow_start_after_idle=1 (READ-ONLY in vast containers), cwnd collapses to the initial window
# each round, so a fat frame pays several slow-start RTTs of ramp every round. A daemon posts a tiny
# {"op":"noop"} on the forward socket whenever it has idled > period, keeping the connection active
# so cwnd never resets. Receivers ALWAYS skip noop frames, so a keepwarm-off peer is unaffected. Real
# sends + the noop share one lock so they never interleave on the socket; the noop acquires it
# NON-BLOCKING (a held lock = a real send in flight = the leg isn't idle, so skip this tick).
V4_KEEPWARM = bool(int(os.environ.get("V4_KEEPWARM", "0") or 0))
V4_KEEPWARM_MS = float(os.environ.get("V4_KEEPWARM_MS", "150") or 150)


class _KeepWarm:
    """Keep one SENDING socket's cwnd warm. All real forward sends go through .send() (locked); the
    daemon posts a noop only when the leg has been idle > period and no real send holds the lock.
    attach() swaps the underlying socket under the same lock — _fwd_open rebuilds a dead leg
    mid-ring, and the noop thread must not keep writing to the corpse."""

    def __init__(self, sock):
        self.sock = sock
        self.lock = threading.Lock()
        self.last = time.monotonic()
        self._stop = False
        self.on = V4_KEEPWARM and sock is not None
        if self.on:
            self.period = V4_KEEPWARM_MS / 1000.0
            threading.Thread(target=self._run, daemon=True, name="v4-keepwarm").start()

    def attach(self, sock):
        with self.lock:
            self.sock = sock
            self.last = time.monotonic()

    def send(self, msg):
        if not self.on:
            return send_msg(self.sock, msg)
        with self.lock:
            n = send_msg(self.sock, msg)
            self.last = time.monotonic()
        return n                                             # bytes on the wire, as send_msg reports

    def _run(self):
        while not self._stop:
            time.sleep(self.period / 2)
            if time.monotonic() - self.last < self.period:
                continue
            if self.lock.acquire(blocking=False):
                try:
                    if self.sock is not None:
                        send_msg(self.sock, {"op": "noop"})
                        self.last = time.monotonic()
                except Exception:                                # a dead socket is the serve loop's problem
                    pass
                finally:
                    self.lock.release()

    def stop(self):
        self._stop = True


# ── sampling: the tail turns the last-position logits into a token ─────────────────────────────────

def sample_token(logits_row, temp=0.0, gen=None):
    """[vocab] -> int id. Greedy argmax at temp 0 (deployed default, bit-exact and the parity bar —
    the reference's own sample() is an argmax at temperature 0); temperature samples from
    softmax(logits/temp) with an optional torch.Generator for determinism. Ties break to the lowest
    index, exactly like the reference's argmax."""
    if temp and temp > 0:
        probs = (logits_row.float() / float(temp)).softmax(-1)
        return int(torch.multinomial(probs, 1, generator=gen).item())
    return int(logits_row.argmax().item())


# ── DSpark tail seam ───────────────────────────────────────────────────────────────────────────────
# V4 ships a TRAINED speculator (the DSpark MTP stages) that lives on the TAIL, not on the
# coordinator: forward_spec() needs `main_hidden` — the mean-pooled hidden of the last few main
# layers — which only the tail holds. So a V4 drafted round does not stream aux down the ring the way
# K3's DSpark did; the tail drafts locally and returns the draft with the token.
#
# THE SEAM: an object with
#     on_chunk(msg, st, out) -> dict | None
# where `msg` is the received step frame, `st` the tail Stage (st.tail_main_hidden() is the
# [b, s, 3*dim] the MTP stages consume) and `out` the reply dict already holding {"token": ...} (and
# "tokens" when the job is spec-armed). Whatever it returns is merged into the reply. It is called
# ONLY on a step frame of a job whose reset armed dspark, so a greedy ring never touches it.
#
# The real one is v4_dspark_draft.RingDrafter, which a --dspark tail builds for itself on the first
# drafted job (_tail_drafter). Setting this module hook OVERRIDES that — it is the injection point a
# test scripts a tail with, and it stays None in every deployment.
TAIL_DRAFTER = None


# ── the job horizon: what v4_ref_slim's indexer skip needs from the serve path ─────────────────────
# v4_ref_slim (V4_REF_SLIM) skips the Indexer's SCORING while every compressed slot is selected
# anyway, and may additionally skip the Indexer's COMPRESSOR — which is STATE, not a query — but only
# for a job guaranteed never to leave that regime. Guess that wrong in the SHORT direction and the
# indexer re-engages past `index_topk * ratio` against a half-filled cache and picks the wrong keys,
# silently. So the horizon travels with the job, in the reset frame that already opens it, and every
# stage sets it before the first step of that job lands.
#
# THE SAFETY DIRECTION IS ASYMMETRIC, AND EVERYTHING HERE LEANS THE SAFE WAY:
#   over-declare  -> the compressor keeps advancing -> correct at ANY length, costs 2 cheap GEMMs.
#   under-declare -> the compressor is skipped and the indexer may still re-engage -> WRONG.
#   absent (None) -> "unknown", which v4_ref_slim already treats as keep-advanced, i.e. the safe end.
# So a reset frame from an older coordinator (no "max_pos" key) degrades to correct-but-unoptimised,
# never to wrong, and a margin that is too FAT costs nothing but a missed optimisation.

# Speculative overshoot: a spec/dspark round feeds `[cur] + drafts` at the committed position, so the
# ring transiently touches positions past the last COMMITTED one even though a rejected draft is
# rolled back. The committed length is prompt + max_new; this covers the draft block on top of it.
# 64 is far above any block the ring actually offers (coordinate_spec defaults K=4, the shipped DSpark
# MTP drafts 3) and is deliberately fat: see the asymmetry above.
_SPEC_POS_MARGIN = 64


def _job_max_pos(prompt_ids, max_new, spec_margin=0):
    """The job's GUARANTEED-not-to-be-exceeded maximum absolute position, for the reset frame.

    Upper bound, never an estimate — `prompt + max_new` is the committed ceiling the coordinator loops
    enforce, and `spec_margin` covers the draft tokens a speculative round puts on the wire above it."""
    return len(prompt_ids) + int(max_new) + int(spec_margin)


def _set_job_horizon(max_pos):
    """Hand v4_ref_slim this job's horizon (None = unknown, which it treats as the safe answer).

    Called on EVERY reset, including with None, so one job's horizon can never leak into the next: the
    value is process-wide, matching model.py's other globals. Unconditional — v4_ref_slim is import-
    light and pure-Python, and with V4_REF_SLIM off nothing reads what this sets, so the cost of not
    branching on the env here is one module-global assignment per job."""
    import v4_ref_slim
    v4_ref_slim.set_job_max_pos(max_pos)


# ── stage server: one V4 layer block, fire-forward ────────────────────────────────────────────────

def _dial_window(timeout):
    """The forward-dial retry WINDOW in seconds: V4_DIAL_RETRY_S when set, else the stage `timeout`
    itself. Keyed to an existing constant rather than a bare magic number so it scales with how long
    a stage is allowed to take, and comfortably covers a neighbour that is still loading weights."""
    return DIAL_RETRY_S or float(timeout)


def _dial(host, port, timeout, retry_s=None):
    """Dial host:port, retrying for a time WINDOW while the peer is still coming up (see DIAL_RETRY_S).
    A stage binds+listens BEFORE it dials, so a generous window lets the whole chain resolve whatever
    order the stages were launched in. Each individual connect is capped at DIAL_CONNECT_TIMEOUT so a
    black-holing WAN box cannot consume the window in one blocked connect."""
    connect_timeout = min(float(timeout), DIAL_CONNECT_TIMEOUT)
    window = float(retry_s) if retry_s is not None else _dial_window(timeout)
    deadline = time.time() + window
    last = None
    while True:
        try:
            s = socket.create_connection((host, int(port)), timeout=connect_timeout)
            s.setsockopt(*NODELAY)
            # OS-level TCP keepalive: a stage dials its successor right after load, then the socket
            # sits IDLE until the first job flows down the chain. Some WAN routes (seen live FR->SE,
            # OVH egress) drop an idle connection in under a minute, so the first forward send hits a
            # dead pipe. Keepalive probes every 15s keep the mapping warm through that idle window.
            s.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            for _opt, _val in (("TCP_KEEPIDLE", 15), ("TCP_KEEPINTVL", 15), ("TCP_KEEPCNT", 4)):
                if hasattr(socket, _opt):
                    try:
                        s.setsockopt(socket.IPPROTO_TCP, getattr(socket, _opt), _val)
                    except OSError:
                        pass
            s.settimeout(float(timeout))                # a fat frame must not trip the CONNECT timeout
            return s
        except OSError as e:
            last = e
            if time.time() >= deadline:
                break
            time.sleep(0.25)
    raise RuntimeError(f"v4 stage: could not connect to {host}:{port} within {window:.0f}s "
                       f"({type(last).__name__}: {last})")


def _fwd_open(kw, nxt, timeout, msg, tag="[s]"):
    """Forward a job-OPENING frame, rebuilding the leg if it died while the ring was idle (#156,
    ported from m25_pipe — k3_pipe predates the fix and a V4 ring would inherit the bug).

    A stage dials its successor BEFORE it binds its own listener, and every stage in the ring loads
    weights concurrently — so whichever finishes first dials a successor whose engine has not bound
    yet. The sidecar carrying that leg accepts the local conn anyway, fails to reach the far engine,
    and closes it (sidecar/main.go runForward / runInbound). The leg is write-only, so nothing
    observes that: the ring then sits idle for the rest of the pull holding a dead socket that every
    readiness check still reads as healthy.

    The FIRST WRITE of the next job is what discovers it, and that write is the reset. Losing it
    there is UNRECOVERABLE — the coordinator reads the TAIL, so no error can reach it and the job can
    only die on the stall watchdog. The 2026-07-29 capstone ring lost EVERY job that way: six stages
    READY, every socket healthy, 0 tokens, GPU idle.

    A reset carries no state — it IS the job barrier — and a receipt sweep is likewise idempotent
    per job, so re-sending on a fresh link is exactly equivalent to having opened the job on it. ONE
    retry: a successor that is genuinely gone must still surface as a dead ring rather than a spin.
    Returns the (possibly new) forward socket."""
    try:
        kw.send(msg)
        return kw.sock
    except _LEG_ERRORS as e:
        if nxt is None:                                  # no address to re-dial (in-process ring): the
            raise                                        # caller's supervision owns it
        print(f"{tag} forward leg was dead at job open ({type(e).__name__}); "
              f"rebuilding -> {nxt} and re-sending {msg.get('op')!r}", flush=True)
    try:
        if kw.sock is not None:
            kw.sock.close()                              # the retry dials a FRESH link; the dead one's
    except OSError:                                      # fd must not leak across jobs
        pass
    kw.attach(None)
    sock = _dial(*nxt.rsplit(":", 1), timeout=timeout)    # a re-dial that fails raises into the serve
    kw.attach(sock)                                       # loop's supervision, exactly as before
    if SWARM_TOKEN is not None:
        # the fresh link must greet like the launch-time dial did, or a token-mode successor
        # (rightly) drops it as a stranger and the self-heal this function exists for never lands
        send_msg(sock, {"op": "hello_pred", "token": SWARM_TOKEN})
    kw.send(msg)
    return sock


def ring_args(ckpt_dir):
    """The ModelArgs a V4 process is built from, with the fields the shipped config omits filled in.

    ONLY where the checkpoint's own config.json is SILENT, or where the env says so explicitly. That
    distinction is the whole design: V4's release declares neither max_seq_len nor max_batch_size, so
    the ring supplies 8192 / 1 (V4_MAX_SEQ / V4_MAX_BATCH) — but a config that DOES declare them means
    it, and overwriting those would build stages whose caches and freqs_cis table disagree with the
    model they are graded against. (The CPU parity fixtures declare both; that is how they stay
    bit-exact against the oracle while a real ring gets 8k.)

    Neither field changes numerics — both only SIZE things. `precompute_freqs_cis` derives its YaRN
    correction from original_seq_len, not from the table length, and every read of a compressed region
    is sliced by position (`kv_cache[:bsz, :end_pos // ratio]`), so a longer buffer is a bigger buffer
    and nothing else. It is memory and reach, not answers."""
    V4 = _v4()
    args = V4.config(ckpt_dir)
    with open(os.path.join(ckpt_dir or V4.V4_DIR, "config.json")) as f:
        declared = json.load(f)
    for field, env, default in (("max_seq_len", "V4_MAX_SEQ", V4_MAX_SEQ),
                                ("max_batch_size", "V4_MAX_BATCH", V4_MAX_BATCH)):
        if os.environ.get(env):
            setattr(args, field, int(os.environ[env]))
        elif field not in declared:
            setattr(args, field, default)
    return args


def _is_return_hello(msg):
    """A coordinator-return greeting THIS swarm accepts. When SHARD_SWARM_TOKEN is set the token
    VALUE is compared — a greeting of the right shape with the wrong (or no) token is a stranger.
    The shape-only check shipped first and was a hole: any scanner speaking the frame codec could
    have adopted the return channel of a token-'gated' ring."""
    if not (isinstance(msg, dict) and msg.get("op") == "hello_return"):
        return False
    return SWARM_TOKEN is None or msg.get("token") == SWARM_TOKEN


def _is_pred_hello(msg):
    """Predecessor greeting, token value compared under the same rule as _is_return_hello."""
    if not (isinstance(msg, dict) and msg.get("op") == "hello_pred"):
        return False
    return SWARM_TOKEN is None or msg.get("token") == SWARM_TOKEN


def serve_stage(stage, nstages, lo, hi, port, nxt=None, *, ckpt_dir=None, args=None, device=None,
                receipts=None, key_path=None, timeout=600.0, bind="127.0.0.1", ready=None,
                ret_relay=None, dspark=False):
    """Serve one contiguous layer block [lo:hi) in the fire-forward ring.

    head (stage 0)      embeds token ids -> h [b, s, 4, dim], runs its layers, forwards (h, ids).
    middle              runs its layers on the received (h, ids), forwards them.
    tail (stage n-1)    runs its layers, then hc_head + final norm + lm_head + SAMPLE, and returns
                        the token id on the coordinator-return channel.

    `ret_relay` (a multi-GPU TAIL box only) makes this the box-ingress RELAY: it is a normal forward
    stage for its own layers, but the coordinator-return tunnel terminates at THIS box's ENG_IN while
    the token is produced on the box's last-GPU stage, so it also bridges the return — dialing the box
    tail's return channel at `ret_relay` (loopback) and pumping every frame the tail sends straight out
    to the coordinator. Only the box ingress of a G>1 tail box passes it.

    `dspark` builds the tail's MTP stages (the trained speculator) at load time; per-JOB arming is the
    reset's `dspark` flag. `ready` is an optional threading.Event set once the stage is listening (the
    selftest waits on it instead of sleeping). `args` (a ModelArgs) skips the on-disk config for an
    in-process ring; `ckpt_dir` loads real weights."""
    V4 = _v4()
    head, tail = (stage == 0), (stage == nstages - 1)
    args = args if args is not None else ring_args(ckpt_dir)
    dev = device or getattr(V4, "dev", "cuda")
    receipts = RECEIPTS if receipts is None else receipts
    key_path = key_path or NODE_KEY_PATH
    if str(dev).startswith("cuda"):
        # generate.py:92 does this and the reference NEEDS it: model.py's lru_cached index helpers
        # (get_window_topk_idxs:261, get_compress_topk_idxs:275, get_dspark_topk_idxs:744) build
        # their topk_idxs with a bare `torch.arange` at FORWARD time, on the AMBIENT default device
        # — outside any `with torch.device(...)` the Stage's constructor used. Without this the
        # first attention hands a CPU int32 tensor to a CUDA kernel. It is process-wide and it is
        # set HERE, in the serve path, rather than at import: a library import that moves every
        # caller's default device is a trap, and the CPU parity suite shares this process.
        torch.set_default_device(dev)
        # generate.py:77's other half, equally load-bearing at FORWARD time: kernel.py's GEMM
        # wrappers allocate their output at torch.get_default_dtype() (fp8_gemm:270, fp4_gemm:533),
        # and the tilelang kernels are compiled for bf16 C — under the fp32 process default the
        # first quantized linear dies with "input C dtype mismatch, expected bfloat16" (observed:
        # it killed the head stage's prefill on the first live ring, 2026-08-01). Guarded by the
        # same cuda condition so the CPU suite keeps its fp32 default.
        torch.set_default_dtype(torch.bfloat16)
    with _BUILD_LOCK:                                   # process-wide torch/reference globals — see the lock
        st = V4.Stage(lo, hi, args, head=head, tail=tail, dspark=(dspark and tail), device=dev)
        if ckpt_dir is not None:
            st.load(ckpt_dir)
    node_key = load_or_make_node_key(key_path) if receipts else None
    print(f"[s{stage}] {st}", flush=True)
    # THE LEVER AUDIT, here and not later: everything a stage installs is installed by now, and every
    # number this process is about to produce is about whatever configuration it actually reached.
    # Requested vs LIVE-observed, per lever, to the log this launch already redirects stderr into.
    # Raises under V4_LEVERS_STRICT rather than serving a ring whose measurement would be meaningless.
    v4_levers.report(side=v4_levers.STAGE, stage=st)

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((bind, port))
    srv.listen(4)
    # non-tail stages dial the forward leg at startup (a dead --next at boot is a launcher bug); the
    # sidecar tunnels FWD_RING to the next stage's inbound. Kernel accepts the handshake as soon as
    # the peer is listening, so this completes before the peer calls accept().
    nxt_sock = _dial(*nxt.rsplit(":", 1), timeout=timeout) if (not tail and nxt) else None
    if nxt_sock is not None and SWARM_TOKEN is not None:
        send_msg(nxt_sock, {"op": "hello_pred", "token": SWARM_TOKEN})
    print(f"[s{stage}] listening {bind}:{port}"
          + (f" -> {nxt}" if nxt_sock is not None else " (tail)")
          + (f" pub={pub_b64(node_key)}" if node_key is not None else ""), flush=True)
    if ready is not None:
        ready.set()

    try:
        if tail:
            _serve_tail(st, srv, lo, hi, node_key, receipts, timeout, ckpt_dir=(ckpt_dir if dspark
                                                                               else None))
        elif ret_relay is not None:
            _serve_relay_ingress(st, stage, srv, nxt_sock, nxt, head, lo, hi, node_key, receipts,
                                 timeout, ret_relay)
        else:
            _serve_forward(st, stage, srv, nxt_sock, nxt, head, lo, hi, node_key, receipts, timeout)
    finally:
        # Teardown: this stage serves no more jobs, so drop the horizon it was carrying. In a real
        # ring that is one process per stage and the clear is cosmetic; in the IN-PROCESS shape (the
        # selftest runs every stage as a thread beside the coordinator and the oracle) the horizon is
        # a shared global, and a dead stage thread must not leave one job's value behind it.
        _set_job_horizon(None)


_STRAY = (ConnectionError, OSError, ValueError, KeyError, TypeError, struct.error)


def _accept_pred(srv, timeout):
    """Accept the predecessor, tolerating stray connections on the public engine port. The port is
    internet-facing (a scanner racing the real predecessor for accept() sends e.g. an HTTP probe,
    whose bytes parse as a garbage frame length and would crash the stage — cascading the ring). So
    select() and only recv a READABLE connection: the real predecessor emits keep-warm noops within
    seconds, while a silent scanner never becomes readable and is ignored; a connection that delivers
    an unparseable/oversized frame is dropped and we keep waiting. token-mode returns after the
    hello_pred greeting; token-less returns the first frame as `queued`."""
    pending = []
    while True:
        ready, _, _ = select.select([srv] + pending, [], [])
        if srv in ready:
            conn, _ = srv.accept()
            conn.setsockopt(*NODELAY)
            conn.settimeout(timeout)
            pending.append(conn)
            continue
        conn = next(c for c in pending if c in ready)
        pending.remove(conn)
        try:
            first = recv_msg(conn)
        except _STRAY as e:
            print(f"[s] dropped stray inbound: {type(e).__name__}", flush=True)
            try:
                conn.close()
            except Exception:
                pass
            continue
        if SWARM_TOKEN is not None:
            if not _is_pred_hello(first):
                print("[s] dropped inbound w/o valid greeting", flush=True)
                try:
                    conn.close()
                except Exception:
                    pass
                continue
            return conn, None
        return conn, first


def _warm_until_accept(nxt_sock, period=5.0):
    """Tick a noop on the forward leg every `period`s until stopped. A stage dials its successor right
    after loading, then blocks in srv.accept() waiting for its predecessor — which does not connect
    until a job flows all the way down from the coordinator. Through that wait the just-dialed forward
    connection sits IDLE, and some WAN paths drop an idle connection fast (seen live: OVH egress on one
    box's OUTBOUND and another host's INBOUND both reset idle conns in ~50s), so the very first forward
    send hits a dead pipe and crashes the stage — cascading the whole ring. Keeping real noop traffic
    flowing from the moment of dial holds the mapping warm on both ends. Returns a stop() callable; the
    caller MUST stop() before the forward loop touches nxt_sock (no concurrent send)."""
    if nxt_sock is None:
        return lambda: None
    ev = threading.Event()

    def run():
        while not ev.wait(period):
            try:
                send_msg(nxt_sock, {"op": "noop"})
            except OSError:
                return
    t = threading.Thread(target=run, daemon=True)
    t.start()

    def stop():
        ev.set()
        t.join(timeout=2)
    return stop


def _serve_forward(st, stage, srv, nxt_sock, nxt, head, lo, hi, node_key, receipts, timeout):
    """Head/middle serve loop: process a frame, forward it down the ring, never answer the pipe.

    The HEAD's predecessor IS the coordinator, so it gets a `reaccept` closure: a coordinator that
    exits (a bench that finished, a restart) must not take the ring down with it — the head re-accepts
    a reconnecting one and the rest of the ring stays warm behind it (see _forward_loop). A middle
    stage passes no reaccept: its predecessor is the head, which now survives, so a predecessor loss
    there is a genuine upstream death and should still cascade."""
    stop_warm = _warm_until_accept(nxt_sock)   # keep the just-dialed forward leg warm through accept-wait
    try:
        conn, queued = _accept_pred(srv, timeout)
    finally:
        stop_warm()                            # stop BEFORE the forward loop touches nxt_sock
    print(f"[s{stage}] predecessor connected", flush=True)
    reaccept = (lambda: _accept_pred(srv, timeout)) if head else None
    _forward_loop(st, stage, nxt_sock, nxt, head, lo, hi, node_key, receipts, conn, queued, timeout,
                  reaccept=reaccept)


def _forward_loop(st, stage, nxt_sock, nxt, head, lo, hi, node_key, receipts, conn, queued,
                  timeout=600.0, reaccept=None):
    """The head/middle process-and-forward loop over an already-accepted predecessor `conn` (with an
    optional already-read first frame `queued`). Split out of _serve_forward so the box-ingress relay
    reuses it verbatim for the DOWN direction while a pump thread carries the return UP.

    `reaccept` (head only) makes a predecessor disconnect SURVIVABLE instead of fatal: the head reads
    only from the coordinator and writes only down the ring, so a lost predecessor is a lost
    coordinator, and re-accepting one keeps the whole ring warm across a bench exit or a coordinator
    restart. Left None (middles, the relay ingress) a disconnect re-raises and cascades, which is the
    right answer for a real upstream death."""
    signer = None
    kw = _KeepWarm(nxt_sock)                                   # cwnd keep-warm on the forward leg (opt-in)
    tag = f"[s{stage}]"
    epoch = 0                                                  # newest speculation generation seen (fence)
    timer = _timer(tag, ("recv", "pre", "fwd", "out", "send"), getattr(st, "device", None))
    with torch.no_grad():
        while True:
            if queued is not None:
                msg, queued = queued, None
            else:
                try:
                    msg = recv_msg(conn)
                except _LEG_ERRORS as e:
                    if reaccept is None:                       # genuine upstream death: cascade (correct)
                        raise
                    # The head's predecessor IS the coordinator. It exited (a finished bench) or is
                    # restarting; the forward leg to the rest of the ring is still up (kept warm, and
                    # _fwd_open heals it if it idled out), so DO NOT die — re-accept a reconnecting
                    # coordinator and keep every stage warm. A reset opens each job, so nothing partial
                    # survives the gap. This is the head half of surviving a coordinator restart; the
                    # tail's return channel is the other half (_serve_tail's re-accept thread).
                    print(f"{tag} coordinator disconnected ({type(e).__name__}); re-accepting "
                          f"— ring stays warm", flush=True)
                    try:
                        conn.close()
                    except OSError:
                        pass
                    conn, queued = reaccept()
                    print(f"{tag} coordinator reconnected", flush=True)
                    continue
            op = msg.get("op")
            if op == "noop":                                  # keep-warm tick from the predecessor: skip
                continue
            if op == "reset":
                timer.start()                                 # a reset opens a job: time it on its own
                st.reset()
                _set_job_horizon(msg.get("max_pos"))          # v4_ref_slim; absent key => None => safe
                st._spec = bool(msg.get("spec"))              # arm the stage snapshot/rollback (step 5)
                st._dspark = bool(msg.get("dspark"))          # arm the tail's DSpark drafter (step 4)
                epoch = 0
                signer = (ReceiptSigner(node_key, msg.get("swarm_id", "swarm"),
                                        msg.get("job_id", "job"), lo, hi, nonce=msg.get("nonce"))
                          if receipts else None)
                _fwd_open(kw, nxt, timeout, msg, tag)          # propagate reset UNCHANGED down the ring
                continue
            if op == "receipt":                               # job done: append my receipt, pass on
                timer.report()                                # the job barrier: one timing line per job
                if signer is not None:
                    msg.setdefault("receipts", []).append({"stage": stage, **signer.finalize()})
                _fwd_open(kw, nxt, timeout, msg, tag)          # the sweep opens a leg that idled all job
                continue
            if op == "step":
                timer.lap("recv", msg)
                if _fenced(msg, epoch):                       # stale speculation: pass it on, untouched
                    msg["fenced"] = True
                    kw.send(msg)
                    continue
                epoch = max(epoch, int(msg.get("epoch", 0)))
                if "cpos" in msg:                             # settled frontier: free the spent snapshots
                    st.commit(int(msg["cpos"]))
                if head:
                    ids = _ids_tensor(msg["ids"])             # the coordinator sends a plain list
                    h = st.embed(ids)                         # token ids -> h [b, s, hc_mult, dim]
                    in_b = _payload_bytes(h, ids) if signer is not None else None
                else:
                    h, ids, in_b = _recv_hids(msg, signer)    # upcasts fp8; in_b hashes the wire bytes
                start_pos = int(msg["start_pos"])
                timer.lap("pre")
                h = st.forward(h, ids, start_pos)
                timer.sync()
                timer.lap("fwd")
                frame, out_b = _make_step_frame(h, ids, start_pos, signer)  # fp8-packs when V4_FP8_WIRE
                for k in _PASSTHRU:                           # protocol header rides every hop
                    if k in msg:
                        frame[k] = msg[k]
                if signer is not None:
                    signer.observe(in_b, out_b)
                timer.lap("out")
                timer.lap("send", kw.send(frame))
                timer.frame(h.shape[1])
                continue
            if op == "stop":
                kw.stop()
                try:
                    kw.send(msg)                              # tear the ring down front-to-back
                except OSError:
                    pass
                conn.close()
                return
            raise RuntimeError(f"s{stage}: unknown op {op!r}")


def _tail_bringup(srv, timeout):
    """Identify the two tail-inbound streams WITHOUT ever block-recv'ing a silent one. The
    predecessor connection is accepted early (the upstream stage dialed at launch) but stays silent
    until the first job frame flows the ring; the coordinator-return greets immediately. So select
    on the accepted set and only recv a connection that is READABLE — a plain accept-then-recv here
    deadlocks: the tail would block reading the silent predecessor while the coordinator blocks
    waiting for its ret_ok. Returns (ret, pred, queued)."""
    ret = pred = queued = None
    pending = []
    while ret is None or pred is None:
        ready, _, _ = select.select([srv] + pending, [], [])
        if srv in ready:
            conn, _ = srv.accept()
            conn.setsockopt(*NODELAY)
            conn.settimeout(timeout)
            pending.append(conn)
            continue
        conn = next(c for c in pending if c in ready)
        pending.remove(conn)
        try:
            first = recv_msg(conn)
        except _STRAY as e:                                   # public port: drop a scanner/HTTP probe
            print(f"[tail] dropped stray inbound: {type(e).__name__}", flush=True)
            try:
                conn.close()
            except Exception:
                pass
            continue
        if _is_return_hello(first):
            ret = conn
            send_msg(ret, "ret_ok")                           # the coordinator blocks on this ack
        elif _is_pred_hello(first):                           # token-mode predecessor greeting
            pred = conn
        else:                                                 # token-less predecessor: first frame is job data
            pred, queued = conn, first
    return ret, pred, queued


def _tail_logit_rows(st, h, start_pos):
    """The tail's logits, ONE ROW PER CHUNK POSITION, each computed at the SAME GEMM shape greedy
    decode uses. -> [ [b, vocab] fp32, ... ] with one entry per position the reply answers for.

    THIS IS THE LOSSLESSNESS OF SPECULATION, and it is a shape rule, not an approximation:
    `hc_head` -> `norm` -> `ParallelHead` over a chunk runs its fp32 GEMMs at M = b*s where greedy
    decode runs them at M = b, and a different M reassociates the reduction — measured at ~4e-7 in
    step 2. That is invisible right up until an argmax NEAR-TIE flips, at which point the speculated
    stream stops being the greedy stream at random, rarely, and unfalsifiably. With every logits row
    the engine ever computes taken at M = b (greedy decode, the verify rows here), spec == greedy
    holds BY CONSTRUCTION and the CPU selftest's bit-exact bar actually proves something.

    PREFILL (start_pos == 0) is the one place the whole chunk goes in at once, and it must: the
    reference's own `Transformer.forward` collapses and norms the entire prompt before
    `ParallelHead` slices to the last position, so matching the oracle means doing exactly that. It
    is also where `full_logits=False` earns its keep — at V4's shape a 4096-token prefill's full
    logits are [1, 4096, 129280] fp32 = 2 GiB, of which one row is wanted."""
    if start_pos == 0:
        return [st.logits_all(h, full_logits=False)]
    return [st.logits_all(h[:, j:j + 1], full_logits=False) for j in range(h.shape[1])]


def _tail_drafter(st, ckpt_dir, cache):
    """The drafter a dspark-armed job hands its chunks to, built ONCE per tail process.

    TAIL_DRAFTER (the module seam) wins when a test injects one; otherwise a real
    v4_dspark_draft.RingDrafter is built over this tail's own Stage and its `mtp.*` loaded from the
    same checkpoint dir the layers came from. Lazily, on the first dspark job: a greedy ring never
    pays for the MTP stages (3 extra Blocks with their own MoE at V4's shape), and a checkpoint
    without `mtp.*` is a loud failure at the moment someone asks for drafting rather than at boot."""
    if TAIL_DRAFTER is not None:
        return TAIL_DRAFTER
    if cache.get("drafter") is None:
        cache["drafter"] = _dspark().ring_drafter(st, ckpt_dir)
        print(f"[tail] dspark drafter {cache['drafter'].tail}", flush=True)
    return cache["drafter"]


class _RetChannel:
    """The tail's coordinator-return socket, SWAPPABLE under a lock. The serve loop blocks reading its
    predecessor, so it cannot itself accept a reconnecting coordinator; a background thread does, and
    swaps the live socket in here while the loop keeps sending through `send()`. A send to a channel
    whose coordinator has gone is DROPPED, not fatal — the coordinator that died is not reading it, and
    the next reset on a freshly accepted channel re-opens the job. One lock, held across the send, so a
    swap never races a half-written frame onto the wire."""

    def __init__(self, sock):
        self.sock = sock
        self.lock = threading.Lock()

    def send(self, msg):
        with self.lock:
            try:
                return send_msg(self.sock, msg)
            except _LEG_ERRORS:
                return 0                                       # coordinator gone; the loop keeps serving

    def swap(self, sock):
        with self.lock:
            old, self.sock = self.sock, sock
        return old

    def close(self):
        with self.lock:
            sock = self.sock
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass


def _tail_return_reaccept(srv, chan, timeout):
    """Own srv.accept() after bring-up so a coordinator RESTART is survivable. A restarting
    coordinator re-dials the return channel (hello_return); the tail's serve loop is blocked reading
    its predecessor and cannot accept it, and without this the coordinator's connect_ring would hang
    forever waiting on a ret_ok — the ring would be alive but unreachable. This daemon accepts the
    reconnect, SWAPS it into `chan` FIRST (so by the time the coordinator has its ret_ok, the tail is
    already answering on the new socket) and only then acks. The predecessor never re-connects here
    (the upstream stage stays up while the head survives); a stray/scanner frame is dropped."""
    while True:
        try:
            ready, _, _ = select.select([srv], [], [])
            if srv not in ready:
                continue
            conn, _ = srv.accept()
            conn.setsockopt(*NODELAY)
            conn.settimeout(timeout)
        except OSError:
            return                                             # srv closed on stop/teardown: thread exits
        try:
            first = recv_msg(conn)
        except _STRAY as e:
            print(f"[tail] dropped stray reconnect: {type(e).__name__}", flush=True)
            try:
                conn.close()
            except OSError:
                pass
            continue
        if not _is_return_hello(first):
            print("[tail] dropped non-return reconnect on the tail port", flush=True)
            try:
                conn.close()
            except OSError:
                pass
            continue
        old = chan.swap(conn)                                  # visible BEFORE the coordinator proceeds
        try:
            send_msg(conn, "ret_ok")                           # the coordinator's connect_ring blocks on this
        except _LEG_ERRORS:
            pass
        if old is not None:
            try:
                old.close()
            except OSError:
                pass
        print("[tail] coordinator-return re-accepted — ring survived a coordinator restart", flush=True)


def _serve_tail(st, srv, lo, hi, node_key, receipts, timeout, ckpt_dir=None):
    """Tail serve loop. Accepts BOTH inbound streams (predecessor + coordinator-return) on the one
    engine port, classified by the hello_return greeting, then serves: run the block, collapse the
    hyper-connections, sample, and send the token id back on the return channel.

    The return channel is a swappable `_RetChannel`: a background thread re-accepts a reconnecting
    coordinator and swaps its socket in, so the ring survives a coordinator restart instead of dying
    the moment the first coordinator's return socket closes (the tail half of the resilience the head's
    predecessor re-accept provides).

    `ckpt_dir` is set only when the stage was built --dspark: it is where the drafter's `mtp.*` come
    from, and passing it is what makes a drafted ring possible at all."""
    ret, pred, queued = _tail_bringup(srv, timeout)
    print("[tail] predecessor + coord-return connected", flush=True)
    chan = _RetChannel(ret)
    threading.Thread(target=_tail_return_reaccept, args=(srv, chan, timeout), daemon=True,
                     name="v4-tail-reaccept").start()

    signer = None
    temp, gen = 0.0, None                                    # sampling arm, (re)set per job by the reset
    drafter, built = None, {}                                # per-job arm, process-lifetime drafter
    epoch = 0                                                # newest speculation generation seen (fence)
    timer = _timer("[tail]", ("recv", "pre", "fwd", "out", "logits", "draft", "send"),
                   getattr(st, "device", None))
    with torch.no_grad():
        while True:
            msg = queued if queued is not None else recv_msg(pred)
            queued = None
            op = msg.get("op")
            if op == "noop":                                  # keep-warm tick from the predecessor: skip
                continue
            if op == "reset":
                timer.start()                                 # a reset opens a job: time it on its own
                st.reset()
                _set_job_horizon(msg.get("max_pos"))          # v4_ref_slim; absent key => None => safe
                st._spec = bool(msg.get("spec"))              # arm the stage's snapshot/rollback
                st._dspark = bool(msg.get("dspark"))          # arm the taps the drafter consumes
                epoch = 0
                try:
                    drafter = _tail_drafter(st, ckpt_dir, built) if st._dspark else None
                    if drafter is not None:                   # streamed s=1 frames vs one chunk
                        drafter.pipelined = bool(msg.get("pipelined"))
                except Exception as e:  # noqa: BLE001 — any drafter fault is this JOB's, not the ring's
                    # A dspark job on a tail that cannot draft — launched without --dspark, or a
                    # checkpoint carrying no mtp.* — must fail the JOB and leave the ring standing.
                    # Raising here kills the serve loop, and the coordinator only ever reads the TAIL,
                    # so it would see a stall on this job and on every job after it. The reset ack is
                    # the one channel back, and a non-"ok" ack is already a hard failure there.
                    print(f"[tail] dspark unavailable: {type(e).__name__}: {e}", flush=True)
                    chan.send({"ok": False, "error": f"{type(e).__name__}: {e}"})
                    continue
                signer = (ReceiptSigner(node_key, msg.get("swarm_id", "swarm"),
                                        msg.get("job_id", "job"), lo, hi, nonce=msg.get("nonce"))
                          if receipts else None)
                temp = float(msg.get("temp", 0.0))
                gen = (torch.Generator(device="cpu").manual_seed(int(msg["seed"]))
                       if temp > 0 and msg.get("seed") is not None else None)
                chan.send("ok")
                continue
            if op == "receipt":
                timer.report()                                # the job barrier: one timing line per job
                if signer is not None:
                    msg.setdefault("receipts", []).append({"stage": "tail", **signer.finalize()})
                chan.send(msg.get("receipts", []))
                continue
            if op == "step":
                timer.lap("recv", msg)
                if _fenced(msg, epoch):                       # stale speculation: answer, compute nothing
                    send_msg(ret, {"fenced": True, "epoch": int(msg.get("epoch", 0)),
                                   "pos": int(msg["start_pos"])})
                    continue
                epoch = max(epoch, int(msg.get("epoch", 0)))
                if "cpos" in msg:                             # settled frontier: free the spent snapshots
                    st.commit(int(msg["cpos"]))
                h, ids, in_b = _recv_hids(msg, signer)        # upcasts fp8; in_b hashes the wire bytes
                start_pos = int(msg["start_pos"])
                timer.lap("pre")
                h = st.forward(h, ids, start_pos)
                timer.sync()
                timer.lap("fwd")
                if signer is not None:
                    signer.observe(in_b, _payload_bytes(h, ids))
                timer.lap("out")
                rows = _tail_logit_rows(st, h, start_pos)     # hc_head + norm + lm_head, one row per pos
                out = {"token": sample_token(rows[-1][0], temp, gen)}
                if getattr(st, "_spec", False):               # spec: model's greedy token at EVERY chunk pos
                    out["tokens"] = [int(r[0].argmax()) for r in rows]
                if "epoch" in msg:                            # pipelined: the reply names the frame it judges
                    out["epoch"], out["pos"] = int(msg["epoch"]), start_pos
                timer.sync()
                timer.lap("logits")
                if drafter is not None:                       # dspark: draft the next block locally
                    out.update(drafter.on_chunk(msg, st, out) or {})
                timer.sync()
                timer.lap("draft")
                timer.lap("send", chan.send(out))
                timer.frame(h.shape[1])
                continue
            if op == "stop":
                chan.send({"token": None})
                pred.close()
                chan.close()
                return
            raise RuntimeError(f"tail: unknown op {op!r}")


def _dial_return(addr, timeout):
    """Dial an intra-box return channel to the box TAIL and greet it exactly as the coordinator
    would — hello_return, then block on its ret_ok. Retries for the full dial window while the tail
    is still coming up (it loads its weights after the ingress binds). Returns the socket the tail
    replies on (reset ack, tokens, receipts)."""
    host, port = addr.rsplit(":", 1)
    s = _dial(host, port, timeout)
    s.settimeout(timeout)
    send_msg(s, {"op": "hello_return", "token": SWARM_TOKEN} if SWARM_TOKEN is not None
             else {"op": "hello_return"})
    recv_msg(s)                                                # the box tail acks (ret_ok)
    return s


def _serve_relay_ingress(st, stage, srv, nxt_sock, nxt, head, lo, hi, node_key, receipts, timeout,
                         ret_relay):
    """Box-ingress serve loop for a MULTI-GPU TAIL box. The sidecar delivers the coordinator-return
    tunnel to this box's ENG_IN (same one -inbound as the predecessor's job frames), but the token is
    produced on the box's LAST-GPU stage. So this stage:
      * runs its own layers and forwards frames down the intra-box loopback chain — byte-identical to
        a plain forward stage (it appends its receipt too),
      * bridges the return: it dials the box tail's return over loopback (`ret_relay`) and pumps every
        frame the tail sends (reset ack, tokens, the receipt list) straight out to the coordinator.
    The box tail's serve loop is UNCHANGED — it just gets its coordinator-return dialed from here
    instead of the WAN. Reached only when a G>1 box holds the global tail."""
    ret_up, pred, queued = _tail_bringup(srv, timeout)         # coordinator-return + predecessor at ENG_IN
    ret_down = _dial_return(ret_relay, timeout)                # our return leg to the box tail (loopback)

    def _pump():
        try:
            while True:
                send_msg(ret_up, recv_msg(ret_down))           # box tail -> coordinator, frame for frame
        except OSError:                                        # ConnectionError on stop / ring torn down
            pass
    threading.Thread(target=_pump, daemon=True).start()
    print(f"[s{stage}] relay ingress: coord-return <-> box tail {ret_relay}", flush=True)
    _forward_loop(st, stage, nxt_sock, nxt, head, lo, hi, node_key, receipts, pred, queued, timeout)


# ── coordinator: drive greedy generation over the ring ─────────────────────────────────────────────

def _hostport(s):
    h, _, p = s.rpartition(":")
    return h or "127.0.0.1", int(p)


def connect_ring(head, tail, timeout=600.0, token=None, retry_s=300):
    """Dial the head engine (pipe) + the return tunnel to the tail (ret), with retries — the daemon
    starts the coordinator while other stages may still be pulling weights. Mirrors
    shard.coordinate.connect_ring: hello_return classifies the tail-side stream, the token (env-only)
    rides both greetings when set."""
    deadline = time.time() + retry_s
    last = None
    while time.time() < deadline:
        pipe = ret = None
        try:
            pipe = socket.create_connection(_hostport(head), timeout=timeout)
            pipe.setsockopt(*NODELAY)
            ret = socket.create_connection(_hostport(tail), timeout=timeout)
            ret.setsockopt(*NODELAY)
            ret.settimeout(timeout)
            if token:
                send_msg(pipe, {"op": "hello_pred", "token": token})
                send_msg(ret, {"op": "hello_return", "token": token})
            else:
                send_msg(ret, {"op": "hello_return"})
            recv_msg(ret)                                     # tail acks the return channel (ret_ok)
            return pipe, ret
        except Exception as e:                                # noqa: BLE001 — any dial fault = retry
            last = e
            for s in (pipe, ret):
                if s is not None:
                    try:
                        s.close()
                    except OSError:
                        pass
            time.sleep(min(2.0, max(0.25, deadline - time.time())))
    raise ConnectionError(f"v4 ring not reachable after {retry_s}s: {type(last).__name__}: {last}")


def _sweep_receipts(pipe, ret, layer_count, nonce):
    """Sweep the ring once at job end and verify coverage — fail-closed (C10). wire_receipt strips the
    post-sign `stage` debug tag, which is in the signed preimage of nothing and would break the
    signature if a verifier saw it. Returns (receipts, receipts_ok)."""
    send_msg(pipe, {"op": "receipt", "receipts": []})
    recs = recv_msg(ret) or []
    if not recs or layer_count is None:
        return recs, None
    wired = [wire_receipt(r) for r in recs]
    try:
        verify_coverage(wired, int(layer_count), expected_nonce=nonce, check_chain=True)
        return recs, True
    except ReceiptError:
        return recs, False


def coordinate(pipe, ret, prompt_ids, max_new, *, eos_ids=(), nonce=None, swarm_id="swarm",
               job_id="job", layer_count=None, receipts=False, temp=0.0, seed=0, timeout=600.0,
               on_token=None):
    """Greedy sequential decode over the fire-forward ring. Weightless: the head embeds, the tail
    samples, and this loop only threads token ids and the settlement nonce over the sockets.

    Returns {ok, tokens, prompt_tokens, receipts, receipts_ok}. `receipts` sweeps the ring once at
    the end and verifies coverage against `layer_count` and the job nonce (fail-closed, C10)."""
    ret.settimeout(timeout)
    send_msg(pipe, {"op": "reset", "swarm_id": swarm_id, "job_id": job_id, "nonce": nonce,
                    "temp": float(temp), "seed": int(seed),
                    "max_pos": _job_max_pos(prompt_ids, max_new)})   # greedy: no draft overshoot
    ack = recv_msg(ret)                                       # the tail acks the whole ring is reset
    if not (ack == "ok" or (isinstance(ack, dict) and ack.get("ok"))):
        raise RuntimeError(f"v4 ring reset not acked: {ack!r}")

    eos = set(eos_ids)
    ids = list(prompt_ids)
    pos = 0
    toks = []
    for _ in range(max_new):
        send_msg(pipe, {"op": "step", "ids": [ids], "start_pos": pos})
        pos += len(ids)
        rep = recv_msg(ret)
        tid = int(rep["token"]) if isinstance(rep, dict) else int(rep)
        toks.append(tid)
        if on_token is not None:
            on_token(tid)
        if tid in eos:
            break
        ids = [tid]

    recs, receipts_ok = [], None
    if receipts:
        recs, receipts_ok = _sweep_receipts(pipe, ret, layer_count, nonce)
    return {"ok": True, "tokens": toks, "prompt_tokens": len(prompt_ids),
            "receipts": recs, "receipts_ok": receipts_ok}


class _RepeatDrafter:
    """The zero-dependency fallback proposer: repeat the last committed token K times. It is a bad
    drafter (accepts only on a genuine repetition) but a VALID one — spec-decode is lossless whatever
    the proposal is, so this exercises the accept/commit/rewind loop without pulling in ngram_draft."""

    def propose(self, seq, k):
        return [seq[-1] if seq else 0] * k


def _drafter_propose(drafter, ng):
    """Normalize a proposer into `propose(ids, K) -> [K token ids]`. Accepts an object with .propose
    (phase0/ngram_draft.NgramDrafter), a bare callable, or None (build the n-gram drafter, falling
    back to _RepeatDrafter where phase0/ngram_draft.py is not deployed)."""
    if drafter is None:
        try:
            from ngram_draft import NgramDrafter
            drafter = NgramDrafter(ng=ng, margin=0)
        except ImportError:
            drafter = _RepeatDrafter()
    return drafter.propose if hasattr(drafter, "propose") else drafter


def coordinate_spec(pipe, ret, prompt_ids, max_new, *, eos_ids=(), nonce=None, swarm_id="swarm",
                    job_id="job", layer_count=None, receipts=False, timeout=600.0, on_token=None,
                    K=4, ng=3, drafter=None):
    """SPECULATIVE decode over the fire-forward ring — the g-lever that beats the transport ceiling.
    Each round proposes K draft tokens, sends the (cur + drafts) chunk through the ring in ONE
    traversal, the tail returns the model's greedy token at EVERY chunk position, and we commit the
    accepted draft prefix + one correction: byte-identical to greedy decode (lossless), at up to K+1
    committed tokens per WAN round-trip.

    THE ROLLBACK IS THE STAGE'S. A rejected draft leaves every stage's window ring and compressor
    accumulators ahead of the committed sequence; re-opening the next chunk at the last committed
    position is what makes `Stage._seek` restore its pre-chunk snapshot and replay the accepted
    prefix. This loop's job is to compute that position correctly — `pos = len(ids) - 1`, never the
    end of the rejected draft.

    Returns coordinate()'s dict plus spec stats {rounds, generated, g, accept_hist}."""
    plan_verify_round = _dspark().plan_verify_round        # ONE accept rule, shared with the tail
    propose = _drafter_propose(drafter, ng)
    ret.settimeout(timeout)
    send_msg(pipe, {"op": "reset", "swarm_id": swarm_id, "job_id": job_id, "nonce": nonce,
                    "temp": 0.0, "seed": 0, "spec": True,
                    "max_pos": _job_max_pos(prompt_ids, max_new, K + 1)})   # + the draft chunk
    ack = recv_msg(ret)
    if not (ack == "ok" or (isinstance(ack, dict) and ack.get("ok"))):
        raise RuntimeError(f"v4 spec ring reset not acked: {ack!r}")

    eos = set(eos_ids)
    ids = list(prompt_ids)                                    # full committed sequence (prompt + gen)
    toks = []                                                 # generated tokens only
    rounds, accepted_total = 0, 0
    hist = {}

    # prefill: forward the whole prompt as one chunk, take the first generated token
    send_msg(pipe, {"op": "step", "ids": [ids], "start_pos": 0})
    rep = recv_msg(ret)
    pos = len(ids)                                            # absolute position of `cur` (not yet fed)
    cur = int(rep["token"] if isinstance(rep, dict) else rep)
    ids.append(cur)
    toks.append(cur)
    if on_token is not None:
        on_token(cur)

    while len(toks) < max_new and cur not in eos:
        drafts = propose(ids, K)                              # K proposed continuations of `cur`
        rounds += 1
        send_msg(pipe, {"op": "step", "ids": [[cur] + drafts], "start_pos": pos})
        r = recv_msg(ret)["tokens"]                           # model greedy token AFTER each chunk pos (K+1)
        n, committed = plan_verify_round(drafts, r)           # longest matching prefix + the correction
        accepted_total += n
        hist[n] = hist.get(n, 0) + 1
        stop = False
        for t in committed:
            toks.append(int(t))
            ids.append(int(t))
            if on_token is not None:
                on_token(int(t))
            if int(t) in eos or len(toks) >= max_new:
                stop = True
                break
        cur = ids[-1]
        pos = len(ids) - 1                                    # `cur` sits here, to be fed next round
        if stop:
            break

    recs, receipts_ok = [], None
    if receipts:
        recs, receipts_ok = _sweep_receipts(pipe, ret, layer_count, nonce)
    gen = len(toks)
    return {"ok": True, "tokens": toks, "prompt_tokens": len(prompt_ids),
            "receipts": recs, "receipts_ok": receipts_ok,
            "rounds": rounds, "generated": gen, "accepted": accepted_total,
            "g": (gen / rounds) if rounds else float(gen), "accept_hist": hist}


# CONFIDENCE-GATED ADAPTIVE SEND-LENGTH (coordinate_dspark). The DSpark drafter emits a per-position
# confidence score with every block (`conf`, RingDrafter ships it). On a round the drafter itself
# expects to lose early, sending the whole block spends chunk positions the ring will compute and
# throw away; on a round it expects a long run, sending the whole block banks the most tokens per
# traversal. So this gate TRUNCATES the block the coordinator OFFERS to the confidence-predicted
# survival prefix. It is LOSSLESS by construction and not by luck: the tail verifies exactly the
# chunk it receives and both ends run plan_verify_round over the SAME sent drafts, so the committed
# stream is byte-identical whatever the send-length — the gate can only ever change HOW MANY tokens a
# round banks, never WHICH. That makes it a pure throughput knob: a mis-set threshold costs
# acceptance, never a wrong token.
#
# THE THRESHOLD NEEDS CALIBRATION AND THE GATE IS OFF UNTIL IT HAS IT. Measured on the CPU reference,
# `conf` is a RAW score near 0 that goes NEGATIVE — it is the confidence head's logit, not a
# probability — so there is no universal cutoff and a cumulative-product-of-probabilities rule would
# be meaningless on it. The rule here is therefore a per-position raw-score floor (keep the leading
# run whose conf stays >= thresh, stop at the first below — a later low score cannot matter once an
# earlier position is predicted to reject), and the floor is what an operator calibrates to the REAL
# model's conf distribution. TODO(calibrate): on a live GPU ring run coordinate_dspark UNGATED with a
# `conf_probe` and accumulate (conf_i, accepted?) pairs — accepted positions are 0..n-1, the rejected
# one is n — then set V4_DSPARK_CONF_THRESH to the conf below which P(accept) falls under ~0.5 (or
# wherever the round-cost/accept trade crosses). Research puts the win at ~+15-25%; that number is a
# measurement this gate makes reachable, not a promise it keeps on its own.
V4_DSPARK_CONF_GATE = os.environ.get("V4_DSPARK_CONF_GATE", "0") not in ("", "0")   # OFF until calibrated
V4_DSPARK_CONF_THRESH = float(os.environ.get("V4_DSPARK_CONF_THRESH", "0") or 0)    # raw-score floor
V4_DSPARK_CONF_MIN = int(os.environ.get("V4_DSPARK_CONF_MIN", "1") or 1)            # always offer >= this many


def _conf_send_len(confs, thresh, min_send):
    """How many of a drafted block's tokens to SEND this round, from its per-position confidence.

    `confs[i]` is the drafter's raw confidence for draft `i` (see the note above — a logit, can be
    negative). Keep the longest leading prefix whose conf stays >= `thresh`; stop at the first draft
    below it (a survival prefix — once a position is predicted to reject, every later one is moot).
    Floored at `min_send` (0 lets a low-confidence round fall back to a bare greedy step) and capped
    at the block length. An empty block (round 1) sends 0. Lossless whatever it returns."""
    if not confs:
        return 0
    k = 0
    for c in confs:
        if float(c) < thresh:
            break
        k += 1
    return min(len(confs), max(min_send, k))


def coordinate_dspark(pipe, ret, prompt_ids, max_new, *, eos_ids=(), nonce=None, swarm_id="swarm",
                      job_id="job", layer_count=None, receipts=False, timeout=600.0, on_token=None,
                      conf_gate=None, conf_thresh=None, conf_min=None, conf_probe=None):
    """DSPARK speculative decode over the fire-forward ring — the headline drafted path.

    Same propose->verify->accept->rollback contract coordinate_spec proves, with the proposer moved
    to where its input already lives. V4 ships a TRAINED speculator (the MTP stages) whose input is
    `main_hidden`, 24 KiB per position at V4's shape, held only by the tail — so unlike K3's DSpark,
    nothing is streamed down the ring to draft with. The tail drafts LOCALLY and returns the block
    with the token, which is why this coordinator takes no drafter argument at all: it is weightless
    even by speculation's standards, and a drafted round costs the wire a few dozen extra bytes.

    THE ROUND. Reset arms `spec` (the per-position replies + the stage rollback) AND `dspark` (the
    taps + the drafter). Prefill returns the first token and primes the mtp window. Round 1 is the
    BARE `[cur]` chunk — the drafter deliberately produces no block at prefill (v4_dspark_draft's
    FIRST ROUND paragraph) — and every round after sends `[cur] + drafts` where `drafts` came back
    with the previous round's reply.

    BOTH ENDS RUN THE ACCEPT RULE, ON THE SAME INPUTS, AND MUST AGREE. The tail has to, to know how
    far to advance the drafter before it can answer; this loop has to, to know what to emit. So the
    reply carries the tail's own `n` and it is ASSERTED against ours: a mismatch means the two halves
    of a lossless protocol have diverged, and the drafter is now conditioning on a history the ring
    never took. That is a job-killing bug, not a degradation, and it fails loudly here rather than
    surfacing as quietly worse acceptance forever after.

    CONFIDENCE GATE (see the note above `_conf_send_len`). When armed, the block the coordinator
    OFFERS each round is truncated to the drafter's own confidence-predicted survival prefix — fewer
    positions on a round it expects to lose early, the full block on a round it expects a long run.
    Lossless (the tail verifies only what it receives); off unless `conf_gate` / V4_DSPARK_CONF_GATE
    says so, because the raw-score threshold has to be calibrated to the real model first. `conf_probe`,
    if given, is called `conf_probe(confs, n)` each drafted round with the FULL block's confidence and
    the number accepted — the hook a calibration run records the conf-vs-accept curve through.

    Returns coordinate()'s dict plus spec stats {rounds, drafted, generated, accepted, g,
    accept_hist, sent, send_hist} — `drafted` being the rounds that carried a real block, `sent` the
    total draft tokens actually offered (< the drafted total when the gate trims), which is how a
    selftest tells "the drafter proposed nothing" apart from "the drafter proposed and was rejected"
    and how a bench reads the gate's effect."""
    plan_verify_round = _dspark().plan_verify_round        # ONE accept rule, shared with the tail
    gate = V4_DSPARK_CONF_GATE if conf_gate is None else bool(conf_gate)
    thresh = V4_DSPARK_CONF_THRESH if conf_thresh is None else float(conf_thresh)
    min_send = V4_DSPARK_CONF_MIN if conf_min is None else int(conf_min)
    v4_levers.note("V4_DSPARK_CONF_GATE", gate)          # what the loop RAN with, for the audit
    v4_levers.note("V4_PIPELINED_SPEC", False)
    # The serial round sends a block and waits for it; there is no hint to skip a draft with, so lazy
    # drafting is not merely off here, it is unreachable. Recording False is what makes an operator
    # who set V4_LAZY_DRAFT=1 on a serial ring see a MISMATCH instead of "no-job-yet" forever.
    v4_levers.note("V4_LAZY_DRAFT", False)
    ret.settimeout(timeout)
    send_msg(pipe, {"op": "reset", "swarm_id": swarm_id, "job_id": job_id, "nonce": nonce,
                    "temp": 0.0, "seed": 0, "spec": True, "dspark": True,
                    # the tail's MTP block size is not knowable here — the fat margin is the answer
                    "max_pos": _job_max_pos(prompt_ids, max_new, _SPEC_POS_MARGIN)})
    ack = recv_msg(ret)
    if not (ack == "ok" or (isinstance(ack, dict) and ack.get("ok"))):
        raise RuntimeError(f"v4 dspark ring reset not acked: {ack!r}")

    eos = set(eos_ids)
    ids = list(prompt_ids)                                    # full committed sequence (prompt + gen)
    toks = []                                                 # generated tokens only
    rounds, accepted_total = 0, 0
    hist = {}

    # prefill: the whole prompt as one chunk. The tail also builds the mtp window from its taps here.
    send_msg(pipe, {"op": "step", "ids": [ids], "start_pos": 0})
    rep = recv_msg(ret)
    pos = len(ids)                                            # absolute position of `cur` (not yet fed)
    cur = int(rep["token"] if isinstance(rep, dict) else rep)
    ids.append(cur)
    toks.append(cur)
    if on_token is not None:
        on_token(cur)

    block, confs = [], []                                     # round 1 is the bare [cur] chunk (no block)
    d2s = []                                                  # the block's runner-up column (V4_DRAFT_TOP2)
    rescue_by_depth = {}                                      # cancel depth -> [runner-up hits, cancels]
    drafted, sent = 0, 0                                       # rounds that carried a block; drafts offered
    send_hist = {}
    while len(toks) < max_new and cur not in eos:
        rounds += 1
        # Offer the whole block, or — armed and calibrated — its confidence-predicted survival prefix.
        drafts = block[:_conf_send_len(confs, thresh, min_send)] if gate else block
        drafted += bool(drafts)
        sent += len(drafts)
        send_hist[len(drafts)] = send_hist.get(len(drafts), 0) + 1
        send_msg(pipe, {"op": "step", "ids": [[cur] + drafts], "start_pos": pos})
        rep = recv_msg(ret)
        if "n" not in rep:
            raise RuntimeError(
                "v4 dspark: the tail's reply carries no accept length, so nothing is drafting on it "
                "— launch the tail stage with --dspark so it builds the MTP speculator")
        n, committed = plan_verify_round(drafts, rep["tokens"])
        if int(rep["n"]) != n:
            raise RuntimeError(
                f"v4 dspark: the tail accepted {rep['n']} of round {rounds}'s {len(drafts)} drafts "
                f"and this coordinator accepted {n}, off the same drafts and the same replies. The "
                f"two ends of a lossless round have diverged, so the drafter is advancing over a "
                f"history the ring is not taking — one accept rule, and it is plan_verify_round.")
        accepted_total += n
        hist[n] = hist.get(n, 0) + 1
        if n < len(drafts) and n < len(d2s):                  # THE TREE GATE: a cancel with a d2 to
            slot = rescue_by_depth.setdefault(n + 1, [0, 0])  # judge — would the drafter's runner-up
            slot[1] += 1                                      # at the missed slot have rescued it?
            slot[0] += int(d2s[n] == int(rep["tokens"][n]))   # beta (docs/V4_TREE_VERDICT.md)
        if conf_probe is not None and block:                  # the FULL block's conf vs what accepted
            conf_probe(list(confs), n)
        stop = False
        for t in committed:
            toks.append(int(t))
            ids.append(int(t))
            if on_token is not None:
                on_token(int(t))
            if int(t) in eos or len(toks) >= max_new:
                stop = True
                break
        cur = ids[-1]
        pos = len(ids) - 1                                    # `cur` sits here — the stage rewinds to it
        block = [int(t) for t in (rep.get("draft") or [])]    # the block drafted off what we committed
        confs = [float(c) for c in (rep.get("conf") or [])]   # its per-position confidence (gate input)
        d2s = [int(t) for t in (rep.get("d2") or [])]         # its runner-up column, same alignment
        if stop:
            break

    recs, receipts_ok = [], None
    if receipts:
        recs, receipts_ok = _sweep_receipts(pipe, ret, layer_count, nonce)
    gen = len(toks)
    return {"ok": True, "tokens": toks, "prompt_tokens": len(prompt_ids),
            "receipts": recs, "receipts_ok": receipts_ok,
            "rounds": rounds, "drafted": drafted, "generated": gen, "accepted": accepted_total,
            "g": (gen / rounds) if rounds else float(gen), "accept_hist": hist,
            "rescue_by_depth": {d: tuple(v) for d, v in sorted(rescue_by_depth.items())},
            "sent": sent, "send_hist": send_hist}


# ── PIPELINED speculation: stream the block as s=1 frames instead of verifying it as one chunk ──────
# The design, the throughput model and the correctness gate are docs/V4_PIPELINED_SPEC.md. In one
# paragraph: coordinate_dspark sends `[cur] + drafts` as ONE frame and waits, so exactly one chunk is
# ever in flight (5 of 6 stages idle) and every stage replays the chunk position by position
# (v4_stage.forward's start_pos>0 loop) — which is why the drafted path merely ties greedy. Streaming
# the same block as separate s=1 frames back-to-back deletes both costs at once: stage k works on
# token i while stage k-1 works on token i+1, and no frame is ever longer than one position.
#
# OPT-IN, and the serial path is untouched: V4_PIPELINED_SPEC (env) or a job's `pipelined` flag.
V4_PIPELINED_SPEC = bool(int(os.environ.get("V4_PIPELINED_SPEC", "0") or 0))
# The in-flight cap, and it is the SAME number as the stage's rollback ring (v4_stage's
# V4_SPEC_DEPTH): a rejection W frames downstream has to rewind across all W, and a stage refuses to
# rewind past its oldest live checkpoint. Streaming more than the stage can undo is the one way this
# path can fail loudly, so the two defaults are read from one env var.
V4_SPEC_DEPTH = int(os.environ.get("V4_SPEC_DEPTH", "16") or 16)
# LAZY DRAFTING. A pipelined round consumes ONE block and the tail drafts one on every committed
# frame, so it pays for `g` of them and throws `g - 1` away — measured on the 6-stage EU ring, that
# discarded drafting is ~28 of the tail's 37.4 ms per frame and the tail is the ring's binding stage.
# Armed, the coordinator hints each frame with the token it has already streamed at the NEXT position
# and the tail skips the block wherever that hint can only mean "accepted, and not the round's last"
# — see `_hints` for the proof, and `RingDrafter._on_chunk_pipelined` for what it does instead.
# Opt-in (V4_LAZY_DRAFT / the `lazy=` argument) and default OFF: an unhinted frame always drafts, so
# the flag off is the eager path frame for frame. It is read by the COORDINATOR — the tail reacts to
# the hints on the frames, not to its own environment — and it rides ENG_ENV anyway so that an
# operator exporting it on the launcher box gets it wherever v4_pipe runs, stage or coordinator.
V4_LAZY_DRAFT = bool(int(os.environ.get("V4_LAZY_DRAFT", "0") or 0))
# THE REFILL FLOOR. The shipped round consumes a fresh block only when the pipeline has DRAINED to
# the frontier (`horizon == c`), so in-flight saws 6,5,4,3,2,6 and the ring idles between teeth —
# while the tail is already drafting a fresh block on EVERY committed frame and the coordinator
# throws every mid-run one away. The floor is the in-flight level at or below which a reply's block
# is consumed instead of discarded: 1 is the shipped drain-only behaviour, frame for frame;
# `dspark_block_size` (5) pins the pipe at block+1 by refilling on every reply.
#
# WHY THIS IS A LEVER NOW when docs/V4_MULTIBLOCK_VERDICT.md §4 measured the rolling top-up at +3%
# and declined to build it: that price was a property of the OLD drafter (q5 = 0.680, g = 4.92).
# The lever's value is almost entirely a function of q at the block's DEEPEST index — the one new
# position a mid-run block adds is always its deepest draft — and the current ring's drafter
# measures far better there (g = 11.13 on the 07-31 six-stage ring). Re-priced against that ring,
# floor=5 models ~+21% where the old drafter modelled +3% (phase0/v4_ngram_econ.py). The family is
# NON-MONOTONE — floor=2 prices BELOW floor=1, because the first thing a deeper floor changes is the
# trial-depth mix (deepest-draft-only) and the fill gain only pays past that — so the operating
# point is a measurement, not a monotone knob to max out.
#
# THE HONEST PRICE, stated where the knob is read. A mid-run block's deep drafts condition (through
# the Markov head) on ITS OWN block prefix, not on the tokens already in flight at those positions;
# when the two disagree, the topped-up frames answer a history the ring is not taking and are
# wasted, and every one of them also enlarges the discarded future behind a rejection. The loop
# therefore scores topped-up frames apart from drain-refill frames (`topup_accept_by_depth` vs
# `accept_by_depth`) and counts the overlap disagreement directly (`topup_disagree`), so the first
# real ring run measures the assumption the +21% rests on instead of inheriting it. It also defeats
# lazy drafting in proportion: a reply at or below the floor consumes a block, so the hint proof
# licenses fewer skips the higher the floor sits (none at all from floor 5), and the tail pays the
# drafting the refills consume — `drafts_issued` against `frames` is that bill, per run.
V4_REFILL_FLOOR = int(os.environ.get("V4_REFILL_FLOOR", "1") or 1)


class _PipeSpecState:
    """The only state the sender thread and the receiver loop share, and the lock over it.

    `epoch` is written by the receiver (on a cancel) and read by the sender (to drop frames of a
    cancelled speculation); `outstanding` is appended by the sender at the moment a frame really goes
    on the wire and popped by the receiver as replies arrive, so it is the exact FIFO of frames the
    ring owes an answer for. `pending` counts frames handed to the sender and not yet accounted for by
    EITHER path (replied, or dropped unsent), which is what tells the receiver when the ring has
    nothing left to say — a count the receiver alone could not keep, because it does not know which
    queued frames the fence swallowed."""

    def __init__(self):
        self.lock = threading.Lock()
        self.epoch = 0
        self.outstanding = collections.deque()
        self.pending = 0
        self.unsent = 0                                       # frames the fence dropped before the wire


class _FrameSender(threading.Thread):
    """Push step frames at the head without ever blocking the receiver.

    The sender is a thread and not just a send() call in the loop for one reason: the accept loop must
    never be parked in a socket write while a reply is waiting to be read. It is also where the
    SENDER-SIDE half of the epoch fence lives — a cancel fires while a whole block may still be queued
    here, and those frames must not reach the wire. The check and the `outstanding` append happen under
    the same lock, so a frame is recorded as owed a reply exactly when it is about to be sent: dropping
    one can never leave the receiver waiting for an answer that will never come."""

    def __init__(self, pipe, state):
        super().__init__(daemon=True, name="v4-pipe-sender")
        self.pipe = pipe
        self.st = state
        self.q = queue.Queue()
        self.err = None
        self._stopped = False

    def put(self, frame, epoch):
        with self.st.lock:
            self.st.pending += 1
        self.q.put((frame, epoch))

    def stop(self):
        """Drain and join. The RECEIVER calls this the moment it stops generating, not just at the
        end, and that ordering is load-bearing: `pending` is incremented by the caller but decremented
        by whichever of the two paths (a reply, or the fence swallowing the frame unsent) gets there,
        so a receiver that checked "is anything still owed" while the sender had a fenced frame in hand
        would read a stale non-zero and park in a recv nothing will ever answer. Joining first settles
        the count. Idempotent — the end-of-job call is a no-op after the stop-generating one."""
        if self._stopped:
            return
        self._stopped = True
        self.q.put(None)
        self.join(timeout=10)

    def run(self):
        while True:
            item = self.q.get()
            if item is None:
                return
            frame, ep = item
            with self.st.lock:
                if ep != self.st.epoch:                       # fenced: a cancelled block never ships
                    self.st.pending -= 1
                    self.st.unsent += 1
                    continue
                self.st.outstanding.append((int(frame["start_pos"]), ep))
            try:
                send_msg(self.pipe, frame)
            except Exception as e:                            # noqa: BLE001 — the receiver reports it
                self.err = e
                return


def coordinate_dspark_pipelined(pipe, ret, prompt_ids, max_new, *, eos_ids=(), nonce=None,
                                swarm_id="swarm", job_id="job", layer_count=None, receipts=False,
                                timeout=600.0, on_token=None, depth=None, lazy=None, floor=None):
    """DSPARK speculative decode, PIPELINED — the same lossless round, streamed instead of chunked.

    Same contract as coordinate_dspark (same reset, same drafter, same accept rule, same emitted
    stream) and a different shape on the wire. Instead of one `[cur] + drafts` frame per round:

        frame(cur, pos=c)            -> reply {m_c, block}   m_c is the model's token at c+1
        frame(m_c, pos=c+1)  ┐
        frame(d1,  pos=c+2)  │  streamed back-to-back, never waiting; <= W in flight
        ...                  ┘       -> one reply per frame, in position order

    so the D stages fill (stage k runs token i while stage k-1 runs token i+1) and every frame is
    genuinely s=1 — no stage ever runs `forward`'s per-position replay loop.

    LOSSLESSNESS, and why it is not an argument about speculation at all: every token this emits is
    `m_p`, the tail's own greedy token from a frame at position `p` whose entire history is committed,
    computed at M=b through ParallelHead exactly as `coordinate()` computes it (_tail_logit_rows).
    Accepting a draft does not commit the DRAFT, it commits `m_p` — which happens to equal it. So on
    the accepted path the ring sees precisely the frame sequence greedy decode would send it, plus
    rejected frames that Stage._seek undoes before the next accepted frame is processed. Same frames,
    same shapes, same order => same tokens, bit for bit.

    THE CANCEL. When the reply for `p` disagrees with the token already streamed at `p+1`, everything
    behind it is a discarded future: commit `m_p` at `p+1`, bump the epoch (see the epoch fence above),
    and stream the correction frame AT `p+1` — whose `start_pos` IS the rewind command, because
    Stage.forward seeks before it does anything else and the checkpoint ring is W deep. No cancel op
    crosses the wire and no stage needs one.

    AND THE FRESH BLOCK COMES WITH THE REJECTION. The reply that reveals the rejection is the reply of
    a COMMITTED frame, so the tail advanced its drafter over `m_p` — the correction token — and drafted
    a block off that correct history before answering. The coordinator can therefore stream
    `[correction] + block` immediately: the pipeline refills in the same breath it drains, rather than
    idling a full traversal waiting for a new block to be proposed.

    BOTH ENDS STILL RUN THE ACCEPT RULE AND MUST AGREE, as in the serial path: the tail's `acc` (its
    own judgment that this frame's position is committed, from its incremental frontier) is asserted
    against this loop's. A divergence means the drafter is conditioning on a history the ring never
    took, and it fails the job loudly instead of quietly rotting acceptance.

    THE TAIL DRAFTS ONCE PER COMMITTED FRAME AND THIS LOOP CONSUMES ONE BLOCK PER ROUND, so the
    eager path pays for `drafts_issued` blocks and streams `drafted` of them — the ratio is the
    drafting the round threw away, and on the real ring the discarded share is the tail's whole
    margin over the next slowest stage. `lazy` (V4_LAZY_DRAFT) closes it; see `_hints`.

    Returns coordinate()'s dict plus {frames, drafted, drafts_issued, lazy, floor, generated,
    accepted, cancels, cycles, g, accept_hist, accept_by_depth, topups, topup_frames,
    topup_accept_by_depth, topup_agree, topup_disagree, max_inflight, mean_inflight,
    inflight_time_avg, decode_wall_s, frames_per_token, block_len, stale_replies,
    unsent_frames} — `max_inflight` being the pipeline depth actually achieved, i.e. whether the
    thing pipelined at all, and `drafts_issued` what the tail was billed for to achieve it.

    `mean_inflight` is EVENT-weighted (one sample per judged reply) and overstates the fill a ring
    actually held — a level is weighted by how many replies it survived, not by how long it lasted,
    and on the live ring that bias measured ~6.5%. `inflight_time_avg` is the same span integrated
    over the wall clock between every change of frontier or horizon, from the first streamed frame
    to the commit that stops the run, over exactly `decode_wall_s` seconds — it is the number
    Little's law wants (fill = frame rate x time-in-span), so it is what the pricing model takes.
    `frames_per_token` is frames streamed per token emitted (the waste multiple, >= 1.0), and
    `block_len` the widest block a reply actually carried — the trained width unless the tail was
    built with V4_DSPARK_BLOCK.

    `accept_by_depth` maps a draft's DEPTH — how far ahead of the frontier it was streamed, 1 for a
    block's first draft, B for its last — to `(hits, trials)`, counting only frames actually judged.
    It exists because `g` alone cannot tell a drafter that is uniformly mediocre from one that is
    sharp up close and blind far out, and only the second shape is changed by feeding the pipeline
    differently; it is also the measured curve the pricing model takes in place of a two-point fit
    (phase0/v4_ngram_econ.py). Truth/correction frames are depth 0 and are not predictions, so they
    are never scored.

    `floor` (V4_REFILL_FLOOR — see the note at the constant) is the in-flight level at or below
    which a reply's fresh block is consumed. Frames a mid-run refill streams are scored in
    `topup_accept_by_depth`, apart from `accept_by_depth`, because they are a different bet: a
    mid-run block's deep drafts condition on its own prefix rather than on the frames already in
    flight, and `topup_agree`/`topup_disagree` count exactly how often those two histories differ
    where they overlap. At floor=1 every topup counter is structurally zero and the round is the
    shipped one, frame for frame, hints included."""
    W = int(depth or V4_SPEC_DEPTH)
    F = int(floor if floor is not None else V4_REFILL_FLOOR)
    if F < 1:
        raise ValueError(f"v4 pipelined dspark: refill floor {F} — the pipe refills at or below the "
                         f"floor, so it must be at least 1 (1 = the shipped drain-only behaviour)")
    lazy = V4_LAZY_DRAFT if lazy is None else bool(lazy)
    # A coordinator lever has no rebound method for an audit to inspect -- `lazy` is an ARGUMENT to
    # this loop -- so the loop states what it ran with and v4_levers reports THAT, not a second read
    # of the env it already believed. This is the lever that was set on six stages for a night.
    v4_levers.note("V4_LAZY_DRAFT", lazy)
    v4_levers.note("V4_SPEC_DEPTH", W)
    v4_levers.note("V4_PIPELINED_SPEC", True)
    v4_levers.note("V4_REFILL_FLOOR", F)
    ret.settimeout(timeout)
    send_msg(pipe, {"op": "reset", "swarm_id": swarm_id, "job_id": job_id, "nonce": nonce,
                    "temp": 0.0, "seed": 0, "spec": True, "dspark": True, "pipelined": True})
    ack = recv_msg(ret)
    if not (ack == "ok" or (isinstance(ack, dict) and ack.get("ok"))):
        raise RuntimeError(f"v4 dspark ring reset not acked: {ack!r}")

    eos = set(eos_ids)
    ids = list(prompt_ids)                                    # full committed sequence (prompt + gen)
    toks = []                                                 # generated tokens only

    # PREFILL is still one multi-token frame — it is the only place the whole chunk goes in at once
    # (the reference's own collapse-then-slice), and it is where the tail builds its mtp window.
    send_msg(pipe, {"op": "step", "ids": [ids], "start_pos": 0, "epoch": 0})
    rep = recv_msg(ret)
    if not isinstance(rep, dict) or "acc" not in rep:
        raise RuntimeError(
            "v4 pipelined dspark: the tail's prefill reply carries no `acc`, so nothing is drafting "
            "on it in pipelined mode — launch the tail stage with --dspark (and check it is a build "
            "new enough to honour the reset's `pipelined` flag)")
    cur = int(rep["token"])
    c = len(ids)                                              # absolute position of the first new token
    ids.append(cur)
    toks.append(cur)
    if on_token is not None:
        on_token(cur)
    stop = cur in eos or len(toks) >= max_new

    st8 = _PipeSpecState()
    sender = _FrameSender(pipe, st8)
    sender.start()
    sent = {}                                                 # pos -> token fed there in THIS epoch
    ddepth = {}                                               # pos -> how far ahead it was drafted
    dsrc = {}                                                 # pos -> "block" | "topup"
    dalt = {}                                                 # pos -> the drafter's runner-up there
    horizon = c - 1                                           # highest position a frame has been sent for
    frames = drafted = accepted = cancels = run = stale = issued = 0
    topups = topup_frames = topup_agree = topup_disagree = 0
    hist, depths = {}, []
    by_depth, tu_by_depth = {}, {}                            # draft depth -> [hits, trials], per source
    rescue_by_depth = {}                                      # cancel depth -> [runner-up hits, cancels]
    prevhint = set()                                          # positions sent a `dprev` PREDICTION
    blen = 0                                                  # widest block a reply actually carried
    # THE TIME-WEIGHTED FILL. `depths` samples the span once per judged reply, so `mean_inflight`
    # weights a level by how many replies it survived rather than by how long it HELD, and that
    # bias measured ~6.5% high on the live ring. `tick` instead integrates level x wall time across
    # every change of `c` or `horizon`: the clock starts at the first streamed frame and freezes on
    # the commit that sets `stop`, so the post-EOS drain dilutes nothing.
    tick = {"t": None, "area": 0.0, "span": 0.0}

    def _mark():
        """Advance the fill integral to NOW at the CURRENT level — called BEFORE c/horizon moves."""
        now = time.monotonic()
        if tick["t"] is not None:
            dt = now - tick["t"]
            tick["area"] += (horizon - c + 1) * dt
            tick["span"] += dt
        tick["t"] = now

    def _feed(pos, tok, nxt=None, prev=False, dep=0, src="block", alt=None):
        """Stream one s=1 frame. `cpos` is the settled watermark the stages free checkpoints against —
        `c - 1`, never `c`: the deepest rewind this coordinator can still ask for is `c` itself (a
        cancel commits at some p+1 > c and re-opens THERE), so anything ending at or before `c - 1` is
        provably unreachable, and anything that could cover `c` survives.

        `nxt` is THE LAZY-DRAFT HINT and it is only ever set when this loop can prove two things about
        the frame at `pos + 1`: that it is already decided (`nxt` IS the token being streamed there)
        and that this frame's reply therefore cannot be the one that consumes a block. See
        `_hints` — every frame the proof does not cover is sent without a hint, which the tail reads
        as "draft".

        `prev` is the SAME hint for the one frame `nxt` structurally cannot reach: the last of a
        burst, whose successor comes out of the block this burst has not been answered with yet. See
        the DPREV comment where it is set.

        `dep` is how far ahead of the frontier this token was DRAFTED — 1 for a block's first draft,
        B for its last, 0 for a truth/correction frame, which is not a prediction at all — and `src`
        is which refill streamed it. Both exist for one reason: `accept_by_depth` and
        `topup_accept_by_depth` in the return dict. Acceptance is not one number — it decays with
        draft depth, and a topped-up frame at the same depth is a different bet from a drain-refill
        one — and a run that reported one blended acceptance could not say which of the two the
        refill floor is spending. Nothing reads them to make a decision; they only report.

        `alt` is the drafter's RUNNER-UP for this position (the reply's `d2` column, sliced exactly
        as the draft was) — carried purely so a cancel can score it (`rescue_by_depth`). Nothing is
        ever fed from it."""
        nonlocal horizon, frames, topup_frames
        sent[pos] = int(tok)
        ddepth[pos] = int(dep)
        dsrc[pos] = src
        dalt[pos] = None if alt is None else int(alt)
        if src == "topup":
            topup_frames += 1
        _mark()                                               # the old level held until this frame
        horizon = max(horizon, pos)
        frames += 1
        f = {"op": "step", "ids": [[int(tok)]], "start_pos": pos, "epoch": st8.epoch, "cpos": c - 1}
        if lazy and nxt is not None:
            f["dnxt"] = int(nxt)
        if lazy and prev:
            f["dprev"] = True
            prevhint.add(pos)
        sender.put(f, st8.epoch)

    def _hints(span):
        """One lazy-draft hint per position of the in-flight span `c .. c+len(span)-1`, in order.

        `span` is the tokens that will occupy those positions once this burst is on the wire; the
        hint for position `p` is the token at `p + 1`, or None where this loop cannot prove the
        frame's reply will not be asked for a block.

        THE PROOF THE HINT STANDS ON. This loop consumes a block from the reply of position `p` in
        exactly three cases, which are the `need` predicate below: nothing is speculated at `p + 1`
        (`fed is None`), what is speculated there disagrees with the model's token and the round
        cancels, or in-flight has sagged to the refill floor (`horizon - p <= F` at the reply).
        After this burst the in-flight span is `c .. H` with `H = c + len(span) - 1`, CONTIGUOUS, and
        within an epoch horizon only ever grows — a cancel resets it, but a cancel also fences every
        frame behind it, and a fenced frame's reply is dropped before it can be judged, so no hint on
        it is ever acted on by this loop. So for a frame at `p <= H - F - 1`: `sent[p + 1]` exists
        (contiguity), and in-flight at its reply is `horizon_then - p >= H - p >= F + 1 > F`, so its
        reply cannot be the one that refills. The only way it can be the round's last is a
        disagreement at `p + 1` — which is exactly what `dnxt` lets the tail test for itself, off the
        model's own token, without this loop having to predict acceptance.

        The last `F + 1` positions get no hint, and the tail drafts on them: the deepest because its
        successor is not decided yet (it comes out of a block not yet returned), the rest because
        their replies may sit at or below the floor and be asked to refill. At the shipped floor of 1
        that is exactly the old two — the drain and the frame past it. That is the conservative half
        of the rule: an unhinted frame always drafts, so a hint that is merely MISSING costs
        throughput and can never cost a block the round needed. It is also why a deeper floor defeats
        lazy drafting by degrees — the guard swallows the whole span from floor 5 — which is part of
        the floor's price and is visible per run as `drafts_issued` against `frames`."""
        return [span[i + 1] if i <= len(span) - F - 2 else None for i in range(len(span))]

    if not stop:
        _feed(c, cur)                                         # the cur frame: its reply carries block 1
    while True:
        with st8.lock:
            waiting = st8.pending
        if waiting == 0:
            if stop:
                break
            raise RuntimeError(                               # would otherwise be a silent hang
                f"v4 pipelined dspark: nothing in flight at position {c} with {len(toks)} of "
                f"{max_new} tokens generated — the pipeline emptied without stopping")
        try:
            rep = recv_msg(ret)
        except Exception:
            if sender.err is not None:
                raise RuntimeError(f"v4 pipelined dspark: the frame sender died: "
                                   f"{type(sender.err).__name__}: {sender.err}") from sender.err
            raise
        with st8.lock:
            pos, ep = st8.outstanding.popleft()
            st8.pending -= 1
        if int(rep.get("pos", pos)) != pos or int(rep.get("epoch", ep)) != ep:
            raise RuntimeError(
                f"v4 pipelined dspark: reply {rep.get('pos')}@e{rep.get('epoch')} for the frame "
                f"{pos}@e{ep} — the return channel reordered or dropped a frame, and every accept "
                f"after it would be attributed to the wrong position")
        if rep.get("fenced") or ep != st8.epoch:              # THE FENCE: a cancelled future's reply
            stale += 1
            continue
        if stop:                                              # draining after EOS/max_new: judge nothing
            continue
        if pos != c:
            raise RuntimeError(f"v4 pipelined dspark: reply for position {pos} with the committed "
                               f"frontier at {c} — replies must land in committed order")
        # A reply is only allowed to be BELIEVED when the frame that produced it was on the committed
        # path — otherwise its greedy token answers a history the ring is not taking. Two independent
        # ways to know, and both must say yes: this loop's own record of what it fed at `pos` against
        # what it committed there, and the tail's `acc` from its own incremental frontier. The first is
        # an invariant of the fence (a rejected frame is fenced and its reply dropped before it can get
        # here); the second is the same cross-check the serial round asserts on `n`.
        if sent.get(pos) != ids[pos]:
            raise RuntimeError(
                f"v4 pipelined dspark: about to judge the reply of the frame at {pos}, which fed "
                f"{sent.get(pos)!r} while the committed token there is {ids[pos]!r} — a fenced frame "
                f"reached the accept path, and every token after it would answer a discarded history")
        if not rep.get("acc"):
            raise RuntimeError(
                f"v4 pipelined dspark: the tail judged the frame at {pos} speculative and this "
                f"coordinator judged it committed, off the same frame and the same replies. The two "
                f"ends of a lossless round have diverged, so the drafter is advancing over a history "
                f"the ring is not taking — one accept rule, and it is plan_verify_round.")
        m = int(rep["tokens"][0])                             # the model's token at pos+1, at M=b
        fed = sent.get(pos + 1)                               # what we streamed there, if anything yet
        fdep = ddepth.get(pos + 1, 0)                         # 0 == a truth frame, i.e. not a trial
        _mark()                                               # the old level held until this reply
        ids.append(m)
        c = pos + 1
        toks.append(m)
        if on_token is not None:
            on_token(m)
        if fdep:                                              # score the prediction at ITS draft depth
            book = tu_by_depth if dsrc.get(pos + 1) == "topup" else by_depth
            slot = book.setdefault(fdep, [0, 0])
            slot[1] += 1
            if fed == m:
                slot[0] += 1
            elif dalt.get(pos + 1) is not None:               # THE TREE GATE: a cancel with a d2 to
                slot2 = rescue_by_depth.setdefault(fdep, [0, 0])   # judge — would the runner-up have
                slot2[1] += 1                                 # rescued this very cancel? beta is
                slot2[0] += int(dalt[pos + 1] == m)           # hits/trials (docs/V4_TREE_VERDICT.md)
        if fed == m:
            accepted += 1
            run += 1
        stop = m in eos or len(toks) >= max_new
        if stop:
            with st8.lock:                                    # fence what is still queued: the run is
                st8.epoch += 1                                # over, so nothing more reaches the ring
            sender.stop()                                     # settle `pending` before draining on it
            continue
        blk = [int(t) for t in (rep.get("draft") or [])]
        alt2 = [int(t) for t in (rep.get("d2") or [])]        # runner-up per slot, aligned with blk
        issued += bool(blk)                                   # what the TAIL paid for, block for block
        blen = max(blen, len(blk))
        # THE REPLY THE ROUND REFILLS FROM, named: the next block has to come off THIS reply —
        # because nothing is speculated past the frontier (`fed is None`), or because the cancel
        # below discards everything that was (`fed != m`), or because in-flight has sagged to the
        # refill floor (`horizon - c + 1 <= F`; at the shipped floor of 1 that is the old drain,
        # `horizon == c`, exactly). It is the condition the block is streamed under, hoisted so the
        # lazy-draft hint can be audited against it.
        need = fed is None or fed != m or horizon - c + 1 <= F
        if lazy and need and "draft" not in rep and pos not in prevhint:
            # LAZY DRAFTING SKIPPED A BLOCK THE ROUND NEEDED. A reply with no `draft` key AT ALL is a
            # tail that skipped one (a `_done` tail sends `"draft": []`, which is a different thing
            # and legal). It would not corrupt a token — the round falls back to feeding the truth one
            # position at a time — but it would silently delete the speculation, so a `dnxt` hint,
            # which is a FACT about a frame already streamed, fails loudly when it turns out wrong.
            #
            # `dprev` is a PREDICTION and is exempt, deliberately. It says "your successor is the head
            # of the block you return one frame from now", which holds unless the NEXT burst is
            # degenerate — a burst of one puts the drain back on the frontier, i.e. on this very
            # frame. That cannot happen while the drafter's block length is fixed (it is: the
            # pipelined path refuses the one lever that trims it, see the confGate raise in
            # `_coord_cli`), but the cost of being wrong should be a round of greedy, not a dead job
            # in the middle of a generation. So the prediction degrades and the fact raises.
            #
            # Gated on `lazy` because with the flag off no hint was sent, and a tail that answers a
            # committed frame with no block is then simply a tail that is not drafting — the eager
            # path's own business, and it ran to completion before this lever existed.
            raise RuntimeError(
                f"v4 pipelined dspark: the reply for the frame at {pos} carries no block and this "
                f"round needs one (frontier {c}, horizon {horizon}) — the tail skipped a draft the "
                f"lazy-draft hint did not license it to skip")
        # THE SPAN THIS BURST LEAVES IN FLIGHT: the token committed at `c` followed by as much of the
        # block as the in-flight cap admits (`(pos+2+i) - c >= W` with `c == pos+1` cuts it at W-1).
        # `m` is what sits at `c` on every path — it is what the two branches below feed there, and on
        # the accepted path `fed == m` was already streamed there. Hints are one per position of it.
        nblk = min(len(blk), max(W - 1, 0))
        hints = _hints([m] + blk[:nblk])
        if fed is None:                                       # nothing speculated here: feed the truth
            _feed(c, m, hints[0])
        elif fed != m:                                        # REJECT -> cancel, rewind, correct
            cancels += 1
            hist[run] = hist.get(run, 0) + 1
            run = 0
            with st8.lock:
                st8.epoch += 1
            for p in [p for p in sent if p > pos]:            # the discarded future
                del sent[p]
                ddepth.pop(p, None)
                dsrc.pop(p, None)
                dalt.pop(p, None)
            _mark()                                           # the fenced span held until the cancel
            horizon = pos
            # start_pos == c IS the rewind command. The block on THIS reply was drafted off `m`, the
            # correction — the docstring's FRESH BLOCK COMES WITH THE REJECTION — so the span is the
            # correction plus that block, exactly as after a drain.
            _feed(c, m, hints[0])
        # Stream the fresh block when this reply opened the speculation — after a cancel, after the
        # first frame, after a full-accept run whose last reply lands the horizon back on the
        # frontier — or, above the shipped floor of 1, when in-flight has merely SAGGED to the floor.
        # Mid-run replies carry a block either way (drafted one position further along); below the
        # floor it is discarded exactly as before, at or under it the block's NEW positions are
        # streamed and the ones already in flight are never re-streamed — a frame on the wire cannot
        # be recalled, and the disagreement between it and the fresh block's re-prediction is the
        # measurement (`topup_agree`/`topup_disagree`), not a decision.
        if blk and horizon - c + 1 <= F:
            base = horizon                                    # deepest frame already in flight
            topup = base > c                                  # a refill past frames still unjudged
            drafted += 1
            topups += topup
            for i, d in enumerate(blk[:nblk]):
                p = pos + 2 + i
                if p <= base:                                 # in flight: measure agreement, feed nothing
                    topup_agree += int(d) == sent[p]
                    topup_disagree += int(d) != sent[p]
                    continue
                # DPREV, on the last frame of a floor-1 burst and only there. `_hints` cannot reach
                # it: its successor at `H + 1` is the first draft of the block the frame at `H - 1`
                # has not returned yet. But at floor 1 `H - 1` IS the drain (`max(c, H-1)`, and
                # `H-1 >= c` because `nblk >= 1`), so this loop already knows it will consume that
                # reply's block and stream it from `H + 1` — and the tail holds it, because an
                # unhinted drain drafts. `dprev` says exactly that: "your successor is the head of
                # the block you returned one frame ago". The tail re-checks its half (that it really
                # did draft, at exactly `H - 1`, non-empty) and falls back to drafting when it cannot.
                #
                # `nblk >= 2` IS THE SOUNDNESS CONDITION AND IT IS NOT COSMETIC. A burst of one leaves
                # the drain ON the frontier (`max(c, H-1) == c`), which is a frame already in flight —
                # so the frame this hint lands on would be the NEXT burst's drain, and a drain that
                # skips is a round with no block. Bursts are `min(block, W-1)` long and the drafter's
                # block length does not move within a job, so requiring it here requires it of the
                # next burst too; at W=2 the lever simply stays off, and the coordinator's loud check
                # is what fires if that reasoning is ever wrong.
                #
                # `F == 1` IS PART OF THAT SOUNDNESS: above the shipped floor the refill reply is not
                # the drain — it is whichever reply sags to the floor first — so "the block you
                # returned one frame ago" stops naming the block this loop will stream from, and the
                # prediction is withheld rather than made cleverer. The frames dprev would have
                # covered simply draft, which a deeper floor needs of them anyway.
                _feed(p, d, hints[i + 1], prev=(F == 1 and i == nblk - 1 and nblk >= 2),
                      dep=p - c, src="topup" if topup else "block",
                      alt=alt2[i] if i < len(alt2) else None)
        # THE PIPELINE FILL, measured: positions c..horizon all have a frame in the ring and none of
        # them has been judged yet (replies land in order and this one was for c-1), so that span IS
        # the number of frames in flight. 1 means the pipeline never filled and this is greedy decode
        # with extra steps; B+1 is the block fully streamed.
        depths.append(horizon - c + 1)
        sent.pop(pos - 1, None)                               # settled two positions back: never read again
        ddepth.pop(pos - 1, None)
        dsrc.pop(pos - 1, None)
        dalt.pop(pos - 1, None)
        prevhint.discard(pos)

    hist[run] = hist.get(run, 0) + 1
    with st8.lock:                                            # drop anything the fence left queued
        st8.epoch += 1
    sender.stop()
    if sender.err is not None:
        raise RuntimeError(f"v4 pipelined dspark: the frame sender died: "
                           f"{type(sender.err).__name__}: {sender.err}") from sender.err
    recs, receipts_ok = [], None
    if receipts:
        recs, receipts_ok = _sweep_receipts(pipe, ret, layer_count, nonce)
    gen = len(toks)
    cycles = cancels + 1
    return {"ok": True, "tokens": toks, "prompt_tokens": len(prompt_ids),
            "receipts": recs, "receipts_ok": receipts_ok,
            "frames": frames, "drafted": drafted, "drafts_issued": issued, "lazy": lazy,
            "floor": F, "generated": gen, "accepted": accepted,
            "cancels": cancels, "cycles": cycles, "rounds": cycles,
            "g": (gen / cycles) if cycles else float(gen), "accept_hist": hist,
            "accept_by_depth": {d: tuple(v) for d, v in sorted(by_depth.items())},
            "topups": topups, "topup_frames": topup_frames,
            "topup_accept_by_depth": {d: tuple(v) for d, v in sorted(tu_by_depth.items())},
            "topup_agree": topup_agree, "topup_disagree": topup_disagree,
            "rescue_by_depth": {d: tuple(v) for d, v in sorted(rescue_by_depth.items())},
            "max_inflight": max(depths) if depths else 0,
            "mean_inflight": round(sum(depths) / len(depths), 2) if depths else 0.0,
            "inflight_time_avg": round(tick["area"] / tick["span"], 2) if tick["span"] > 0 else 0.0,
            "decode_wall_s": round(tick["span"], 3),
            "frames_per_token": round(frames / gen, 3) if gen else 0.0,
            "block_len": blen,
            "stale_replies": stale, "unsent_frames": st8.unsent}


# ── layer tiling: consume a V4 profile via plan_ring ───────────────────────────────────────────────

def even_tiling(n_layers, nstages):
    """Contiguous, balanced tiling of [0:n_layers) across nstages — the front stages take the extra
    layer when it doesn't divide. The offline fallback (and the tiny-ring test split) when there is
    no measured pool to plan against."""
    if nstages < 1 or nstages > n_layers:
        raise ValueError(f"cannot tile {n_layers} layers over {nstages} stages")
    base, extra = divmod(n_layers, nstages)
    ranges, lo = [], 0
    for k in range(nstages):
        hi = lo + base + (1 if k < extra else 0)
        ranges.append((lo, hi))
        lo = hi
    return ranges


def plan_layer_ranges(nodes, rtt, model_id=V4_MODEL_ID, *, per_gpu=False, **kw):
    """Place the 43-layer ring across a measured pool. Delegates to shard.plan.plan_ring with the V4
    engine profile, so v4_pipe never re-derives the memory model. Returns
    [{id, index, lo, hi, head, tail, layers}, ...] or None when the pool cannot hold the model.

    shard/plan.py keys its profiles by CATALOG MODEL ID (PROFILES: M25_PROFILE, K3_PROFILE) and
    profile_for() raises on an unknown id rather than falling back to another model's calibration.
    There is no V4 entry yet — it needs MEASURED numbers (per-layer VRAM at native FP4, the FP4
    load transient, ms/layer, and a boundary payload of hc_mult*dim*2 = 32 KiB/token, 4x a plain
    hidden state), which is a ring-phase job. Until then this raises with that message and
    even_tiling() is the offline fallback.

    plan_ring places layers on NODES (a box announces its AGGREGATE free VRAM across its GPUs). With
    per_gpu=True each node's block is then split into one contiguous sub-block per local GPU (read
    from the node's "gpus" field, default 1) via split_stages_to_gpus."""
    try:
        from plan import plan_ring, profile_for
    except ImportError:
        from shard.plan import plan_ring, profile_for
    try:
        profile_for(model_id)
    except ValueError as e:
        raise RuntimeError(
            f"no engine profile for {model_id!r} in shard/plan.py PROFILES ({e}) — register a "
            f"measured V4_PROFILE (layer_vram_mb / load_peak_extra_mb / layer_ms_base / "
            f"decode_bytes = hc_mult*dim*2) before planning a V4 ring; even_tiling() is the "
            f"offline fallback") from e
    plan = plan_ring(nodes, rtt, model=model_id, **kw)
    if not plan:
        return None
    if not per_gpu:
        return plan["stages"]
    return split_stages_to_gpus(plan["stages"], {nd["id"]: int(nd.get("gpus", 1)) for nd in nodes})


# ── sidecar wiring: the direct-return topology (payload-agnostic, mirrors m25_scatter_pipe) ─────────

def ring_sidecar_spec(k, n, maddrs, ret_maddr=None):
    """The per-stage sidecar forward/inbound/allow set for the direct-return topology, byte-identical
    in shape to k3_pipe's / m25_scatter_pipe's (the sidecar carries {h, ids} exactly as it carries
    M2.5's single hidden — it never looks inside a frame). Returns (inbound, forwards, allow):
      * every non-tail stage forwards FWD_RING -> the next stage's maddr,
      * the HEAD also forwards FWD_RET -> the tail (the coordinator-return tunnel),
      * every stage but the head takes an -inbound to its local engine, and admits only its
        predecessor's PeerId (the tail also admits the head's — the return stream arrives as head)."""
    inbound = f"127.0.0.1:{ENG_IN}" if k > 0 else ""
    forwards = []
    if k < n - 1:
        forwards.append(f"127.0.0.1:{FWD_RING}={maddrs[k + 1]}")
    if k == 0:
        forwards.append(f"127.0.0.1:{FWD_RET}={ret_maddr or maddrs[-1]}")
    allow = None
    if k > 0:
        allow = [_peerid_of(maddrs[k - 1])]
        if k == n - 1:
            allow.append(_peerid_of(maddrs[0]))
    return inbound, forwards, allow


def _peerid_of(maddr):
    return maddr.rsplit("/p2p/", 1)[-1].split("/p2p-circuit")[0]


# The V4_CUDA_GRAPH strings v4_stage._graph_mode resolves to something other than "off". Duplicated
# here rather than imported because stage_launch_cmd is a PURE BUILDER — tests/test_v4_pipe.py runs it
# on a box with no checkpoint and no v4_stage import (see LAZY MODEL IMPORT in the module docstring).
# tests/test_v4_pipe.py asserts the two lists agree, so drift fails loudly instead of silently
# launching a ring that resolves to eager.
GRAPH_MODE_VALUES = frozenset({"0", "1", "island", "on", "whole", "2", "eager"})

# Every engine lever the operator exports locally has to reach the STAGE PROCESSES too — V4 reads
# them once at import, per process, so a lever set only on the launcher box configures nothing.
# m25_scatter_pipe's ENG_ENV carries the same list for the same reason, and its comment records the
# measurement that taught it. The fp8 wire is the worst possible case for this bug: a ring where
# only some processes have it still WORKS (_recv_hids dispatches on the frame, not on the local
# flag), so a missed propagation is invisible except in the byte counters.
#
# THE LIST IS THE WHOLE CLOSURE, NOT THE LEVER OF THE DAY. `v4_pipe.py stage` imports v4_stage, and
# through it v4_moe_grouped / v4_moe_decode / v4_dspark_fast / v4_ref_slim / v4_whole_layer_graph /
# v4_kernels_cpu, and EVERY one of those reads its flags at import. A lever that is merely absent
# from this list is not "off" on the ring — it is off on the stages and on wherever the operator
# thinks it is, which is how a bench measures the baseline and reports the lever.
# test_v4_pipe.py::test_every_v4_lever_reaches_a_launched_stage re-derives this set from the source
# of those modules, so a lever added later fails the suite until it is routed here.
#
# Two names are deliberately NOT here, because stage_launch_cmd already emits them itself and a
# second path could only disagree with the first: V4_DIR / V4_DEV (built from the arguments) and
# V4_CUDA_GRAPH (the validated `cuda_graph=` mode — see _graph_env_conflict for why an operator's
# export of it RAISES rather than quietly winning or quietly losing).
ENG_ENV = [
    "V4_FP8_WIRE",                                                              # wire codec
    "V4_PIPELINED_SPEC", "V4_SPEC_DEPTH", "V4_LAZY_DRAFT", "V4_REFILL_FLOOR",   # pipelined speculation
    "V4_MOE_GROUPED", "V4_MOE_DECODE", "V4_MOE_MULTI", "V4_MOE_MULTI_MAX",      # MoE kernels
    "V4_FP8_GEMV", "V4_FP8_SHARED",                                             # fp8 GEMV path
    "V4_GRAPH_MAX", "V4_MOE_IN_GRAPH",                                          # graph budget / scope
    "V4_DSPARK_FAST", "V4_DSPARK_GRAPH", "V4_DSPARK_MOE", "V4_DSPARK_BLOCK",    # drafter
    "V4_DRAFT_TOP2",                                                            # tree-gate measurement
    "V4_DSPARK_CONF_GATE", "V4_DSPARK_CONF_MIN", "V4_DSPARK_CONF_THRESH",       # adaptive send-length
    "V4_REF_SLIM", "V4_REF_SLIM_NOQAT",                                         # reference-compute slim
    "V4_FAST_VERIFY", "V4_FAST_VERIFY_MAX",                                     # chunked verify
    "V4_KERNELS", "V4_DTYPE", "V4_MAX_SEQ", "V4_MAX_BATCH",                     # stage build-out
    "V4_KEEPWARM", "V4_KEEPWARM_MS",                                            # transport keep-warm
    "V4_DIAL_CONNECT_TIMEOUT", "V4_DIAL_RETRY_S",                               # inter-stage dial
    "V4_TIMING", "V4_TIMING_EVERY",                                             # instrumentation
    "V4_LEVERS_STRICT",                                                         # lever audit
]


def _eng_env():
    return "".join(f"{k}={os.environ[k]} " for k in ENG_ENV if k in os.environ)


def _graph_env_conflict(gp_mode):
    """Refuse a launch whose V4_CUDA_GRAPH export disagrees with the mode it is about to emit.

    V4_CUDA_GRAPH is the one lever that does NOT ride ENG_ENV, because stage_launch_cmd always writes
    it explicitly from `cuda_graph=` — it cannot go missing, so the bug ENG_ENV exists for cannot
    happen to it. What CAN happen is the two disagreeing, and both ways of resolving that silently are
    the failure this engine's flags exist to prevent: let the export win and an explicit
    `cuda_graph=False` latency A/B secretly runs graphed; let the argument win and an operator who
    exported `whole` measures island and reports whole. So say so instead, and make them pick."""
    v = os.environ.get("V4_CUDA_GRAPH")
    if v is not None and v != gp_mode:
        raise ValueError(
            f"V4_CUDA_GRAPH={v!r} is exported on the launcher but this launch would send "
            f"V4_CUDA_GRAPH={gp_mode!r} to the stages. Refusing to pick for you: pass "
            f"cuda_graph={v!r} to mean it, or unset the export to use the launch default. "
            f"(Unlike the ENG_ENV levers this one is always emitted, so it is never dropped — "
            f"only ever ambiguous.)")


def stage_launch_cmd(stage, nstages, lo, hi, *, model_dir="/root/v4", receipts=False,
                     device="cuda", token=None, extra_env="", gpu=None, port=None, nxt_addr=None,
                     ret_relay=None, dspark=False, cuda_graph=True):
    """The shell command a launcher runs to start ONE v4 engine stage over the sidecar. Fully
    DETACHED in every shape (setsid + fd redirect), mirroring k3_pipe.stage_launch_cmd /
    m25_scatter_pipe.stage_cmd — see the note by the return.

    Single-GPU box (gpu/port/nxt_addr all unset): the stage binds ENG_IN and a non-tail stage
    forwards to the LOCAL sidecar forward leg (FWD_RING); the tail has none. The engine binds
    loopback (M25_ENGINE_BIND) so only the local sidecar can reach it.

    Multi-GPU box: pass this stage's local
      * `gpu`      — the local GPU index. CUDA_VISIBLE_DEVICES=<gpu> so the process sees ONLY that
                     card, as cuda:0 (the least-invasive device pin — no engine change).
      * `port`     — its local engine port (ENG_IN for the box's first stage, else ENG_LOCAL_BASE+idx).
      * `nxt_addr` — where it forwards: a LOOPBACK 127.0.0.1:<next local port> to the next GPU on this
                     box, or the WAN 127.0.0.1:FWD_RING leg for the box's last stage.
      * `ret_relay` — set ONLY on the ingress of a G>1 TAIL box: the loopback 127.0.0.1:<box tail local
                     port> it bridges the coordinator-return to (--ret-relay). Unset everywhere else.
      * `dspark`   — build the tail's MTP speculator stages at load (step 4).

    `cuda_graph` (V4_CUDA_GRAPH) is ON by default because the ring is CPU-launch-bound: the partial
    island graphs (v4_stage's _BlockGraphs — the hc_pre/hc_post/norm islands, bit-exact per
    test_v4_stage_graph.py) collapse ~68 of ~240 kernel launches per layer and buy ~+12% steady-state
    single-stream, which is real money on a dispatch-bound serial pipe. It is OFF in the module
    default (a bare import, the CPU parity suite, an in-process ring) and turned ON *here*, in the
    launch path, where it belongs.

    IT IS A MODE, NOT A BOOL: True/"1"/"island" graphs the hc/norm islands (1.22x on the layer),
    "whole" folds in v4_whole_layer_graph's capture-safe attention core and leaves the REAL routed MoE
    eager between two graphs (2.15x on the layer — docs/receipts/v4-whole-layer-graph-20260801.json),
    False/"0" is off. 2.15x is the deployable number; the receipt's 7.31x wall / 12.01x cpu is the
    ceiling with a STUB MoE and is not what a ring gets.

    The mode is spelled out here rather than left to `extra_env` because a launcher that can only
    reach island mode is one forgotten string away from measuring island and reporting whole — the
    exact shape of the m25 failure this engine's flags are all designed against.

    THE COST IS ONE-TIME AND FRONT-LOADED. The first decode token pays a capture cascade — measured
    ~533s on the first live ring — that is NOT the CUDA-graph capture (microseconds) but the tilelang
    JIT autotuning + compiling the sparse-attn / fp8 / fp4 kernels at V4's real shapes, triggered the
    first time each fires inside a capture's warm-up. tilelang memoises those compiled artifacts to
    its on-disk cache, so the compile half is paid ONCE PER BOX and a re-warm (re-launch on the same
    still-rented box) reuses them; only a genuinely fresh box pays it again. The graph capture itself
    is per-process and unavoidable, but it is a fixed first-token tax amortised over the whole
    generation. A launcher that wants the cold first token faster (e.g. a latency A/B) sets
    `cuda_graph=False`; a value passed through `extra_env` still wins over this default.
    """
    port = port or ENG_IN
    if nxt_addr is not None:
        nxt = f"--next {nxt_addr}"
    else:
        nxt = "" if stage == nstages - 1 else f"--next 127.0.0.1:{FWD_RING}"
    rr = f"--ret-relay {ret_relay} " if ret_relay else ""
    ds = "--dspark " if dspark else ""
    rc = "SHARD_RECEIPTS=1 " if receipts else ""
    tk = f"SHARD_SWARM_TOKEN={token} " if token else ""
    cvd = f"CUDA_VISIBLE_DEVICES={int(gpu)} " if gpu is not None else ""
    # graphs ON by default for a GPU launch; placed BEFORE extra_env so an explicit override there
    # (V4_CUDA_GRAPH=0) still wins — bash takes the rightmost assignment of a repeated name.
    # A bool keeps the old spelling; a string passes the mode through, so "whole" is reachable from a
    # launcher argument instead of only from a hand-written extra_env.
    gp_mode = ("1" if cuda_graph is True else "0" if cuda_graph is False else str(cuda_graph))
    if gp_mode not in GRAPH_MODE_VALUES:
        raise ValueError(f"stage_launch_cmd: V4_CUDA_GRAPH={gp_mode!r} is not a mode v4_stage "
                         f"recognises, so the stage would resolve it to OFF. Use one of "
                         f"{sorted(GRAPH_MODE_VALUES)} — a typo here launches a ring that measures "
                         f"eager and reports graphed.")
    _graph_env_conflict(gp_mode)
    gp = f"V4_CUDA_GRAPH={gp_mode} "
    # DETACH (K3 ring Bug 1, 2026-07-29): a bare `nohup <cmd> &` over ssh kept the channel open on the
    # engine's child fds and HUNG the launcher, so a parallel launch of all B*G stages never returned
    # the ssh session. setsid + a full fd redirect fully detaches the engine and ssh returns instantly;
    # its stdout goes to a PER-PORT log so co-located stages never clobber each other's.
    log = f"/root/v4_stage_{port}.log"
    inner = (f"python3 /root/v4_pipe.py stage --stage {stage} --nstages {nstages} --lo {lo} --hi {hi} "
             f"--port {port} {nxt} {rr}{ds}--dir {model_dir} > {log} 2>&1")
    return (f"{rc}{tk}{cvd}{gp}{_eng_env()}{extra_env}V4_DIR={model_dir} V4_DEV={device} "
            f"M25_ENGINE_BIND=127.0.0.1 setsid bash -c '{inner}' </dev/null >/dev/null 2>&1 &")


# ── multi-GPU per box: split node blocks to GPUs, wire loopback intra-box + WAN inter-box ───────────

def local_eng_port(local_index):
    """The engine port for the `local_index`-th stage on a box. Index 0 -> ENG_IN (the box ingress
    the sidecar -inbound and the head coordinator dial); 1.. -> ENG_LOCAL_BASE+idx (loopback only)."""
    return ENG_IN if local_index == 0 else ENG_LOCAL_BASE + local_index


def split_stages_to_gpus(node_stages, gpus):
    """Split a head-first NODE plan into one contiguous sub-block per local GPU.

    `node_stages` is plan_ring/plan_layer_ranges output (or any head-first [{id, lo, hi}...] whose
    blocks tile the model): each entry is a BOX holding a contiguous layer range. `gpus` gives a
    box's GPU count — an int applied to every box, a list aligned to node_stages, or a dict keyed by
    node id. A box's range is tiled EVENLY across its GPUs (front GPUs take the extra layer), so a
    G-GPU box becomes G contiguous sub-blocks. A 1-GPU box is returned unchanged (its one sub-block
    == the node block), so a single-GPU ring is byte-identical.

    Pure tiling: it does NOT enforce a per-GPU layer cap — VRAM feasibility is plan_ring's job
    upstream (the caller announces a box's aggregate VRAM). Returns a FLAT head-first list of
    per-GPU stage specs:
      {"id", "box_index", "gpu", "local_index", "nlocal", "global_index", "nstages",
       "lo", "hi", "layers", "head", "tail", "box_head", "box_tail"}
    where head/tail are the GLOBAL ring ends (the head embeds, the tail samples) and box_head/box_tail
    mark the first/last local stage on a box (its WAN-facing ends)."""
    def _g(k, nd):
        if isinstance(gpus, dict):
            return int(gpus.get(nd["id"], 1))
        if isinstance(gpus, (list, tuple)):
            return int(gpus[k])
        return int(gpus)
    subs = []
    for k, nd in enumerate(node_stages):
        G = _g(k, nd)
        if G < 1:
            raise ValueError(f"box {nd.get('id', k)!r} announced {G} GPUs")
        for j, (slo, shi) in enumerate(even_tiling(nd["hi"] - nd["lo"], G)):
            subs.append({"id": nd["id"], "box_index": k, "gpu": j, "local_index": j, "nlocal": G,
                         "lo": nd["lo"] + slo, "hi": nd["lo"] + shi, "layers": shi - slo,
                         "box_head": j == 0, "box_tail": j == G - 1})
    n = len(subs)
    for g, s in enumerate(subs):
        s["global_index"], s["nstages"] = g, n
        s["head"], s["tail"] = g == 0, g == n - 1
    return subs


def box_stage_wiring(sub):
    """(eng_port, next_addr, link) for one per-GPU stage from split_stages_to_gpus.

    eng_port: the box's first local stage binds ENG_IN (the sidecar -inbound / head coordinator reach
    it); the rest bind loopback-only local ports. next_addr / link:
      * global tail            -> (None, "tail")      no forward; it samples + returns to the coordinator
      * box_tail (not tail)    -> (FWD_RING, "wan")    the box's WAN egress -> next box via the sidecar
      * otherwise              -> (next local port, "loopback")   hand off to the next GPU on THIS box
    """
    eng_port = local_eng_port(sub["local_index"])
    if sub["tail"]:
        return eng_port, None, "tail"
    if sub["box_tail"]:
        return eng_port, f"127.0.0.1:{FWD_RING}", "wan"
    return eng_port, f"127.0.0.1:{local_eng_port(sub['local_index'] + 1)}", "loopback"


def box_return_relay(sub, tail_box_index):
    """The intra-box return-relay target for one per-GPU stage, or None. Only the box INGRESS
    (box_head) of a MULTI-GPU box that holds the global tail relays: the coordinator-return tunnel
    lands at that box's ENG_IN, but the token is produced on its last-GPU stage — so the ingress
    bridges them, dialing the box tail's return over loopback and pumping it out to the coordinator.
    Returns '127.0.0.1:<box tail local port>' for that one stage, None for every other. A 1-GPU tail
    box has box_head == box_tail == the global tail and needs no relay (returns None)."""
    if sub["box_head"] and sub["nlocal"] > 1 and sub["box_index"] == tail_box_index:
        return f"127.0.0.1:{local_eng_port(sub['nlocal'] - 1)}"
    return None


def box_ring_launch(node_stages, gpus, box_maddrs=None, *, model_dir="/root/v4", receipts=False,
                    token=None, ret_maddr=None, dspark=False):
    """The full launch plan for a ring of B boxes x G GPUs = B*G stages but only B WAN hops.

    node_stages: head-first NODE plan (plan_layer_ranges without per_gpu); gpus: per-box GPU count
    (see split_stages_to_gpus); box_maddrs: the B box libp2p multiaddrs (head-first) for the sidecar
    wiring — omit to skip it (topology-only). Returns:
      {"stages":   [{**sub, "eng_port", "next", "link", "ret_relay", "cmd"}...],  # one per (box, gpu)
       "sidecars": [(inbound, forwards, allow)...] | None}            # per box, ring_sidecar_spec at
                                                                      # BOX granularity

    The intra-box GPU stages hand off over loopback and NEVER touch the sidecar, so the sidecar sees
    only the B boxes: ring_sidecar_spec over B nodes gives B-1 forward hops + 1 head->tail return =
    B WAN hops, whatever G is.

    The RETURN path is closed for a G>1 TAIL box: the coordinator-return tunnel is delivered by the
    sidecar to the tail box's -inbound (ENG_IN = its FIRST local stage), but the token is produced on
    its LAST local GPU. That box's ingress carries a `ret_relay` (--ret-relay) and bridges the two over
    loopback, so the sidecar wiring stays BOX-granular and identical to M2.5/K3."""
    subs = split_stages_to_gpus(node_stages, gpus)
    tail_box = subs[-1]["box_index"]                           # the box holding the global tail
    stages = []
    for sub in subs:
        eng_port, nxt, link = box_stage_wiring(sub)
        ret_relay = box_return_relay(sub, tail_box)
        cmd = stage_launch_cmd(sub["global_index"], sub["nstages"], sub["lo"], sub["hi"],
                               model_dir=model_dir, receipts=receipts, token=token,
                               gpu=sub["gpu"], port=eng_port, nxt_addr=nxt, ret_relay=ret_relay,
                               dspark=(dspark and sub["tail"]))
        stages.append({**sub, "eng_port": eng_port, "next": nxt, "link": link,
                       "ret_relay": ret_relay, "cmd": cmd})
    B = len(node_stages)
    sidecars = ([ring_sidecar_spec(b, B, box_maddrs, ret_maddr=ret_maddr) for b in range(B)]
                if box_maddrs is not None else None)
    return {"stages": stages, "sidecars": sidecars}


# ── offline selftest: the whole ring on localhost, CPU, no spend ───────────────────────────────────

def _write_tiny_checkpoint(d, args, model):
    """A real tiny checkpoint in the CONVERTED format the reference's own generate.py consumes:
      <dir>/config.json                 the ModelArgs kwargs (inference/config.json's shape)
      <dir>/model0-mp1.safetensors      rank 0 of a world_size 1 conversion (convert.py's naming)
    so v4_stage.config(dir) + Stage.load(dir) exercise the real loader path against real files.

    Every tensor is CLONED: the reference ties mtp[i].embed/head to the main embed/head, and
    safetensors refuses to save two keys sharing storage. Buffers (kv_cache, kv_state, score_state,
    freqs_cis) are all non-persistent in the reference, so a state_dict is pure parameters and
    carries no warm sequence state into the stages.

    ASSUMPTION TO RECONCILE WITH v4_stage (step 2): that .load() reads model0-mp1.safetensors
    (strict=False, taking only the keys its own layer range owns) and .config() parses config.json
    into ModelArgs. If step 2 chose different names, change them HERE — this is the only place the
    pipe touches checkpoint layout."""
    import dataclasses
    import safetensors.torch as ST
    os.makedirs(d, exist_ok=True)
    tensors = {k: v.detach().clone().contiguous() for k, v in model.state_dict().items()}
    ST.save_file(tensors, os.path.join(d, "model0-mp1.safetensors"))
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump(dataclasses.asdict(args), f)
    return d


def _reference_tokens(model, prompt, max_new):
    """Greedy token ids from the vendored reference Transformer itself — the parity baseline. Same
    cadence as the ring: the whole prompt at start_pos 0, then one token per step. temperature is 0
    in cpu_args(), so the reference's own sample() IS an argmax."""
    ids = list(prompt)
    with torch.inference_mode():
        out, _, _ = model(torch.tensor([ids]))
        tid = int(out.reshape(-1)[-1])
        toks = [tid]
        pos = len(ids)
        while len(toks) < max_new:
            out, _, _ = model(torch.tensor([[tid]]), pos)
            pos += 1
            tid = int(out.reshape(-1)[-1])
            toks.append(tid)
    return toks


def _free_ports(n):
    """N distinct free localhost ports. Bind to :0, read the assignments, close — the stages rebind
    with SO_REUSEADDR, so the brief TIME_WAIT window between here and there is harmless."""
    socks = [socket.socket(socket.AF_INET, socket.SOCK_STREAM) for _ in range(n)]
    ports = []
    for s in socks:
        s.bind(("127.0.0.1", 0))
        ports.append(s.getsockname()[1])
    for s in socks:
        s.close()
    return ports


def _expected_cover(ranges):
    return sorted((lo, hi) for lo, hi in ranges)


def _spawn_ring(d, ranges, tail_box_g=1, dspark=False, tag="s"):
    """Start one localhost ring over a tiny checkpoint and dial it. -> (pipe, ret).

    One stage thread per range — the in-process shape of one box per stage — with the last
    `tail_box_g` of them standing in for a MULTI-GPU tail box, whose coordinator-return terminates at
    the box ingress and is bridged to the box tail over loopback (the --ret-relay path). `dspark`
    builds the tail's MTP speculator, exactly as --dspark does on a real box."""
    n = len(ranges)
    ports = _free_ports(n)
    relay_i = n - tail_box_g                                   # ingress of the multi-GPU tail box (>1)
    events = [threading.Event() for _ in range(n)]
    for i, (lo, hi) in enumerate(ranges):
        nxt = None if i == n - 1 else f"127.0.0.1:{ports[i + 1]}"
        ret_relay = f"127.0.0.1:{ports[-1]}" if (tail_box_g > 1 and i == relay_i) else None
        threading.Thread(target=serve_stage, kwargs=dict(
            stage=i, nstages=n, lo=lo, hi=hi, port=ports[i], nxt=nxt, ckpt_dir=d, device="cpu",
            receipts=True, key_path=f"{d}/{tag}{i}.key", ret_relay=ret_relay, dspark=dspark,
            ready=events[i]), daemon=True).start()
    for e in events:
        e.wait(120)
    tail_port = ports[relay_i] if tail_box_g > 1 else ports[-1]   # return lands at the tail box ingress
    return connect_ring(f"127.0.0.1:{ports[0]}", f"127.0.0.1:{tail_port}", timeout=120)


def _dspark_tiling(args, nstages):
    """A tiling whose TAIL owns EVERY dspark target layer, which even_tiling has no reason to know.

    `Stage.tail_main_hidden()` refuses a split target range — the drafter consumes all the taps
    concatenated, so they have to land on ONE stage — and at the toy shape even_tiling leaves the
    first target on the middle stage. That is not a test artifact: it is the placement constraint a
    real V4 ring has (its tail owns at least max(targets) - min(targets) + 1 layers), and the
    eventual V4_PROFILE has to encode it or the planner will place a ring that cannot draft."""
    lo = min(args.dspark_target_layer_ids)
    if nstages != 3 or lo < 2:
        raise ValueError(f"the drafted selftest ring wants 3 stages and targets from layer 2 up "
                         f"(got {nstages} stages, first target {lo})")
    return [(0, lo // 2), (lo // 2, lo), (lo, args.n_layers)]


def selftest(nstages=3, prompt=(168, 15, 493, 72, 22), max_new=6, tail_box_g=1):
    """Offline, CPU, no spend: full v4_pipe rings on localhost (head embed + middles + tail sample +
    weightless coordinator over real shard.transport sockets), decoding three ways, and every one of
    them checked BIT-IDENTICAL against the vendored reference Transformer's own greedy decode.

      greedy   coordinate()          one token per traversal — the baseline the other two must match
      spec     coordinate_spec()     a coordinator-side drafter, deliberately a BAD one (repeat the
                                     last token), so nearly every round is rejected and every one of
                                     those REWINDS each stage's window ring + compressor accumulators
      dspark   coordinate_dspark()   V4's own trained MTP speculator, drafting on the tail

    Losslessness is the whole claim of speculative decoding and it is precisely what is graded here:
    the same tokens as the reference, bit for bit, with the receipts still settling over the ring.

    `tail_box_g` > 1 folds the last G stages into ONE multi-GPU tail box: the coordinator-return
    terminates at that box's ingress and is bridged over loopback to the box tail (the --ret-relay
    path a real G>1 tail box runs), proving the return relay against the same parity bar."""
    import tempfile
    import v4_ref_cpu as R
    os.environ["SHARD_RECEIPTS"] = "1"
    global RECEIPTS
    RECEIPTS = True
    # cpu_args()'s OWN default shape: compress_ratios is pinned to n_layers + n_mtp_layers, so the
    # layer count is not ours to pick — read it back off the args instead.
    args = R.cpu_args()
    n_layers = args.n_layers
    d = tempfile.mkdtemp(prefix="v4pipe_")
    model = R.build_oracle(args)
    _write_tiny_checkpoint(d, args, model)                     # write BEFORE decoding: a fresh model
    os.environ["V4_DIR"] = d                                   # never carries a used sequence's state

    ref_tokens = _reference_tokens(model, list(prompt), max_new)
    ranges = even_tiling(n_layers, nstages)
    pipe, ret = _spawn_ring(d, ranges, tail_box_g)
    r = coordinate(pipe, ret, list(prompt), max_new, nonce="settle-nonce-0", receipts=True,
                   layer_count=n_layers, timeout=120)
    s = coordinate_spec(pipe, ret, list(prompt), max_new, nonce="spec-nonce-0", receipts=True,
                        layer_count=n_layers, timeout=120, K=4, drafter=_RepeatDrafter())

    def perfect(seq, K):
        """The other extreme: propose the reference's OWN continuation, so every draft is accepted.

        A bad drafter only ever exercises rejection; this forces the full-accept path — K+1 tokens
        committed in one traversal, and a next chunk that opens exactly where the stage already
        stands (the rollback's zero-cost no-op). It is not circular: a draft is accepted only when
        the RING's own reply at that position equals it, so a wrong ring rejects it and the
        correction it commits instead diverges from the reference all the same."""
        nxt = ref_tokens[len(seq) - len(prompt):][:K]
        return list(nxt) + [0] * (K - len(nxt))
    p = coordinate_spec(pipe, ret, list(prompt), max_new, nonce="spec-nonce-1", receipts=True,
                        layer_count=n_layers, timeout=120, K=2, drafter=perfect)
    send_msg(pipe, {"op": "stop"})

    d_ranges = _dspark_tiling(args, nstages)                   # the drafted ring: its own tail placement
    d_pipe, d_ret = _spawn_ring(d, d_ranges, tail_box_g, dspark=True, tag="d")
    k = coordinate_dspark(d_pipe, d_ret, list(prompt), max_new, nonce="dspark-nonce-0",
                          receipts=True, layer_count=n_layers, timeout=120)
    q = coordinate_dspark_pipelined(d_pipe, d_ret, list(prompt), max_new, nonce="dspark-nonce-1",
                                    receipts=True, layer_count=n_layers, timeout=120)
    # LAZY DRAFTING, at three pipeline depths on the same warm ring, each against ITS OWN eager
    # baseline at the SAME depth. The depth decides WHICH frame drains the pipeline and is therefore
    # what the hint is derived from, so one depth proving lossless would prove the arithmetic at one
    # depth — and comparing a lazy run at W against an eager run at some other W would compare two
    # different rounds. depth=2 is the degenerate one where the drain sits on the frontier and the
    # lever is expected to stay off rather than be clever.
    #
    # WHAT THIS CANNOT SHOW, and where the proof that matters lives: at random weights the toy's own
    # drafter is wrong every round, so every reply IS the round's last and there is nothing a hint
    # can license skipping. This is the zero-accept regime — it proves the lever is lossless through a
    # rewind on every round and costs nothing, not that it skips anything. The forced-accept regime,
    # where the skipping actually happens, needs a scripted block and lives in
    # tests/test_v4_pipe.py::test_lazy_drafting_on_a_real_ring_is_bit_identical_at_every_depth.
    y = {w: coordinate_dspark_pipelined(d_pipe, d_ret, list(prompt), max_new, nonce=f"eager-{w}",
                                        receipts=True, layer_count=n_layers, timeout=120, depth=w)
         for w in (2, 3, 16)}
    z = {w: coordinate_dspark_pipelined(d_pipe, d_ret, list(prompt), max_new, nonce=f"lazy-{w}",
                                        receipts=True, layer_count=n_layers, timeout=120,
                                        depth=w, lazy=True)
         for w in (2, 3, 16)}
    # THE REFILL FLOOR, on the same warm ring, at the settings that bracket it (2 = one past the
    # shipped drain, 5 = refill on every reply). At random weights every block is rejected at its
    # first draft, so in-flight never exceeds the floor mid-run and a mid-run refill never fires:
    # each floor arm must therefore run the SHIPPED round frame for frame — same stream, same frame
    # count, same cancels, same fill, zero topups. That is the lever's own floor ("a raised floor
    # with nothing to keep up costs nothing"), proven on sockets through a rewind on every round.
    # The regime where the floor actually tops up mid-run needs a block that ACCEPTS, which random
    # weights cannot give; it lives in tests/test_v4_pipe.py against a scripted oracle drafter.
    fl = {f: coordinate_dspark_pipelined(d_pipe, d_ret, list(prompt), max_new, nonce=f"floor-{f}",
                                         receipts=True, layer_count=n_layers, timeout=120, floor=f)
          for f in (2, 3, 5)}
    send_msg(d_pipe, {"op": "stop"})

    def cover(res, rs):
        return sorted((c["layer_start"], c["layer_end"])
                      for c in res["receipts"]) == _expected_cover(rs)

    checks = {
        # A random model's greedy stream usually falls into a fixed point within a token or two, and
        # "the speculated stream equals the greedy stream" says almost nothing when both are one
        # token repeated: every drafter is perfect on a constant. The prompt is picked so the
        # reference's continuation is 6 distinct tokens, and that is asserted rather than assumed.
        "reference_stream_is_a_fingerprint": (len(set(ref_tokens)) >= max(4, max_new - 1)),
        "full_ring_matches_reference": (r["tokens"] == ref_tokens),
        "receipts_settle": (r["receipts_ok"] is True),
        "coverage_tiles_all_layers": cover(r, ranges),
        "spec_lossless_vs_greedy": (s["tokens"] == ref_tokens),
        "spec_receipts_settle": (s["receipts_ok"] is True and cover(s, ranges)),
        # the rejection path above rewinds every round; this is the acceptance path — more than one
        # token committed per traversal, which is the entire point of the lever
        "spec_full_accept_commits_a_block": (p["tokens"] == ref_tokens and p["g"] > 1.0
                                             and max(p["accept_hist"]) == 2),
        "dspark_lossless_vs_greedy": (k["tokens"] == ref_tokens),
        "dspark_receipts_settle": (k["receipts_ok"] is True and cover(k, d_ranges)),
        # A MACHINERY bar, not an acceptance bar. At random weights a trained drafter has nothing to
        # be right about, so demanding g > 1 here would be demanding luck; what must hold is that real
        # draft blocks crossed the ring and were verified — every round after the deliberately bare
        # first one carries one — and that the stream came out exact anyway. Acceptance is a
        # measurement for real weights on a real ring. Losslessness is provable here, and here it is.
        "dspark_drafted_real_blocks": (k["rounds"] > 1 and k["drafted"] == k["rounds"] - 1),
        # PIPELINED: the same drafted ring, streamed as s=1 frames instead of verified as one chunk.
        # Same bar and one more — the emitted stream must equal the reference AND the serial dspark
        # path token for token, because "we restructured the wire and the answer changed" is the only
        # way this lever can be wrong.
        "pipelined_lossless_vs_greedy": (q["tokens"] == ref_tokens),
        "pipelined_matches_the_serial_dspark_path": (q["tokens"] == k["tokens"]),
        "pipelined_receipts_settle": (q["receipts_ok"] is True and cover(q, d_ranges)),
        "pipelined_actually_pipelined": (q["max_inflight"] > 1),
        # LAZY DRAFTING, against the eager round at the same depth: the reference's stream, the same
        # blocks consumed off the same cancels at the same depth, and never MORE drafting. See the
        # comment above `y` for what the zero-accept regime can and cannot show.
        "lazy_lossless_vs_greedy_at_every_depth": all(v["tokens"] == ref_tokens for v in z.values()),
        "lazy_runs_the_same_round_as_eager": all(
            z[w]["drafted"] == y[w]["drafted"] and z[w]["cancels"] == y[w]["cancels"]
            and z[w]["frames"] == y[w]["frames"] and z[w]["max_inflight"] == y[w]["max_inflight"]
            for w in z),
        "lazy_never_drafts_more_than_eager": all(z[w]["drafts_issued"] <= y[w]["drafts_issued"]
                                                 for w in z),
        "lazy_receipts_settle": all(v["receipts_ok"] is True and cover(v, d_ranges) for v in z.values()),
        # THE REFILL FLOOR in the zero-accept regime: lossless at every setting, and byte-identical
        # to the shipped drain-only round — see the comment above `fl` for why identity is exactly
        # what this regime must show. `stale + unsent` is compared as a sum: the split between a
        # fenced frame dropped before the wire and one answered after depends on sender-thread
        # timing, the sum is the deterministic count of frames the fence discarded.
        "floor_lossless_at_every_setting": all(v["tokens"] == ref_tokens for v in fl.values()),
        "floor_at_zero_accept_is_the_shipped_round": all(
            v["frames"] == q["frames"] and v["cancels"] == q["cancels"]
            and v["max_inflight"] == q["max_inflight"]
            and v["stale_replies"] + v["unsent_frames"] == q["stale_replies"] + q["unsent_frames"]
            and v["topups"] == 0 and v["topup_frames"] == 0
            for v in fl.values()),
        "floor_receipts_settle": all(v["receipts_ok"] is True and cover(v, d_ranges)
                                     for v in fl.values()),
        # the depth histogram: at zero accept every judged draft is a depth-1 miss off a cancel
        # refill, so the block book must carry trials and no hits, and the topup book must be empty
        "accept_by_depth_scores_the_misses": all(
            v["accept_by_depth"] and all(h == 0 for h, _ in v["accept_by_depth"].values())
            and v["topup_accept_by_depth"] == {} for v in fl.values()),
    }
    tag = f", tail box = {tail_box_g} GPUs (return relay)" if tail_box_g > 1 else ""
    print("\n=== V4 pipe offline selftest (CPU tiny config) ===")
    print(f"  ring:   {nstages} stages over {n_layers} layers {ranges}{tag}")
    print(f"  dspark: {d_ranges} — the tail owns targets {tuple(args.dspark_target_layer_ids)}, "
          f"block={args.dspark_block_size}")
    print(f"  tokens (ring)   {r['tokens']}\n  tokens (spec)   {s['tokens']}")
    print(f"  tokens (dspark) {k['tokens']}\n  tokens (ref)    {ref_tokens}")
    print(f"  spec   rounds={s['rounds']} accepted={s['accepted']} g={s['g']:.2f} "
          f"hist={s['accept_hist']}  (repeat-last drafter: rejects, so every round rewinds)")
    print(f"  spec   rounds={p['rounds']} accepted={p['accepted']} g={p['g']:.2f} "
          f"hist={p['accept_hist']}  (perfect drafter: full accepts, no rewind)")
    print(f"  dspark rounds={k['rounds']} drafted={k['drafted']} accepted={k['accepted']} "
          f"g={k['g']:.2f} hist={k['accept_hist']}")
    print(f"  tokens (pipe)   {q['tokens']}")
    print(f"  pipe   frames={q['frames']} cycles={q['cycles']} accepted={q['accepted']} "
          f"cancels={q['cancels']} g={q['g']:.2f} inflight max={q['max_inflight']} "
          f"mean={q['mean_inflight']} stale={q['stale_replies']} unsent={q['unsent_frames']}")
    print("  drafts issued/consumed  "
          + "   ".join(f"W={w} eager {y[w]['drafts_issued']}/{y[w]['drafted']} "
                       f"lazy {z[w]['drafts_issued']}/{z[w]['drafted']}" for w in z)
          + "   (zero accept: every block is consumed, so there is nothing to skip)")
    print("  floor  " + "   ".join(
        f"F={f} frames={v['frames']} cancels={v['cancels']} topups={v['topups']} "
        f"by_depth={v['accept_by_depth']}" for f, v in fl.items()))
    for name, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {name}")
    ok = all(checks.values())
    print(f"\n  {'ALL PASS' if ok else 'FAILURES PRESENT'}", flush=True)
    os._exit(0 if ok else 1)


# ── CLI ────────────────────────────────────────────────────────────────────────────────────────────

def _emit(tag, **fields):
    sys.stdout.write(tag + " " + json.dumps(fields) + "\n")
    sys.stdout.flush()


def _encode_prompt(tok, job):
    """Job messages -> token ids. V4's tokenizer_config.json ships NO chat_template (unlike K3's), so
    there is no apply_chat_template to lean on: the prompt format lives in the vendored
    deepseek_v4_ref/encoding/encoding_dsv4.py, which renders messages (incl. tools + thinking mode)
    to a string that already carries the BOS token — hence add_special_tokens=False."""
    if not job.get("messages"):
        return job["promptIds"]
def _vendored(name):
    """Locate a vendored reference tree, in the repo AND on a deployed box.

    The repo keeps upstream trees under vendor/ so the engine directories hold only our code. A
    rented stage does NOT have that layout: deploy copies the modules flat into /root with the
    reference tree beside them, which is the whole reason the engines are flat modules. So try the
    sibling first (deployed), then the repo's vendor/. Resolving only one way silently breaks the
    other, and the box is the one that would not be caught by a test run."""
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (os.path.join(here, name),
                 os.path.join(here, os.pardir, os.pardir, "vendor", name)):
        if os.path.isdir(cand):
            return os.path.normpath(cand)
    return os.path.join(here, name)


    enc_dir = os.path.join(_vendored("deepseek_v4_ref"), "encoding")
    if enc_dir not in sys.path:
        sys.path.insert(0, enc_dir)
    from encoding_dsv4 import encode_messages
    text = encode_messages(job["messages"], "thinking" if job.get("thinking") else "chat",
                           reasoning_effort=job.get("reasoningEffort"))
    return tok.encode(text, add_special_tokens=False)


def _coord_cli(a):
    """The node-daemon serving entrypoint (mirrors shard.coordinate): dial the ring, read jobs as
    JSON lines on stdin, drive decode, and emit the SHARD_JOB_* stdout contract.

    THE PERSISTENT-COORDINATOR PATTERN. This is meant to be a LONG-LIVED process fed a STREAM of job
    lines and never sent EOF: the node daemon holds it open and writes each job as it arrives, so one
    warm coordinator serves the ring for its whole lifetime — no reconnect, no re-reset between jobs.
    A single job's fault keeps it alive (see the except below). And now, crucially, EOF is survivable
    too: when stdin closes this returns and its sockets drop, but the ring no longer cascades — the
    head re-accepts a reconnecting coordinator and the tail re-accepts its return channel (see
    _serve_forward / _serve_tail). So a bench that runs, exits, and re-runs reconnects to the SAME
    warm ring instead of paying a full ~25min re-warm each time — the biggest iteration-speed win on
    the V4 path. To stream jobs by hand rather than tear down between them, keep stdin open (e.g. feed
    a FIFO) rather than piping a here-doc that EOFs after the last line."""
    os.environ.setdefault("V4_DIR", a.dir)
    layer_count = ring_args(a.dir).n_layers                   # same view of the config the stages built
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.dir, trust_remote_code=True)
    except Exception as e:  # noqa: BLE001
        _emit("SHARD_JOB_FATAL", error=f"tokenizer load failed: {e}")
        return 1
    eos = tok.eos_token_id
    eos_ids = tuple(eos) if isinstance(eos, (list, tuple)) else ((eos,) if eos is not None else ())
    pipe, ret = connect_ring(a.head, a.tail, timeout=a.timeout, token=SWARM_TOKEN,
                             retry_s=a.connect_retry)
    # The COORDINATOR-side half of the lever audit, on stderr so the SHARD_JOB_* stdout contract the
    # node daemon parses stays clean. Its most valuable line is the WRONG PROCESS one: V4_LAZY_DRAFT
    # and V4_PIPELINED_SPEC are read HERE and nowhere else, and an operator who set them on the
    # stages instead has, until now, had no signal at all.
    v4_levers.report(side=v4_levers.COORDINATOR)
    _emit("SHARD_COORD_READY", head=a.head, tail=a.tail, receipts=a.receipts)
    audited = False

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            job = json.loads(line)
            job_id = job["jobId"]
        except (ValueError, KeyError) as e:
            _emit("SHARD_JOB_FATAL", error=f"unparseable job line: {e}")
            continue
        max_new = max(1, min(int(job.get("maxNew") or 512), 4096))
        _emit("SHARD_JOB_START", jobId=job_id, maxNew=job.get("maxNew"))
        state = {"n": 0, "t0": None, "tft": None}

        def _on_token(_tid, _job=job_id, _st=state):
            _st["n"] += 1
            if _st["tft"] is None:                            # first-token latency (prefill + one traversal)
                _st["tft"] = time.time() - _st["t0"]
            _emit("SHARD_JOB_TOKEN", jobId=_job, delta=_st["n"])
        try:
            prompt_ids = _encode_prompt(tok, job)
            state["t0"] = time.time()
            if job.get("dspark"):                             # V4's own trained speculator, on the tail
                # PIPELINED is opt-in per job or per process, and the serial path stays the default:
                # the two emit the same stream, but only one of them has been measured on a real ring.
                # The confidence gate trims the tail's OFFERED BLOCK LENGTH, which only the serial
                # path has -- pipelined streams one s=1 frame per position and never sends a block --
                # so the conf_* knobs are passed to the serial coordinator only. They are not
                # silently dropped: a job that asks for both is told, because a gate that does
                # nothing would read on the ring as "confidence gating did not help".
                pipelined = bool(V4_PIPELINED_SPEC or job.get("pipelined"))
                wants_conf = (job.get("confGate") if job.get("confGate") is not None
                              else V4_DSPARK_CONF_GATE)
                if pipelined and wants_conf:
                    raise ValueError(
                        "confGate/V4_DSPARK_CONF_GATE gates the serial DSpark path's block length; "
                        "the pipelined coordinator streams s=1 frames and has no block to trim. "
                        "Run one or the other, not both.")
                if pipelined:
                    r = coordinate_dspark_pipelined(
                        pipe, ret, prompt_ids, max_new, eos_ids=eos_ids,
                        nonce=job.get("nonce"), swarm_id=job.get("swarmId") or "swarm",
                        job_id=job_id, layer_count=layer_count, receipts=a.receipts,
                        timeout=a.timeout, on_token=_on_token)
                else:
                    r = coordinate_dspark(pipe, ret, prompt_ids, max_new, eos_ids=eos_ids,
                                          nonce=job.get("nonce"), swarm_id=job.get("swarmId") or "swarm",
                                          job_id=job_id, layer_count=layer_count, receipts=a.receipts,
                                          timeout=a.timeout, on_token=_on_token,
                                          conf_gate=job.get("confGate"),  # None -> V4_DSPARK_CONF_* env
                                          conf_thresh=job.get("confThresh"),
                                          conf_min=job.get("confMin"))
            elif job.get("spec"):                             # coordinator-side drafter (n-gram)
                r = coordinate_spec(pipe, ret, prompt_ids, max_new, eos_ids=eos_ids,
                                    nonce=job.get("nonce"), swarm_id=job.get("swarmId") or "swarm",
                                    job_id=job_id, layer_count=layer_count, receipts=a.receipts,
                                    timeout=a.timeout, on_token=_on_token,
                                    K=int(job.get("specK", 4)), ng=int(job.get("specNg", 3)))
            else:
                r = coordinate(pipe, ret, prompt_ids, max_new, eos_ids=eos_ids,
                               nonce=job.get("nonce"), swarm_id=job.get("swarmId") or "swarm",
                               job_id=job_id, layer_count=layer_count, receipts=a.receipts,
                               temp=float(job.get("temperature", 0.0)), timeout=a.timeout,
                               on_token=_on_token)
            elapsed = time.time() - state["t0"]
            ngen = len(r["tokens"])
            _emit("SHARD_JOB_DONE", jobId=job_id, ok=True,
                  response=tok.decode(r["tokens"], skip_special_tokens=True),
                  tokensGenerated=ngen,
                  tokPerSec=round(ngen / elapsed, 3) if elapsed > 0 else None,
                  firstTokenMs=round((state["tft"] or 0) * 1000, 1),
                  elapsedS=round(elapsed, 2),
                  spec=bool(job.get("spec") or job.get("dspark")), dspark=bool(job.get("dspark")),
                  g=r.get("g"), rounds=r.get("rounds"), acceptHist=r.get("accept_hist"),
                  receipts=[wire_receipt(rr) for rr in (r["receipts"] or [])],
                  receiptsOk=r["receipts_ok"], nonce=job.get("nonce"))
        except Exception as e:  # noqa: BLE001
            # Keep the persistent coordinator ALIVE on a single job fault (do not exit): exiting closes
            # the pipe to the head, which disconnects it and cascades the ring down. A fresh reset on
            # the next job re-inits every still-connected stage, so a transient hiccup costs one job,
            # not the warm ring.
            _emit("SHARD_JOB_FATAL", jobId=job_id, error=f"{type(e).__name__}: {e}")
            continue
        if not audited:
            # ONCE, after the first job COMPLETES and OUTSIDE its try. The coordinator levers are
            # ARGUMENTS to a loop, so only a finished round turns "requested" into an observed fact
            # (v4_levers.note); the startup audit could only report what the env resolved to.
            #
            # Both halves of the placement are load-bearing. Outside the try, because inside it a
            # STRICT finding would be caught as a job fault -- the finished generation discarded, a
            # FATAL emitted for a job that worked, `audited` never set, and the whole thing repeated
            # for every job after it, with no path that ever clears. And `audited` first, so even a
            # raise cannot leave the flag unset. Under STRICT this ends the coordinator loudly, once,
            # which is the honest response to "the round you just ran was not the one you asked for".
            audited = True
            v4_levers.report(side=v4_levers.COORDINATOR)
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("stage", help="serve one V4 layer block in the fire-forward ring")
    s.add_argument("--stage", type=int, required=True)
    s.add_argument("--nstages", type=int, required=True)
    s.add_argument("--lo", type=int, required=True)
    s.add_argument("--hi", type=int, required=True)
    s.add_argument("--port", type=int, default=ENG_IN)
    s.add_argument("--next", default=None, dest="next")
    s.add_argument("--ret-relay", default=None, dest="ret_relay",
                   help="multi-GPU tail box ingress only: loopback addr of the box tail's return "
                        "channel to bridge the coordinator-return to")
    s.add_argument("--dspark", action="store_true",
                   help="tail only: build the DSpark MTP speculator stages (step 4)")
    s.add_argument("--dir", default=os.environ.get("V4_DIR", "/root/v4"))
    s.add_argument("--device", default=None)
    s.add_argument("--bind", default=os.environ.get("M25_ENGINE_BIND", "127.0.0.1"))
    s.add_argument("--receipts", action="store_true")

    c = sub.add_parser("coord", help="drive jobs over a formed ring (stdin JSON lines)")
    c.add_argument("--head", default=f"127.0.0.1:{ENG_IN}")
    c.add_argument("--tail", default=f"127.0.0.1:{FWD_RET}")
    c.add_argument("--dir", default=os.environ.get("V4_DIR", "/root/v4"))
    c.add_argument("--receipts", action="store_true")
    c.add_argument("--timeout", type=int, default=600)
    c.add_argument("--connect-retry", type=int, default=300, dest="connect_retry")

    sub.add_parser("selftest", help="offline CPU full-ring parity proof (no GPU, no spend)")
    sub.add_parser("selftest-relay",
                   help="offline CPU parity proof with a MULTI-GPU tail box (return relay path)")
    a = ap.parse_args()

    if a.cmd == "stage":
        serve_stage(a.stage, a.nstages, a.lo, a.hi, a.port, nxt=a.next, ckpt_dir=a.dir,
                    device=a.device, receipts=(a.receipts or RECEIPTS), bind=a.bind,
                    ret_relay=a.ret_relay, dspark=a.dspark)
    elif a.cmd == "coord":
        sys.exit(_coord_cli(a))
    elif a.cmd == "selftest":
        selftest()
    elif a.cmd == "selftest-relay":
        selftest(nstages=3, tail_box_g=2)


if __name__ == "__main__":
    main()
