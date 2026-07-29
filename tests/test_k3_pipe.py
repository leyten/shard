"""Kimi-K3 full-ring pipe: the whole pipeline is bit-identical to Moonshot's model, end to end.

tests/test_k3_stage.py proved a Stage layer-range reproduces the reference. This file extends that
proof THROUGH the real fire-forward ring: a head that embeds token ids, M middle stages, a tail that
runs final-norm + output-AttnRes + lm_head and SAMPLES, and a weightless coordinator that drives
greedy decode over real shard.transport sockets — carrying BOTH boundary tensors (h, block_residual)
across every hop. The bar is BIT-IDENTICAL sampled token ids against KimiLinearForCausalLM's own
greedy decode. Plus: receipts settle (C10 wire_receipt) over the full ring, and reset gives a warm
ring a cold ring's answer.

Everything runs on CPU against a tiny random-init K3 (mixed KDA/MLA, AttnRes append mid-ring) — no
GPU, no network beyond localhost, no weight downloads.

Run: python3 -m pytest tests/test_k3_pipe.py -q
"""
import os
import socket
import sys
import threading
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "phase0"))

torch = pytest.importorskip("torch")
pytest.importorskip("einops")                              # the vendored reference imports it at module scope
pytest.importorskip("safetensors.torch")
K3 = pytest.importorskip("k3_stage")
KP = pytest.importorskip("k3_pipe")

try:
    from receipt import ReceiptError, verify_coverage, wire_receipt  # noqa: E402
except ImportError:
    from shard.receipt import ReceiptError, verify_coverage, wire_receipt  # noqa: E402

PROMPT = [3, 9, 17, 2, 41]
NEW = 5


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


class _Ring:
    """A localhost k3_pipe ring over a tiny checkpoint: N stage-server threads + a coordinator dial.
    `coordinate()` reuses the SAME warm ring across calls (each call resets), so a second job on the
    ring is the reset/warm path."""

    def __init__(self, ckpt_dir, cfg, ranges, receipts=False, tail_box_g=1):
        self.ranges = ranges
        self.n = len(ranges)
        self.layer_count = cfg.num_hidden_layers
        ports = _free_ports(self.n)
        relay_i = self.n - tail_box_g                          # ingress of a multi-GPU tail box (>1)
        events = [threading.Event() for _ in range(self.n)]
        self.threads = []
        for i, (lo, hi) in enumerate(ranges):
            nxt = None if i == self.n - 1 else f"127.0.0.1:{ports[i + 1]}"
            # a G>1 tail box: its ingress bridges the coordinator-return to the box tail (last stage)
            ret_relay = f"127.0.0.1:{ports[-1]}" if (tail_box_g > 1 and i == relay_i) else None
            t = threading.Thread(target=KP.serve_stage, kwargs=dict(
                stage=i, nstages=self.n, lo=lo, hi=hi, port=ports[i], nxt=nxt, ckpt_dir=ckpt_dir,
                device="cpu", receipts=receipts, key_path=f"{ckpt_dir}/s{i}.key",
                ret_relay=ret_relay, ready=events[i]), daemon=True)
            t.start()
            self.threads.append(t)
        for e in events:
            assert e.wait(30), "a stage never came up"
        # the coordinator-return terminates at the tail box INGRESS (the relay), not the global tail
        tail_port = ports[relay_i] if tail_box_g > 1 else ports[-1]
        self.pipe, self.ret = KP.connect_ring(f"127.0.0.1:{ports[0]}",
                                               f"127.0.0.1:{tail_port}", timeout=60)
        self.receipts = receipts

    def coordinate(self, prompt, max_new, nonce=None, temp=0.0, seed=0):
        return KP.coordinate(self.pipe, self.ret, prompt, max_new, nonce=nonce,
                             receipts=self.receipts, layer_count=self.layer_count, temp=temp,
                             seed=seed, timeout=60)

    def close(self):
        try:
            KP.send_msg(self.pipe, {"op": "stop"})
        except OSError:
            pass
        for t in self.threads:
            t.join(timeout=5)


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    """(checkpoint dir, config, reference model) — one 6-layer random-init K3, built once."""
    cfg = KP._tiny_config(6)
    d = str(tmp_path_factory.mktemp("k3pipe"))
    d, model = KP.write_tiny_checkpoint(d, cfg)
    os.environ["K3_DIR"] = d
    return d, cfg, model


# ── the headline: the full ring reproduces Moonshot's own whole-model greedy decode ────────────────

@pytest.mark.parametrize("ranges", [
    [(0, 2), (2, 4), (4, 6)],          # 3 stages: head embeds, KDA/MLA middle, MLA tail
    [(0, 1), (1, 3), (3, 5), (5, 6)],  # 4 stages: dense head, mixed middles, MLA tail
    [(0, 3), (3, 6)],                  # 2 stages, split mid-block (crosses a fat block_residual)
    [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)],   # one layer per stage — every hop is a boundary
])
def test_full_ring_greedy_matches_reference_whole_model(tiny, ranges):
    """THE proof: head-embed + M stages + tail-lm_head, (h, block_residual) + per-stage state crossing
    the real transport, sampled token-for-token against KimiLinearForCausalLM. Bit-identical."""
    d, cfg, model = tiny
    ref = KP._reference_tokens(cfg, model, PROMPT, NEW)
    ring = _Ring(d, cfg, ranges)
    try:
        got = ring.coordinate(list(PROMPT), NEW)["tokens"]
    finally:
        ring.close()
    assert got == ref, f"{ranges}: ring {got} != ref {ref}"


def test_streamed_tokens_arrive_one_per_step(tiny):
    """The coordinator streams a token per ring round (the SHARD_JOB_TOKEN cadence)."""
    d, cfg, model = tiny
    ring = _Ring(d, cfg, [(0, 3), (3, 6)])
    seen = []
    try:
        KP.coordinate(ring.pipe, ring.ret, list(PROMPT), NEW, timeout=60,
                      on_token=seen.append)
    finally:
        ring.close()
    assert len(seen) == NEW


# ── multi-GPU tail box: the return relay closes the last-GPU -> coordinator leg ─────────────────────

@pytest.mark.parametrize("ranges,tail_box_g", [
    ([(0, 2), (2, 4), (4, 6)], 2),          # box0 head (1 GPU), box1 tail (2 GPUs): ingress relay + box tail
    ([(0, 1), (1, 3), (3, 5), (5, 6)], 3),  # box1 tail (3 GPUs): ingress relay + a middle + box tail
    ([(0, 3), (3, 6)], 2),                   # a single 2-GPU box that is BOTH head and tail
])
def test_multigpu_tail_box_relays_return_and_matches_reference(tiny, ranges, tail_box_g):
    """The return path for a G>1 TAIL box: the coordinator-return lands at the box ingress (ENG_IN),
    the token is produced on the box's last-GPU stage, and the ingress bridges them over loopback.
    The ring must stay BIT-IDENTICAL to Moonshot's whole-model greedy decode — proving the relay is
    a transparent tunnel, not a numerics change. (The box tail's serve loop is untouched; only the
    ingress relays.)"""
    d, cfg, model = tiny
    ref = KP._reference_tokens(cfg, model, PROMPT, NEW)
    ring = _Ring(d, cfg, ranges, tail_box_g=tail_box_g)
    try:
        got = ring.coordinate(list(PROMPT), NEW)["tokens"]
    finally:
        ring.close()
    assert got == ref, f"{ranges} (tail_box_g={tail_box_g}): ring {got} != ref {ref}"


def test_multigpu_tail_receipts_settle_through_the_relay(tiny):
    """Receipts sweep through the relay: the ingress appends its OWN receipt down the loopback chain
    (it is a compute stage), the box tail aggregates the set and sends it back on the return channel,
    and the ingress pumps that set out to the coordinator. Coverage still tiles every layer and the
    chain settles under the nonce."""
    d, cfg, model = tiny
    nonce = "relay-settle-nonce"
    ring = _Ring(d, cfg, [(0, 2), (2, 4), (4, 6)], receipts=True, tail_box_g=2)
    try:
        r = ring.coordinate(list(PROMPT), NEW, nonce=nonce)
    finally:
        ring.close()
    assert r["receipts_ok"] is True
    assert len(r["receipts"]) == 3, "ingress-relay stage's own receipt must be in the settled set"
    wired = [wire_receipt(rr) for rr in r["receipts"]]
    verify_coverage(wired, cfg.num_hidden_layers, expected_nonce=nonce, check_chain=True)


# ── receipts settle over the full ring (C10) ───────────────────────────────────────────────────────

def test_full_ring_receipts_settle_and_c10(tiny):
    """Every stage signs its two-tensor boundary hash-chain over the job; the coordinator sweeps the
    ring once and settles. wire_receipt (C10) strips the post-sign `stage` debug tag, so the RAW
    receipts fail verify (the tag is in the signed preimage) and the WIRE receipts pass."""
    d, cfg, model = tiny
    nonce = "settle-nonce-abc"
    ring = _Ring(d, cfg, [(0, 2), (2, 4), (4, 6)], receipts=True)
    try:
        r = ring.coordinate(list(PROMPT), NEW, nonce=nonce)
    finally:
        ring.close()

    assert r["receipts_ok"] is True, "the coordinator's own coverage verify must pass"
    raw = r["receipts"]
    assert len(raw) == 3 and all("stage" in rr for rr in raw), "each stage tags its receipt post-sign"

    # C10 red: the un-stripped debug tag is in the signed body -> verify fails closed
    with pytest.raises(ReceiptError):
        verify_coverage(raw, cfg.num_hidden_layers, expected_nonce=nonce, check_chain=True)
    # C10 green: stripped, the full ring settles — coverage tiles [0:L), nonce matches, chain holds
    wired = [wire_receipt(rr) for rr in raw]
    verify_coverage(wired, cfg.num_hidden_layers, expected_nonce=nonce, check_chain=True)


def test_receipts_bind_to_the_settlement_nonce(tiny):
    """The coordinator threads the settlement nonce into the reset; every stage signs it, so a
    receipt set verified against the WRONG nonce fails closed (anti-replay)."""
    d, cfg, model = tiny
    ring = _Ring(d, cfg, [(0, 3), (3, 6)], receipts=True)
    try:
        r = ring.coordinate(list(PROMPT), NEW, nonce="the-real-nonce")
    finally:
        ring.close()
    wired = [wire_receipt(rr) for rr in r["receipts"]]
    verify_coverage(wired, cfg.num_hidden_layers, expected_nonce="the-real-nonce", check_chain=True)
    with pytest.raises(ReceiptError, match="nonce"):
        verify_coverage(wired, cfg.num_hidden_layers, expected_nonce="a-different-nonce",
                        check_chain=True)


def test_receipt_chain_attests_the_two_tensor_boundary(tiny):
    """A K3 stage hashes (h, block_residual) at input and output, so the receipt chain
    (out_root == in_root across the ring) proves the AttnRes stack crossed intact — a property
    M2.5's single-tensor chain does not have. The chain must hold on the lossless localhost wire."""
    d, cfg, model = tiny
    ring = _Ring(d, cfg, [(0, 2), (2, 4), (4, 6)], receipts=True)
    try:
        r = ring.coordinate(list(PROMPT), NEW, nonce="n")
    finally:
        ring.close()
    wired = sorted((wire_receipt(rr) for rr in r["receipts"]), key=lambda x: x["layer_start"])
    for a, b in zip(wired, wired[1:]):
        assert a["out_root"] == b["in_root"], "boundary hash-chain broke across a hop"


# ── reset: a warm ring answers like a cold one ─────────────────────────────────────────────────────

def test_reset_gives_a_warm_ring_a_cold_rings_answer(tiny):
    """A second, different job on the same warm ring must equal that job on a fresh ring — the reset
    at the head of coordinate() drops all per-stage MLA KV + KDA recurrent/conv state."""
    d, cfg, model = tiny
    other = [7, 7, 7, 1, 2]
    ref = KP._reference_tokens(cfg, model, PROMPT, NEW)
    ring = _Ring(d, cfg, [(0, 2), (2, 4), (4, 6)])
    try:
        ring.coordinate(other, NEW)                          # a first, different job warms the state
        warm = ring.coordinate(list(PROMPT), NEW)["tokens"]  # reset + the job we can check
    finally:
        ring.close()
    assert warm == ref, f"warm-after-reset {warm} != cold {ref}"


# ── the pure builders: tiling + the payload-agnostic sidecar wiring ────────────────────────────────

def test_even_tiling_covers_all_93_layers_contiguously():
    ranges = KP.even_tiling(93, 19)
    assert len(ranges) == 19
    assert ranges[0][0] == 0 and ranges[-1][1] == 93
    for (a_lo, a_hi), (b_lo, b_hi) in zip(ranges, ranges[1:]):
        assert a_hi == b_lo, "tiling has a gap or overlap"
    assert max(hi - lo for lo, hi in ranges) - min(hi - lo for lo, hi in ranges) <= 1, "unbalanced"


def test_even_tiling_rejects_impossible_splits():
    with pytest.raises(ValueError):
        KP.even_tiling(3, 4)


# ── the forward-dial retry window: a stage keeps dialing a neighbour that is still loading ──────────

def test_dial_retry_window_covers_a_slow_stage_load_and_is_configurable():
    """Bug 1 (live K3 ring 2026-07-29): the old fixed ~30s dial window gave up on a 14-stage ring
    mid-launch. The window now derives from the stage `timeout` (the 600s ceiling a stage already
    tolerates on a frame), so it comfortably covers a neighbour taking MINUTES to load — and
    K3_DIAL_RETRY_S overrides it for an even larger ring."""
    assert KP._dial_window(600) == 600.0                   # keyed to timeout, not a magic number
    assert KP._dial_window(600) >= 300                     # covers a stage that is minutes from ready
    assert KP._dial_window(600) > 30                        # strictly beyond the old fixed window
    old = KP.DIAL_RETRY_S
    try:
        KP.DIAL_RETRY_S = 1200.0
        assert KP._dial_window(600) == 1200.0               # explicit override wins
    finally:
        KP.DIAL_RETRY_S = old


def test_dial_gives_up_on_a_time_window_not_a_fixed_try_count():
    """_dial loops on a wall-clock deadline: a small retry_s raises after ~that window (proving it is
    not the old fixed 120x0.25s count), so a black-holing peer can't be dialed forever — while the
    LARGE default window keeps a boot-time dialer retrying through a slow neighbour's whole load
    instead of aborting the launch."""
    port = _free_ports(1)[0]                                # a free, UNBOUND port -> connect refused
    t0 = time.time()
    with pytest.raises(RuntimeError):
        KP._dial("127.0.0.1", port, timeout=5, retry_s=0.5)
    dt = time.time() - t0
    assert 0.4 <= dt < 5, f"dial window not honored: {dt:.2f}s"


def test_ring_sidecar_spec_is_the_direct_return_topology():
    """Every non-tail stage forwards the ring; the HEAD also forwards the coordinator-return to the
    tail; each inbound stage admits only its predecessor (the tail also admits the head, whose
    return stream arrives as its RemotePeer). Byte-shape-identical to m25_scatter_pipe."""
    m = [f"/ip4/10.0.0.{k}/tcp/29600/p2p/PEER{k}" for k in range(3)]
    inb0, fwd0, allow0 = KP.ring_sidecar_spec(0, 3, m)
    assert inb0 == "" and allow0 is None                     # head: no inbound, open
    assert f"127.0.0.1:{KP.FWD_RING}={m[1]}" in fwd0         # ring -> s1
    assert f"127.0.0.1:{KP.FWD_RET}={m[2]}" in fwd0          # return -> tail
    inb1, fwd1, allow1 = KP.ring_sidecar_spec(1, 3, m)
    assert inb1 == f"127.0.0.1:{KP.ENG_IN}"
    assert fwd1 == [f"127.0.0.1:{KP.FWD_RING}={m[2]}"] and allow1 == ["PEER0"]
    inb2, fwd2, allow2 = KP.ring_sidecar_spec(2, 3, m)
    assert inb2 == f"127.0.0.1:{KP.ENG_IN}" and fwd2 == []   # tail: inbound only, no forward
    assert allow2 == ["PEER1", "PEER0"]                      # predecessor + head (the return stream)


def test_ring_sidecar_spec_honors_an_explicit_return_maddr():
    m = [f"/ip4/10.0.0.{k}/tcp/29600/p2p/PEER{k}" for k in range(3)]
    _, fwd0, _ = KP.ring_sidecar_spec(0, 3, m, ret_maddr="/relay/circuit/p2p/PEER2")
    assert f"127.0.0.1:{KP.FWD_RET}=/relay/circuit/p2p/PEER2" in fwd0


def test_plan_layer_ranges_tiles_the_93_layer_ring_via_k3_profile():
    """The 93-layer tiling comes from shard.plan.plan_ring under the K3 engine profile — k3_pipe
    never re-derives the VRAM model. A pool of 96 GB cards places a head-first ring covering [0:93)."""
    n = 22                                                   # ~5 K3 layers/96GB card => >=19 needed for 93
    nodes = [{"id": f"box{i}", "free_vram_mb": 96000.0, "subnet": f"10.{i}.0.0/24",
              "total_vram_mb": 98304.0} for i in range(n)]
    rtt = [[0.0 if i == j else 12.0 for j in range(n)] for i in range(n)]
    stages = KP.plan_layer_ranges(nodes, rtt)
    assert stages is not None, "a 22x96GB pool must be able to hold K3"
    stages = sorted(stages, key=lambda s: s["index"])
    assert stages[0]["head"] and stages[0]["lo"] == 0
    assert stages[-1]["tail"] and stages[-1]["hi"] == 93
    cursor = 0
    for s in stages:
        assert s["lo"] == cursor, "planned ranges must tile with no gap/overlap"
        cursor = s["hi"]
        assert s["layers"] <= 5, "a 96 GB K3 stage holds at most 5 layers (G0-proven)"
    assert cursor == 93
