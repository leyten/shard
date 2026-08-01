"""DeepSeek-V4-Flash ring: the PROTOCOL under phase0/v4_pipe.py, then the whole ring end to end.

Sections 1-8 have no model in them at all. They drive the real frame codec, receipt chain, socket
lifecycle and coordinator loops against FAKE stages that do trivial arithmetic on tensors — so they
run on any box, in seconds, with no weights and no checkpoint:

  * the step frame carries h [b,s,4,dim] AND the routing ids, bit-exact, bf16 and fp8,
  * the receipt hash-chain over that two-part payload closes across a real socket (out_root of one
    stage == in_root of the next), fp8 wire included — the fp8 path hashes the PACKED wire bytes on
    both sides precisely so the chain survives a lossy activation,
  * the public engine port survives a scanner racing the real predecessor for accept(),
  * a forward leg that died while the ring was idle is rebuilt at job open (#156) — the failure that
    cost the 2026-07-29 capstone ring every one of its jobs,
  * the keep-warm noop never interleaves with a real send on the same socket,
  * the SPEC accept/commit/rewind arithmetic, against a scripted tail,
  * the DSPARK round, likewise: who drafts, which chunk goes on the wire, and the tripwire that fires
    when the tail's accept length disagrees with the coordinator's.

Section 9 is the headline, and it is the same proof `python3 phase0/v4_pipe.py selftest` prints:
real v4_stage stages over a real tiny checkpoint, decoding through real sockets, BIT-IDENTICAL to
the vendored reference Transformer's own greedy decode — greedy, coordinator-drafted (rejecting AND
accepting) and DSpark-drafted, all three against that one bar. It imports v4_ref_cpu/v4_stage inside
its fixture, so a box without them skips that section and still runs everything above it.

Run: python3 -m pytest tests/test_v4_pipe.py -q
"""
import os
import select
import socket
import struct
import sys
import threading
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "phase0"))

torch = pytest.importorskip("torch")
VP = pytest.importorskip("v4_pipe")          # protocol only — v4_stage/v4_ref_cpu stay unimported

send_msg, recv_msg = VP.send_msg, VP.recv_msg

try:
    # ReceiptError MUST come from the same module as verify_coverage: in a full-suite process an
    # earlier test's sys.path setup makes the flat `receipt` importable, so `shard.receipt` and
    # `receipt` are two live module objects for one file — and an except clause holding the OTHER
    # module's ReceiptError silently stops catching (CI caught this; a file-scoped run cannot).
    from receipt import (load_or_make_node_key, verify_coverage, wire_receipt, ReceiptSigner,
                         ReceiptError)
except ImportError:
    from shard.receipt import (load_or_make_node_key, verify_coverage, wire_receipt, ReceiptSigner,
                               ReceiptError)

B, S, HC, D = 1, 3, 4, 8                      # a V4 boundary payload in miniature: [b, s, hc_mult, dim]


def _h(seed=0.0):
    """A boundary tensor with distinguishable, outlier-bearing values (fp8 e4m3 has ~2 decimal digits
    of mantissa, so equal-magnitude noise would make a scale bug invisible)."""
    t = torch.arange(B * S * HC * D, dtype=torch.float32).reshape(B, S, HC, D) / 10.0 + seed
    t[0, 0, 0, 0] = 40.0                      # one big channel: the per-tensor scale has to cover it
    return t.to(torch.bfloat16)


def _ids(n=S):
    return torch.arange(100, 100 + n, dtype=torch.int64).reshape(B, n)


def _pair(timeout=5):
    a, b = socket.socketpair()
    a.settimeout(timeout)
    b.settimeout(timeout)
    return a, b


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def _drain(sock, seconds):
    """Every frame arriving on sock within `seconds` (select-polled, never blocks past the end)."""
    out, end = [], time.monotonic() + seconds
    while time.monotonic() < end:
        if select.select([sock], [], [], 0.01)[0]:
            out.append(recv_msg(sock))
    return out


@pytest.fixture
def fp8(request, monkeypatch):
    """Parametrized fp8 wire: the same assertions must hold with V4_FP8_WIRE on and off."""
    monkeypatch.setattr(VP, "V4_FP8_WIRE", request.param)
    return request.param


# ── 1. the step frame: h + the routing ids, over the real transport ────────────────────────────────

@pytest.mark.parametrize("fp8", [False, True], indirect=True)
def test_step_frame_roundtrip(fp8):
    """_make_step_frame -> wire -> _recv_hids. bf16 is BIT-exact; fp8 is within one per-tensor-scale
    quantization step. In BOTH modes the ids come back bit-exact int64 (they are indices into a hash
    function and a vocab table — a quantized token id is not a token id), and the sender's out_root
    digest equals the receiver's in_root digest over the same bytes."""
    a, b = _pair()
    h, ids = _h(), _ids()
    signer = object()                          # only its not-None-ness is read by the helpers
    try:
        frame, out_b = VP._make_step_frame(h, ids, 7, signer)
        send_msg(a, frame)
        got = recv_msg(b)
        h2, ids2, in_b = VP._recv_hids(got, signer)
    finally:
        a.close()
        b.close()

    assert torch.equal(ids2, ids) and ids2.dtype == torch.int64, "routing ids must cross untouched"
    assert int(got["start_pos"]) == 7
    assert in_b == out_b, "receipt digest differs across the hop — the chain would break"
    if fp8:
        assert got["h"].dtype == torch.float8_e4m3fn and "h8" in got
        err = (h2.float() - h.float()).abs().max().item()
        assert err <= 40.0 / 448.0 * 8, f"fp8 wire error {err} beyond one per-tensor scale step"
        assert h2.dtype == torch.bfloat16
    else:
        assert "h8" not in got
        assert torch.equal(h2, h), "bf16 wire must be bit-exact"


def test_step_frame_hashes_the_ids_not_just_the_hidden():
    """The V4 delta over K3/M2.5: rewriting the ids alone — which silently re-routes every hash MoE
    layer downstream — must change the payload digest. A hidden-state-only chain could not see it."""
    h, ids = _h(), _ids()
    other = ids.clone()
    other[0, 0] += 1
    assert VP._payload_bytes(h, ids) != VP._payload_bytes(h, other)
    assert VP._payload_bytes(h, ids) == VP._payload_bytes(h, ids.clone())


# ── 2. the receipt chain over a real socket, with fake stages ──────────────────────────────────────

class _FakeStage:
    """Payload plumbing, not a model: forward() perturbs h deterministically, embed() makes an h out
    of ids, logits_all() makes a [b,s,vocab] whose argmax is a function of the ids. Enough to prove
    frames, digests and receipts move correctly; nothing here is DeepSeek's math."""

    def __init__(self, k=1.0, vocab=32):
        self.k = k
        self.vocab = vocab
        self.n = 0

    def reset(self):
        self.n = 0

    def embed(self, ids):
        return (ids.float().unsqueeze(-1).unsqueeze(-1).expand(-1, -1, HC, D) / 100.0).to(torch.bfloat16)

    def forward(self, h, ids, start_pos):
        self.n += 1
        return (h.float() + self.k).to(torch.bfloat16)

    def logits_all(self, h):
        lg = torch.zeros(h.shape[0], h.shape[1], self.vocab)
        lg[..., 3] = 1.0
        return lg


def _run_chain(tmp_path, nstages=3, layers=9, chunks=2):
    """Drive `chunks` step frames through a chain of fake stages linked by socketpairs, each stage
    signing the real ReceiptSigner chain over the real payload digests. Returns the receipt list in
    ring order, exactly as the coordinator's sweep would collect it."""
    ranges = VP.even_tiling(layers, nstages)
    stages = [_FakeStage(k=i + 1.0) for i in range(nstages)]
    signers = [ReceiptSigner(load_or_make_node_key(str(tmp_path / f"s{i}.key")), "swarm", "job",
                             lo, hi, nonce="n0") for i, (lo, hi) in enumerate(ranges)]
    links = [_pair() for _ in range(nstages)]      # link i: stage i-1 (or the coordinator) -> stage i

    for c in range(chunks):
        ids = _ids()
        send_msg(links[0][0], {"op": "step", "ids": ids.tolist(), "start_pos": c})
        for i, st in enumerate(stages):
            msg = recv_msg(links[i][1])
            if i == 0:                             # head: embed, then hash what it will forward
                ids_i = VP._ids_tensor(msg["ids"])
                h = st.embed(ids_i)
                in_b = VP._payload_bytes(h, ids_i)
            else:
                h, ids_i, in_b = VP._recv_hids(msg, signers[i])
            h = st.forward(h, ids_i, int(msg["start_pos"]))
            if i == nstages - 1:                   # tail: no forward leg, hash the plain payload
                signers[i].observe(in_b, VP._payload_bytes(h, ids_i))
            else:
                frame, out_b = VP._make_step_frame(h, ids_i, int(msg["start_pos"]), signers[i])
                signers[i].observe(in_b, out_b)
                send_msg(links[i + 1][0], frame)
    for a, b in links:
        a.close()
        b.close()
    return [{"stage": i, **s.finalize()} for i, s in enumerate(signers)]


@pytest.mark.parametrize("fp8", [False, True], indirect=True)
def test_receipt_chain_over_socketpair(tmp_path, fp8):
    """out_root of stage k == in_root of stage k+1 across the whole ring, and the set settles under
    verify_coverage (tiles every layer, carries the nonce, chain intact).

    fp8 ON is the interesting half: the activation is LOSSY, so the chain can only hold because both
    sides hash the same PACKED wire bytes (sender hashes what it packs, receiver hashes what it got)
    rather than the pre/post-quantization tensors."""
    recs = _run_chain(tmp_path)
    wired = [wire_receipt(r) for r in recs]
    for a, b in zip(wired, wired[1:]):
        assert a["out_root"] == b["in_root"], f"chain broke at [{a['layer_start']}:{a['layer_end']}]"
    verify_coverage(wired, 9, expected_nonce="n0", check_chain=True)
    assert all(r["n_chunks"] == 2 for r in wired), "every stage must attest every chunk"


# ── 3. the public engine port survives a scanner ───────────────────────────────────────────────────

def test_accept_pred_drops_scanner(monkeypatch):
    """A scanner racing the real predecessor for accept(): its HTTP probe's first 8 bytes parse as a
    preposterous frame length, which without the guard crashes the stage and cascades the ring.
    _accept_pred must drop it and still accept the predecessor that connects afterwards."""
    monkeypatch.setattr(VP, "SWARM_TOKEN", None)
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    srv.listen(4)
    port = srv.getsockname()[1]
    box = {}

    t = threading.Thread(target=lambda: box.update(zip(("conn", "queued"), VP._accept_pred(srv, 5))),
                         daemon=True)
    t.start()

    scanner = socket.create_connection(("127.0.0.1", port), timeout=5)
    scanner.sendall(b"GET / HTTP/1.1\r\nHost: x\r\n\r\n")     # unparseable as a frame
    time.sleep(0.2)
    pred = socket.create_connection(("127.0.0.1", port), timeout=5)
    send_msg(pred, {"op": "step", "ids": [[1, 2]], "start_pos": 0})   # token-less: first frame is job data
    t.join(10)
    assert not t.is_alive(), "_accept_pred never returned — the scanner wedged it"
    assert box["queued"]["op"] == "step", "the predecessor's first frame must be queued, not lost"
    for s in (scanner, pred, box["conn"], srv):
        s.close()


def test_accept_pred_token_mode_requires_the_greeting(monkeypatch):
    """C2 token mode: an inbound that parses but carries no valid hello_pred is dropped too, so a
    stage never adopts a foreign connection on an internet-facing port."""
    monkeypatch.setattr(VP, "SWARM_TOKEN", "f" * 32)
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    srv.listen(4)
    port = srv.getsockname()[1]
    box = {}
    t = threading.Thread(target=lambda: box.update(zip(("conn", "queued"), VP._accept_pred(srv, 5))),
                         daemon=True)
    t.start()

    stray = socket.create_connection(("127.0.0.1", port), timeout=5)
    send_msg(stray, {"op": "step", "ids": [[1]], "start_pos": 0})      # valid frame, no greeting
    time.sleep(0.2)
    pred = socket.create_connection(("127.0.0.1", port), timeout=5)
    send_msg(pred, {"op": "hello_pred", "token": "f" * 32})
    t.join(10)
    assert not t.is_alive() and box["queued"] is None
    for s in (stray, pred, box["conn"], srv):
        s.close()


def test_tail_bringup_classifies_both_inbound_streams(monkeypatch):
    """The tail holds TWO streams on its one engine port. The coordinator-return greets immediately;
    the predecessor stays SILENT until the first job frame flows the ring — so bringup must select()
    rather than block-recv, or the tail reads the silent predecessor while the coordinator waits for
    its ret_ok and the ring deadlocks at formation."""
    monkeypatch.setattr(VP, "SWARM_TOKEN", None)
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    srv.listen(4)
    port = srv.getsockname()[1]
    box = {}
    t = threading.Thread(target=lambda: box.update(zip(("ret", "pred", "queued"),
                                                       VP._tail_bringup(srv, 5))), daemon=True)
    t.start()

    pred = socket.create_connection(("127.0.0.1", port), timeout=5)   # connects first, says nothing
    time.sleep(0.2)
    ret = socket.create_connection(("127.0.0.1", port), timeout=5)
    ret.settimeout(5)
    send_msg(ret, {"op": "hello_return"})
    assert recv_msg(ret) == "ret_ok", "the coordinator's return must be acked while pred is silent"
    send_msg(pred, {"op": "step", "ids": [[5]], "start_pos": 0})
    t.join(10)
    assert not t.is_alive()
    assert box["queued"]["op"] == "step"
    for s in (pred, ret, box["ret"], box["pred"], srv):
        s.close()


# ── 4. #156: a forward leg that died while the ring was idle ───────────────────────────────────────

def test_fwd_open_rebuilds_dead_leg():
    """The capstone-ring failure at helper granularity. The stage's forward leg is closed under it
    while the ring is idle; nothing reads a write-only leg, so the RESET is the write that discovers
    it — and losing a reset is unrecoverable (the coordinator reads the TAIL, so no error can reach
    it). _fwd_open must rebuild the leg and deliver the frame EXACTLY once."""
    port = _free_port()
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", port))
    srv.listen(4)
    got = []
    stop = threading.Event()

    def serve():
        first = True
        while not stop.is_set():
            try:
                c, _ = srv.accept()
            except OSError:
                return
            if first:                                   # the sidecar that accepted, failed to reach
                first = False                           # the far engine, and hung up
                c.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
                c.close()                               # RST, so the engine's next write fails on
                continue                                # the spot rather than one frame later
            c.settimeout(5)
            try:
                while True:
                    got.append(recv_msg(c))
            except (OSError, EOFError):
                pass

    t = threading.Thread(target=serve, daemon=True)
    t.start()

    dead = VP._dial("127.0.0.1", port, timeout=5)        # the leg dialed at launch...
    # ...which the "sidecar" then dropped. Wait for the RST to actually LAND before writing: a
    # pending socket error makes the fd readable, and only then is the next send guaranteed to fail.
    # Without this the test races the kernel and occasionally writes into a leg that still looks
    # alive — which is the same "healthy socket, dead peer" illusion the fix exists for, just on the
    # test's side of the clock.
    deadline = time.time() + 5
    while not select.select([dead], [], [], 0.05)[0] and time.time() < deadline:
        pass
    kw = VP._KeepWarm(dead)
    reset = {"op": "reset", "swarm_id": "s", "job_id": "j", "nonce": "n"}
    new_sock = VP._fwd_open(kw, f"127.0.0.1:{port}", 5, reset, "[s0]")
    assert new_sock is not dead, "the dead leg must be replaced, not reused"
    VP._fwd_open(kw, f"127.0.0.1:{port}", 5, {"op": "receipt", "receipts": []}, "[s0]")

    deadline = time.time() + 5
    while len(got) < 2 and time.time() < deadline:
        time.sleep(0.02)
    stop.set()
    for s in (new_sock, srv):
        s.close()
    assert [m["op"] for m in got] == ["reset", "receipt"], f"delivery not exactly-once: {got}"
    assert got[0]["nonce"] == "n", "the rebuilt leg must carry the ORIGINAL reset, not a stub"


def test_fwd_open_is_a_passthrough_on_a_healthy_leg():
    """No rebuild, no extra frame, when the leg is fine — the retry is a recovery path, not a policy."""
    a, b = _pair()
    kw = VP._KeepWarm(a)
    same = VP._fwd_open(kw, "127.0.0.1:1", 1, {"op": "reset"}, "[s0]")
    assert same is a
    assert recv_msg(b) == {"op": "reset"}
    a.close()
    b.close()


def test_fwd_open_reraises_when_there_is_no_address_to_redial():
    """An in-process ring (selftest, tests) has no --next to re-dial; the fault must surface to the
    caller's supervision rather than being swallowed into a silent no-op."""
    a, b = _pair()
    b.close()
    a.close()
    kw = VP._KeepWarm(a)
    with pytest.raises(OSError):
        VP._fwd_open(kw, None, 1, {"op": "reset"}, "[s0]")


# ── 5-6. keeping an idle leg alive: the accept-wait ticker and the cwnd keep-warm ──────────────────

def test_warm_until_accept_ticks_and_stops():
    """A stage dials its successor at launch, then blocks in accept() waiting for its predecessor —
    which does not connect until a job flows down from the coordinator. Live, WAN paths dropped that
    idle leg in ~50s, so the first forward send crashed the stage. Noops must flow from the moment of
    dial, and MUST stop before the forward loop touches the socket (no concurrent send)."""
    a, b = _pair()
    stop = VP._warm_until_accept(a, period=0.02)
    frames = _drain(b, 0.15)
    stop()
    time.sleep(0.05)
    after = _drain(b, 0.1)
    a.close()
    b.close()
    assert len(frames) >= 2 and all(f == {"op": "noop"} for f in frames), frames
    assert after == [], "the ticker kept sending after stop() — it would race the forward loop"


def test_warm_until_accept_is_a_noop_without_a_leg():
    assert VP._warm_until_accept(None)() is None       # the tail has no forward leg


def test_keepwarm_lock_discipline(monkeypatch):
    """Real sends and the keep-warm noop share ONE lock: without it two threads' sendall() calls
    interleave partial frames on the same socket and the codec explodes / payloads scramble. The
    noop's acquire is NON-BLOCKING, so a held lock (a real send in flight = the leg is not idle)
    skips the tick instead of queueing behind it."""
    monkeypatch.setattr(VP, "V4_KEEPWARM", True)
    monkeypatch.setattr(VP, "V4_KEEPWARM_MS", 1.0)
    N = 120
    a, b = _pair(timeout=10)
    got, noops, err = [], [0], []

    def rx():
        try:
            while len(got) < N:
                f = recv_msg(b)
                if f == {"op": "noop"}:
                    noops[0] += 1
                else:
                    got.append(f)
        except Exception as e:                          # a decode error IS the corruption bug
            err.append(e)

    t = threading.Thread(target=rx, daemon=True)
    t.start()
    kw = VP._KeepWarm(a)
    assert kw.on, "keep-warm did not arm — the test would prove nothing"
    t0 = time.monotonic()
    try:
        for i in range(N):                              # header + tensor blob: multi-segment on the wire
            frame, _ = VP._make_step_frame(_h(float(i)), _ids(), i, None)
            kw.send(frame)
            if i % 20 == 0:
                time.sleep(0.003)                       # yield so the noop thread interleaves for real
    finally:
        wall = time.monotonic() - t0
        kw.stop()
    t.join(10)
    a.close()
    b.close()
    assert not err, f"receiver hit a decode error (stream corrupted): {err[0]}"
    assert len(got) == N and [int(f["start_pos"]) for f in got] == list(range(N)), "frames reordered"
    assert noops[0] >= 1, "the noop thread never interleaved — the test exercised nothing"
    assert wall < 5.0, f"real sends stalled behind the noop path: {wall:.2f}s"


def test_keepwarm_off_by_default_and_attach_swaps_the_socket():
    """Default OFF (no thread, no bytes) so a plain ring is byte-identical; attach() is what lets
    _fwd_open hand the daemon a rebuilt leg instead of leaving it writing to the corpse."""
    a, b = _pair()
    kw = VP._KeepWarm(a)
    assert not kw.on and _drain(b, 0.08) == []
    c, d = _pair()
    kw.attach(c)
    kw.send({"op": "reset"})
    assert recv_msg(d) == {"op": "reset"}
    assert _drain(b, 0.05) == [], "a send after attach() must not go to the old socket"
    for s in (a, b, c, d):
        s.close()


# ── 7. the spec loop's arithmetic, against a scripted tail (step 5 pins the stage half) ────────────

class _ScriptedRing:
    """head+tail as one thread over two socketpairs: it reads the coordinator's frames off the pipe
    and answers on the return channel from a script. Nothing computes — the point is to hand
    coordinate_spec exactly the per-position tokens a real tail WOULD return, and check what it
    commits."""

    def __init__(self, first_token, replies):
        self.pipe_a, self.pipe_b = _pair(timeout=10)      # coordinator -> head
        self.ret_a, self.ret_b = _pair(timeout=10)        # tail -> coordinator
        self.first_token = first_token
        self.replies = list(replies)
        self.chunks = []                                  # (ids, start_pos) per step frame seen
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        try:
            while True:
                msg = recv_msg(self.pipe_b)
                op = msg.get("op")
                if op == "reset":
                    send_msg(self.ret_a, "ok")
                elif op == "step":
                    self.chunks.append((list(msg["ids"][0]), int(msg["start_pos"])))
                    if len(self.chunks) == 1:             # prefill: one token back, like a real tail
                        send_msg(self.ret_a, {"token": self.first_token})
                    else:
                        r = self.replies.pop(0)
                        send_msg(self.ret_a, {"token": r[-1], "tokens": r})
                else:
                    return
        except (OSError, EOFError, IndexError):
            return

    def close(self):
        for s in (self.pipe_a, self.pipe_b, self.ret_a, self.ret_b):
            try:
                s.close()
            except OSError:
                pass


def _spec(first_token, replies, drafts, max_new, K=4, eos_ids=()):
    ring = _ScriptedRing(first_token, replies)
    script = list(drafts)
    try:
        r = VP.coordinate_spec(ring.pipe_a, ring.ret_b, [1, 2, 3], max_new, K=K, eos_ids=eos_ids,
                               timeout=10, drafter=lambda ids, k: list(script.pop(0)))
    finally:
        ring.close()
    return r, ring.chunks


def test_coordinate_spec_full_accept_commits_k_plus_one():
    """Every draft matches: the round commits K accepted drafts + one free correction = K+1 tokens
    for ONE ring traversal. That is the whole point of the lever."""
    drafts = [[11, 12, 13, 14]]
    replies = [[11, 12, 13, 14, 15]]                      # the model's greedy token AFTER each chunk pos
    r, chunks = _spec(first_token=10, replies=replies, drafts=drafts, max_new=6)
    assert r["tokens"] == [10, 11, 12, 13, 14, 15]
    assert r["accepted"] == 4 and r["accept_hist"] == {4: 1} and r["rounds"] == 1
    assert r["g"] == 6.0                                  # 6 committed tokens in 1 drafted round
    assert chunks[0] == ([1, 2, 3], 0), "prefill sends the whole prompt at position 0"
    assert chunks[1] == ([10, 11, 12, 13, 14], 3), "the chunk is [cur]+drafts at cur's position"


def test_coordinate_spec_partial_accept_commits_prefix_plus_correction():
    """Longest MATCHING PREFIX only: draft 3 diverges, so drafts[3:] are discarded even though
    draft 4 happens to be right. Committed = the 2 accepted + the model's own token at position 2."""
    drafts = [[11, 12, 99, 14], [77, 78, 79, 80]]
    replies = [[11, 12, 13, 14, 15], [21, 22, 23, 24, 25]]
    r, chunks = _spec(first_token=10, replies=replies, drafts=drafts, max_new=5)
    assert r["tokens"] == [10, 11, 12, 13, 21]
    assert r["accept_hist"] == {2: 1, 0: 1} and r["accepted"] == 2
    assert chunks[1] == ([10, 11, 12, 99, 14], 3)
    # the REWIND: after committing [11,12,13] the sequence is [1,2,3,10,11,12,13], so `cur`=13 sits
    # at absolute position 6 — the next chunk must be fed there, NOT at the end of the rejected draft
    assert chunks[2] == ([13, 77, 78, 79, 80], 6)


def test_coordinate_spec_zero_accept_still_commits_one():
    """A useless drafter costs a traversal but never correctness: the round still commits the model's
    own next token, so spec degrades to plain greedy decode rather than stalling."""
    drafts = [[99, 98, 97, 96], [95, 94, 93, 92]]
    replies = [[11, 0, 0, 0, 0], [12, 0, 0, 0, 0]]
    r, chunks = _spec(first_token=10, replies=replies, drafts=drafts, max_new=3)
    assert r["tokens"] == [10, 11, 12]
    assert r["accept_hist"] == {0: 2} and r["accepted"] == 0
    assert chunks[2] == ([11, 95, 94, 93, 92], 4), "one committed token advances `cur` by exactly one"


def test_coordinate_spec_stops_at_eos_inside_a_commit():
    """EOS lands in the MIDDLE of an accepted prefix: the loop must stop there and never emit the
    tokens behind it, even though the tail already computed them."""
    drafts = [[11, 12, 13, 14]]
    replies = [[11, 12, 13, 14, 15]]
    r, _ = _spec(first_token=10, replies=replies, drafts=drafts, max_new=10, eos_ids=(12,))
    assert r["tokens"] == [10, 11, 12], "tokens after the EOS must not be committed"
    assert r["generated"] == 3


def test_coordinate_spec_honors_max_new_mid_commit():
    drafts = [[11, 12, 13, 14]]
    replies = [[11, 12, 13, 14, 15]]
    r, _ = _spec(first_token=10, replies=replies, drafts=drafts, max_new=3)
    assert r["tokens"] == [10, 11, 12]


def test_spec_drafter_fallbacks_are_valid_proposers():
    """coordinate_spec takes any object with .propose(ids, K) or a bare callable; with neither, it
    builds the n-gram drafter, falling back to repeat-last where phase0/ngram_draft.py isn't
    deployed. All of them must return exactly K ids — losslessness does not depend on quality."""
    assert VP._RepeatDrafter().propose([1, 2, 3], 4) == [3, 3, 3, 3]
    assert len(VP._drafter_propose(None, 3)([1, 2, 3, 1, 2, 3], 4)) == 4
    assert VP._drafter_propose(lambda ids, k: [7] * k, 3)([1], 2) == [7, 7]


# ── 7b. the DSPARK round: the tail drafts, and both ends must agree on the accept ──────────────────

class _ScriptedDsparkRing(_ScriptedRing):
    """The tail half of a drafted round, scripted: each reply carries the model's per-position tokens,
    the tail's own accept length `n` and the block it drafted for the NEXT round.

    `n` is computed with the real accept rule, off the drafts it actually received — which is the
    point: the coordinator computes the same thing independently and the two are asserted equal.
    `lie` forces a divergence, the failure a silent drafter desync would otherwise look like."""

    def __init__(self, first_token, script, lie=None):
        self.script = list(script)                        # (tokens, next_draft) per drafted round
        self.lie = lie
        self.accept = VP._dspark().plan_verify_round      # the coordinator's rule, on the tail's side
        super().__init__(first_token, [])

    def _run(self):
        try:
            while True:
                msg = recv_msg(self.pipe_b)
                op = msg.get("op")
                if op == "reset":
                    send_msg(self.ret_a, "ok")
                elif op == "step":
                    ids, pos = list(msg["ids"][0]), int(msg["start_pos"])
                    self.chunks.append((ids, pos))
                    if len(self.chunks) == 1:
                        send_msg(self.ret_a, {"token": self.first_token})
                        continue
                    tokens, draft = self.script.pop(0)
                    n = self.accept(ids[1:], tokens)[0]
                    send_msg(self.ret_a, {"token": tokens[-1], "tokens": tokens,
                                          "draft": list(draft),
                                          "n": self.lie if self.lie is not None else n})
                else:
                    return
        except (OSError, EOFError, IndexError):
            return


def _dspark_round(first_token, script, max_new, lie=None, eos_ids=()):
    ring = _ScriptedDsparkRing(first_token, script, lie=lie)
    try:
        r = VP.coordinate_dspark(ring.pipe_a, ring.ret_b, [1, 2, 3], max_new, eos_ids=eos_ids,
                                 timeout=10)
    finally:
        ring.close()
    return r, ring.chunks


def test_coordinate_dspark_first_round_is_bare_then_drafts_the_tails_block():
    """The round structure. The drafter deliberately produces no block at prefill, so round 1 is
    `[cur]` alone (the accept rule's degenerate case, one committed token); every later round sends
    `[cur] + the drafts the tail returned last time`. Nothing the coordinator invents — it does not
    own a drafter at all, which is the whole point of drafting on the tail."""
    script = [([21], [31, 32, 33]),                       # bare round: one reply, one commit
              ([31, 32, 99, 0], [41, 42, 43]),           # 2 of 3 accepted, then the correction
              ([41, 0, 0, 0], [51, 52, 53])]             # 1 of 3, and max_new lands mid-commit
    r, chunks = _dspark_round(first_token=10, script=script, max_new=6)
    assert chunks[0] == ([1, 2, 3], 0), "prefill sends the whole prompt at position 0"
    assert chunks[1] == ([10], 3), "round 1 is the bare [cur] chunk — no drafts exist yet"
    assert chunks[2] == ([21, 31, 32, 33], 4), "round 2 sends the tail's block behind cur"
    assert r["tokens"] == [10, 21, 31, 32, 99, 41]
    assert r["accept_hist"] == {0: 1, 2: 1, 1: 1} and r["accepted"] == 3
    assert r["rounds"] == 3 and r["drafted"] == 2, "only the bare first round carried no block"
    # the REWIND: [1,2,3,10,21,31,32,99] leaves cur=99 at absolute position 7, and the next chunk
    # must open THERE — not at the end of the draft the ring rejected
    assert chunks[3] == ([99, 41, 42, 43], 7)


def test_coordinate_dspark_full_accept_commits_the_block_plus_the_bonus():
    """A fully accepted block of 3 commits 4 tokens for ONE traversal, and the next chunk opens where
    the stage already stands — the rollback's no-op path."""
    script = [([21], [31, 32, 33]), ([31, 32, 33, 44], [51, 52, 53])]
    r, chunks = _dspark_round(first_token=10, script=script, max_new=6)
    assert r["tokens"] == [10, 21, 31, 32, 33, 44]
    assert r["accept_hist"] == {0: 1, 3: 1} and r["g"] == 3.0
    assert chunks[2] == ([21, 31, 32, 33], 4)


def test_coordinate_dspark_fails_loudly_when_the_tail_accepts_differently():
    """The protocol's own tripwire. The tail runs the accept rule to advance its drafter and the
    coordinator runs it to decide what to emit; if they ever disagree, the drafter is conditioning on
    a history the ring did not take, and every later draft looks plausible while being wrong. That is
    a job-killing bug, so it fails HERE rather than as quietly worse acceptance forever."""
    script = [([21], [31, 32, 33]), ([31, 32, 99, 0], [41, 42, 43])]
    with pytest.raises(RuntimeError, match="two ends of a lossless round have diverged"):
        _dspark_round(first_token=10, script=script, max_new=6, lie=3)


def test_coordinate_dspark_stops_at_eos_inside_a_commit():
    """EOS in the middle of an accepted block: the tokens behind it are never emitted, even though
    the tail computed them and the drafter has already advanced past them."""
    script = [([21], [31, 32, 33]), ([31, 32, 33, 44], [51, 52, 53])]
    r, _ = _dspark_round(first_token=10, script=script, max_new=10, eos_ids=(32,))
    assert r["tokens"] == [10, 21, 31, 32] and r["generated"] == 4


# ── 8. the pure builders: tiling, per-GPU split, sidecar wiring, planner seam ──────────────────────

def test_even_tiling_covers_all_43_layers_contiguously():
    ranges = VP.even_tiling(VP.N_LAYERS, 6)
    assert len(ranges) == 6
    assert ranges[0][0] == 0 and ranges[-1][1] == VP.N_LAYERS
    for (a_lo, a_hi), (b_lo, b_hi) in zip(ranges, ranges[1:]):
        assert a_hi == b_lo, "tiling has a gap or overlap"
    assert max(hi - lo for lo, hi in ranges) - min(hi - lo for lo, hi in ranges) <= 1, "unbalanced"


def test_even_tiling_rejects_impossible_splits():
    with pytest.raises(ValueError):
        VP.even_tiling(3, 4)


def test_split_stages_to_gpus_tiles_three_boxes_of_two_gpus():
    """43 layers over 3 boxes x 2 GPUs = 6 stages but only 3 WAN hops: the per-GPU sub-blocks tile
    the box block, and head/tail are the GLOBAL ring ends while box_head/box_tail are its WAN ends."""
    nodes = [{"id": f"box{i}", "lo": lo, "hi": hi}
             for i, (lo, hi) in enumerate(VP.even_tiling(VP.N_LAYERS, 3))]
    subs = VP.split_stages_to_gpus(nodes, 2)
    assert len(subs) == 6
    assert subs[0]["head"] and subs[-1]["tail"]
    assert sum(s["head"] for s in subs) == 1 and sum(s["tail"] for s in subs) == 1
    cursor = 0
    for s in subs:
        assert s["lo"] == cursor, "per-GPU sub-blocks must tile with no gap/overlap"
        cursor = s["hi"]
    assert cursor == VP.N_LAYERS
    assert [s["box_index"] for s in subs] == [0, 0, 1, 1, 2, 2]
    assert [s["box_head"] for s in subs] == [True, False] * 3


def test_box_wiring_keeps_intra_box_hops_off_the_sidecar():
    """Only the box's LAST local stage takes a WAN leg; the others hand off over loopback ports the
    sidecar never touches. That is what makes B boxes x G GPUs cost B hops, not B*G."""
    nodes = [{"id": f"box{i}", "lo": lo, "hi": hi}
             for i, (lo, hi) in enumerate(VP.even_tiling(VP.N_LAYERS, 3))]
    subs = VP.split_stages_to_gpus(nodes, 2)
    links = [VP.box_stage_wiring(s) for s in subs]
    assert [l[2] for l in links] == ["loopback", "wan", "loopback", "wan", "loopback", "tail"]
    assert [l[0] for l in links] == [VP.ENG_IN, VP.ENG_LOCAL_BASE + 1] * 3
    assert links[0][1] == f"127.0.0.1:{VP.ENG_LOCAL_BASE + 1}"
    assert links[1][1] == f"127.0.0.1:{VP.FWD_RING}"
    assert links[-1][1] is None, "the global tail has no forward leg"


def test_box_return_relay_only_on_a_multigpu_tail_box_ingress():
    """The coordinator-return lands at the tail box's ENG_IN but the token is produced on its last
    GPU: exactly one stage bridges them, and a 1-GPU tail box needs no bridge at all."""
    nodes = [{"id": f"box{i}", "lo": lo, "hi": hi}
             for i, (lo, hi) in enumerate(VP.even_tiling(VP.N_LAYERS, 3))]
    subs = VP.split_stages_to_gpus(nodes, 2)
    relays = [VP.box_return_relay(s, subs[-1]["box_index"]) for s in subs]
    assert relays == [None, None, None, None, f"127.0.0.1:{VP.ENG_LOCAL_BASE + 1}", None]
    solo = VP.split_stages_to_gpus(nodes, [2, 2, 1])
    assert all(VP.box_return_relay(s, solo[-1]["box_index"]) is None for s in solo)


def test_ring_sidecar_spec_is_the_direct_return_topology():
    """Every non-tail stage forwards the ring; the HEAD also forwards the coordinator-return to the
    tail; each inbound stage admits only its predecessor (the tail also admits the head, whose return
    stream arrives as its RemotePeer). Byte-shape-identical to k3_pipe / m25_scatter_pipe — the
    sidecar never looks inside a frame, so the V4 payload changes nothing here."""
    m = [f"/ip4/10.0.0.{k}/tcp/29600/p2p/PEER{k}" for k in range(3)]
    inb0, fwd0, allow0 = VP.ring_sidecar_spec(0, 3, m)
    assert inb0 == "" and allow0 is None                      # head: no inbound, open
    assert f"127.0.0.1:{VP.FWD_RING}={m[1]}" in fwd0          # ring -> s1
    assert f"127.0.0.1:{VP.FWD_RET}={m[2]}" in fwd0           # return -> tail
    inb1, fwd1, allow1 = VP.ring_sidecar_spec(1, 3, m)
    assert inb1 == f"127.0.0.1:{VP.ENG_IN}"
    assert fwd1 == [f"127.0.0.1:{VP.FWD_RING}={m[2]}"] and allow1 == ["PEER0"]
    inb2, fwd2, allow2 = VP.ring_sidecar_spec(2, 3, m)
    assert inb2 == f"127.0.0.1:{VP.ENG_IN}" and fwd2 == []    # tail: inbound only, no forward
    assert allow2 == ["PEER1", "PEER0"]                       # predecessor + head (the return stream)


def test_box_ring_launch_emits_detached_per_gpu_commands():
    """The launcher's product: one setsid-detached command per (box, GPU) — a bare `nohup ... &` over
    ssh held the engine's fds and hung a parallel launch — each pinned to its own CUDA device and
    logging to a PER-PORT file so co-located stages never clobber each other."""
    nodes = [{"id": f"box{i}", "lo": lo, "hi": hi}
             for i, (lo, hi) in enumerate(VP.even_tiling(VP.N_LAYERS, 3))]
    maddrs = [f"/ip4/10.0.0.{k}/tcp/29600/p2p/PEER{k}" for k in range(3)]
    out = VP.box_ring_launch(nodes, 2, maddrs, receipts=True, token="tok")
    assert len(out["stages"]) == 6 and len(out["sidecars"]) == 3, "sidecars stay BOX-granular"
    for k, st in enumerate(out["stages"]):
        assert st["cmd"].startswith("SHARD_RECEIPTS=1 SHARD_SWARM_TOKEN=tok ")
        assert f"CUDA_VISIBLE_DEVICES={st['gpu']} " in st["cmd"]
        assert "setsid bash -c" in st["cmd"] and "</dev/null" in st["cmd"]
        assert f"--stage {k} --nstages 6" in st["cmd"]
        assert f"/root/v4_stage_{st['eng_port']}.log" in st["cmd"]
    assert "--next" not in out["stages"][-1]["cmd"], "the tail forwards nowhere"
    assert "--ret-relay" in out["stages"][4]["cmd"], "the tail box ingress bridges the return"


def test_plan_layer_ranges_names_the_missing_v4_profile():
    """shard/plan.py keys engine profiles by catalog model id and REFUSES an unknown one rather than
    planning V4 at another model's calibration (the admit-then-OOM failure the measured numbers exist
    to prevent). There is no V4 entry yet — the error has to say so, and name even_tiling as the
    offline fallback, instead of surfacing as a bare KeyError three frames down."""
    nodes = [{"id": "b0", "free_vram_mb": 96000.0, "subnet": "10.0.0.0/24"}]
    with pytest.raises(RuntimeError, match="no engine profile"):
        VP.plan_layer_ranges(nodes, [[0.0]])
    with pytest.raises(RuntimeError, match="even_tiling"):
        VP.plan_layer_ranges(nodes, [[0.0]])


def test_tail_drafter_seam_is_unset_by_default():
    """Step 4 fills TAIL_DRAFTER; until then the tail must not reach for a drafter it does not have,
    and a greedy ring must never touch the seam at all."""
    assert VP.TAIL_DRAFTER is None


# ── 9. the real ring: head embed -> middles -> tail sample, against the reference itself ───────────
# Everything above runs with no model. This last section is the same proof `python3 phase0/v4_pipe.py
# selftest` prints, wired into the suite: real v4_stage stages over a real tiny checkpoint, and the
# bar is BIT-IDENTICAL token ids against the vendored reference Transformer's own greedy decode.
# v4_ref_cpu / v4_stage are imported INSIDE the fixture, so a box without them skips this section and
# still runs the protocol tests.

PROMPT = [3, 9, 17, 2, 41]
NEW = 5


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    """(checkpoint dir, args, reference tokens) — one toy V4, built and decoded ONCE.

    The reference is decoded here and its answer cached, not re-run per test, because the reference
    Transformer is STATEFUL ACROSS SEQUENCES: a prefill only rewrites the compression blocks it
    fills, so the second sequence through one model instance starts from the first's tail (see
    v4_ref_cpu's module docstring). One oracle, one sequence — the ring is the side that gets to
    prove it can answer the same question twice (test_reset_gives_a_warm_ring_a_cold_rings_answer)."""
    R = pytest.importorskip("v4_ref_cpu")
    pytest.importorskip("v4_stage")
    pytest.importorskip("safetensors.torch")
    args = R.cpu_args()
    d = str(tmp_path_factory.mktemp("v4pipe"))
    model = R.build_oracle(args)
    VP._write_tiny_checkpoint(d, args, model)      # write BEFORE decoding: the stages must not
    return d, args, VP._reference_tokens(model, PROMPT, NEW)      # inherit a used sequence's state


class _Ring:
    """A localhost v4_pipe ring over the tiny checkpoint: N stage-server threads + a coordinator
    dial. coordinate() reuses the SAME warm ring across calls (each call resets), so a second job on
    the ring is the reset/warm path."""

    def __init__(self, ckpt_dir, args, ranges, receipts=False, tail_box_g=1, dspark=False):
        self.n = len(ranges)
        self.layer_count = args.n_layers
        self.receipts = receipts
        ports = VP._free_ports(self.n)
        relay_i = self.n - tail_box_g                          # ingress of a multi-GPU tail box (>1)
        events = [threading.Event() for _ in range(self.n)]
        self.threads = []
        for i, (lo, hi) in enumerate(ranges):
            nxt = None if i == self.n - 1 else f"127.0.0.1:{ports[i + 1]}"
            ret_relay = f"127.0.0.1:{ports[-1]}" if (tail_box_g > 1 and i == relay_i) else None
            t = threading.Thread(target=VP.serve_stage, kwargs=dict(
                stage=i, nstages=self.n, lo=lo, hi=hi, port=ports[i], nxt=nxt, ckpt_dir=ckpt_dir,
                device="cpu", receipts=receipts, key_path=f"{ckpt_dir}/s{i}.key",
                ret_relay=ret_relay, dspark=dspark, ready=events[i]), daemon=True)
            t.start()
            self.threads.append(t)
        for e in events:
            assert e.wait(120), "a stage never came up"
        # the coordinator-return terminates at the tail box INGRESS (the relay), not the global tail
        tail_port = ports[relay_i] if tail_box_g > 1 else ports[-1]
        self.pipe, self.ret = VP.connect_ring(f"127.0.0.1:{ports[0]}", f"127.0.0.1:{tail_port}",
                                              timeout=60)

    def coordinate(self, prompt, max_new, nonce=None):
        return VP.coordinate(self.pipe, self.ret, prompt, max_new, nonce=nonce,
                             receipts=self.receipts, layer_count=self.layer_count, timeout=60)

    def spec(self, prompt, max_new, nonce=None, **kw):
        return VP.coordinate_spec(self.pipe, self.ret, prompt, max_new, nonce=nonce,
                                  receipts=self.receipts, layer_count=self.layer_count,
                                  timeout=60, **kw)

    def dspark(self, prompt, max_new, nonce=None):
        return VP.coordinate_dspark(self.pipe, self.ret, prompt, max_new, nonce=nonce,
                                    receipts=self.receipts, layer_count=self.layer_count,
                                    timeout=60)

    def close(self):
        try:
            send_msg(self.pipe, {"op": "stop"})
        except OSError:
            pass
        for t in self.threads:
            t.join(timeout=10)


@pytest.mark.parametrize("ranges", [
    [(0, 3), (3, 6), (6, 8)],                                  # 3 stages: window / compressed / indexed mix
    [(0, 4), (4, 8)],                                          # 2 stages, split mid-stack
    [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8)],   # every layer a boundary
])
def test_full_ring_greedy_matches_reference(tiny, ranges):
    """THE proof: head-embed + M stages + tail lm_head, h [b,s,4,dim] AND the routing ids crossing
    real transport sockets every hop, sampled token-for-token against the reference Transformer.
    Bit-identical — so nothing about splitting the model across boxes changes the answer."""
    d, args, ref = tiny
    ring = _Ring(d, args, ranges)
    try:
        got = ring.coordinate(list(PROMPT), NEW)["tokens"]
    finally:
        ring.close()
    assert got == ref, f"{ranges}: ring {got} != ref {ref}"


def test_full_ring_receipts_settle_and_c10(tiny):
    """Every stage signs its (h || ids) hash-chain over the job; the coordinator sweeps the ring once
    and settles. wire_receipt (C10) strips the post-sign `stage` debug tag, so the RAW receipts fail
    verify (the tag is in the signed preimage) and the WIRE receipts pass."""
    d, args, _ref = tiny
    ring = _Ring(d, args, [(0, 3), (3, 6), (6, 8)], receipts=True)
    try:
        r = ring.coordinate(list(PROMPT), NEW, nonce="settle-nonce-abc")
    finally:
        ring.close()
    assert r["receipts_ok"] is True, "the coordinator's own coverage verify must pass"
    raw = r["receipts"]
    assert len(raw) == 3 and all("stage" in rr for rr in raw), "each stage tags its receipt post-sign"
    with pytest.raises(ReceiptError):                          # C10 red: the debug tag breaks the sig
        verify_coverage(raw, args.n_layers, expected_nonce="settle-nonce-abc", check_chain=True)
    wired = [wire_receipt(rr) for rr in raw]                   # C10 green: stripped, the ring settles
    verify_coverage(wired, args.n_layers, expected_nonce="settle-nonce-abc", check_chain=True)
    with pytest.raises(ReceiptError, match="nonce"):           # and it binds to THIS job's nonce
        verify_coverage(wired, args.n_layers, expected_nonce="another-nonce", check_chain=True)


def test_fp8_wire_keeps_the_receipt_chain_on_a_real_ring(tiny, monkeypatch):
    """The fp8 wire halves the bytes/hop at the cost of a LOSSY activation — so the receipt chain can
    only survive because both sides hash the PACKED wire bytes. On a real ring: out_root == in_root
    across every hop with check_chain=True, no relaxation. (The tokens themselves are NOT asserted
    here: fp8 transport is lossy by construction, and a random-init toy model has near-uniform logits
    that a one-ulp nudge flips. Whether it is lossless ENOUGH on the real weights is a measurement
    for the ring phase, not something a toy fixture can answer.)"""
    monkeypatch.setattr(VP, "V4_FP8_WIRE", True)               # the stage threads read the module global
    d, args, _ref = tiny
    ring = _Ring(d, args, [(0, 3), (3, 6), (6, 8)], receipts=True)
    try:
        r = ring.coordinate(list(PROMPT), NEW, nonce="fp8-nonce")
    finally:
        ring.close()
    assert r["receipts_ok"] is True, "the fp8 wire broke the receipt chain"
    wired = sorted((wire_receipt(rr) for rr in r["receipts"]), key=lambda x: x["layer_start"])
    for a, b in zip(wired, wired[1:]):
        assert a["out_root"] == b["in_root"], "chain broke across an fp8 hop"


def test_multigpu_tail_box_relays_return_and_matches_reference(tiny):
    """The return path for a G>1 TAIL box: the coordinator-return lands at the box ingress (ENG_IN),
    the token is produced on the box's last-GPU stage, and the ingress bridges them over loopback.
    Bit-identical still — the relay is a transparent tunnel, not a numerics change — and the ingress
    stage's OWN receipt is in the settled set (it is a compute stage, not just a bridge)."""
    d, args, ref = tiny
    ring = _Ring(d, args, [(0, 3), (3, 6), (6, 8)], receipts=True, tail_box_g=2)
    try:
        r = ring.coordinate(list(PROMPT), NEW, nonce="relay-nonce")
    finally:
        ring.close()
    assert r["tokens"] == ref, f"relay ring {r['tokens']} != ref {ref}"
    assert r["receipts_ok"] is True and len(r["receipts"]) == 3


def test_spec_ring_is_lossless_both_ways(tiny):
    """SPECULATION ON A REAL RING IS BIT-IDENTICAL TO GREEDY — both halves of the protocol.

    A drafter that is always WRONG (repeat-last) makes every round a rejection, so every round drives
    Stage._seek: restore the pre-chunk snapshot, replay the accepted prefix, re-open at the committed
    position. A drafter that is always RIGHT (the reference's own continuation) makes every round a
    full accept, so the ring commits K+1 tokens per traversal and never rewinds at all. Both must
    emit exactly the reference's tokens.

    IT IS THE PER-POSITION LOGITS THAT MAKE THE SECOND ONE POSSIBLE. A draft is accepted only where
    the tail's own reply at that position equals it, so a full accept is the assertion "the verify
    rows ARE the greedy rows" — bitwise, at every position of the chunk. Batch the chunk's logits
    into one [b, s, vocab] GEMM instead and the fp32 reassociation shifts them ~4e-7, which flips an
    argmax near-tie at random: acceptance would quietly rot and the stream would stop being greedy
    without anything failing."""
    d, args, ref = tiny
    ring = _Ring(d, args, [(0, 3), (3, 6), (6, 8)], receipts=True)
    try:
        bad = ring.spec(list(PROMPT), NEW, nonce="spec-bad", K=4, drafter=VP._RepeatDrafter())
        good = ring.spec(list(PROMPT), NEW, nonce="spec-good", K=2,
                         drafter=lambda seq, K: (ref[len(seq) - len(PROMPT):][:K] + [0] * K)[:K])
    finally:
        ring.close()
    assert bad["tokens"] == ref, f"rejected-round ring {bad['tokens']} != greedy {ref}"
    assert good["tokens"] == ref, f"accepted-round ring {good['tokens']} != greedy {ref}"
    assert bad["receipts_ok"] is True and good["receipts_ok"] is True
    assert max(good["accept_hist"]) == 2, "the perfect drafter was not fully accepted — verify rows " \
                                          "differ from the greedy rows they claim to replace"
    assert good["g"] > bad["g"], "a perfect drafter must commit more per traversal than a useless one"


def test_full_ring_dspark_matches_reference(tiny):
    """THE drafted ring, end to end: the tail runs V4's own MTP speculator over its `main_hidden`,
    returns the block with the token, and the emitted stream is the reference's greedy stream.

    Acceptance is NOT asserted: at random weights a trained drafter has nothing to be right about, so
    g > 1 here would be luck rather than evidence. What is asserted is that the machinery really ran
    — every round after the deliberately bare first one carried a real draft block through the whole
    ring and was verified — and that the stream is exact anyway. That is the property speculation must
    have; the acceptance rate is a measurement for real weights."""
    d, args, ref = tiny
    ring = _Ring(d, args, VP._dspark_tiling(args, 3), receipts=True, dspark=True)
    try:
        r = ring.dspark(list(PROMPT), NEW, nonce="dspark-nonce")
    finally:
        ring.close()
    assert r["tokens"] == ref, f"dspark ring {r['tokens']} != greedy {ref}"
    assert r["receipts_ok"] is True, "a drafted job must still settle its receipts"
    assert r["rounds"] > 1 and r["drafted"] == r["rounds"] - 1, \
        f"the drafter never proposed a block: {r['rounds']} rounds, {r['drafted']} drafted"


def test_a_second_dspark_job_on_a_warm_ring_is_a_cold_rings_answer(tiny):
    """Two drafted jobs back to back on one warm ring — the shape a served ring actually runs in.

    A drafted job leaves MORE behind than a greedy one: the stages carry a spec checkpoint and a
    rewound KV, and the tail's drafter carries an mtp window and a position cursor that only means
    anything for the sequence it was built on. The drafter is rebuilt for no job — it is a process
    lifetime object — so if the per-job reset missed any of that, the second job would draft off the
    first one's history and could not come back with the reference's stream."""
    d, args, ref = tiny
    ring = _Ring(d, args, VP._dspark_tiling(args, 3), dspark=True)
    try:
        first = ring.dspark([7, 7, 7, 1, 2, 9], NEW)               # a different job, warming the state
        warm = ring.dspark(list(PROMPT), NEW)
    finally:
        ring.close()
    assert first["tokens"] != ref, "the warming job must be a DIFFERENT sequence to prove anything"
    assert warm["tokens"] == ref, f"warm drafted ring {warm['tokens']} != cold greedy {ref}"


def test_dspark_job_on_a_greedy_ring_fails_the_job_not_the_ring(tiny):
    """Ask a tail that was launched WITHOUT --dspark to draft: the job dies, the ring lives.

    The tail is the only thing the coordinator reads, so an exception in its serve loop is not a
    failed job — it is a ring that stalls this job and every job after it, which is how the 2026-07-29
    capstone ring was lost. A drafter that cannot be built (no embedding, or a checkpoint with no
    `mtp.*`) therefore answers the RESET with a failure the coordinator raises on, and the next
    greedy job goes through the same warm ring untouched."""
    d, args, ref = tiny
    ring = _Ring(d, args, VP._dspark_tiling(args, 3))          # no dspark=True: no MTP stages here
    try:
        with pytest.raises(RuntimeError, match="not acked"):
            ring.dspark(list(PROMPT), NEW)
        assert ring.coordinate(list(PROMPT), NEW)["tokens"] == ref, "the ring did not survive"
    finally:
        ring.close()


def test_dspark_tiling_puts_every_target_layer_on_the_tail(tiny):
    """The placement constraint a drafted ring has and a greedy one does not: `main_hidden` is the
    taps of ALL dspark target layers concatenated, so they must land on ONE stage — the tail. A plan
    that splits them serves greedily and refuses to draft (Stage.tail_main_hidden), which is why the
    V4_PROFILE this ring is eventually planned with has to encode it."""
    _d, args, _ref = tiny
    ranges = VP._dspark_tiling(args, 3)
    lo, hi = ranges[-1]
    assert all(lo <= t < hi for t in args.dspark_target_layer_ids)
    assert ranges[0][0] == 0 and hi == args.n_layers
    for (_a, a_hi), (b_lo, _b) in zip(ranges, ranges[1:]):
        assert a_hi == b_lo, "the drafted tiling still has to tile"


def test_ring_args_fills_only_what_the_shipped_config_omits(tiny, tmp_path, monkeypatch):
    """V4's release declares neither max_seq_len nor max_batch_size, so ModelArgs' 4096/4 silently
    size every kv_cache, the freqs_cis table and the drafter's end-of-context guard — an 8k ring would
    stop drafting at ~4090. ring_args fills them, but ONLY where the config is silent: a checkpoint
    that states them means it, and overriding those would build stages whose caches disagree with the
    model they are graded against (which is exactly the CPU parity fixtures' case)."""
    import dataclasses
    import json
    d, args, _ref = tiny
    full = dataclasses.asdict(args)
    silent = {k: v for k, v in full.items() if k not in ("max_seq_len", "max_batch_size")}
    quiet = tmp_path / "silent"
    quiet.mkdir()
    (quiet / "config.json").write_text(json.dumps(silent))

    got = VP.ring_args(str(quiet))
    assert (got.max_seq_len, got.max_batch_size) == (VP.V4_MAX_SEQ, VP.V4_MAX_BATCH)
    kept = VP.ring_args(d)
    assert (kept.max_seq_len, kept.max_batch_size) == (args.max_seq_len, args.max_batch_size)

    monkeypatch.setenv("V4_MAX_SEQ", "777")               # an explicit env override always wins
    assert VP.ring_args(str(quiet)).max_seq_len == 777


def test_reset_gives_a_warm_ring_a_cold_rings_answer(tiny):
    """A second, different job on the same warm ring must equal that job on a fresh ring — the reset
    at the head of coordinate() drops every stage's KV cache AND the sparse-attention compressor
    accumulators, which the reference only ever partially rewrites on a re-prefill."""
    d, args, ref = tiny
    ring = _Ring(d, args, [(0, 3), (3, 6), (6, 8)])
    try:
        ring.coordinate([7, 7, 7, 1, 2, 9], NEW)               # a first, different job warms the state
        warm = ring.coordinate(list(PROMPT), NEW)["tokens"]    # reset + the job we can check
    finally:
        ring.close()
    assert warm == ref, f"warm-after-reset {warm} != cold {ref}"


def test_swarm_token_value_is_actually_compared(monkeypatch):
    """The hole a live-ring audit found: greetings were checked for SHAPE only, so any scanner
    speaking the codec could greet its way past a token-'gated' ring. The token VALUE must gate."""
    monkeypatch.setattr(VP, "SWARM_TOKEN", "s3cret")
    assert VP._is_pred_hello({"op": "hello_pred", "token": "s3cret"}), "the right token must pass"
    assert VP._is_return_hello({"op": "hello_return", "token": "s3cret"})
    assert not VP._is_pred_hello({"op": "hello_pred", "token": "wrong"}), "a wrong token is a stranger"
    assert not VP._is_pred_hello({"op": "hello_pred"}), "a missing token is a stranger"
    assert not VP._is_return_hello({"op": "hello_return", "token": "wrong"})
    # right shape, wrong op — never a match
    assert not VP._is_pred_hello({"op": "hello_return", "token": "s3cret"})
    # token-less ring: shape alone gates (a token=None deployment is unchanged)
    monkeypatch.setattr(VP, "SWARM_TOKEN", None)
    assert VP._is_pred_hello({"op": "hello_pred"}), "token-less ring accepts on shape"
    assert VP._is_pred_hello({"op": "hello_pred", "token": "ignored"})


# ── 10. the fp8 activation wire (V4_FP8_WIRE): bytes, scale axis, and what it refuses ──────────────
# The lever is opt-in and default OFF. Three things have to hold for it to be worth switching on:
# it must actually halve the hop AT V4's shape (not be eaten by framing overhead), its scale axis
# must keep the drafted stream identical to the greedy one, and a frame whose scale sidecar does not
# match its hidden must be refused rather than silently applied.

V4_DIM, V4_HC, V4_BLOCK = 4096, 4, 5          # phase0/deepseek_v4_ref/inference/config.json


def _bench():
    return pytest.importorskip("v4_wire_bench")


def test_fp8_wire_halves_the_hop_and_framing_overhead_does_not_eat_it():
    """The bytes, at the SHIPPED config, through the real transport codec.

    The s=1 PIPELINED DECODE frame is the one to be suspicious about: it is the smallest frame the
    ring sends, so it is where fixed per-frame overhead would have the best chance of swamping the
    saving. It does not — V4's boundary is 4 hyper-connection streams x 4096 x 2 B = 32 KiB for a
    SINGLE token, against a 168 B JSON header. That is the V4 difference: on a model with one
    residual stream this frame would be 8 KiB and the argument would be closer."""
    WB = _bench()
    for s, floor in ((1, 1.95), (V4_BLOCK + 1, 1.95), (2048, 1.99)):
        bf16, fp8 = WB._frames(s, V4_DIM, V4_HC)
        nb, nf = WB._wire_len(bf16), WB._wire_len(fp8)
        assert nb / nf >= floor, f"s={s}: fp8 wire only {nb/nf:.3f}x ({nb} -> {nf})"
    bf16, _ = WB._frames(1, V4_DIM, V4_HC)
    payload = V4_HC * V4_DIM * 2 + 8                        # h + one int64 id
    overhead = WB._wire_len(bf16) - payload
    assert overhead < 0.01 * WB._wire_len(bf16), \
        f"framing overhead {overhead} B is >1% of the s=1 decode frame — metadata would dominate"


def test_bf16_cannot_prefill_v4s_own_max_seq_in_one_frame_but_fp8_can():
    """A live constraint, not a curiosity: V4_MAX_SEQ defaults to 8192 and one bf16 prefill frame at
    8192 positions is 268,501,190 B against transport's 268,435,456 B MAX_FRAME. The ring cannot
    prefill its own advertised context in a single frame on the bf16 wire; recv_msg refuses the
    length before it ever allocates. fp8 moves the ceiling past it with room to spare."""
    WB = _bench()
    T = WB.T
    assert WB._wire_len(WB._frames(VP.V4_MAX_SEQ, V4_DIM, V4_HC)[0]) > T.MAX_FRAME
    assert WB._wire_len(WB._frames(VP.V4_MAX_SEQ, V4_DIM, V4_HC)[1]) < T.MAX_FRAME
    assert WB.max_single_frame_prefill(V4_DIM, V4_HC, False) < VP.V4_MAX_SEQ
    assert WB.max_single_frame_prefill(V4_DIM, V4_HC, True) > VP.V4_MAX_SEQ


def _packed_bytes(t):
    q, s = VP._pack_t(t)
    return q.view(torch.uint8).numpy().tobytes(), s.view(torch.uint8).numpy().tobytes()


def test_fp8_scale_axis_makes_the_packing_chunk_size_invariant():
    """THE reason the scale reduces over `dim` alone, and it is a correctness property.

    DSpark sends the SAME position inside chunks of different lengths — the first round of a
    generation is a bare [cur] chunk (s=1), every round after carries a whole block
    (s=dspark_block_size+1). If the scale reduced over the sequence axis, a position's transmitted
    bytes would depend on which other positions happened to share its frame, and the drafted stream
    would stop being bit-identical to the greedy stream for a reason that has nothing to do with the
    model. Reducing over `dim` makes position p's packing a function of position p alone."""
    torch.manual_seed(0)
    h = torch.randn(1, 6, HC, 128, dtype=torch.bfloat16)
    h[0, 3, 1, 7] = 60.0                                     # an outlier that only ONE position owns
    h[0, 0, 0, 0] = -250.0                                   # and another, on a different position
    whole_q, whole_s = _packed_bytes(h)
    per = V4_HC and h[0, 0].numel()                          # bytes of one position's packed hidden
    for p in range(h.size(1)):
        alone_q, alone_s = _packed_bytes(h[:, p:p + 1])
        assert whole_q[p * per:(p + 1) * per] == alone_q, \
            f"position {p} packs differently inside a 6-chunk than alone — spec != greedy"
        assert whole_s[p * HC * 2:(p + 1) * HC * 2] == alone_s

    # the contrast, so the choice is documented by a failing alternative: a scale reduced over the
    # WHOLE tensor (what a per-tensor codec does) is NOT invariant, and the outliers above are
    # exactly the case that breaks it.
    def pertensor(t):
        f = t.detach().float()
        sc = (f.abs().amax() / 448.0).clamp(min=1e-8).to(torch.bfloat16)
        return (f / sc.float()).clamp(-448.0, 448.0).to(torch.float8_e4m3fn) \
            .view(torch.uint8).numpy().tobytes()
    assert pertensor(h)[3 * per:4 * per] != pertensor(h[:, 3:4]), \
        "the per-tensor contrast is supposed to differ — this test proves nothing if it does not"


def test_fp8_pack_saturates_and_survives_zeros_and_hostile_dtypes():
    """Numerics the codec must not get wrong, none of which the happy path exercises.

    The dtype cases are the subtle ones: `clamp(min=1e-8)` coerces its bound to the tensor's dtype,
    and in float16 1e-8 underflows to 0.0 — so an all-but-zero row would divide by zero and ship a
    frame of NaN. Taking the amax in fp32 is what makes the floor mean 1e-8 in every input dtype."""
    for dt in (torch.bfloat16, torch.float32, torch.float16):
        z = torch.zeros(1, 2, HC, 64, dtype=dt)
        q, s = VP._pack_t(z)
        assert not q.float().isnan().any(), f"all-zeros packed to NaN in {dt}"
        assert torch.equal(VP._unpack_t(q, s).float(), torch.zeros(1, 2, HC, 64)), \
            f"zeros must round-trip EXACTLY in {dt}"
        tiny_vals = torch.full((1, 1, HC, 64), 1e-7, dtype=dt)
        q, s = VP._pack_t(tiny_vals)
        assert not q.float().isnan().any(), f"denormal-scale input packed to NaN in {dt}"

    # saturation: the division maps amax to exactly 448, and torch's e4m3fn cast turns anything
    # past ~464 into NaN rather than clamping. The clamp is what makes that unreachable.
    torch.manual_seed(0)
    for mag in (1e-4, 1.0, 1e4, 1e30):
        t = (torch.randn(1, 2, HC, 64) * mag).to(torch.bfloat16)
        q, s = VP._pack_t(t)
        assert not q.float().isnan().any(), f"magnitude {mag} produced NaN"
        assert q.float().abs().max() <= 448.0
        assert not VP._unpack_t(q, s).float().isnan().any()

    # ONE inf must stay CONTAINED. An infinite amax makes that scale infinite, and then every finite
    # value sharing it divides to 0 while the inf itself divides to NaN. Its own (position, stream)
    # row cannot be saved — the scale has to cover the inf — but the blast radius must stop there,
    # which is a second dividend of reducing over `dim`: a scale reduced over the whole tensor would
    # take the entire chunk, every position and every stream, down with it.
    t = torch.randn(1, 2, HC, 64, dtype=torch.bfloat16)
    clean = VP._unpack_t(*VP._pack_t(t)).float()
    t[0, 0, 0, 5] = float("inf")
    got = VP._unpack_t(*VP._pack_t(t)).float()
    assert not got.isnan().any(), "one inf turned its row into NaN instead of saturating"
    spoiled = torch.zeros(1, 2, HC, dtype=torch.bool)
    spoiled[0, 0, 0] = True
    assert torch.equal(got[~spoiled], clean[~spoiled]), \
        "one inf changed the packing of rows it does not share a scale with"

    # a chunk spanning a huge dynamic range still round-trips finite
    t = torch.randn(1, 4, HC, 64, dtype=torch.bfloat16)
    t[0, 0] *= 2.0 ** 30
    q, s = VP._pack_t(t)
    assert not VP._unpack_t(q, s).float().isnan().any()


def test_recv_refuses_a_step_frame_whose_scale_sidecar_does_not_match():
    """A dropped or mismatched `h8` must be a dead edge, not a silent 1/scale corruption.

    Without the cross-check an fp8 `h` that arrived with no scale would sail through
    Stage.forward's `h.to(self.dtype)` as a tensor scaled by ~1/scale — no exception, no NaN, and an
    INTACT receipt chain (both sides hash the same bytes, so the chain cannot see it). ValueError is
    in _STRAY, so the serve loop resets the edge like any other malformed frame."""
    h, ids = _h(), _ids()
    frame, _ = VP._make_step_frame(h, ids, 0, None)
    assert "h8" not in frame                                 # default OFF: the bf16 wire is untouched
    VP.V4_FP8_WIRE = True
    try:
        frame, _ = VP._make_step_frame(h, ids, 0, None)
    finally:
        VP.V4_FP8_WIRE = False
    assert VP._recv_hids(dict(frame), None)[0].dtype == torch.bfloat16, "the happy path must decode"

    stripped = {k: v for k, v in frame.items() if k != "h8"}
    with pytest.raises(ValueError, match="h8"):
        VP._recv_hids(stripped, None)
    with pytest.raises(ValueError, match="h8"):               # a bf16 h with a scale is just as wrong
        VP._recv_hids({**frame, "h": h}, None)
    with pytest.raises(ValueError, match="h8"):               # right dtype, wrong shape
        VP._recv_hids({**frame, "h8": frame["h8"][:, :1]}, None)
    with pytest.raises(ValueError, match="h8"):               # the old scalar-scale wire
        VP._recv_hids({**frame, "h8": 0.25}, None)


def test_dspark_stream_still_equals_the_greedy_stream_under_the_fp8_wire(tiny, monkeypatch):
    """THE speculation question, and the one a lossy wire can genuinely break.

    coordinate_dspark's whole contract is that the drafted stream is bit-identical to the greedy
    one. The fp8 wire is lossy, so that can only survive RELATIVE to the same wire — and it does,
    because the drafter is tail-LOCAL and eats the post-wire tensor: the quantization lands once,
    upstream of the fork into (verify rows, drafter taps), so drafter and verifier are perturbed
    identically and the accept rule never rejects on account of the codec. What would break it is a
    scale that depends on chunk length, since a bare [cur] round is s=1 and a drafted round is
    s=block_size+1; test_fp8_scale_axis_makes_the_packing_chunk_size_invariant pins that axis."""
    d, args, _ref = tiny
    monkeypatch.setattr(VP, "V4_FP8_WIRE", True)             # the stage threads read the module global
    ring = _Ring(d, args, VP._dspark_tiling(args, 3), dspark=True)
    try:
        drafted = ring.dspark(list(PROMPT), NEW)
        greedy = ring.coordinate(list(PROMPT), NEW)["tokens"]
    finally:
        ring.close()
    assert drafted["tokens"] == greedy, (
        f"the fp8 wire desynced speculation from greedy: {drafted['tokens']} != {greedy}")
    assert drafted["g"] >= 1.0

