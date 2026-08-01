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
    when the tail's accept length disagrees with the coordinator's,
  * the PIPELINED round (7c): the block streamed as separate s=1 frames, the epoch fence on both the
    sending and the receiving side, and the cancel/rewind arithmetic — against a scripted tail that
    DERIVES its accept from the same frontier rule the real one runs.

Sections 9 and 10 are the headline, and section 9 is the same proof `python3 phase0/v4_pipe.py
selftest` prints: real v4_stage stages over a real tiny checkpoint, decoding through real sockets,
BIT-IDENTICAL to the vendored reference Transformer's own greedy decode — greedy, coordinator-drafted
(rejecting AND accepting) and DSpark-drafted, all against that one bar. Section 10 holds PIPELINED
speculation to it too, in every regime that can move the answer: zero accept, full accept, a cancel
landing exactly on a compression boundary, and a stale in-flight frame of a cancelled epoch. They
import v4_ref_cpu/v4_stage inside their fixture, so a box without them skips both and still runs
everything above.

Run: python3 -m pytest tests/test_v4_pipe.py -q
"""
import os
import select
import socket
import struct
import sys
import threading
import time
import traceback

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
        self.reset_msg = None                             # the frame that opened the job
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        try:
            while True:
                msg = recv_msg(self.pipe_b)
                op = msg.get("op")
                if op == "reset":
                    self.reset_msg = msg
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
                    self.reset_msg = msg
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


# ── 7c. the PIPELINED round: s=1 frames streamed, epoch-fenced, judged one reply at a time ────────


class _ScriptedPipeRing:
    """The tail half of a PIPELINED drafted round, scripted: one reply per streamed `s=1` frame.

    It is handed the model's own greedy continuation as `truth` (absolute position -> token) and
    answers each frame the way a real tail does: the frame at `p` returns the model's token at `p+1`.
    Two things are DERIVED rather than scripted, and that is the point of the harness —

      `acc`   the incremental frontier rule (`p == cfront+1 and the token == the greedy we produced
              at cfront`), the same rule RingDrafter._on_chunk_pipelined runs. The coordinator asserts
              its own judgment against it, so a scripted `acc` would assert nothing.
      the reply of an OFF-PATH frame is deliberate garbage (`_WRONG`), because a coordinator that ever
              believes one would commit a token off a history the ring did not take.

    `blocks(q)` supplies the drafts a committed frame at `q` returns (for positions q+2..q+B+1), which
    is where a test scripts full accept, partial accept or zero accept. The epoch FENCE is honoured
    exactly as the stages honour it: a frame older than the newest epoch seen is answered with a
    `fenced` reply and nothing is computed or advanced."""

    _WRONG = 90000

    def __init__(self, truth, blocks):
        self.truth = dict(truth)                          # absolute pos -> the model's token there
        self.blocks = blocks                              # committed pos -> the block drafted there
        self.pipe_a, self.pipe_b = _pair(timeout=10)
        self.ret_a, self.ret_b = _pair(timeout=10)
        self.frames = []                                  # (ids, start_pos, epoch) per step frame seen
        self.fenced = []                                  # the frames the fence dropped
        self.lie = None                                   # force an `acc` divergence
        self.err = None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        cfront = mfront = None
        epoch = 0
        try:
            while True:
                msg = recv_msg(self.pipe_b)
                op = msg.get("op")
                if op == "reset":
                    send_msg(self.ret_a, "ok")
                    continue
                if op != "step":
                    return
                pos, ep = int(msg["start_pos"]), int(msg.get("epoch", 0))
                self.frames.append((list(msg["ids"][0]), pos, ep))
                if msg.get("fenced") or ep < epoch:
                    self.fenced.append(pos)
                    send_msg(self.ret_a, {"fenced": True, "epoch": ep, "pos": pos})
                    continue
                epoch = max(epoch, ep)
                if pos == 0:                              # prefill: seeds the frontier, drafts nothing
                    p = len(msg["ids"][0])
                    cfront, mfront = p - 1, self.truth[p]
                    send_msg(self.ret_a, {"token": mfront, "tokens": [mfront], "acc": True,
                                          "epoch": ep, "pos": pos})
                    continue
                acc = (pos == cfront + 1 and int(msg["ids"][0][0]) == mfront)
                m = self.truth[pos + 1] if acc else self._WRONG
                out = {"token": m, "tokens": [m], "epoch": ep, "pos": pos,
                       "acc": (self.lie if self.lie is not None else acc)}
                if acc:
                    cfront, mfront = pos, m
                    blk = self.blocks(pos)
                    if blk is not None:
                        out["draft"] = list(blk)
                send_msg(self.ret_a, out)
        except (OSError, EOFError) as e:                  # the coordinator closed: the run is over
            self.err = e
        except Exception as e:                            # noqa: BLE001 — a script thread that dies
            self.err = e                                  # would otherwise surface as a recv timeout
            traceback.print_exc()

    def close(self):
        for s in (self.pipe_a, self.pipe_b, self.ret_a, self.ret_b):
            try:
                s.close()
            except OSError:
                pass


PIPE_PROMPT = [1, 2, 3]
# The "greedy stream" the scripted tail defends: positions 3.. of a 3-token prompt.
TRUTH = {p: 100 + p for p in range(3, 40)}


def _pipelined(blocks, max_new, eos_ids=(), lie=None, truth=None, **kw):
    ring = _ScriptedPipeRing(truth or TRUTH, blocks)
    ring.lie = lie
    try:
        r = VP.coordinate_dspark_pipelined(ring.pipe_a, ring.ret_b, list(PIPE_PROMPT), max_new,
                                           eos_ids=eos_ids, timeout=10, **kw)
    finally:
        ring.close()
    assert not isinstance(ring.err, (KeyError, IndexError, AssertionError)), \
        f"the scripted tail itself died: {type(ring.err).__name__}: {ring.err}"
    return r, ring


def _perfect_block(q, n=3, truth=None):
    """The drafts a committed frame at `q` returns when the drafter is RIGHT: the model's own tokens
    at q+2..q+n+1, which is where advance_and_draft's block sits."""
    t = truth or TRUTH
    return [t[q + 2 + i] for i in range(n)]


def test_pipelined_streams_the_block_as_separate_s1_frames():
    """THE LEVER, on the wire. The prompt still goes in as one prefill frame; everything after it is a
    SINGLE-POSITION frame at its own start_pos, streamed contiguously — never a `[cur]+drafts` chunk
    that each stage would have to replay position by position."""
    r, ring = _pipelined(_perfect_block, max_new=6)
    assert r["tokens"] == [TRUTH[p] for p in range(3, 9)]
    assert ring.frames[0] == ([1, 2, 3], 0, 0), "prefill is still the whole prompt at position 0"
    steps = ring.frames[1:]
    assert all(len(ids) == 1 for ids, _, _ in steps), "a pipelined frame is s=1, always"
    assert [p for _, p, _ in steps[:4]] == [3, 4, 5, 6], "frames stream contiguously by position"
    assert [ids[0] for ids, _, _ in steps[:4]] == [TRUTH[p] for p in range(3, 7)]


def test_pipelined_full_accept_never_cancels_and_fills_the_pipeline():
    """A drafter that is always right: no cancel ever fires, every draft is accepted, and the depth
    reached is block+1 frames in the ring at once — which is the entire point of the lever."""
    r, ring = _pipelined(_perfect_block, max_new=10)
    assert r["tokens"] == [TRUTH[p] for p in range(3, 13)]
    assert r["cancels"] == 0 and r["accept_hist"] == {r["accepted"]: 1}, "one uninterrupted cycle"
    assert r["max_inflight"] == 4, "block of 3 + the frame at the frontier should all be in flight"
    assert r["accepted"] >= 6 and r["g"] == r["generated"], "no rollback: every token in one cycle"
    # The only frames the fence ever touched are the ones still in the ring when max_new landed —
    # they are drained, not believed, which is why a stopped run cannot emit a token past its limit.
    assert r["stale_replies"] + r["unsent_frames"] > 0


def test_pipelined_partial_accept_cancels_at_the_first_divergence():
    """The block is right for one position and wrong after it. The reply that reveals it commits the
    MODEL's token there (never the draft), fences the frames already streamed behind it, and re-opens
    at that position — whose start_pos IS the rewind command every stage obeys."""
    def blocks(q):
        b = _perfect_block(q)
        return [b[0], 777, 778]                       # d1 right, then divergence
    r, ring = _pipelined(blocks, max_new=8)
    assert r["tokens"] == [TRUTH[p] for p in range(3, 11)], "lossless whatever the drafter proposed"
    assert r["cancels"] >= 1 and r["stale_replies"] >= 1, "the fence dropped the discarded future"
    assert r["accepted"] >= 1, "the matching prefix of each block was accepted"
    assert sorted({e for _, _, e in ring.frames}) == list(range(r["cancels"] + 1)), \
        "the epoch is a dense generation counter, one bump per cancel"
    # THE REWIND, read off the wire: the first frame of every new epoch opens BEHIND the last frame of
    # the one it cancelled, and it carries the model's own token — that start_pos is the whole rollback
    # command, which is why no cancel op crosses the ring.
    first, last = {}, {}
    for ids, pos, ep in ring.frames[1:]:
        first.setdefault(ep, (ids[0], pos))
        last[ep] = pos
    for ep in sorted(first)[1:]:
        tok, pos = first[ep]
        assert pos <= last[ep - 1], f"epoch {ep} opened at {pos}, ahead of the discarded {last[ep-1]}"
        assert tok == TRUTH[pos], "a correction frame carries the model's token, never the draft"


def test_pipelined_zero_accept_degrades_to_greedy_and_stays_lossless():
    """A useless drafter costs wasted ring work and never correctness: every block is rejected at its
    first position, so each reply commits the model's own token and the stream is still greedy's."""
    r, _ = _pipelined(lambda q: [901, 902, 903], max_new=6)
    assert r["tokens"] == [TRUTH[p] for p in range(3, 9)]
    assert r["accepted"] == 0, "nothing the drafter proposed was ever right"
    # One cycle per committed token — every block is rejected at its first position, so every reply
    # after the first cancels and re-opens. (The last one is not counted: max_new ends the run before
    # the rejection it would have been.)
    assert r["cancels"] == 3 and r["accept_hist"] == {0: 4}


def test_pipelined_stops_at_eos_inside_a_block():
    """EOS lands mid-block, with the frames behind it already in the ring: they must be drained and
    their tokens never emitted."""
    r, _ = _pipelined(_perfect_block, max_new=10, eos_ids=(TRUTH[6],))
    assert r["tokens"] == [TRUTH[3], TRUTH[4], TRUTH[5], TRUTH[6]]
    assert r["generated"] == 4


def test_pipelined_honors_max_new_mid_block():
    r, _ = _pipelined(_perfect_block, max_new=3)
    assert r["tokens"] == [TRUTH[3], TRUTH[4], TRUTH[5]]


@pytest.mark.parametrize("max_new,eos", [(1, ()), (8, (TRUTH[3],))])
def test_pipelined_run_that_ends_at_the_prefill_streams_nothing(max_new, eos):
    """The degenerate ends — max_new=1, and an EOS the prefill itself produced. Nothing is ever
    streamed, so the sender is started and stopped without a frame, and the drain must terminate on an
    empty pipeline rather than parking in a recv for a reply nobody owes."""
    r, ring = _pipelined(_perfect_block, max_new=max_new, eos_ids=eos)
    assert r["tokens"] == [TRUTH[3]] and r["frames"] == 0
    assert [f[1] for f in ring.frames] == [0], "only the prefill frame reached the ring"


def test_pipelined_caps_the_frames_in_flight_at_the_stages_rollback_depth():
    """The stage refuses to rewind past its oldest live checkpoint (v4_stage's W-deep ring), so the
    coordinator must never stream deeper than that. depth=2 leaves one draft behind the frontier."""
    r, _ = _pipelined(_perfect_block, max_new=8, depth=2)
    assert r["tokens"] == [TRUTH[p] for p in range(3, 11)]
    assert r["max_inflight"] <= 2


def test_pipelined_fails_loudly_when_the_tail_disagrees_about_the_accept():
    """The protocol tripwire, pipelined. The tail's drafter advances on its own incremental accept and
    the coordinator commits on its own; if the two ever disagree the drafter is conditioning on a
    history the ring never took, and every later block looks plausible while being wrong."""
    with pytest.raises(RuntimeError, match="two ends of a lossless round have diverged"):
        _pipelined(_perfect_block, max_new=6, lie=False)


def test_pipelined_needs_a_tail_that_is_actually_drafting():
    """No `acc` in the prefill reply = the tail did not honour the reset's `pipelined` flag (or is not
    drafting at all). Fail the job at the first frame rather than mis-reading every reply after it."""
    class _Mute(_ScriptedPipeRing):
        def _run(self):
            try:
                while True:
                    msg = recv_msg(self.pipe_b)
                    if msg.get("op") == "reset":
                        send_msg(self.ret_a, "ok")
                    elif msg.get("op") == "step":
                        send_msg(self.ret_a, {"token": TRUTH[3], "tokens": [TRUTH[3]]})
                    else:
                        return
            except (OSError, EOFError):
                return
    ring = _Mute(TRUTH, _perfect_block)
    try:
        with pytest.raises(RuntimeError, match="carries no `acc`"):
            VP.coordinate_dspark_pipelined(ring.pipe_a, ring.ret_b, list(PIPE_PROMPT), 4, timeout=10)
    finally:
        ring.close()


def test_sender_side_epoch_fence_drops_queued_frames_without_stranding_the_receiver():
    """The other half of the fence, unit-tested because a live toy ring drains the queue too fast to
    hit it: when a cancel fires, frames already QUEUED for a now-dead epoch must never reach the wire,
    and dropping one must not leave the receiver waiting for a reply that will never come. The check
    and the `outstanding` append are under one lock precisely so that accounting cannot drift."""
    a, b = _pair(timeout=2)
    state = VP._PipeSpecState()
    sender = VP._FrameSender(a, state)
    state.epoch = 1                                    # the cancel already happened
    sender.put({"op": "step", "ids": [[7]], "start_pos": 9, "epoch": 0}, 0)     # stale: must be dropped
    sender.put({"op": "step", "ids": [[8]], "start_pos": 9, "epoch": 1}, 1)     # live: must be sent
    sender.start()
    got = _drain(b, 0.4)
    sender.stop()
    assert [m["ids"] for m in got] == [[[8]]], "a fenced frame reached the wire"
    with state.lock:
        assert state.pending == 1, "the live frame is owed exactly one reply"
        assert state.unsent == 1 and list(state.outstanding) == [(9, 1)]
    for s in (a, b):
        s.close()
# ── 7d. the job horizon the reset frame carries for v4_ref_slim ───────────────────────────────────
# v4_ref_slim's indexer skip may drop the Indexer's COMPRESSOR — which is state, not a query — but
# only for a job that provably never leaves the select-all regime. Under-declare the horizon and the
# indexer re-engages against a half-filled cache and picks the wrong keys, silently. So the bar these
# pin is one-directional: the declared `max_pos` must UPPER-BOUND every absolute position the ring is
# actually asked for, on all three coordinators, including the speculative overshoot a rejected draft
# puts on the wire before it is rolled back.

def _max_end_pos(chunks):
    """The furthest absolute position the ring was actually driven to, over every step frame seen."""
    return max(pos + len(ids) for ids, pos in chunks)


def test_greedy_reset_declares_a_horizon_that_bounds_every_position():
    ring = _ScriptedRing(first_token=10, replies=[[11], [12], [13]])   # one per post-prefill step
    try:
        VP.coordinate(ring.pipe_a, ring.ret_b, [1, 2, 3], 4, timeout=10)
    finally:
        ring.close()
    assert ring.reset_msg["max_pos"] == 3 + 4, "greedy has no draft overshoot: prompt + max_new, exact"
    assert ring.reset_msg["max_pos"] >= _max_end_pos(ring.chunks)


def test_spec_reset_horizon_covers_the_draft_chunk_on_top_of_max_new():
    """A spec round puts `[cur] + K drafts` on the wire at the committed position, so the ring touches
    positions past the last COMMITTED one even when the draft is rejected. K+1 covers exactly that."""
    drafts = [[11, 12, 13, 14], [21, 22, 23, 24]]
    replies = [[11, 12, 13, 14, 15], [21, 22, 23, 24, 25]]
    ring = _ScriptedRing(first_token=10, replies=replies)
    script = list(drafts)
    try:
        VP.coordinate_spec(ring.pipe_a, ring.ret_b, [1, 2, 3], 6, K=4, timeout=10,
                           drafter=lambda ids, k: list(script.pop(0)))
    finally:
        ring.close()
    assert ring.reset_msg["max_pos"] == 3 + 6 + 5
    assert ring.reset_msg["max_pos"] >= _max_end_pos(ring.chunks), "horizon must bound the wire"


def test_dspark_reset_horizon_bounds_a_tail_drafted_block_of_unknown_size():
    """The coordinator does not own the drafter here — the tail's MTP block size is not knowable at
    reset — so the margin is deliberately fat rather than derived. Over-declaring only forgoes an
    optimisation; under-declaring is a silent wrong answer."""
    script = [([21], [31, 32, 33]), ([31, 32, 33, 44], [51, 52, 53])]
    ring = _ScriptedDsparkRing(first_token=10, script=script)
    try:
        VP.coordinate_dspark(ring.pipe_a, ring.ret_b, [1, 2, 3], 6, timeout=10)
    finally:
        ring.close()
    assert ring.reset_msg["max_pos"] == 3 + 6 + VP._SPEC_POS_MARGIN
    assert VP._SPEC_POS_MARGIN >= 8, "the margin must clear any block a real tail drafts"
    assert ring.reset_msg["max_pos"] >= _max_end_pos(ring.chunks)


def test_job_horizon_absent_from_the_frame_is_the_SAFE_answer():
    """A reset frame with no `max_pos` — an older coordinator, or a hand-built frame — must land as
    None, which v4_ref_slim reads as "horizon unknown" and answers by KEEPING the compressor advanced.
    Correct at any length; only the guaranteed-short optimisation is forgone."""
    slim = pytest.importorskip("v4_ref_slim")
    try:
        VP._set_job_horizon({"op": "reset"}.get("max_pos"))
        assert slim._JOB_MAX_POS is None
        VP._set_job_horizon(512)
        assert slim._JOB_MAX_POS == 512
        VP._set_job_horizon(None)                          # teardown clears it: no leak into next job
        assert slim._JOB_MAX_POS is None
    finally:
        slim.set_job_max_pos(None)


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


def test_launch_defaults_cuda_graph_on_with_an_opt_out():
    """A GPU ring launch turns the partial island graphs ON by default (the ring is CPU-launch-bound
    and the graphs are bit-exact, per test_v4_stage_graph.py) while the module default stays OFF for
    a bare import / the CPU parity suite / an in-process ring. The env is placed BEFORE extra_env, so
    an explicit V4_CUDA_GRAPH there still wins — bash takes the rightmost assignment of a name."""
    on = VP.stage_launch_cmd(0, 3, 0, 15)
    assert "V4_CUDA_GRAPH=1 " in on and on.index("V4_CUDA_GRAPH=1 ") < on.index("V4_DIR=")
    off = VP.stage_launch_cmd(0, 3, 0, 15, cuda_graph=False)
    assert "V4_CUDA_GRAPH=0 " in off
    override = VP.stage_launch_cmd(0, 3, 0, 15, extra_env="V4_CUDA_GRAPH=0 ")
    assert override.index("V4_CUDA_GRAPH=1 ") < override.index("V4_CUDA_GRAPH=0 ")


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
        self.head_addr = f"127.0.0.1:{ports[0]}"
        self.tail_addr = f"127.0.0.1:{tail_port}"
        self.pipe, self.ret = VP.connect_ring(self.head_addr, self.tail_addr, timeout=60)

    def coordinate(self, prompt, max_new, nonce=None):
        return VP.coordinate(self.pipe, self.ret, prompt, max_new, nonce=nonce,
                             receipts=self.receipts, layer_count=self.layer_count, timeout=60)

    def spec(self, prompt, max_new, nonce=None, **kw):
        return VP.coordinate_spec(self.pipe, self.ret, prompt, max_new, nonce=nonce,
                                  receipts=self.receipts, layer_count=self.layer_count,
                                  timeout=60, **kw)

    def dspark(self, prompt, max_new, nonce=None, **kw):
        return VP.coordinate_dspark(self.pipe, self.ret, prompt, max_new, nonce=nonce,
                                    receipts=self.receipts, layer_count=self.layer_count,
                                    timeout=60, **kw)

    def pipelined(self, prompt, max_new, nonce=None, **kw):
        return VP.coordinate_dspark_pipelined(self.pipe, self.ret, prompt, max_new, nonce=nonce,
                                              receipts=self.receipts, layer_count=self.layer_count,
                                              timeout=60, **kw)

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


def test_ring_survives_a_coordinator_disconnect(tiny):
    """A coordinator that DROPS its sockets — a finished bench, a crash, an EOF, a restart — must not
    take the warm ring down with it. The head reads only from the coordinator and the tail answers
    only it, so before this guard the first coordinator's close cascaded the whole ring (a ~25min
    re-warm per bench, the single biggest iteration tax). Now the head re-accepts a reconnecting
    coordinator's pipe and the tail's background thread re-accepts its return channel and swaps it in,
    so a BRAND-NEW coordinator dialed onto the SAME still-running stages decodes the same stream.

    The disconnect is modelled exactly as the real one: the sockets are closed with NO stop op (a
    stop would tear the ring down on purpose), then connect_ring re-dials the very same head/tail
    addresses — which is what a re-run bench or a restarted node daemon does."""
    d, args, ref = tiny
    ring = _Ring(d, args, [(0, 3), (3, 6), (6, 8)])
    try:
        assert ring.coordinate(list(PROMPT), NEW)["tokens"] == ref, "baseline job diverged"
        # the coordinator vanishes WITHOUT a stop op: just drop both sockets, as a crash/EOF does
        ring.pipe.close()
        ring.ret.close()
        # a fresh coordinator dials the SAME warm ring — head re-accept + tail return re-accept
        ring.pipe, ring.ret = VP.connect_ring(ring.head_addr, ring.tail_addr, timeout=60)
        assert ring.coordinate(list(PROMPT), NEW, nonce="after-reconnect")["tokens"] == ref, \
            "the ring did not survive the coordinator reconnect"
        # and it is genuinely still warm: a THIRD job on the reconnected coordinator still answers
        assert ring.coordinate(list(PROMPT), NEW)["tokens"] == ref, "second post-reconnect job diverged"
    finally:
        ring.close()


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


def test_conf_send_len_is_a_survival_prefix():
    """The gate's arithmetic, in isolation: keep the leading run of drafts whose raw confidence stays
    >= thresh, stop at the first below (once a position is predicted to reject, the rest is moot),
    floored at min_send and capped at the block. `conf` is a raw score, so this thresholds it
    directly rather than treating it as a probability."""
    L = VP._conf_send_len
    assert L([], 0.0, 1) == 0                                  # round 1: no block, nothing to send
    assert L([0.9, 0.8, 0.7], -9.9, 1) == 3                   # all above the floor: the whole block
    assert L([0.9, -0.1, 0.8], 0.0, 1) == 1                   # position 2 predicted to reject: send 1
    assert L([-0.2, 0.9, 0.9], 0.0, 1) == 1                   # position 1 low but min_send holds it at 1
    assert L([-0.2, 0.9, 0.9], 0.0, 0) == 0                   # min_send 0: a bare greedy round
    assert L([0.9, 0.9, 0.9], 9.9, 1) == 1                    # nothing clears a high floor: min_send
    assert L([0.5, 0.5], 0.5, 1) == 2                         # the floor is inclusive (>=)


def test_dspark_confidence_gate_is_lossless_and_adapts_send_length(tiny):
    """The confidence gate truncates the OFFERED block, never the answer. Run the SAME drafted ring
    three ways — ungated, gated with an unreachable floor (every block trims to min_send=1), and
    gated with a floor nothing clears from below (every block sent whole) — and all three must emit
    the reference's greedy stream. That is losslessness by construction: the tail verifies exactly
    what it receives, so the send-length is a throughput knob and can never move a token. The `sent`
    counts prove the knob actually MOVED: trimming sends strictly fewer draft tokens than sending the
    blocks whole."""
    d, args, ref = tiny
    ring = _Ring(d, args, VP._dspark_tiling(args, 3), dspark=True)
    probe = []
    try:
        base = ring.dspark(list(PROMPT), NEW)
        trim = ring.dspark(list(PROMPT), NEW, conf_gate=True, conf_thresh=float("inf"), conf_min=1,
                           conf_probe=lambda confs, n: probe.append((len(confs), n)))
        whole = ring.dspark(list(PROMPT), NEW, conf_gate=True, conf_thresh=float("-inf"))
    finally:
        ring.close()
    for name, r in (("ungated", base), ("trimmed", trim), ("whole-block", whole)):
        assert r["tokens"] == ref, f"{name} dspark stream {r['tokens']} != greedy {ref}"
    # the floor is unreachable, so every drafted round trimmed to exactly one offered draft
    assert max(trim["send_hist"], default=0) <= 1 and trim["sent"] == trim["drafted"]
    # and nothing clears -inf from below, so the whole-block run offered strictly more than the trim
    assert whole["sent"] > trim["sent"], f"the gate did not vary send-length: {whole['sent']} vs {trim['sent']}"
    # the probe saw every drafted round's FULL block confidence and its accept count
    assert probe and len(probe) == trim["drafted"]
    assert all(clen == args.dspark_block_size and 0 <= n <= clen for clen, n in probe)


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


# ── 10. PIPELINED speculation on the real ring: same stream, streamed ──────────────────────────────
# The lever docs/V4_PIPELINED_SPEC.md specifies, against the same bar everything above it meets: the
# reference Transformer's own greedy tokens, bit for bit. What is new here is the SHAPE of the traffic
# (B+1 separate s=1 frames in flight instead of one chunk), which is exactly the thing that could
# silently change the answer — a stage rewinding across compression boundaries with several frames of
# speculation already past it. Every test in this section is the same assertion in a different regime.

NEW_LONG = 10
# A prompt whose greedy continuation is FIVE DISTINCT tokens (the selftest's, and asserted below).
# The module's own PROMPT collapses into a fixed point after three tokens, which is fine for "does
# the ring reproduce the reference" and useless for "did corrupting the ring change the answer" — a
# repeated token is reachable from a poisoned state too. Any test whose RED half is a divergence has
# to run on a stream that can actually diverge.
FP_PROMPT = [168, 15, 493, 72, 22]


@pytest.fixture(scope="module")
def fp_ref(tiny):
    """The toy model's greedy stream for FP_PROMPT, and the assertion that it is a fingerprint."""
    R = pytest.importorskip("v4_ref_cpu")
    _d, args, _ref = tiny
    ref = VP._reference_tokens(R.build_oracle(args), FP_PROMPT, NEW)
    assert len(set(ref)) == NEW, f"FP_PROMPT is no longer a fingerprint: {ref}"
    return ref


@pytest.fixture(scope="module")
def long_ref(tiny):
    """The toy model's greedy stream, further out: 10 tokens, so a cancel can be placed on a
    compression boundary a 5-token run never reaches (ratio 4 compresses at position 7, 11, ...).

    A FRESH oracle for the same reason the `tiny` fixture builds one and caches its answer — the
    reference Transformer is stateful across sequences. build_oracle is seeded, so these are the same
    weights the ring is serving out of the checkpoint."""
    R = pytest.importorskip("v4_ref_cpu")
    _d, args, _ref = tiny
    return VP._reference_tokens(R.build_oracle(args), PROMPT, NEW_LONG)


class _BlockScriptDrafter:
    """A TAIL_DRAFTER double that keeps the REAL drafter and replaces only what it PROPOSES.

    Acceptance is the one thing a random-weight toy model cannot produce: a trained speculator with
    nothing to be right about drafts noise, so every block is rejected and the accepted path never
    runs. Substituting the block with the reference's own continuation forces it — and it is not
    circular, because a draft is accepted only where the RING's own reply at that position equals it.
    A wrong ring rejects the substituted block and the correction it commits instead diverges from the
    reference all the same (the same argument `perfect()` makes in the selftest).

    Everything else — the frontier rule, the advance, the position discipline, the reply protocol — is
    the real RingDrafter's, built lazily off the tail Stage the first frame hands it."""

    def __init__(self, ckpt_dir, block_for):
        self.ckpt_dir = ckpt_dir
        self.block_for = block_for                 # (start_pos, the real block) -> the block to send
        self.pipelined = False
        self._inner = None

    def on_chunk(self, msg, st, out):
        if self._inner is None:
            self._inner = VP._dspark().ring_drafter(st, self.ckpt_dir)
        self._inner.pipelined = self.pipelined
        r = self._inner.on_chunk(msg, st, out) or {}
        if r.get("draft"):
            r = dict(r, draft=list(self.block_for(int(msg["start_pos"]), r["draft"])))
        return r


def _oracle_blocks(ref, poison_at=()):
    """Blocks carrying the reference's own tokens for the positions they propose — except at
    `poison_at`, where a token the model never produces forces a rejection AT that exact position (and
    therefore a rewind to it)."""
    bad = set(poison_at)

    def at(p):
        i = p - len(PROMPT)
        return 1 if p in bad else (ref[i] if 0 <= i < len(ref) else 1)

    return lambda start_pos, real: [at(start_pos + 2 + i) for i in range(len(real))]


def test_pipelined_ring_matches_the_reference_and_the_serial_path(tiny):
    """THE PROOF, on real stages: the drafted block streamed as separate s=1 frames emits the
    reference Transformer's greedy tokens, bit for bit — and the same tokens the SERIAL drafted path
    emits, on the same warm ring. Restructuring how the block travels is allowed to change the
    throughput and nothing else.

    At random weights the MTP drafter is wrong essentially every time, so this is the ZERO-ACCEPT
    regime: every block is rejected at its first position, every rejection fences the frames already
    streamed behind it and rewinds the stages W deep. It is the path that had to be proven safe."""
    d, args, ref = tiny
    ring = _Ring(d, args, VP._dspark_tiling(args, 3), receipts=True, dspark=True)
    try:
        serial = ring.dspark(list(PROMPT), NEW, nonce="pipe-serial")
        piped = ring.pipelined(list(PROMPT), NEW, nonce="pipe-nonce")
    finally:
        ring.close()
    assert piped["tokens"] == ref, f"pipelined ring {piped['tokens']} != greedy {ref}"
    assert piped["tokens"] == serial["tokens"], "pipelined and serial dspark disagree"
    assert piped["receipts_ok"] is True, "a pipelined job must still settle its receipts"
    assert piped["max_inflight"] > 1, "nothing was ever pipelined — this is greedy with extra steps"
    assert piped["cancels"] > 0 and piped["stale_replies"] > 0, \
        "the rejection path (and with it the W-deep rewind) never ran"


def test_pipelined_ring_full_accept_fills_the_pipeline_and_never_rewinds(tiny):
    """The OTHER extreme, forced with the reference's own continuation as the block: every draft is
    accepted, so no cancel ever fires, the ring holds block+1 frames at once, and the stream is still
    exactly greedy's. This is the regime the throughput projection assumes and the one the toy model's
    own drafter cannot reach."""
    d, args, ref = tiny
    double = _BlockScriptDrafter(d, _oracle_blocks(ref))
    ring = _Ring(d, args, VP._dspark_tiling(args, 3), receipts=True, dspark=True)
    try:
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(VP, "TAIL_DRAFTER", double)
            r = ring.pipelined(list(PROMPT), NEW, nonce="pipe-accept")
    finally:
        ring.close()
    assert r["tokens"] == ref, f"full-accept pipelined ring {r['tokens']} != greedy {ref}"
    assert r["cancels"] == 0, "a correct block was rejected — the verify rows are not the greedy rows"
    assert r["max_inflight"] == args.dspark_block_size + 1, \
        f"the pipeline never filled: {r['max_inflight']} frames deep"
    assert r["receipts_ok"] is True


@pytest.mark.parametrize("boundary", [7, 11])
def test_pipelined_cancel_on_a_compression_boundary_is_lossless(tiny, long_ref, boundary):
    """A CANCEL THAT LANDS EXACTLY ON A COMPRESSION BOUNDARY — the zero-margin case the whole W-deep
    rollback rests on.

    cpu_args compresses at ratio 4 and ratio 8, so slot j is written at position (j+1)*ratio-1: 7 is a
    boundary for BOTH, 11 for the ratio-4 layers. Poisoning the draft at exactly that position makes
    the reply that judges it reject, which commits the model's token there and re-opens the ring AT
    the boundary — with the frames streamed past it already folded into the compressor accumulators.
    The stage's snapshot deliberately does not cover the compressed region (v4_stage._snapshot: a
    poisoned slot is always rewritten before its first read, an argument that is position-local and
    therefore depth-invariant). If that argument were wrong at any depth, this is where it shows."""
    d, args, _ref = tiny
    double = _BlockScriptDrafter(d, _oracle_blocks(long_ref, poison_at=(boundary,)))
    ring = _Ring(d, args, VP._dspark_tiling(args, 3), receipts=True, dspark=True)
    try:
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(VP, "TAIL_DRAFTER", double)
            r = ring.pipelined(list(PROMPT), NEW_LONG, nonce=f"pipe-b{boundary}")
    finally:
        ring.close()
    assert r["tokens"] == long_ref, f"rewind onto boundary {boundary}: {r['tokens']} != {long_ref}"
    assert r["cancels"] == 1, "the poisoned draft did not cause exactly one rewind"
    assert r["receipts_ok"] is True


def _drive_pipelined(ring, prompt, steps, *, stale_at=None):
    """Drive a pipelined ring frame by frame, SYNCHRONOUSLY, so a test can inject a frame the
    coordinator would never send. Each frame carries the model's own token at its position, so every
    one of them is on the committed path and the emitted stream must be greedy's.

    -> (tokens, the reply to the injected frame or None)."""
    send_msg(ring.pipe, {"op": "reset", "swarm_id": "swarm", "job_id": "job", "nonce": None,
                         "temp": 0.0, "seed": 0, "spec": True, "dspark": True, "pipelined": True})
    assert recv_msg(ring.ret) == "ok"
    send_msg(ring.pipe, {"op": "step", "ids": [list(prompt)], "start_pos": 0, "epoch": 1})
    toks = [int(recv_msg(ring.ret)["tokens"][0])]
    pos, injected = len(prompt), None
    for i in range(steps):
        if stale_at == i:
            # THE STALE FRAME: an epoch the ring has already moved past, opened BEHIND the frontier
            # and carrying a token the model never produced. Believed, it would _seek() every stage
            # back onto a live checkpoint and rewrite a committed position's KV with that token —
            # which is precisely what an in-flight frame of a cancelled speculation is.
            send_msg(ring.pipe, {"op": "step", "ids": [[7]], "start_pos": pos - 1, "epoch": 0})
            injected = recv_msg(ring.ret)
        send_msg(ring.pipe, {"op": "step", "ids": [[toks[-1]]], "start_pos": pos, "epoch": 1})
        toks.append(int(recv_msg(ring.ret)["tokens"][0]))
        pos += 1
    return toks, injected


def test_stale_epoch_frame_is_dropped_by_the_ring_and_changes_nothing(tiny, fp_ref):
    """THE EPOCH FENCE, end to end on real stages: a frame from a cancelled generation is answered
    `fenced` and computes nothing — no forward, no checkpoint, no KV write — so the stream is
    bit-identical to the run that never saw it.

    The second half is the anti-vacuity check, and it is the half that matters: with `_fenced` stubbed
    out the SAME injected frame corrupts the stream. So the green above is the fence working, not the
    frame being harmless."""
    d, args, _ref = tiny
    ring = _Ring(d, args, VP._dspark_tiling(args, 3), dspark=True)
    try:
        clean, none_ = _drive_pipelined(ring, FP_PROMPT, NEW - 1)
        fenced, reply = _drive_pipelined(ring, FP_PROMPT, NEW - 1, stale_at=2)
    finally:
        ring.close()
    assert none_ is None and clean == fp_ref, f"the manual driver itself is wrong: {clean}"
    assert reply == {"fenced": True, "epoch": 0, "pos": len(FP_PROMPT) + 1}, \
        f"the stale frame was not fenced: {reply!r}"
    assert fenced == fp_ref, f"a fenced frame changed the stream: {fenced} != {fp_ref}"

    ring2 = _Ring(d, args, VP._dspark_tiling(args, 3), dspark=True)
    try:
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(VP, "_fenced", lambda msg, epoch: False)   # believe the stale frame
            poisoned, _r = _drive_pipelined(ring2, FP_PROMPT, NEW - 1, stale_at=2)
    finally:
        ring2.close()
    assert poisoned != fp_ref, ("the injected frame is harmless even unfenced, so this test proves "
                                "nothing about the fence")


def test_pipelined_and_serial_dspark_jobs_interleave_on_one_warm_ring(tiny):
    """The two coordinators share a ring and a drafter object, and each arms its own mode at reset.
    Running them back to back is how a served box actually meets them, and it is where a leaked
    frontier (`_cfront`/`_mfront`) or a mode left armed from the previous job would show up as a
    second job that cannot reproduce a cold ring's answer."""
    d, args, ref = tiny
    ring = _Ring(d, args, VP._dspark_tiling(args, 3), dspark=True)
    try:
        first = ring.pipelined([7, 7, 7, 1, 2, 9], NEW)            # a different sequence, warming it
        mid = ring.dspark(list(PROMPT), NEW)
        warm = ring.pipelined(list(PROMPT), NEW)
    finally:
        ring.close()
    assert first["tokens"] != ref, "the warming job must be a DIFFERENT sequence to prove anything"
    assert mid["tokens"] == ref and warm["tokens"] == ref


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


def test_job_horizon_reaches_the_stages_and_is_dropped_at_teardown(tiny):
    """The SEAM, end to end: a job's horizon has to arrive at every stage's reference, not just at the
    coordinator's frame. The stages here are threads beside the test, so v4_ref_slim's process-wide
    global IS the thing they set — after the job it holds this job's horizon, and after the stages are
    torn down it is back to None, so nothing leaks into whatever runs next in the process."""
    slim = pytest.importorskip("v4_ref_slim")
    d, args, ref = tiny
    slim.set_job_max_pos(None)
    ring = _Ring(d, args, [(0, 4), (4, 8)])
    try:
        got = ring.coordinate(list(PROMPT), NEW)["tokens"]
        assert got == ref, "the horizon must not change the answer"
        assert slim._JOB_MAX_POS == len(PROMPT) + NEW, "the stages never saw the job's horizon"
    finally:
        ring.close()
    assert slim._JOB_MAX_POS is None, "a torn-down stage left a horizon behind it"


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
