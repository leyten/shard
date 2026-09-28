"""An in-process scripted pipeline for the pspec coordinator (shard/pspec/coord.py).

D stage threads in a row, the coordinator's `Link` on the ends: `send` takes a shard-frame/2 frame
into the head, `recv` hands back the tail's shard-frame/2 reply. Every message crosses as canonical
wire bytes (encode -> decode), so the coordinator is held to the real schema. Per-leg latency is
injectable. It stands in for F4's fake mesh / the node server until they exist; the seam is the same
`Link`, so the coordinator does not change when those plug in.

THE MODEL is tests/oracle.py's: T is the greedy stream, and the token at row p is
oracle_token(T, p) = T[p+1] when the causal history every stage actually holds for rows 0..p is
exactly T[0..p]. Any other history answers a deterministic function of that history: half the time
the plausible T[p+1] (a real model often re-predicts the same token after a perturbation, which is
what makes "accept past the first mismatch" a live bug), otherwise GARBAGE + p, a token no T and no
scripted draft contains. So a coordinator that believes an off-path reply, or a stage left holding a
stale row after a rewind, sooner or later emits a token the greedy stream does not have;
losslessness is `tokens == toy_greedy(...)`, assertable to the token.

THE STAGES model what a v2 node does to a session's KV, strictly:
  * fence: a frame whose sep is below the newest seen (or already fenced) is forwarded header-only
    and the tail answers `fen` (sid/rte/seq/sep kept);
  * commit ON RECEIPT, before the seek (the strictest legal order): rows < cpos become final;
  * seek: a frame past the stage's length is a gap; a frame before it truncates, and truncating
    below the committed frontier is refused;
  * spec window: a frame may end at most W rows past the committed frontier;
  * lease: a frame may not end past max_pos (run_pspec leases exactly the rows the job can use,
    prompt + max_new - 1).
A refusal kills the pipeline and the coordinator's next recv raises ConnectionError naming it.

THE TAIL judges `acc` with the shipped pspec.accept.Frontier (the same rule a real tail runs, so the
coordinator's cross-check asserts something), opens the `eid` to get the frame's ids, checks that no
frame claims a cpos past the frontier the tail itself has committed, checks the frame `kind`
(P only for the prompt; T only for a single committed row, which must be on the path; a chunk is S;
with an uncapped cpos, T is exactly the single-row frame at cpos), and answers a block from
`blocks(q)` (the tail drafter: proposals for q+2.. where q is its new frontier, V4's convention)
on every committed frame.
"""
import queue
import threading
import time
from array import array
from hashlib import blake2b

from oracle import oracle_token

from shard import frame as F
from shard.pspec import coord
from shard.pspec.accept import Frontier

GARBAGE = 2_000_000          # an off-path row answers GARBAGE + position
WRONG = 5_000                # scripted wrong drafts are WRONG + position
SID = "00000000000000a1"


def identity(b):
    """The test `seal`: the eid rides as plaintext (the AEAD belongs to the lease, not to pspec)."""
    return bytes(b)


def _ints(t):
    if t is None:
        return []
    a = array("i")
    a.frombytes(t.data)
    return list(a)


def _i32(vals):
    return F.Tensor("i32", (len(vals),), array("i", [int(v) for v in vals]))


def _wire(msg):
    return F.decode(b"".join(F.encode(msg)))


def truth(n, seed=1234):
    """A greedy stream of n distinct-enough tokens in 2..99 (oracle.novel_T's range)."""
    import random
    rng = random.Random(seed)
    return [rng.randrange(2, 100) for _ in range(n)]


def toy_greedy(T, P, max_new, eos=()):
    """The reference: greedy decode of the toy model, directly (next = oracle token)."""
    ids, out = list(T[:P]), []
    while True:
        t = oracle_token(T, len(ids) - 1)
        ids.append(t)
        out.append(t)
        if t in set(eos) or len(out) >= max_new:
            return out


def _at(T, p):
    return T[p] if p < len(T) else T[-1]


def alpha_miss(alpha, seed=0):
    """A deterministic per-(position, depth) coin: miss with probability 1 - alpha."""
    def miss(p, d):
        h = int.from_bytes(blake2b(f"{seed}/{p}/{d}".encode(), digest_size=8).digest(), "little")
        return h / 2 ** 64 >= alpha
    return miss


def tail_blocks(T, K, miss=None):
    """The tail drafter: blocks(q) -> proposals for q+2..q+K+1 (right unless miss(pos, depth))."""
    def blocks(q):
        return [(WRONG + p if miss and miss(p, d) else _at(T, p))
                for d, p in enumerate(range(q + 2, q + 2 + K), 1)]
    return blocks


def entry_drafter(T, K, miss=None):
    """An entry-side drafter: drafter(ids) -> proposals for len(ids)..len(ids)+K-1."""
    def drafter(ids):
        s = len(ids)
        return [(WRONG + p if miss and miss(p, d) else _at(T, p))
                for d, p in enumerate(range(s, s + K), 1)]
    return drafter


class StageRefused(Exception):
    pass


class _State:
    def __init__(self):
        self.hist = []               # the input row at each position this stage holds (its KV)
        self.good = 0                # len of the longest prefix of hist equal to T
        self.committed = -1          # final-row frontier: rows < committed never roll back
        self.sep_seen = 0


class ScriptedPipeline:
    """D in-process stages implementing coord.Link. Use as a context manager (joins the threads)."""

    def __init__(self, T, *, depth=3, W=16, leg_s=0.0, blocks=None, lie=None, judge=True,
                 frontier=Frontier, sid=SID, rte=0, max_pos=None, tight_cpos=True):
        self.T, self.D, self.W, self.leg_s = list(T), int(depth), int(W), float(leg_s)
        self.max_pos, self.tight_cpos = max_pos, tight_cpos
        self.blocks, self.lie, self.judge = blocks, lie, judge
        self.sid, self.rte = sid, rte
        self.st = [_State() for _ in range(self.D)]
        self.front = frontier()
        self.q = [queue.Queue() for _ in range(self.D)]
        self.ret = queue.Queue()
        self.log = []                # every frame the head received, in order
        self.fenced = []             # (stage, pos, sep) of frames a stage fenced
        self.err = None
        self._lock = threading.Lock()
        self.threads = [threading.Thread(target=self._worker, args=(k,), daemon=True)
                        for k in range(self.D)]
        for t in self.threads:
            t.start()

    # ── the Link ─────────────────────────────────────────────────────────────────────────────────
    def send(self, msg):
        m = _wire(msg)
        if m["op"] != "f":
            raise ValueError(f"the head got a non-frame {m['op']!r}")
        if (m["kind"] == "P") != (not self.log):
            raise StageRefused(f"head: kind {m['kind']} on frame #{len(self.log)} (P is the prompt)")
        rows = _ints(m["x"])
        self.log.append({k: m[k] for k in ("seq", "sep", "pos", "n", "kind", "cpos", "lg", "fen")}
                        | {"toks": rows})
        self.q[0].put((m, rows, [True] * len(rows)))

    def recv(self, timeout):
        end = time.monotonic() + timeout
        while True:
            if self.err is not None:
                raise ConnectionError(f"scripted pipeline: {type(self.err).__name__}: {self.err}")
            rem = end - time.monotonic()
            if rem <= 0:
                raise TimeoutError("scripted pipeline: no reply")
            try:
                return self.ret.get(timeout=min(rem, 0.05))
            except queue.Empty:
                continue

    # ── stages ───────────────────────────────────────────────────────────────────────────────────
    def _worker(self, k):
        while True:
            item = self.q[k].get()
            if item is None:
                if k + 1 < self.D:
                    self.q[k + 1].put(None)
                return
            if self.err is not None:
                continue
            if self.leg_s:
                time.sleep(self.leg_s)
            try:
                out = self._step(k, *item)
            except Exception as e:                            # noqa: BLE001 - surfaced by recv
                with self._lock:
                    self.err = self.err or e
                continue
            (self.q[k + 1] if k + 1 < self.D else self.ret).put(out)

    def _step(self, k, hdr, rows, ok):
        s, tail = self.st[k], k == self.D - 1
        if hdr["fen"] or hdr["sep"] < s.sep_seen:
            self.fenced.append((k, hdr["pos"], hdr["sep"]))
            hdr = dict(hdr, fen=True)
            if tail:
                return _wire(F.reply(sid=hdr["sid"], rte=hdr["rte"], seq=hdr["seq"], sep=hdr["sep"],
                                     pos=hdr["pos"], n=hdr["n"], fen=True))
            return hdr, None, None
        s.sep_seen = max(s.sep_seen, hdr["sep"])
        s.committed = max(s.committed, hdr["cpos"])
        pos, n = hdr["pos"], hdr["n"]
        if pos > len(s.hist):
            raise StageRefused(f"stage {k}: gap, frame at {pos} past length {len(s.hist)}")
        if pos < len(s.hist):
            if pos < s.committed:
                raise StageRefused(f"stage {k}: rewind to {pos} below the committed frontier "
                                   f"{s.committed}")
            del s.hist[pos:]
            s.good = min(s.good, pos)
        if pos + n > s.committed + self.W:
            raise StageRefused(f"stage {k}: frame [{pos}, {pos + n}) past committed {s.committed} "
                               f"+ spec window {self.W}")
        if self.max_pos is not None and pos + n > self.max_pos:
            raise StageRefused(f"stage {k}: frame [{pos}, {pos + n}) past max_pos {self.max_pos}")
        s.hist.extend(rows)
        while s.good < min(len(s.hist), len(self.T)) and s.hist[s.good] == self.T[s.good]:
            s.good += 1
        ok = [ok[j] and pos + j < s.good for j in range(n)]
        return self._reply(hdr, rows, ok) if tail else (hdr, rows, ok)

    def _offpath(self, p):
        """The model's token at an off-path row p: a deterministic function of the recent history the
        tail holds. Half the time it is the PLAUSIBLE token (a real model often re-predicts the same
        continuation after a perturbation), so an accept rule that reads past the first mismatch
        finds matches there; otherwise GARBAGE + p."""
        h = blake2b(repr((p, self.st[-1].hist[max(0, p - 3):p + 1])).encode(), digest_size=2)
        return oracle_token(self.T, p) if h.digest()[0] & 1 else GARBAGE + p

    def _reply(self, hdr, rows, ok):
        pos, n = hdr["pos"], hdr["n"]
        toks = [oracle_token(self.T, pos + j) if ok[j] else self._offpath(pos + j) for j in range(n)]
        acc = blk = None
        if hdr["eid"] is not None and self.judge:
            ids, _ = F.eid_unpack(hdr["eid"], n)
            if ids != rows:
                raise StageRefused(f"tail: eid ids {ids} disagree with the rows fed {rows}")
            if hdr["kind"] == "P":
                self.front.prefill(n, toks[-1])
                acc = True
            else:
                if hdr["cpos"] > self.front.cfront + 1:
                    raise StageRefused(f"tail: cpos {hdr['cpos']} past the committed frontier "
                                       f"{self.front.cfront + 1}")
                acc, _k = self.front.judge(pos, ids, toks)
                if acc and self.blocks is not None:
                    blk = self.blocks(self.front.cfront) or None
                self._check_kind(hdr, acc)
                if self.lie is not None:
                    acc = self.lie
        lg = hdr["lg"]
        out = toks if lg == "all" else toks[-1:] if lg == "last" else []
        return _wire(F.reply(sid=hdr["sid"], rte=hdr["rte"], seq=hdr["seq"], sep=hdr["sep"],
                             pos=pos, n=n, toks=_i32(out), acc=acc,
                             blk=None if blk is None else _i32(blk)))

    def _check_kind(self, hdr, acc):
        kind, pos, n = hdr["kind"], hdr["pos"], hdr["n"]
        if n > 1 and kind != "S":
            raise StageRefused(f"tail: a {n}-row frame at {pos} marked {kind}: drafts ride in it")
        if kind == "T" and not acc:
            raise StageRefused(f"tail: a truth frame at {pos} is off the committed path")
        if self.tight_cpos and (kind == "T") != (n == 1 and pos == hdr["cpos"]):
            raise StageRefused(f"tail: kind {kind} for the {n}-row frame at {pos} with cpos "
                               f"{hdr['cpos']}: T is exactly the committed row at the frontier")

    def close(self):
        self.q[0].put(None)
        for t in self.threads:
            t.join(5)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def run_pspec(T, P, max_new, *, depth=3, W=16, leg_s=0.0, blocks=None, drafter=None,
              policy="chunk", floor=1, eos=(), seal=identity, lie=None, judge=True,
              frontier=Frontier, mod=coord, timeout=20.0):
    """One job: T[:P] prompt, a fresh D-stage pipeline leased exactly the rows the job can use.
    -> (result, pipeline, session). `mod` swaps in a mutated coordinator module
    (tests/test_pspec_mutants.py)."""
    with ScriptedPipeline(T, depth=depth, W=W, leg_s=leg_s, blocks=blocks, lie=lie, judge=judge,
                          frontier=frontier, max_pos=P + max_new - 1) as pipe:
        s = mod.Session(T[:P], max_new, sid=SID, W=W, floor=floor, policy=policy, eos_ids=eos,
                        drafter=drafter, seal=seal)
        r = mod.run(pipe, s, timeout=timeout)
        if pipe.err is not None:
            raise AssertionError(f"the pipeline refused: {pipe.err}")
        return r, pipe, s


def run_acked(T, P, max_new, lag, *, W=8, depth=3, policy="pertoken", blocks=None, drafter=None,
              frontier=Frontier, mod=coord, timeout=20.0):
    """A job whose gateway acknowledges the committed stream `lag` tokens behind the entry (it acks
    on every committed token). -> (result, pipeline, session, built) where built[seq] is the ack
    the session held when it built that frame."""
    built, box = {0: P}, {}

    def gateway(_tok):
        s = box["s"]
        s.ack(max(len(s.prompt), len(s.ids) - lag))

    with ScriptedPipeline(T, depth=depth, W=W, blocks=blocks, frontier=frontier,
                          max_pos=P + max_new - 1, tight_cpos=False) as pipe:
        s = mod.Session(T[:P], max_new, sid=SID, W=W, policy=policy, drafter=drafter, seal=identity,
                        acked=P, on_token=gateway)
        box["s"] = s
        judge = s.on_reply

        def on_reply(rep):
            frames = judge(rep)
            built.update((fr["seq"], s.acked) for fr in frames)
            return frames
        s.on_reply = on_reply
        r = mod.run(pipe, s, timeout=timeout)
        if pipe.err is not None:
            raise AssertionError(f"the pipeline refused: {pipe.err}")
        return r, pipe, s, built
