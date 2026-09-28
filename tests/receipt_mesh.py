"""Scripted receipt scenarios for shard-receipt/2 (BUILD-PLAN P1, library part).

A pure-CPU mesh that emits the exact checkpoint streams a set of hops produces, so
`verify_coverage_v2` and settlement can be attacked before the node server exists (M1 wires the
library into it). Nothing here is a replica of the receipt logic: every hop is a real
`ToyRuntime` stage (shard/runtime/toy.py) holding a real `HopLedger`, rows cross hops as encoded
`shard.frame` messages and are hashed with `frame.wire_row`, a takeover really imports a KVBlob
and runs `rebuild_from` + rebuild segments, and a rebuild really fetches `log_rows`.

The entry is a pipelined speculative coordinator in miniature: prefill chunks, then frames of
[committed token, K drafts] kept `depth` deep in flight, the accept rule (commit truth tokens
while the draft matched, bump `sep` on the first miss), cpos = F = c-1 on every frame. Its drafter
guesses from a fault-free reference run and misses on purpose every `miss` positions, so rows are
recomputed after rejections exactly as on a ring. Waves run every in-flight frame through the
whole path; several sessions in one wave share each hop's step (batching).

Device mix: hops alternate the toy's "chunk_invariant" and "adversarial" numerics (the latter's
bytes depend on segment shape, like float kernels), so every path mixes numerics classes, as a
CUDA card and a Mac would. fp8 is a real per-row-scaled lossy codec over the toy's integers.

Faults and bugs are scripted: kill a hop (optionally losing its in-flight checkpoint), take over
from a KV mirror, rebuild from a predecessor's log, reroute to another path shape, and node bugs
for the mutation suite (fold at c, accept any sender, forge rebuild inputs, lie about a token or
the imported KV).
"""
from __future__ import annotations

import copy
import hashlib
import struct
from collections import deque

import numpy as np
from cryptography.hazmat.primitives.asymmetric import ed25519

from shard import frame as FR
from shard import receipt2 as R2
from shard.runtime.api import KVBlob, Rows, Segment, StageSpec, StepIn
from shard.runtime.toy import P as FIELD
from shard.runtime.toy import ToyRuntime, ToyWeights

MODEL = "toy-mesh"
PATTERN = "SSGSSSGSSGSS"             # 12 layers, global attention at 2, 6, 9
KINDS = tuple("swa" if c == "S" else "global" for c in PATTERN)
L = len(PATTERN)
WINDOW = 6
H, V = 4, 24
W = 16                               # spec window of every lease
MAX_POS = 1024
WEIGHTS = ToyWeights(PATTERN, hidden=H, vocab=V, seed=7)
MANIFEST = WEIGHTS.artifact_cid
MODEL_INFO = {"layer_kinds": list(KINDS), "window": WINDOW}


def key(name: str) -> ed25519.Ed25519PrivateKey:
    return ed25519.Ed25519PrivateKey.from_private_bytes(hashlib.sha256(b"p1/" + name.encode()).digest())


def pub(name_or_priv) -> str:
    priv = key(name_or_priv) if isinstance(name_or_priv, str) else name_or_priv
    return R2._pub(priv)


class NodeDead(Exception):
    pass


class FrameRefused(Exception):
    """C20: a session's frame from anyone but the lease's predecessor."""


# ── the toy wire codecs ───────────────────────────────────────────────────────────────────────
def enc_rows(x: np.ndarray, codec: str) -> dict:
    """int32 [n, H] toy rows -> frame payload. bf16: a lossless bit carrier ([n, 2H] u16 halves).
    fp8: 1 byte per value + a per-row f32 scale (lossy, like e4m3 with row scales)."""
    x = np.ascontiguousarray(x, dtype=np.int32)
    n = x.shape[0]
    if codec == "bf16":
        return {"x": FR.Tensor("bf16", (n, 2 * H), memoryview(x.view(np.uint16).copy()))}
    s = np.maximum(x.max(axis=1).astype(np.float32) / np.float32(255), np.float32(1e-3)).astype(np.float32)
    q = np.clip(np.rint(x / s[:, None]), 0, 255).astype(np.uint8)
    return {"x": FR.Tensor("fp8", (n, H), memoryview(q.copy())), "xs": FR.Tensor("f32", (n,), memoryview(s.copy()))}


def dec_rows(msg) -> np.ndarray:
    x = msg["x"]
    if x.dtype == "bf16":
        return np.frombuffer(bytes(x.data), dtype=np.int32).reshape(x.shape[0], H).copy()
    q = np.frombuffer(bytes(x.data), dtype=np.uint8).reshape(x.shape[0], H).astype(np.float64)
    s = np.frombuffer(bytes(msg["xs"].data), dtype=np.float32).astype(np.float64)
    return np.clip(np.rint(q * s[:, None]), 0, FIELD - 1).astype(np.int32)


def wire(msg) -> bytes:
    """encode + decode: what the next hop really receives."""
    return FR.decode(b"".join(bytes(p) for p in FR.encode(msg)))


def ids_tensor(ids) -> FR.Tensor:
    return FR.Tensor("i32", (len(ids),), struct.pack(f"<{len(ids)}i", *ids))


def ids_of(t) -> list:
    b = bytes(t.data) if isinstance(t, FR.Tensor) else bytes(t)
    return list(struct.unpack(f"<{len(b) // 4}i", b))


# ── a node: one holder of one span ────────────────────────────────────────────────────────────
class Sess:
    def __init__(self):
        self.ledger = None
        self.lease = None
        self.alog = {}           # p -> (codec, x row bytes, xs row bytes) of the output rows it sent
        self.sep = 0
        self.mirror_to = None
        self.mirrored = 0


class Backup:
    """A backup's view of one primary's session: the mirrored KV and its ack ledger."""

    def __init__(self, node, job, primary, rte, ps):
        self.node, self.job, self.primary = node, job, primary
        self.blobs = []
        self.rows = {}           # p -> KV bytes
        self.copies = {}         # checkpoint position -> primary's kv chain (from the ckpt copies)
        self.ledger = R2.HopLedger(node.priv, job_id=job.job_id, sid=job.sid, nonce=job.nonce,
                                   manifest_cid=MANIFEST, numerics_id=node.nid, rte=rte, lo=primary.lo,
                                   hi=primary.hi, role="backup", pos_lo=ps, kv=True, log="mirror",
                                   of=primary.pub)
        self.acked = ps
        self.pending = []
        self.ok = True

    def receive(self, blob: KVBlob, ckpts):
        self.blobs.append(blob)
        for p in range(blob.lo_pos, blob.hi_pos):
            self.rows[p] = blob.pos_bytes(p)
        for ck in ckpts:
            if ck["kind"] == "grid" and "kv" in ck["logs"]:
                self.copies[ck["pos"]] = ck["logs"]["kv"]["chain"]
        if self.ledger.folded < blob.hi_pos:
            self.pending += self.ledger.commit(blob.hi_pos, lambda lo, hi: [self.rows[p] for p in range(lo, hi)])
        keep = []
        for ack in self.pending:                      # ack only what matches the primary's checkpoint
            want = self.copies.get(ack["pos"])
            if want is None:
                keep.append(ack)
            elif self.ok and want == ack["logs"]["kv"]["chain"]:
                self.job.deliver(ack)
                self.acked = ack["pos"]
            else:
                self.ok = False
        self.pending = keep

    def blob(self, upto: int) -> KVBlob:
        rows = []
        for i in range(len(self.blobs[0].layers)):
            parts = [b.rows[i] for b in self.blobs]
            rows.append(Rows.concat(parts).slice(0, upto))
        b0 = self.blobs[0]
        return KVBlob(0, upto, b0.layers, tuple(rows), b0.layout_id, b0.numerics_id)


class Node:
    def __init__(self, name, lo, hi, mode="chunk_invariant", bugs=()):
        self.name, self.lo, self.hi = name, lo, hi
        self.priv = key(name)
        self.pub = pub(self.priv)
        spec = StageSpec(MODEL, lo, hi, is_head=lo == 0, is_tail=hi == L, max_sessions=8,
                         max_ctx=MAX_POS, spec_window=W)
        self.rt = ToyRuntime(spec, pattern=PATTERN, window=WINDOW, hidden=H, vocab=V, mode=mode,
                             max_tokens_per_step=256)
        self.rt.load(WEIGHTS)
        self.nid = self.rt.describe().numerics_id
        self.kv = any(k == "global" for k in KINDS[lo:hi])
        self.head, self.tail = lo == 0, hi == L
        self.role = "solo" if self.head and self.tail else "head" if self.head else "tail" if self.tail else "mid"
        self.S: dict[str, Sess] = {}
        self.backing: dict[tuple, Backup] = {}
        self.alive = True
        self.bugs = set(bugs)
        self.lie_at = None

    def __repr__(self):
        return f"Node({self.name} [{self.lo},{self.hi}))"

    def kv_rows(self, rsid):
        def f(lo, hi):
            b = self.rt.export_state(rsid, lo, hi)
            return [b.pos_bytes(p) for p in range(lo, hi)]
        return f if self.kv else None

    def new_ledger(self, job, rte, ps):
        s = self.S[job.sid]
        s.ledger = R2.HopLedger(self.priv, job_id=job.job_id, sid=job.sid, nonce=job.nonce,
                                manifest_cid=MANIFEST, numerics_id=self.nid, rte=rte, lo=self.lo,
                                hi=self.hi, role=self.role, pos_lo=ps, salt=job.salt,
                                prompt_len=job.P, kv=self.kv)

    def process(self, batch):
        """batch: [(job, msg, sender_pub)] -> [(job, out msg or reply)]. One runtime step."""
        if not self.alive:
            raise NodeDead(self)
        segs, keep = [], []
        for job, msg, sender in batch:
            s = self.S[job.sid]
            if sender != s.lease["pred"] and "accept_any_sender" not in self.bugs:
                raise FrameRefused(f"{self.name}: frame for {job.sid} from a non-predecessor")
            if msg["rte"] != s.lease["rte"] or msg["sep"] < s.sep:
                continue                              # stale lease / fenced: computes nothing
            s.sep = msg["sep"]
            rsid = int(job.sid, 16)
            if msg["cpos"] >= 0:
                self.rt.commit(rsid, msg["cpos"])
                self._fold(job, s, msg["cpos"], [])     # §4.2 step 5: commit on arrival
            n = msg["n"]
            if self.head:
                x = Rows("i32", np.array(ids_of(msg["x"]), dtype=np.int32))
            else:
                x = Rows("i32", dec_rows(msg))
            kind = {"P": "prefill", "R": "rebuild"}.get(msg["kind"], "decode" if n == 1 else "verify")
            segs.append(Segment(rsid, msg["pos"], n, x, kind=kind,
                                tokens=msg["lg"] if self.tail else "none"))
            keep.append((job, msg, kind))
        if not segs:
            return []
        outs = self.rt.step(StepIn(tuple(segs))).outs
        res = []
        for (job, msg, kind), o in zip(keep, outs):
            res.append((job, self._after(job, msg, kind, o)))
        return res

    def _after(self, job, msg, kind, o):
        s, pos, n = self.S[job.sid], msg["pos"], msg["n"]
        cx_in = msg["cx"]
        in_rows = None if self.head else [FR.wire_row(msg, j) for j in range(n)]
        ids = ids_of(msg["x"]) if self.head else (ids_of(msg["eid"]) if self.tail else None)
        if self.tail:
            toks = [None] * n
            if o.tokens is not None:
                drawn = [int(t) for t in o.tokens.data]
                for j, t in enumerate(drawn):
                    toks[n - len(drawn) + j] = t
            out = FR.reply(sid=job.sid, rte=msg["rte"], seq=msg["seq"], sep=msg["sep"], pos=pos, n=n,
                           toks=ids_tensor([t for t in toks if t is not None]))
            out_rows = None
            if self.lie_at is not None and self.lie_at - 1 in range(pos, pos + n):
                toks = list(toks)                          # the tail SIGNS a token it did not return
                j = self.lie_at - 1 - pos
                toks[j] = (toks[j] + 1) % V
        else:
            codec = (job.cx_prefill if msg["kind"] == "P" else job.cx) if self.head else cx_in
            hdr = {k: msg[k] for k in ("sid", "rte", "seq", "sep", "pos", "n", "kind", "cpos", "lg")}
            out = FR.frame(**hdr, cx=codec, eid=msg["eid"], **enc_rows(o.hidden.data, codec))
            out_rows = [FR.wire_row(out, j) for j in range(n)]
            xs = out["xs"]
            for j in range(n):
                s.alog[pos + j] = (codec, bytes(out["x"].row(j)), bytes(xs.row(j)) if xs is not None else b"")
            toks = None
        self._record(s, pos, n, kind, cx_in, in_rows, out_rows, ids, toks)
        self._fold(job, s, msg["cpos"], [])            # and again for this frame's own prompt rows
        return out

    def _fold(self, job, s, cpos, _):
        F = cpos + (1 if "fold_at_c" in self.bugs else 0)
        rsid = int(job.sid, 16)
        cks = s.ledger.commit(F, self.kv_rows(rsid))
        for ck in cks:
            job.deliver(ck)
        if s.mirror_to is not None and s.ledger.folded > s.mirrored:
            blob = self.rt.export_state(rsid, s.mirrored, s.ledger.folded)
            s.mirrored = s.ledger.folded
            s.mirror_to.receive(blob, cks)

    def _record(self, s, pos, n, kind, codec, in_rows, out_rows, ids, toks):
        lg = s.ledger
        if pos < lg.folded:                           # only a node that folded early gets here:
            if "fold_at_c" not in self.bugs:          # the honest ledger refuses the rewrite
                raise AssertionError("rewrite below the folded frontier")
            k = lg.folded - pos                       # the buggy one keeps its early leaf
            if k >= n:
                return
            pos, n = pos + k, n - k
            in_rows = in_rows and in_rows[k:]
            out_rows = out_rows and out_rows[k:]
            ids = ids and ids[k:]
            toks = toks and toks[k:]
        lg.record(pos, n, kind=kind, codec=codec, in_rows=in_rows, out_rows=out_rows, ids=ids, toks=toks)


# ── the entry: a miniature pipelined speculative coordinator + failover driver ────────────────
class Job:
    def __init__(self, path, *, name="job", prompt_len=150, n_new=300, K=3, depth=2, miss=5,
                 cx="bf16", cx_prefill="bf16", chunk=32, backups=None, nonce=None, T=None):
        self.job_id = f"{name}-id"
        self.sid = hashlib.sha256(name.encode()).hexdigest()[:16]
        self.nonce = nonce or hashlib.sha256(b"nonce/" + name.encode()).hexdigest()[:32]
        self.salt = hashlib.sha256(b"salt/" + name.encode()).digest()
        self.entry = key("entry/" + name)
        self.P, self.n_new, self.K, self.depth, self.miss = prompt_len, n_new, K, depth, miss
        self.cx, self.cx_prefill, self.chunk = cx, cx_prefill, chunk
        rng = np.random.default_rng(int(self.sid[:8], 16))
        self.prompt = [int(t) for t in rng.integers(0, V, size=prompt_len)]
        self.T = T if T is not None else reference_tokens(self.prompt, n_new + 64)
        self.ids = list(self.prompt)
        self.sep = self.seq = self.rte = 0
        self.sent = {}
        self.pre = 0
        self.opt = None           # optimistic next row of the in-flight speculation
        self.inflight = deque()
        self.ckpts, self.leases, self.rtes, self.paths = [], [], [], []
        self.path = list(path)
        self.backups = dict(backups or {})        # primary node -> backup node (rte 0)
        self.dead_node = None
        self.lie_at = None        # the entry commits a token the tail never produced at this index
        self.finished = False
        self._open_rte(0, self.path, seated=[])
        for n in self.path:
            self._open_session(n)

    @property
    def c(self):
        return len(self.ids)

    @property
    def F(self):
        return self.c - 1

    def deliver(self, ck):
        self.ckpts.append(ck)

    # leases + path log bookkeeping
    def _open_rte(self, ps, path, seated, backups=None):
        r = len(self.rtes)
        hops = []
        for n in path:
            h = {"pubkey": n.pub, "lo": n.lo, "hi": n.hi, "role": n.role}
            b = (backups if backups is not None else self.backups).get(n)
            if b is not None:
                h["backup"] = b.pub
                self.leases.append({"sid": self.sid, "rte": r, "pubkey": b.pub, "lo": n.lo, "hi": n.hi,
                                    "role": "backup"})
            hops.append(h)
            self.leases.append({"sid": self.sid, "rte": r, "pubkey": n.pub, "lo": n.lo, "hi": n.hi,
                                "role": n.role})
        self.paths.append(list(path))
        self.rtes.append({"rte": r, "pos_start": ps, "pos_end": None, "cx": self.cx,
                          "cx_prefill": self.cx_prefill, "hops": hops, "dead": [], "seated": seated})
        self.rte = r

    def _lease(self, n, i, path):
        return {"rte": self.rte, "pred": path[i - 1].pub if i else self.entry_peer,
                "next": path[i + 1].pub if i + 1 < len(path) else None}

    entry_peer = "entry"

    def on_death(self, j):
        raise AssertionError(f"{j.dead_node} died and the scenario scripted no failover")

    def _open_session(self, n):
        n.S[self.sid] = Sess()
        n.rt.open(int(self.sid, 16), MAX_POS, W)
        i = self.path.index(n)
        n.S[self.sid].lease = self._lease(n, i, self.path)
        n.new_ledger(self, self.rte, self.rtes[-1]["pos_start"])
        b = self.backups.get(n)
        if b is not None:
            bk = Backup(b, self, n, self.rte, self.rtes[-1]["pos_start"])
            b.backing[(self.sid, n.pub)] = bk
            n.S[self.sid].mirror_to = bk

    # frames
    def can_send(self):
        if self.finished or self.dead_node is not None or len(self.inflight) >= self.depth:
            return False
        if self.pre < self.P:
            return True
        if self.opt is None:          # waiting for the first token
            return False
        return self.opt < self.P + self.n_new

    def draft(self, p):
        t = self.T[p] if p < len(self.T) else 0
        return (t + 1) % V if p % self.miss == 0 else t

    def build(self):
        self.seq += 1
        if self.pre < self.P:
            p0, p1 = self.pre, min(self.P, self.pre + self.chunk)
            self.pre = p1
            ids = self.prompt[p0:p1]
            kind, lg, pos, cpos = "P", "last" if p1 == self.P else "none", p0, min(self.F, p1)
        else:
            pos = self.opt
            n = self.K + 1
            ids = [self.ids[pos] if pos < self.c else self.draft(pos)] + [self.draft(pos + j) for j in range(1, n)]
            kind, lg, cpos = ("S" if self.K else "T"), ("all" if self.K else "last"), self.F
            self.opt = pos + n
        for j, t in enumerate(ids):
            self.sent[pos + j] = t
        msg = FR.frame(sid=self.sid, rte=self.rte, seq=self.seq, sep=self.sep, pos=pos, n=len(ids),
                       kind=kind, cpos=cpos, lg=lg, cx="ids", x=ids_tensor(ids), eid=bytes(ids_tensor(ids).data))
        return msg

    def on_reply(self, msg, rep):
        if rep is None:
            return
        if rep["sep"] != self.sep or msg["rte"] != self.rte:
            return                                     # stale (fenced) speculation
        pos, n = msg["pos"], msg["n"]
        toks = ids_of(rep["toks"])
        if msg["kind"] == "P":
            if msg["lg"] == "last":
                self.ids.append(toks[-1])
                self.opt = self.F
            return
        if pos != self.F:
            raise AssertionError(f"reply at {pos} but F={self.F}")
        for j in range(n):
            t = toks[j] if self.c != self.lie_at else (toks[j] + 1) % V
            self.ids.append(t)
            if j + 1 < n and self.sent[pos + j + 1] != t:
                self.sep += 1                          # rejection: fence everything in flight
                self.opt = self.F
                return
        nxt = pos + n                                  # all accepted: is the next frame's guess right?
        if self.inflight and self.sent.get(nxt) != self.ids[nxt]:
            self.sep += 1
            self.opt = self.F

    # faults
    def failover(self, new_path, *, dead=(), seated=None, backups=None):
        """Close the current rte at F and seat `new_path` (lease_update rte+1, truncate_to=F).
        seated: {node: (tier, replaced node or None)} for nodes new to the path."""
        F, e = self.F, self.rtes[-1]
        e["pos_end"] = F
        e["dead"] = [n.pub for n in dead]
        old = self.path
        cont = [n for n in new_path if n in old and n not in dead]
        for n in old:
            if n in dead:
                continue
            n.rt.commit(int(self.sid, 16), F)          # lease_update(truncate_to=F) is a commit to F
            for ck in n.S[self.sid].ledger.close(F, final=False, kv_rows=n.kv_rows(int(self.sid, 16))):
                self.deliver(ck)
            if n not in cont:
                n.rt.close(int(self.sid, 16))
        seated = dict(seated or {})
        info = []
        self.path = list(new_path)
        self.sep += 1
        self.inflight.clear()
        self.opt = F
        self.backups = dict(backups or {})
        fwd = {}
        for i, n in enumerate(new_path):
            if n in cont:
                continue
            tier, repl = seated[n]
            nxt = new_path[i + 1] if i + 1 < len(new_path) else None
            forward = nxt is not None and seated.get(nxt, ("",))[0] == "chain"
            info.append(self._seat(n, i, tier, repl, F, fwd, forward))
        self._open_rte(F, new_path, seated=info)
        for i, n in enumerate(new_path):
            s = n.S[self.sid]
            s.lease = self._lease(n, i, new_path)
            n.new_ledger(self, self.rte, F)
            s.mirror_to = None
            b = self.backups.get(n)
            if b is not None:
                bk = Backup(b, self, n, self.rte, F)
                b.backing[(self.sid, n.pub)] = bk
                s.mirror_to = bk
                s.mirrored = F
        self.dead_node = None

    def _seat(self, n, i, tier, repl, F, fwd, forward):
        rsid = int(self.sid, 16)
        if self.sid in n.S:
            n.rt.close(rsid)
        n.S[self.sid] = Sess()
        n.rt.open(rsid, MAX_POS, W)
        upto = 0
        if tier == "mirror":
            bk = n.backing[(self.sid, repl.pub)]
            g_entry = max([ck["pos"] for ck in self.ckpts if ck["pubkey"] == repl.pub and ck["log"] == "serve"
                           and ck["rte"] == self.rte] + [self.rtes[-1]["pos_start"]])
            upto = min(g_entry, bk.acked)
            n.rt.import_state(rsid, bk.blob(upto), upto)
        a = n.rt.rebuild_from(rsid, F, floor=F)
        fwd_lo = 0 if forward else None
        rb = R2.RebuildLog(n.priv, job_id=self.job_id, sid=self.sid, nonce=self.nonce, manifest_cid=MANIFEST,
                           numerics_id=n.nid, rte=self.rte + 1, lo=n.lo, hi=n.hi, role=n.role, a=a, F=F,
                           upto=upto, tier=tier, replaces=repl.pub if repl is not None else None,
                           kv=n.kv, fwd_lo=fwd_lo, salt=self.salt)
        if upto:
            blob = bk.blob(upto)
            for p in range(upto):
                kvb = blob.pos_bytes(p)
                if "lie_kv" in n.bugs and p == 5:
                    kvb = bytes(len(kvb))               # claims KV it does not hold
                rb.add_kv_import(p, kvb)
        a0 = a // R2.GRID * R2.GRID
        # inputs [a0, F): ids from the entry (head), a chained predecessor's forwarded rows, or the
        # predecessor's activation log via log_rows
        if n.head:
            rows = None
            for p in range(a0, F):
                rb.add_input(p, tok=self.ids[p])
        else:
            pred = self.path[i - 1]
            src = fwd.get(pred.pub) or self._committed_rows(n.lo)
            rows = log_rows(self.sid, src, a0, F)
            for p in range(a0, F):
                row = rows[p][0]
                if "forge_rebuild" in n.bugs and p < a0 + 3:
                    row = bytes(len(row))                # claims inputs it never fetched
                rb.add_input(p, row=row)
        new_fwd = {}
        for s0 in range(a, F, 32):
            s1 = min(F, s0 + 32)
            if n.head:
                x = Rows("i32", np.array(self.ids[s0:s1], dtype=np.int32))
                codec = "ids"
            else:
                x = Rows("i32", np.concatenate([rows[p][1] for p in range(s0, s1)]))
                codec = rows[s0][2]
            o = n.rt.step(StepIn((Segment(rsid, s0, s1 - s0, x, kind="rebuild"),))).outs[0]
            rb.add_segment(s0, s1 - s0, codec)
            if fwd_lo is not None:
                m = FR.frame(sid=self.sid, rte=self.rte + 1, seq=0, sep=0, pos=s0, n=s1 - s0, kind="R",
                             cpos=s1, lg="none", cx="bf16", **enc_rows(o.hidden.data, "bf16"))
                for j in range(s1 - s0):
                    new_fwd[s0 + j] = ("bf16", bytes(m["x"].row(j)), b"")
                    rb.add_output(s0 + j, FR.wire_row(m, j))
        fwd[n.pub] = new_fwd
        n.S[self.sid].alog.update(new_fwd)
        if n.kv:
            blob = n.rt.export_state(rsid, upto, F)
            for p in range(upto, F):
                rb.add_kv(p, blob.pos_bytes(p))
        self.deliver(rb.finish())
        return {"pubkey": n.pub, "tier": tier, "a": a, "upto": upto,
                "replaces": repl.pub if repl is not None else None}

    def _committed_rows(self, b):
        """The committed output rows at layer boundary b: per closed rte, the activation log of the
        hop that sent them (log_fetch goes to whoever held the boundary for those positions)."""
        out = {}
        for e, path in zip(self.rtes, self.paths):
            for n in path:
                if n.hi == b and self.sid in n.S:
                    alog = n.S[self.sid].alog
                    out.update({p: alog[p] for p in range(e["pos_start"], e["pos_end"]) if p in alog})
        return out

    def close(self):
        e = self.rtes[-1]
        e["pos_end"] = self.F
        for n in self.path:
            if n.alive:
                n.rt.commit(int(self.sid, 16), self.F)   # the close carries the final frontier
                for ck in n.S[self.sid].ledger.close(self.F, final=True, kv_rows=n.kv_rows(int(self.sid, 16))):
                    self.deliver(ck)
            else:
                e["dead"].append(n.pub)
        self.finished = True

    def path_log(self):
        body = {"job_id": self.job_id, "sid": self.sid, "nonce": self.nonce, "manifest_cid": MANIFEST,
                "layers": L, "prompt_len": self.P, "W": W, "rtes": copy.deepcopy(self.rtes)}
        return R2.sign_path_log(self.entry, body)

    def bundle(self):
        return {"path_log": self.path_log(), "checkpoints": list(self.ckpts), "leases": list(self.leases),
                "entry_pubkey": pub(self.entry), "nonce": self.nonce, "job_id": self.job_id,
                "tokens": self.ids[self.P:], "job": self}


def log_rows(sid, alog, lo, hi):
    """Serve [lo, hi) of an activation log as log_rows messages (one codec per message), decode
    them as the rebuilding hop does, and return {p: (wire row, int32 row [1, H], codec)}."""
    out, p = {}, lo
    while p < hi:
        codec = alog[p][0]
        q = p
        while q < hi and alog[q][0] == codec and q - p < 1024:
            q += 1
        x = b"".join(alog[t][1] for t in range(p, q))
        kw = {}
        if codec == "bf16":
            kw["x"] = FR.Tensor("bf16", (q - p, 2 * H), x)
        else:
            kw["x"] = FR.Tensor("fp8", (q - p, H), x)
            kw["xs"] = FR.Tensor("f32", (q - p,), b"".join(alog[t][2] for t in range(p, q)))
        m = wire(FR.control("log_rows", sid=sid, pos_lo=p, pos_hi=q, cx=codec, seg=[[p, q - p, codec, "decode"]], **kw))
        rows = dec_rows(m)
        for j in range(q - p):
            out[p + j] = (FR.wire_row(m, j), rows[j:j + 1], codec)
        p = q
    return out


_REF_CACHE = {}


def reference_tokens(prompt, n):
    """The fault-free greedy continuation on one whole-model stage (the drafter's crib)."""
    k = (tuple(prompt), n)
    if k in _REF_CACHE:
        return _REF_CACHE[k]
    rt = ToyRuntime(StageSpec(MODEL, 0, L, is_head=True, is_tail=True, max_ctx=MAX_POS, spec_window=W),
                    pattern=PATTERN, window=WINDOW, hidden=H, vocab=V, max_tokens_per_step=MAX_POS)
    rt.load(WEIGHTS)
    rt.open(1, MAX_POS, W)
    P = len(prompt)
    rt.commit(1, P)
    o = rt.step(StepIn((Segment(1, 0, P, Rows("i32", np.array(prompt, dtype=np.int32)), kind="prefill",
                                tokens="last"),))).outs[0]
    toks = list(prompt) + [int(o.tokens.data[-1])]
    for p in range(P, min(P + n, MAX_POS - 1)):
        rt.commit(1, p)
        o = rt.step(StepIn((Segment(1, p, 1, Rows("i32", np.array([toks[p]], dtype=np.int32)), kind="decode",
                                    tokens="last"),))).outs[0]
        toks.append(int(o.tokens.data[-1]))
    _REF_CACHE[k] = toks
    return toks


# ── the driver ────────────────────────────────────────────────────────────────────────────────
def wave(frames, inject=None):
    """Run each (job, msg) through its job's path, hop by hop; frames of several jobs on the same
    node share one runtime step. Returns the tail replies (None where a hop was dead)."""
    live = [(job, msg, job.entry_peer) for job, msg in frames]
    replies = {id(job): None for job, _ in frames}
    later = []
    hop = 0
    while live or later:
        if not live:
            live, later = later, []
            hop = None
        by_node = {}
        for item in live:
            job, msg, sender = item[:3]
            node = job.path[hop] if hop is not None else item[3]
            by_node.setdefault(node, []).append((job, msg, sender))
        injected = hop is None
        nxt = []
        for node, batch in by_node.items():
            try:
                outs = node.process([(job, wire(msg), sender) for job, msg, sender in batch])
            except NodeDead:
                for job, _, _ in batch:
                    job.dead_node = node
                continue
            for job, out in outs:
                i = job.path.index(node)
                if node.tail:
                    if not injected:
                        replies[id(job)] = wire(out)
                    continue
                if injected:
                    nxt.append((job, out, node.pub, job.path[i + 1]))
                    continue
                nxt.append((job, out, node.pub))
                if inject is not None and inject["node"] is node and inject["seq"] == out["seq"]:
                    hdr = {k: v for k, v in out.items() if k not in ("v", "op", "x", "xs")}
                    rogue = FR.frame(**hdr, **enc_rows((dec_rows(out) + 1) % FIELD, out["cx"]))
                    later.append((job, rogue, inject["sender"], job.path[i + 1]))
        live = nxt
        if hop is not None:
            hop += 1
    return [replies[id(job)] for job, _ in frames]


def run(jobs, events=None, inject=None):
    """Drive jobs to completion. events: [(job, c_threshold, fn(job))], each fired once."""
    events = list(events or [])
    while not all(j.finished for j in jobs):
        senders = [j for j in jobs if j.can_send()]
        if senders:
            frames = [(j, j.build()) for j in senders]
            for (j, msg), rep in zip(frames, wave(frames, inject)):
                j.inflight.append((msg, rep))
        for j in jobs:
            if j.finished or j in senders:
                continue
            if j.inflight:
                msg, rep = j.inflight.popleft()
                if rep is None and j.dead_node is not None:
                    continue                            # lost with the dead hop
                j.on_reply(msg, rep)
            elif j.dead_node is not None:
                handler = j.on_death
                handler(j)
            elif j.c >= j.P + j.n_new or (j.pre >= j.P and j.opt is not None and j.opt >= j.P + j.n_new):
                j.close()
            for ev in list(events):
                if ev[0] is j and j.c >= ev[1]:
                    events.remove(ev)
                    ev[2](j)
    return jobs


def kill(node, job, lose_last=True):
    """kill -9: the node stops; its last checkpoint may still have been in flight (lost)."""
    node.alive = False
    if lose_last:
        mine = [i for i, ck in enumerate(job.ckpts) if ck["pubkey"] == node.pub and ck["log"] == "serve"
                and ck["rte"] == job.rte]
        if mine and job.ckpts[mine[-1]]["kind"] == "grid":
            del job.ckpts[mine[-1]]


def verify(b):
    return R2.verify_coverage_v2(b["path_log"], b["checkpoints"], leases=b["leases"],
                                 entry_pubkey=b["entry_pubkey"], nonce=b["nonce"], job_id=b["job_id"],
                                 **MODEL_INFO)


# ── the scenarios ─────────────────────────────────────────────────────────────────────────────
def path_a(tag, modes=("chunk_invariant", "adversarial", "chunk_invariant", "adversarial"), bugs=None):
    """4 hops, mixed numerics: [0,3) head (G), [3,6) (no global layer), [6,9) (G), [9,12) tail (G)."""
    bugs = bugs or {}
    spans = [(0, 3), (3, 6), (6, 9), (9, 12)]
    return [Node(f"{tag}/h{i}", lo, hi, modes[i], bugs.get(i, ())) for i, (lo, hi) in enumerate(spans)]


def scenario(which, **kw):
    return SCENARIOS[which](**kw)


def sc_plain(**kw):
    """Plain decode: one token per frame, nothing in flight but one frame."""
    job = Job(path_a("plain"), name="plain", K=0, depth=1, n_new=320, **kw)
    run([job])
    return job.bundle()


def sc_spec(bugs=None, inject=None, name="spec", **kw):
    """Pipelined speculation (K=3, two frames in flight) with rejections every few positions."""
    job = Job(path_a(name, bugs=dict(bugs or ())), name=name, K=3, depth=2, miss=4, **kw)
    inj = inject(job) if inject else None
    run([job], inject=inj)
    return job.bundle()


def sc_batch(**kw):
    """Two sessions sharing every hop's step (continuous batching across sessions)."""
    path = path_a("batch")
    a = Job(path, name="batch-a", K=3, depth=2, miss=4, n_new=260)
    b = Job(path, name="batch-b", K=2, depth=2, miss=3, n_new=200, prompt_len=97)
    run([a, b])
    return [a.bundle(), b.bundle()]


def sc_fp8(**kw):
    """fp8 wire (per-row scales) for prefill and decode: lossy, still chained on wire bytes."""
    job = Job(path_a("fp8"), name="fp8", K=3, depth=2, miss=4, cx="fp8", cx_prefill="fp8")
    run([job])
    return job.bundle()


def sc_reroute(**kw):
    """Two route changes: a fat node takes [0,6) (a bigger device, shorter path; rebuilt from the
    entry's ids), then [6,12) is re-split at a boundary no rte had before ([6,8)+[8,12), a chained
    rebuild whose second half is fed by the first half's forwarded rows)."""
    path = path_a("reroute")
    fat = Node("reroute/fat", 0, 6, "chunk_invariant")
    m2 = Node("reroute/m2", 6, 8, "adversarial")
    t2 = Node("reroute/t2", 8, 12, "chunk_invariant")
    job = Job(path, name="reroute", K=3, depth=2, miss=4, n_new=330)

    def first(j):
        drain(j)
        j.failover([fat, path[2], path[3]], seated={fat: ("log", None)})

    def second(j):
        drain(j)
        j.failover([fat, m2, t2], seated={m2: ("chain", None), t2: ("chain", None)})
    run([job], events=[(job, 290, first), (job, 400, second)])
    return job.bundle()


def drain(j):
    while j.inflight:
        msg, rep = j.inflight.popleft()
        j.on_reply(msg, rep)


def sc_takeover(bugs_backup=(), tier="mirror", **kw):
    """kill -9 of [6,9) (global KV) with its in-flight checkpoint lost; its leased backup imports
    the mirrored KV to g_eff and rebuilds [a, F) from [3,6)'s log. The backup runs other numerics
    (adversarial vs the primary's chunk_invariant), so the continuation is valid, not bit-exact."""
    path = path_a("takeover")
    bk = Node("takeover/backup", 6, 9, "adversarial", bugs_backup)
    job = Job(path, name="takeover", K=3, depth=2, miss=4, n_new=330, backups={path[2]: bk})
    job.on_death = lambda j: j.failover([path[0], path[1], bk, path[3]], dead=[path[2]],
                                        seated={bk: (tier, path[2])})
    run([job], events=[(job, 350, lambda j: kill(path[2], j))])
    return job.bundle()


def sc_dead(bugs_new=(), **kw):
    """The tail dies with no backup. A fresh holder rebuilds [0, F) from [6,9)'s log (tier 2); the
    dead tail is paid to its last delivered checkpoint, the positions after it are unchained."""
    path = path_a("dead")
    t2 = Node("dead/tail2", 9, 12, "chunk_invariant", bugs_new)
    job = Job(path, name="dead", K=3, depth=2, miss=4, n_new=330)
    job.on_death = lambda j: j.failover([path[0], path[1], path[2], t2], dead=[path[3]],
                                        seated={t2: ("log", None)})
    run([job], events=[(job, 300, lambda j: kill(path[3], j))])
    return job.bundle()


SCENARIOS = {"plain": sc_plain, "spec": sc_spec, "batch": sc_batch, "fp8": sc_fp8,
             "reroute": sc_reroute, "takeover": sc_takeover, "dead": sc_dead}
