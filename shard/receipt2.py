"""shard-receipt/2: per-position receipts for arbitrary, changing mesh paths (ARCHITECTURE-v2 §7).

v1 (shard/receipt.py) signs one running hash per stage per job. That cannot follow a mesh: paths
change mid-session, a node can die, speculation rewrites rows, and a hop may run on any device
family. v2 attests, per session, hop and route epoch (`rte`), exactly which positions it served,
in a form adjacent hops can be compared on. v1 stays as it is for the v1 rings (M2.5, V4).

LEAVES (§7.1), one per position p, untagged, so a sender's OUT leaf IS its receiver's IN leaf:
  * row leaf  sha256(p ‖ wire row). The wire row is frame.wire_row(): x[i] ‖ xs[i] ‖ tap[i], the
    exact bytes that crossed the link. Both sides hash the same bytes, so the chain holds on the
    fp8 codec and between hops of different numerics (a Mac and a CUDA card on one path).
  * token leaf H_tok(p) = sha256("tok" ‖ salt ‖ p ‖ id). The head's IN log holds the salted token
    ids it received, the tail's OUT log the ones it produced, so the orchestrator compares them
    without ever learning a token (the salt goes to the head and tail only).
  * KV leaf   sha256(p ‖ global-KV rows of the span at p), rows as KVBlob.pos_bytes(p).

TAIL TOKEN LOG. Slot i of the tail's OUT log is H_tok(i) of the token the tail produced at row
i-1, so tail OUT and head IN are index-aligned. Two slots have no production and ECHO the input id
the tail received in `eid` instead: prompt slots (i < P) and the first slot of each rte (the entry
re-feeds the token at F). With the echo, head IN and tail OUT cover the same positions and compare
by chain like any other pair of hops. The token produced at the last row of an rte (index F) is
signed separately as `tok_next` and checked against the next rte's head IN leaf at F.

COMMIT FOLDING. A hop overwrites a row's leaves on every recompute (last write wins) and appends
it to its logs, in position order, only once the frame's cpos (the final-row frontier F = c-1)
has passed it. Rejected speculation and the draft-fed row at F are therefore never hashed into a
log. The tail folds slot i under row i, so it needs row i-1's token, which is final too.

LOGS. Each of IN, OUT, KV is a chunked Merkle log: a binary tree over each 128-slot interval
[128k, 128k+128) at absolute positions (a slot the log does not cover is 32 zero bytes), and a
hash chain over the interval roots, started from the log's first position. Checkpoints carry the
interval roots they seal (not only the chain), because a rebuild, a KV import and the closure
across a route change compare logs over DIFFERENT position ranges, which cumulative chains alone
cannot do. Where a log's range starts or ends inside an interval it ships that interval's leaves
(only at route changes and for rebuilds; a fault-free job ships none). SEG records
[start_pos, n, codec, kind] list the runtime segments that contributed folded rows, in fold order.

CHECKPOINTS `shard-ckpt/2` (§7.2), signed by the hop's node key, go hop -> entry directly (C19):
  kind "grid" at every multiple of 128 the frontier passes, "rte_close" when a route change closes
  the hop's log (lease_update truncate_to=F), "close" at the end of the job, "done" for a rebuild
  log (one checkpoint, at F). `prev_ckpt` links a log's checkpoints, so a dropped one is visible.
  log "serve": the rows a hop served in an rte. log "rebuild": a replacement's rebuild before it
  serves (inputs [a', F) with a' = a rounded down to the grid, the imported KV [0, upto), the KV it
  recomputed [upto, F), and, when it forwards its rebuilt rows to a new successor, its outputs).
  log "mirror": a backup's acknowledgement that it holds and verified a primary's KV.

PATH LOG `shard-path/2`, signed by the entry: per rte the position range, the hops, the codecs,
which hops died during it and how each new hop was seated. `verify_coverage_v2` checks it against
the orchestrator's own leases (the entry cannot invent hops), every checkpoint against the path
log, and every claim against its neighbours; it fails closed with a specific reason, and on
success returns the attested work that `shard.settle2` turns into payouts.

WHAT THE ENTRY CANNOT DO. Invent a hop or backup (leases); move the prompt boundary (head and
tail sign prompt_len); attribute positions past an rte's end; cut a hop's pay by declaring it dead
or withholding its checkpoints. The committed frontier only advances through replies that crossed
every hop, so one hop's checkpoint at X proves every hop of that rte computed the rows below X:
each hop of an rte is paid to the rte's proven frontier (the furthest checkpoint any of its hops
delivered, at most pos_end), a dead one included. A hop that signed its close cannot be called dead.

KNOWN LIMITS (by design, for the audit seam or the doc):
  * receipts prove consistency, not computation: a hop that forwards garbage chains fine and is
    caught by the replay audit (§7.4). Rebuild inputs prove fetching, not computing.
  * a mirror ack proves receipt, not retention: the primary's checkpoint copy rides `mirror`, so
    a backup can ack its chain without storing KV. It gains little: w_mirror is tiny, and a
    takeover pays rebuild credit only from the rule's start given the frontier the backup acked
    or its primary signed it mirrored (`mirrored`), whichever is further.
  * a malicious entry can still force unpaid rebuild work on a hop (declare it dead, re-seat it:
    a key re-seated after its own death is owed no rebuild credit, so self-kill cannot farm it
    under the sticky rejoin of C22). Its serve pay is unaffected.
  * the non-prefill SEG schedule must be identical on every hop of an rte. That holds while
    decode/verify frames are forwarded whole and never coalesced (C15 defers coalescing).
  * fail closed per job: one hop's bad signature or fork voids the job for everyone. The reason
    names the signer or link so c0mpute can attribute it; per-signer isolation is a design fork.
"""
from __future__ import annotations

import base64
import hashlib
import json
import struct

from cryptography.hazmat.primitives.asymmetric import ed25519

try:                                                          # package import (engine repo)
    from .receipt import ReceiptError, _canonical
except ImportError:                                           # flat import (deployed next to the node code)
    from receipt import ReceiptError, _canonical

CKPT_SCHEMA = "shard-ckpt/2"
PATH_SCHEMA = "shard-path/2"
GRID = 128
ZERO = bytes(32)
ROLES = ("head", "mid", "tail", "solo")
SEG_KINDS = ("prefill", "verify", "decode", "rebuild")
CODECS = ("ids", "bf16", "fp8")
_U32 = struct.Struct("!I")


def _h(b: bytes) -> bytes:
    return hashlib.sha256(b).digest()


# ── leaves ────────────────────────────────────────────────────────────────────────────────────
def leaf_row(p: int, row) -> bytes:
    """IN/OUT leaf of a hidden-state wire row (and the KV leaf: p ‖ KVBlob.pos_bytes(p))."""
    return _h(_U32.pack(p) + bytes(row))


leaf_kv = leaf_row


def leaf_tok(salt: bytes, p: int, tok: int) -> bytes:
    """H_tok(p): the salted token leaf. Without the per-session salt anyone could invert it."""
    if len(salt) != 32:
        raise ValueError("the token salt is 32 bytes")
    return _h(b"tok" + salt + _U32.pack(p) + _U32.pack(tok))


# ── chunked Merkle log ────────────────────────────────────────────────────────────────────────
def interval_root(slots) -> bytes:
    """Root of one 128-slot interval (absent slots = ZERO). Fixed depth: no leaf/node ambiguity."""
    level = list(slots)
    if len(level) != GRID:
        raise ValueError(f"an interval has {GRID} slots")
    while len(level) > 1:
        level = [_h(b"\x01" + level[i] + level[i + 1]) for i in range(0, len(level), 2)]
    return level[0]


def _chain0(lo: int) -> bytes:
    return _h(b"shard-log/2" + _U32.pack(lo))


def _chain(c: bytes, k: int, root: bytes) -> bytes:
    return _h(c + _U32.pack(k) + root)


def _seg0(lo: int) -> bytes:
    return _h(b"shard-seg/2" + _U32.pack(lo))


def _seg_step(c: bytes, rec) -> bytes:
    return _h(c + json.dumps(rec, separators=(",", ":")).encode())


def _slots(leaves: dict, k: int) -> list:
    base = k * GRID
    return [leaves.get(base + i, ZERO) for i in range(GRID)]


class SubLog:
    """Writer for one chunked Merkle log over [lo, hi): append leaves in position order."""

    def __init__(self, lo: int):
        self.lo = self.hi = lo
        self.chain = _chain0(lo)
        self._open: dict[int, bytes] = {}
        self._roots: list = []
        self._edge: list = []

    def append(self, leaf: bytes) -> None:
        self._open[self.hi] = leaf
        self.hi += 1
        if self.hi % GRID == 0:
            self._seal(ship=False)

    def _seal(self, ship: bool) -> None:
        k = (self.hi - 1) // GRID
        root = interval_root(_slots(self._open, k))
        self.chain = _chain(self.chain, k, root)
        self._roots.append((k, root))
        if ship or (self.lo > 0 and k == self.lo // GRID):     # a log that starts mid-session ships
            first = min(self._open)                             # its first interval (§ module doc)
            self._edge.append((k, first, [self._open[q] for q in range(first, self.hi)]))
        self._open = {}

    def flush(self, *, final: bool = False, ship_partial: bool = False) -> dict:
        """The checkpoint view: new roots, chain, edge leaves. `final` seals a partial last interval."""
        if final and self._open:
            self._seal(ship=ship_partial)
        d = {"lo": self.lo, "hi": self.hi, "roots": [[k, r.hex()] for k, r in self._roots],
             "chain": self.chain.hex()}
        if self._edge:
            d["leaves"] = [[k, first, [x.hex() for x in lv]] for k, first, lv in self._edge]
        self._roots, self._edge = [], []
        return d


class SegLog:
    def __init__(self, lo: int):
        self.lo = lo
        self.chain = _seg0(lo)
        self._new: list = []

    def append(self, rec) -> None:
        self.chain = _seg_step(self.chain, rec)
        self._new.append(rec)

    def flush(self) -> dict:
        d = {"lo": self.lo, "recs": self._new, "chain": self.chain.hex()}
        self._new = []
        return d


# ── signing ───────────────────────────────────────────────────────────────────────────────────
def _pub(priv: ed25519.Ed25519PrivateKey) -> str:
    return base64.b64encode(priv.public_key().public_bytes_raw()).decode()


def _sign(priv, body: dict, key_field: str = "pubkey") -> dict:
    body = dict(body)
    body[key_field] = _pub(priv)
    body["sig"] = base64.b64encode(priv.sign(_canonical(body))).decode()
    return body


def _check_sig(obj: dict, key_field: str = "pubkey") -> None:
    try:
        pub = ed25519.Ed25519PublicKey.from_public_bytes(base64.b64decode(obj[key_field], validate=True))
        pub.verify(base64.b64decode(obj["sig"], validate=True), _canonical(obj))
    except Exception as e:  # noqa: BLE001 - fail closed on anything
        raise ReceiptError(f"bad signature ({type(e).__name__})") from None


def ckpt_id(ck: dict) -> str:
    """What the next checkpoint's prev_ckpt names: sha256 of the whole signed checkpoint."""
    return hashlib.sha256(json.dumps(ck, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def resign(priv, obj: dict, key_field: str = "pubkey") -> dict:
    """Re-sign an edited checkpoint / path log (tests and tools that model a lying signer)."""
    return _sign(priv, {k: v for k, v in obj.items() if k != "sig"}, key_field)


# ── the hop side: commit-folding ledger ───────────────────────────────────────────────────────
class _Row:
    __slots__ = ("inl", "outl", "seg", "id", "tok")

    def __init__(self, inl, outl, seg, id_, tok):
        self.inl, self.outl, self.seg, self.id, self.tok = inl, outl, seg, id_, tok


class HopLedger:
    """One hop's receipts for one session in one rte ("serve"), or a backup's mirror acks ("mirror").

    The node calls record() for every computed segment, commit(F) with every frame's cpos and
    close() when a lease_update (final=False) or the job close (final=True) ends the log. commit
    and close return the signed checkpoints to send to the entry. `kv_rows(lo, hi)` returns the
    span's global-KV bytes per position for rows being folded (KVBlob.pos_bytes)."""

    def __init__(self, priv, *, job_id, sid, nonce, manifest_cid, numerics_id, rte, lo, hi, role,
                 pos_lo, salt: bytes | None = None, prompt_len: int = 0, kv: bool = False,
                 log: str = "serve", of: str | None = None):
        if log not in ("serve", "mirror"):
            raise ValueError(f"log {log!r}")
        if log == "serve" and role not in ROLES:
            raise ValueError(f"role {role!r}")
        self.priv, self.log, self.role = priv, log, role
        self.headish = log == "serve" and role in ("head", "solo")
        self.tailish = log == "serve" and role in ("tail", "solo")
        if (self.headish or self.tailish) and (salt is None or len(salt) != 32):
            raise ValueError("head and tail ledgers need the session's 32-byte token salt")
        self.salt, self.P, self.pos_lo = salt, prompt_len, pos_lo
        self.meta = {"schema": CKPT_SCHEMA, "job_id": job_id, "sid": sid, "nonce": nonce,
                     "manifest_cid": manifest_cid, "numerics_id": numerics_id, "rte": rte,
                     "lo": lo, "hi": hi, "role": role, "log": log, "pos_lo": pos_lo}
        if of is not None:
            self.meta["of"] = of
        if self.headish or self.tailish:
            self.meta["prompt_len"] = prompt_len     # pins the prefill/decode split the entry claims
        self.subs: dict[str, SubLog] = {}
        if log == "serve":
            self.subs["in"] = SubLog(pos_lo)
            self.subs["out"] = SubLog(pos_lo)
        if kv:
            self.subs["kv"] = SubLog(pos_lo)
        self.seg = SegLog(pos_lo) if log == "serve" else None
        self._next = pos_lo                  # next row to fold
        self._F = pos_lo                     # highest frontier seen
        self._pend: dict[int, _Row] = {}
        self._segs: list = []
        self._last_seg = None
        self._prev_tok = None
        self.n_rows = 0
        self._prev = ""
        self.closed = False
        self.mirrored = None      # a primary: its backup holds KV rows [0, mirrored) (the node sets it)

    @property
    def folded(self) -> int:
        return self._next

    def record(self, pos: int, n: int, *, kind: str, codec: str, in_rows=None, out_rows=None,
               ids=None, toks=None) -> None:
        """A computed segment [pos, pos+n): its wire rows (head: ids; tail: ids from eid and the
        drawn tokens, None where lg drew none). Rows >= pos still pending are dropped first (the
        implicit truncate); a folded row can never be rewritten."""
        if self.closed or self.log != "serve":
            raise ReceiptError("record() on a closed or non-serve ledger")
        if kind not in SEG_KINDS or codec not in CODECS or n < 1:
            raise ValueError(f"bad segment [{pos}, {n}, {codec}, {kind}]")
        if pos < self._next:
            raise ReceiptError(f"segment at {pos} rewrites folded rows (frontier {self._next})")
        end_written = max(self._pend, default=self._next - 1) + 1
        if pos > max(end_written, self._next):
            raise ReceiptError(f"segment at {pos} leaves a gap (rows written to {end_written})")
        for p in [q for q in self._pend if q >= pos]:
            del self._pend[p]
        sid = len(self._segs)
        self._segs.append([pos, n, codec, kind])
        for j in range(n):
            p = pos + j
            inl = leaf_tok(self.salt, p, int(ids[j])) if self.headish else leaf_row(p, in_rows[j])
            outl = None if self.tailish else leaf_row(p, out_rows[j])
            tok = None
            if self.tailish and toks is not None and toks[j] is not None:
                tok = int(toks[j])
            self._pend[p] = _Row(inl, outl, sid, int(ids[j]) if self.tailish else None, tok)
        self.n_rows += n

    def commit(self, F: int, kv_rows=None) -> list:
        """Fold every written row < F (in order, up to the first unwritten one)."""
        if self.closed:
            raise ReceiptError("commit() on a closed ledger")
        self._F = max(self._F, F)
        out = []
        while self._next < self._F:
            end = self._next
            while end < self._F and (self.log == "mirror" or end in self._pend) \
                    and (end == self._next or end % GRID):
                end += 1
            if end == self._next:
                break
            kv = None
            if "kv" in self.subs:
                if kv_rows is None:
                    raise ReceiptError("this span has global KV: commit needs kv_rows")
                kv = list(kv_rows(self._next, end))
                if len(kv) != end - self._next:
                    raise ReceiptError("kv_rows returned the wrong number of positions")
            for i, p in enumerate(range(self._next, end)):
                self._fold(p, None if kv is None else kv[i])
            self._next = end
            if end % GRID == 0:
                out.append(self._ckpt("grid"))
        return out

    def _fold(self, p: int, kv) -> None:
        if kv is not None:
            self.subs["kv"].append(leaf_kv(p, kv))
        if self.log == "mirror":
            return
        row = self._pend.pop(p)
        self.subs["in"].append(row.inl)
        if self.tailish:
            if p == self.pos_lo or p < self.P:
                v = row.id                                   # echo: prompt slot / rte's first slot
            else:
                v = self._prev_tok                           # produced at row p-1
                if v is None:
                    raise ReceiptError(f"tail cannot attest slot {p}: row {p - 1} drew no token")
            self.subs["out"].append(leaf_tok(self.salt, p, v))
            self._prev_tok = row.tok
        else:
            self.subs["out"].append(row.outl)
        if row.seg != self._last_seg:
            self._last_seg = row.seg
            self.seg.append(self._segs[row.seg])

    def close(self, F: int, *, final: bool, kv_rows=None) -> list:
        """End the log at F (lease_update truncate_to=F, or the job close): every row < F must be
        written. Returns the remaining checkpoints, the close last."""
        out = self.commit(F, kv_rows)
        if self._next != F:
            raise ReceiptError(f"close at {F}: rows from {self._next} were never written")
        extra = {}
        if self.tailish and F > self.pos_lo and F >= self.P:   # row F-1 drew a token
            extra["tok_next"] = leaf_tok(self.salt, F, self._need_tok(F)).hex()
        out.append(self._ckpt("close" if final else "rte_close", seal=True, ship=not final, **extra))
        self.closed = True
        return out

    def _need_tok(self, F):
        if self._prev_tok is None:
            raise ReceiptError(f"tail closes at {F} but row {F - 1} drew no token")
        return self._prev_tok

    def _ckpt(self, kind, seal=False, ship=False, **extra) -> dict:
        body = dict(self.meta, kind=kind, pos=self._next, n_rows=self.n_rows, prev_ckpt=self._prev,
                    logs={k: s.flush(final=seal, ship_partial=ship) for k, s in self.subs.items()},
                    **extra)
        if self.seg is not None:
            body["seg"] = self.seg.flush()
        if self.mirrored is not None:
            body["mirrored"] = self.mirrored
        ck = _sign(self.priv, body)
        self._prev = ckpt_id(ck)
        return ck


class RebuildLog:
    """A replacement's rebuild before it serves rte `rte` from F (tier 0-3 seating, §8).

    Inputs [a', F) with a' = a rounded down to the grid (the rows it fetched from the
    predecessor's activation log, or the ids the entry re-sent to a head); SEG records of the
    rebuild segments it ran over [a, F); the global KV it imported [0, upto) and recomputed
    [upto, F); with `fwd_lo`, the rebuilt output rows it forwarded to a new successor [fwd_lo, F).
    One signed checkpoint, kind "done", at F."""

    def __init__(self, priv, *, job_id, sid, nonce, manifest_cid, numerics_id, rte, lo, hi, role,
                 a, F, upto=0, tier="log", replaces=None, kv=False, fwd_lo=None, salt=None):
        if not 0 <= a <= F or not 0 <= upto <= F:
            raise ValueError("need 0 <= a <= F and 0 <= upto <= F")
        self.priv, self.a, self.F, self.upto = priv, a, F, upto
        self.head = role in ("head", "solo")
        if self.head and (salt is None or len(salt) != 32):
            raise ValueError("a head rebuild hashes token leaves: needs the salt")
        self.salt = salt
        a0 = a // GRID * GRID
        self.subs = {"in": SubLog(a0)}
        if fwd_lo is not None:
            if fwd_lo % GRID or fwd_lo < a:
                raise ValueError("forwarded rows start on the grid, at or after a")
            self.subs["out"] = SubLog(fwd_lo)
        if kv:
            self.subs["kvi"] = SubLog(0)
            self.subs["kv"] = SubLog(upto)
        self.seg = SegLog(a)
        self.meta = {"schema": CKPT_SCHEMA, "job_id": job_id, "sid": sid, "nonce": nonce,
                     "manifest_cid": manifest_cid, "numerics_id": numerics_id, "rte": rte,
                     "lo": lo, "hi": hi, "role": role, "log": "rebuild", "pos_lo": a0,
                     "rebuild": {"a": a, "F": F, "upto": upto, "tier": tier, "replaces": replaces}}
        self.n_rows = 0

    def _add(self, name, p, leaf):
        s = self.subs[name]
        if p != s.hi:
            raise ReceiptError(f"rebuild {name} leaf at {p}, expected {s.hi}")
        s.append(leaf)

    def add_input(self, p: int, row=None, tok: int | None = None) -> None:
        self._add("in", p, leaf_tok(self.salt, p, tok) if self.head else leaf_row(p, row))

    def add_output(self, p: int, row) -> None:
        self._add("out", p, leaf_row(p, row))

    def add_kv_import(self, p: int, kv) -> None:
        self._add("kvi", p, leaf_kv(p, kv))

    def add_kv(self, p: int, kv) -> None:
        self._add("kv", p, leaf_kv(p, kv))

    def add_segment(self, start: int, n: int, codec: str) -> None:
        self.seg.append([start, n, codec, "rebuild"])
        self.n_rows += n

    def finish(self) -> dict:
        ends = {"in": self.F, "out": self.F, "kvi": self.upto, "kv": self.F}
        for k, s in self.subs.items():
            if s.hi != ends[k]:
                raise ReceiptError(f"rebuild {k} log ends at {s.hi}, expected {ends[k]}")
        body = dict(self.meta, kind="done", pos=self.F, n_rows=self.n_rows, prev_ckpt="",
                    logs={k: s.flush(final=True, ship_partial=True) for k, s in self.subs.items()},
                    seg=self.seg.flush())
        return _sign(self.priv, body)


def sign_path_log(entry_priv, body: dict) -> dict:
    """The entry signs its path log (key field `entry`)."""
    return _sign(entry_priv, dict(body, schema=PATH_SCHEMA), key_field="entry")




# ── the verifier side ─────────────────────────────────────────────────────────────────────────
def _fail(msg):
    raise ReceiptError(msg)


def _int(v, what, lo=0):
    if type(v) is not int or not lo <= v <= 0xFFFFFFFF:
        _fail(f"{what}: want an int in [{lo}, 2^32), got {str(v)[:24]}")
    return v


def _hex32(v, what) -> bytes:
    if type(v) is not str or len(v) != 64:
        _fail(f"{what}: want 32 bytes hex")
    try:
        return bytes.fromhex(v)
    except ValueError:
        _fail(f"{what}: bad hex")


class _Sub:
    """Verifier view of one sub-log, accumulated over its log's checkpoints."""

    def __init__(self, lo):
        self.lo = self.hi = lo
        self.chain = _chain0(lo)
        self.roots: dict[int, bytes] = {}
        self.leaves: dict[int, bytes] = {}
        self.chains: dict[int, bytes] = {}      # checkpoint position -> chain there
        self._next_k = lo // GRID                # next interval to be sealed

    def full(self, k):
        return k * GRID >= self.lo and (k + 1) * GRID <= self.hi

    def root(self, k):
        return self.roots.get(k) if self.full(k) else None

    def take(self, d, hi, final, ship, where):
        """Fold one checkpoint's view of this sub-log: new interval roots, chain, edge leaves."""
        if type(d) is not dict or set(d) - {"lo", "hi", "roots", "chain", "leaves"}:
            _fail(f"{where}: malformed sub-log")
        if d.get("lo") != self.lo or type(hi) is not int or d.get("hi") != hi or hi < self.hi:
            _fail(f"{where}: covers [{d.get('lo')}, {d.get('hi')}), want [{self.lo}, {hi})")
        self.hi = hi
        roots = d.get("roots")
        if type(roots) is not list:
            _fail(f"{where}: roots")
        new = {}
        for e in roots:
            if type(e) is not list or len(e) != 2 or e[0] != self._next_k:
                _fail(f"{where}: interval roots out of order (want interval {self._next_k})")
            new[e[0]] = _hex32(e[1], where)
            self._next_k += 1
        want_k = self.lo // GRID if hi == self.lo else (-(-hi // GRID) if final else hi // GRID)
        if self._next_k != want_k:
            _fail(f"{where}: seals intervals up to {self._next_k}, the checkpoint needs {want_k}")
        for k, root in new.items():
            self.chain = _chain(self.chain, k, root)
            self.roots[k] = root
        if _hex32(d.get("chain"), where) != self.chain:
            _fail(f"{where}: chain does not fold its interval roots")
        self.chains[hi] = self.chain
        shipped = set()
        for e in d.get("leaves", []):
            if type(e) is not list or len(e) != 3 or type(e[2]) is not list:
                _fail(f"{where}: malformed leaves")
            k, first = e[0], e[1]
            if k not in new:
                _fail(f"{where}: leaves for interval {k!r}, which this checkpoint did not seal")
            lo_k, hi_k = max(k * GRID, self.lo), min((k + 1) * GRID, hi)
            if first != lo_k or len(e[2]) != hi_k - lo_k:
                _fail(f"{where}: interval {k} leaves must cover [{lo_k}, {hi_k})")
            got = {first + i: _hex32(x, where) for i, x in enumerate(e[2])}
            if interval_root(_slots(got, k)) != new[k]:
                _fail(f"{where}: shipped leaves of interval {k} do not match its root")
            self.leaves.update(got)
            shipped.add(k)
        for k in new:
            if ((self.lo > 0 and k == self.lo // GRID) or (ship and not self.full(k))) and k not in shipped:
                _fail(f"{where}: interval {k} is an edge of the log and must ship its leaves")


class _Log:
    """One validated (pubkey, rte, log) checkpoint sequence: linked, gap-free, self-consistent."""

    def __init__(self, key, ckpts):
        self.pubkey, self.rte, self.log = key
        first = ckpts[0]
        self.lo, self.hi, self.role = first.get("lo"), first.get("hi"), first.get("role")
        self.pos_lo = _int(first.get("pos_lo"), "pos_lo")
        for k in ("lo", "hi"):
            _int(first.get(k), k)
        self.fixed = {k: first.get(k) for k in ("lo", "hi", "role", "pos_lo", "numerics_id", "of",
                                                 "rebuild", "prompt_len")}
        self.rebuild = first.get("rebuild")
        by_prev = {}
        for ck in ckpts:
            p = ck.get("prev_ckpt")
            if type(p) is not str:
                _fail(f"{self.who}: checkpoint without prev_ckpt")
            if p in by_prev:
                _fail(f"{self.who}: two checkpoints claim the same predecessor (forked sequence)")
            by_prev[p] = ck
        seq, cur = [], ""
        while cur in by_prev:
            ck = by_prev.pop(cur)
            seq.append(ck)
            cur = ckpt_id(ck)
        if by_prev:
            _fail(f"{self.who}: {len(by_prev)} checkpoint(s) do not link to a delivered predecessor "
                  f"(dropped checkpoint)")
        self.subs: dict[str, _Sub] = {}
        self.seg_recs, self.seg_at = [], {}
        seg_lo = self.rebuild.get("a") if self.log == "rebuild" and type(self.rebuild) is dict \
            else self.pos_lo
        seg_chain = _seg0(_int(seg_lo, f"{self.who} seg lo"))
        grid_next = (self.pos_lo // GRID + 1) * GRID
        n_rows, closed, self.tok_next, self.mirrored = 0, False, None, 0
        for i, ck in enumerate(seq):
            pos = _int(ck.get("pos"), f"{self.who} pos")
            kind = ck.get("kind")
            where = f"{self.who} {kind}@{pos}"
            if closed:
                _fail(f"{where}: checkpoint after the log closed")
            if any(ck.get(k) != v for k, v in self.fixed.items()):
                _fail(f"{where}: identity fields change inside one log")
            if self.log == "rebuild":
                if kind != "done" or len(seq) != 1:
                    _fail(f"{where}: a rebuild log is exactly one 'done' checkpoint")
            elif kind == "grid":
                if pos != grid_next:
                    _fail(f"{where}: out of sequence, the next grid checkpoint is at {grid_next} "
                          f"(dropped or withheld checkpoint)")
                grid_next += GRID
            elif kind in ("close", "rte_close") and self.log == "serve":
                if not max(self.pos_lo, grid_next - GRID) <= pos < grid_next:
                    _fail(f"{where}: close skips the grid checkpoint at {grid_next}")
                closed = True
            else:
                _fail(f"{where}: kind {kind!r} is not valid in a {self.log} log")
            nr = _int(ck.get("n_rows"), f"{where} n_rows")
            if nr < n_rows:
                _fail(f"{where}: n_rows went backwards")
            n_rows = nr
            final, ship = kind != "grid", kind in ("rte_close", "done")
            logs = ck.get("logs")
            if type(logs) is not dict or not logs:
                _fail(f"{where}: logs")
            if i == 0:
                self.subs = {n: _Sub(_int((d or {}).get("lo") if type(d) is dict else None, f"{where} {n}.lo"))
                             for n, d in logs.items()}
            if set(logs) != set(self.subs):
                _fail(f"{where}: the set of sub-logs changes inside one log")
            for name, d in logs.items():
                hi = pos
                if name == "kvi":
                    hi = self.rebuild.get("upto") if type(self.rebuild) is dict else None
                self.subs[name].take(d, hi, final, ship, f"{where} {name}")
            s = ck.get("seg")
            if self.log == "mirror":
                if s is not None:
                    _fail(f"{where}: a mirror log carries no SEG")
            else:
                if type(s) is not dict or s.get("lo") != seg_lo or type(s.get("recs")) is not list:
                    _fail(f"{where}: malformed SEG")
                for rec in s["recs"]:
                    seg_chain = _seg_step(seg_chain, rec)
                    self.seg_recs.append(rec)
                if _hex32(s.get("chain"), where) != seg_chain:
                    _fail(f"{where}: SEG chain does not fold its records")
                self.seg_at[pos] = len(self.seg_recs)
            self.tok_next = ck.get("tok_next")
            if ck.get("mirrored") is not None:
                self.mirrored = max(self.mirrored, _int(ck["mirrored"], f"{where} mirrored"))
        want_lo = {"in": self.pos_lo, "out": self.pos_lo, "kv": self.pos_lo}
        if self.log == "rebuild" and type(self.rebuild) is dict:
            want_lo = {"in": self.pos_lo, "kvi": 0, "kv": self.rebuild.get("upto")}
        for name, sub_ in self.subs.items():
            if name in want_lo and sub_.lo != want_lo[name]:
                _fail(f"{self.who}: sub-log {name} starts at {sub_.lo}, want {want_lo[name]}")
        self.kinds = [ck["kind"] for ck in seq]
        self.end = seq[-1]["pos"]
        self.closed = closed or self.log == "rebuild"
        self.close_kind = seq[-1]["kind"] if self.closed else None

    @property
    def who(self):
        return f"{self.log} log of {str(self.pubkey)[:10]}.. rte {self.rte}"

    def chains(self, name):
        return self.subs[name].chains if name in self.subs else {}


def _seg_check(log: _Log, lo: int, *, P: int, codec_of, where: str):
    """SEG records tile the folded rows from `lo` in fold order, with legal kinds and codecs."""
    recs = log.seg_recs
    for i, r in enumerate(recs):
        if (type(r) is not list or len(r) != 4 or type(r[0]) is not int or type(r[1]) is not int
                or r[1] < 1 or r[2] not in CODECS or r[3] not in SEG_KINDS):
            _fail(f"{where}: SEG record {str(r)[:60]} malformed")
        s, n, codec, kind = r
        if i == 0 and s != lo:
            _fail(f"{where}: SEG lie: the first segment starts at {s}, the log at {lo}")
        if i > 0 and not recs[i - 1][0] < s <= recs[i - 1][0] + recs[i - 1][1]:
            _fail(f"{where}: SEG lie: segment at {s} does not continue {recs[i - 1][:2]}")
        if kind == "decode" and n != 1:
            _fail(f"{where}: SEG lie: decode segment [{s}, {n}] is not one row")
        if kind == "prefill" and s + n > P:
            _fail(f"{where}: SEG lie: prefill segment [{s}, {n}] runs past the prompt ({P})")
        if kind in ("verify", "decode") and s < P - 1:
            _fail(f"{where}: SEG lie: {kind} segment at {s} lies below the prompt ({P})")
        if (kind == "rebuild") != (log.log == "rebuild"):
            _fail(f"{where}: SEG lie: {kind} segment at {s} in a {log.log} log")
        want = codec_of(kind)
        if want is not None and codec != want:
            _fail(f"{where}: SEG lie: segment [{s}, {n}, {codec}, {kind}] but the path log runs "
                  f"{want} for {kind} rows")
    for pos, cnt in log.seg_at.items():
        if pos > lo and (cnt == 0 or recs[cnt - 1][0] + recs[cnt - 1][1] < pos):
            _fail(f"{where}: SEG lie: segment records do not cover the rows folded to {pos}")
        if pos <= lo and cnt:
            _fail(f"{where}: SEG lie: segment records for an empty log")


class _Ref:
    """Attested leaves at one layer boundary (or of one span's KV), merged from several sub-logs
    in layers, newest first: a position takes its value from the newest layer that attests it.
    After a chained recovery a span's state was built on forwarded rows, whose bytes may differ
    from older committed rows at the same positions; a later rebuild must reproduce that state.
    Within one layer (one rte) two attestations must agree."""

    def __init__(self, what):
        self.what = what
        self.layers: list[tuple[dict, dict]] = []

    def layer(self):
        self.layers.append(({}, {}))
        return self

    def add(self, sub: _Sub):
        if not self.layers:
            self.layer()
        roots, leaves = self.layers[-1]
        for k, r in sub.roots.items():
            if sub.full(k) and roots.setdefault(k, r) != r:
                _fail(f"{self.what}: two attestations disagree on interval {k}")
        for p, lf in sub.leaves.items():
            if leaves.setdefault(p, lf) != lf:
                _fail(f"{self.what}: two attestations disagree at position {p}")

    def _leaf(self, p):
        for _, leaves in self.layers:
            if p in leaves:
                return leaves[p]
        return None

    def _ref_root(self, k):
        span = range(k * GRID, (k + 1) * GRID)
        seen = False
        for roots, leaves in self.layers:
            if k in roots and not seen:
                return roots[k]                   # the newest attestation of k is a whole interval
            seen = seen or k in roots or any(p in leaves for p in span)
        if all(self._leaf(p) is not None for p in span):
            return interval_root([self._leaf(p) for p in span])
        return None

    def _root(self, leaves, roots, k):
        if roots is not None and k in roots:
            return roots[k]
        if all(k * GRID + i in leaves for i in range(GRID)):
            return interval_root(_slots(leaves, k))
        return None

    def check(self, sub: _Sub, label: str):
        """Every position of `sub` equals the reference, interval by interval, or fail closed."""
        if sub.hi <= sub.lo:
            return
        for k in range(sub.lo // GRID, -(-sub.hi // GRID)):
            a = sub.root(k) or self._root(sub.leaves, None, k)
            b = self._ref_root(k)
            if a is not None and b is not None:
                if a != b:
                    _fail(f"{label}: interval [{k * GRID}, {(k + 1) * GRID}) differs from {self.what}")
                continue
            for p in range(max(k * GRID, sub.lo), min((k + 1) * GRID, sub.hi)):
                if p not in sub.leaves or self._leaf(p) is None:
                    _fail(f"{label}: position {p} cannot be checked against {self.what} "
                          f"(unattested or no leaves)")
                if sub.leaves[p] != self._leaf(p):
                    _fail(f"{label}: position {p} differs from {self.what}")


def rebuild_rule(kinds, window: int, F: int, upto: int) -> int:
    """a = min(F - (w-1)*n_swa, min over global layers l of upto - (w-1)*i_l), clamped at 0 (§4.1).
    shard.runtime.api.rebuild_start without exact_from, restated so the verifier needs no numpy."""
    w1 = max(window - 1, 0)
    a = F - w1 * sum(k == "swa" for k in kinds)
    below = 0
    for k in kinds:
        if k == "swa":
            below += 1
        else:
            a = min(a, upto - w1 * below)
    return max(0, a)


def verify_coverage_v2(path_log: dict, checkpoints: list, *, leases, entry_pubkey: str, nonce: str,
                       job_id: str, manifest_cid: str, layer_kinds, window: int) -> dict:
    """The job-level check before any v2 pay (§7.3), fail closed: any defect raises ReceiptError
    with a specific reason. On success returns {ok, job_id, sid, layers, prompt_len, pos_end,
    tokens, tokens_unattested, work}, `work` being the attested units settlement pays:
        {pubkey, rte, lo, hi, kind: "serve", prefill: n, decode: n}
        {pubkey, rte, lo, hi, kind: "rebuild", positions: n}
        {pubkey, rte, lo, hi, kind: "mirror", positions: n}
    leases: the ORCHESTRATOR's own records [{sid, rte, pubkey, lo, hi, role, numerics_id?}]
    (role "backup" for a mirror), never taken from the entry. manifest_cid, layer_kinds
    ("global"/"swa" per layer) and window come from the orchestrator's manifest.
    tokens_unattested: generated tokens no live tail signed (a tail died before the rte ended)."""
    try:
        return _Verifier(path_log, checkpoints, leases, entry_pubkey, nonce, job_id, manifest_cid,
                         tuple(layer_kinds), window).run()
    except ReceiptError:
        raise
    except (KeyError, TypeError, ValueError, AttributeError, IndexError, OverflowError,
            RecursionError, struct.error) as e:
        raise ReceiptError(f"malformed receipts ({type(e).__name__}: {str(e)[:120]})") from None


class _Verifier:
    def __init__(self, pl, ckpts, leases, entry_pubkey, nonce, job_id, manifest, kinds, window):
        self.pl, self.ckpts, self.leases = pl, ckpts, leases
        self.entry, self.nonce, self.job_id, self.manifest = entry_pubkey, nonce, job_id, manifest
        self.kinds, self.window, self.L = kinds, window, len(kinds)
        self.work = []
        self.mirror_ok = {}
        self.unattested = 0

    def has_kv(self, lo, hi):
        return any(k == "global" for k in self.kinds[lo:hi])

    def serve(self, r, pk):
        return self.logs.get((pk, r, "serve"))

    # 1. the path log against the orchestrator's leases
    def _path(self):
        pl, L = self.pl, self.L
        if type(pl) is not dict or pl.get("schema") != PATH_SCHEMA:
            _fail("path log: unknown schema")
        if pl.get("entry") != self.entry:
            _fail("path log is not signed by the job's entry")
        _check_sig(pl, "entry")
        if pl.get("nonce") != self.nonce:
            _fail("path log nonce != job nonce (stale or replayed)")
        if pl.get("job_id") != self.job_id or pl.get("layers") != L or pl.get("manifest_cid") != self.manifest:
            _fail("path log is for another job, model or manifest")
        self.sid = pl.get("sid")
        self.P = _int(pl.get("prompt_len"), "prompt_len", 1)
        rtes = pl.get("rtes")
        if type(rtes) is not list or not rtes:
            _fail("path log has no rte")
        issued = self.issued = {}
        for le in self.leases:
            issued[(le["sid"], le["rte"], le["pubkey"], le["lo"], le["hi"], le["role"])] = le
        self.rtes, self.spans, self.hops, self.backups = rtes, [], [], []
        for r, e in enumerate(rtes):
            if e.get("rte") != r:
                _fail(f"path log: rte {e.get('rte')!r} out of order (want {r})")
            ps, pe = _int(e.get("pos_start"), "pos_start"), _int(e.get("pos_end"), "pos_end")
            if ps != (0 if r == 0 else rtes[r - 1]["pos_end"]) or pe < ps:
                _fail(f"rte {r}: positions [{ps}, {pe}) do not continue the previous rte")
            if e.get("cx") not in ("bf16", "fp8") or e.get("cx_prefill") not in ("bf16", "fp8"):
                _fail(f"rte {r}: codecs {e.get('cx')!r}/{e.get('cx_prefill')!r}")
            hops = sorted((dict(h) for h in e.get("hops") or []), key=lambda h: h["lo"])
            cur, seen = 0, set()
            for h in hops:
                pk, lo, hi, role = h["pubkey"], h["lo"], h["hi"], h["role"]
                if lo != cur or hi <= lo:
                    _fail(f"rte {r}: layer coverage broken at {cur} (gap or overlap)")
                cur = hi
                want = "solo" if lo == 0 and hi == L else "head" if lo == 0 else "tail" if hi == L else "mid"
                if role != want:
                    _fail(f"rte {r}: hop [{lo}, {hi}) has role {role!r}, want {want}")
                if pk in seen:
                    _fail(f"rte {r}: duplicate signer {pk[:10]}.. (one identity per rte)")
                seen.add(pk)
                le = issued.get((self.sid, r, pk, lo, hi, role))
                if le is None:
                    _fail(f"rte {r}: no issued lease for hop {pk[:10]}.. [{lo}, {hi}) (invented hop)")
                h["_nid"] = le.get("numerics_id")
            if cur != L:
                _fail(f"rte {r}: layers end at {cur}, the model has {L}")
            bk = {}
            for h in hops:
                b = h.get("backup")
                if b is None:
                    continue
                if b in seen or b in bk:
                    _fail(f"rte {r}: backup {b[:10]}.. is on the path or backs two hops")
                if (self.sid, r, b, h["lo"], h["hi"], "backup") not in issued:
                    _fail(f"rte {r}: no issued backup lease for {b[:10]}.. (invented backup)")
                bk[b] = h
            dead = set(e.get("dead") or [])
            if dead - seen:
                _fail(f"rte {r}: declares dead a pubkey that is not on its path")
            st = [x["pubkey"] for x in (e.get("seated") or [])]
            if set(st) - seen or len(st) != len(set(st)):
                _fail(f"rte {r}: a seated pubkey is not on the path (or seated twice)")
            self.spans.append(hops)
            self.hops.append({h["pubkey"]: h for h in hops})
            self.backups.append(bk)
        self.pos_end = rtes[-1]["pos_end"]
        if self.pos_end < self.P:
            _fail(f"the job ends at {self.pos_end}, before the prompt ({self.P}) was served")

    # 2. checkpoints: signature, job binding, placement in the path log
    def _checkpoints(self):
        groups, seen_ids = {}, set()
        for ck in self.ckpts:
            if type(ck) is dict:
                cid = ckpt_id(ck)
                if cid in seen_ids:
                    continue                      # the same signed checkpoint delivered twice
                seen_ids.add(cid)
            if type(ck) is not dict or ck.get("schema") != CKPT_SCHEMA:
                _fail("checkpoint: unknown schema")
            _check_sig(ck)
            pk, r, lg = ck.get("pubkey"), ck.get("rte"), ck.get("log")
            if ck.get("nonce") != self.nonce:
                _fail(f"checkpoint of {str(pk)[:10]}.. carries nonce {str(ck.get('nonce'))[:16]!r}, "
                      f"not the job's (stale or replayed)")
            if (ck.get("job_id"), ck.get("sid"), ck.get("manifest_cid")) != \
                    (self.job_id, self.sid, self.manifest):
                _fail(f"checkpoint of {str(pk)[:10]}.. is bound to another job/session/manifest")
            if type(r) is not int or not 0 <= r < len(self.rtes):
                _fail(f"checkpoint for unknown rte {r!r}")
            if lg in ("serve", "rebuild"):
                h = self.hops[r].get(pk)
                if h is None:
                    _fail(f"checkpoint signer {str(pk)[:10]}.. is not a hop of rte {r}")
                if (ck.get("lo"), ck.get("hi"), ck.get("role")) != (h["lo"], h["hi"], h["role"]):
                    _fail(f"checkpoint of {pk[:10]}.. attests [{ck.get('lo')}, {ck.get('hi')}) "
                          f"{ck.get('role')}; the path log has [{h['lo']}, {h['hi']}) {h['role']}")
                if h["_nid"] is not None and ck.get("numerics_id") != h["_nid"]:
                    _fail(f"checkpoint of {pk[:10]}.. runs numerics {str(ck.get('numerics_id'))[:16]}, "
                          f"its lease says {str(h['_nid'])[:16]}")
            elif lg == "mirror":
                h = self.backups[r].get(pk)
                if (h is None or ck.get("of") != h["pubkey"] or ck.get("role") != "backup"
                        or (ck.get("lo"), ck.get("hi")) != (h["lo"], h["hi"])):
                    _fail(f"mirror checkpoint of {str(pk)[:10]}.. matches no leased backup of rte {r}")
            else:
                _fail(f"checkpoint log {lg!r}")
            groups.setdefault((pk, r, lg), []).append(ck)
        self.logs = {key: _Log(key, cks) for key, cks in groups.items()}
        wanted = set()
        for r, e in enumerate(self.rtes):
            wanted |= {(h["pubkey"], r, "serve") for h in self.spans[r]}
            wanted |= {(s["pubkey"], r, "rebuild") for s in (e.get("seated") or [])}
            wanted |= {(b, r, "mirror") for b in self.backups[r]}
        extra = set(self.logs) - wanted
        if extra:
            _fail(f"checkpoints the path log does not call for: {sorted(str(k[0])[:10] + '/' + k[2] for k in extra)}")

    def run(self):
        self._path()
        self._checkpoints()
        self.ends = []
        for r in range(len(self.rtes)):
            self._rte(r)
        last_tail = self.serve(len(self.rtes) - 1, self.spans[-1][-1]["pubkey"])
        tokens = self.pos_end + 1 - self.P if last_tail is not None and last_tail.tok_next else 0
        return {"ok": True, "job_id": self.job_id, "sid": self.sid, "layers": self.L,
                "prompt_len": self.P, "pos_end": self.pos_end, "tokens": tokens,
                "tokens_unattested": min(tokens, self.unattested) if tokens else self.unattested,
                "work": self.work}

    def _rte(self, r):
        e, P = self.rtes[r], self.P
        ps, pe = e["pos_start"], e["pos_end"]
        dead = set(e.get("dead") or [])
        seated = {s["pubkey"]: s for s in (e.get("seated") or [])}
        final = r == len(self.rtes) - 1
        ends = {}
        self.ends.append(ends)

        # 3. position coverage: every live hop closes at pos_end, a dead one stops at its g
        for h in self.spans[r]:
            pk, lo, hi, role = h["pubkey"], h["lo"], h["hi"], h["role"]
            lg = self.serve(r, pk)
            if lg is None:
                if pk not in dead:
                    _fail(f"rte {r}: hop {pk[:10]}.. delivered no checkpoint and is not declared dead")
                ends[pk] = (False, ps)
                continue
            if lg.end > pe:
                _fail(f"rte {r}: {pk[:10]}.. signed a checkpoint at {lg.end}, past the rte end {pe}")
            if lg.pos_lo != ps:
                _fail(f"rte {r}: {pk[:10]}.. log starts at {lg.pos_lo}, the rte at {ps}")
            if set(lg.subs) != ({"in", "out", "kv"} if self.has_kv(lo, hi) else {"in", "out"}):
                _fail(f"rte {r}: {pk[:10]}.. keeps logs {sorted(lg.subs)}")
            if pk in dead:
                if lg.closed:
                    _fail(f"rte {r}: {pk[:10]}.. is declared dead but signed the {lg.close_kind} "
                          f"(false death claim)")
                ends[pk] = (False, lg.end)
            else:
                if not lg.closed or lg.end != pe:
                    _fail(f"rte {r}: {pk[:10]}.. checkpoints end at {lg.end} ({lg.kinds[-1]}), the rte "
                          f"at {pe}, and the path log records no death (withheld checkpoint)")
                if lg.close_kind != ("close" if final else "rte_close"):
                    _fail(f"rte {r}: {pk[:10]}.. closed with {lg.close_kind}")
                ends[pk] = (True, pe)
            if role in ("tail", "solo") and lg.closed and pe > ps and pe >= P:
                _hex32(lg.tok_next, f"rte {r}: tail {pk[:10]}.. tok_next")
            if role != "mid" and lg.fixed.get("prompt_len") != P:
                _fail(f"rte {r}: {pk[:10]}.. signed prompt_len {lg.fixed.get('prompt_len')!r}, the path "
                      f"log says {P}")
            if role in ("head", "solo"):
                codec_of = lambda kind: "ids"      # noqa: E731
            else:
                codec_of = lambda kind: e["cx_prefill"] if kind == "prefill" else e["cx"]   # noqa: E731
            _seg_check(lg, ps, P=P, codec_of=codec_of, where=f"rte {r} {pk[:10]}..")

        # 4. chain between adjacent hops at every shared checkpoint, then the token closure
        for a, b in zip(self.spans[r], self.spans[r][1:]):
            la, lb = self.serve(r, a["pubkey"]), self.serve(r, b["pubkey"])
            if la is None or lb is None:
                continue
            oa, ib = la.chains("out"), lb.chains("in")
            for x in sorted(set(oa) & set(ib)):
                if oa[x] != ib[x]:
                    _fail(f"rte {r}: chain break between {a['pubkey'][:10]}.. [{a['lo']}, {a['hi']}) and "
                          f"{b['pubkey'][:10]}.. [{b['lo']}, {b['hi']}) by {x}: an output is not what "
                          f"the next hop attests it received")
        lh, lt = self.serve(r, self.spans[r][0]["pubkey"]), self.serve(r, self.spans[r][-1]["pubkey"])
        if lh is not None and lt is not None:
            ih, ot = lh.chains("in"), lt.chains("out")
            for x in sorted(set(ih) & set(ot)):
                if ih[x] != ot[x]:
                    _fail(f"rte {r}: token closure broken by {x}: the tokens the tail signed are not "
                          f"the tokens the head attests it was fed")
        q = r - 1
        while q > 0 and self.rtes[q]["pos_start"] == self.rtes[q]["pos_end"]:
            q -= 1                                        # an empty rte produced nothing at ps
        if r > 0 and lh is not None and ps in lh.subs["in"].leaves:
            pt = self.serve(q, self.spans[q][-1]["pubkey"])
            if pt is not None and pt.tok_next is not None and lh.subs["in"].leaves[ps].hex() != pt.tok_next:
                _fail(f"rte {r}: token closure broken at {ps}: the rte's first input is not the token "
                      f"the previous tail produced")

        # SEG schedules agree: decode/verify frames are forwarded whole and never coalesced
        # (C15), so every hop of an rte ran the same non-prefill segments (a hop re-splits only
        # prefill). A hop whose shape log differs lies about it, or computed a frame its path did not.
        logged = [(self.serve(r, h["pubkey"]), h) for h in self.spans[r] if self.serve(r, h["pubkey"])]
        longest = max(logged, key=lambda x: x[0].end, default=(None, None))[0]
        if longest is not None:
            ref_pk, ref_end = longest.pubkey, longest.end
            ref = [(x[0], x[1], x[3]) for x in longest.seg_recs if x[3] != "prefill"]
        for lg, h in logged:
            sched = [(x[0], x[1], x[3]) for x in lg.seg_recs if x[3] != "prefill"]
            cut = min(lg.end, ref_end)
            a_, b_ = [x for x in ref if x[0] < cut], [x for x in sched if x[0] < cut]
            if a_ != b_:
                i = next((i for i, (x, y) in enumerate(zip(a_, b_)) if x != y), min(len(a_), len(b_)))
                _fail(f"rte {r}: SEG lie: {h['pubkey'][:10]}.. [{h['lo']}, {h['hi']}) ran segment "
                      f"{b_[i] if i < len(b_) else None} where {ref_pk[:10]}.. ran "
                      f"{a_[i] if i < len(a_) else None}")

        tl, tend = ends[self.spans[r][-1]["pubkey"]]
        if not tl:
            self.unattested += max(0, pe + 1 - max(tend, P))
        proven = max((min(x.end, pe) for x in (self.serve(r, h["pubkey"]) for h in self.spans[r]) if x),
                     default=ps)
        for h in self.spans[r]:
            end = proven
            pre = max(0, min(end, P) - ps)
            self.work.append({"pubkey": h["pubkey"], "rte": r, "lo": h["lo"], "hi": h["hi"],
                              "kind": "serve", "prefill": pre, "decode": max(0, end - ps) - pre})

        # 5. every hop new to the path in this rte shows a rebuild that chains in
        for h in self.spans[r]:
            pk, lo, hi = h["pubkey"], h["lo"], h["hi"]
            ph = self.hops[r - 1].get(pk) if r else None
            cont = ph is not None and (ph["lo"], ph["hi"]) == (lo, hi) and self.ends[r - 1][pk][0]
            if r == 0 or cont:
                if pk in seated:
                    _fail(f"rte {r}: {pk[:10]}.. is seated but continues from the previous rte")
                continue
            if pk not in seated:
                _fail(f"rte {r}: hop {pk[:10]}.. is new to the path but the path log records no seating")
            rb = self.logs.get((pk, r, "rebuild"))
            if rb is None:
                _fail(f"rte {r}: seated hop {pk[:10]}.. delivered no rebuild checkpoint")
            self._rebuild(r, h, seated[pk], rb)

        # mirrors: a backup's acks equal its primary's kv chain at the same positions
        for b, h in self.backups[r].items():
            ml = self.logs.get((b, r, "mirror"))
            if ml is None:
                continue
            if set(ml.subs) != {"kv"} or ml.pos_lo != ps or ml.end > pe:
                _fail(f"rte {r}: mirror log of {b[:10]}.. malformed or past the rte end")
            pl_ = self.serve(r, h["pubkey"])
            prim = pl_.chains("kv") if pl_ is not None else {}
            ok = ps
            for x, c in sorted(ml.chains("kv").items()):
                if x not in prim:
                    continue        # past a dead primary's last delivered checkpoint: proves nothing
                if prim[x] != c:
                    _fail(f"rte {r}: KV lie: backup {b[:10]}.. acked a kv_chain at {x} that is not its "
                          f"primary's")
                ok = x
            self.mirror_ok[(b, r)] = ok
            self.work.append({"pubkey": b, "rte": r, "lo": h["lo"], "hi": h["hi"], "kind": "mirror",
                              "positions": ok - ps})

    def _boundary(self, b, r, what):
        """What was attested at layer boundary b (b = 0: the token boundary) before rte r, newest rte
        first: serve rows, and the rows each rebuild consumed or forwarded there."""
        ref = _Ref(what)
        for q in range(r - 1, -1, -1):
            ref.layer()
            for h in self.spans[q]:
                for name in ("serve", "rebuild"):
                    lg = self.logs.get((h["pubkey"], q, name))
                    if lg is None:
                        continue
                    if h["lo"] == b:
                        ref.add(lg.subs["in"])
                    if b == 0 and name == "serve" and h["hi"] == self.L:
                        ref.add(lg.subs["out"])               # the tail's token log
                    elif b > 0 and h["hi"] == b and "out" in lg.subs:
                        ref.add(lg.subs["out"])
        return ref

    def _kv_history(self, pk, lo, hi, r):
        """The KV a hop attested for its span over its latest continuous run, ending at rte r."""
        ref = _Ref(f"the replaced hop {pk[:10]}..'s KV log")
        while r >= 0:
            h = self.hops[r].get(pk)
            if h is None or (h["lo"], h["hi"]) != (lo, hi):
                break
            lg = self.serve(r, pk)
            if lg is not None and "kv" in lg.subs:
                ref.add(lg.subs["kv"])
            rb = self.logs.get((pk, r, "rebuild"))
            if rb is not None:
                for name in ("kvi", "kv"):
                    if name in rb.subs:
                        ref.add(rb.subs[name])
                break
            r -= 1
        return ref

    def _rebuild(self, r, h, s, rb):
        pk, lo, hi = h["pubkey"], h["lo"], h["hi"]
        F = self.rtes[r]["pos_start"]
        info = rb.rebuild
        who = f"rte {r}: rebuild of {pk[:10]}.. [{lo}, {hi})"
        if type(info) is not dict:
            _fail(f"{who}: no rebuild record")
        a, upto, tier, repl = info.get("a"), info.get("upto"), info.get("tier"), info.get("replaces")
        if (a, upto, tier, repl) != (s.get("a"), s.get("upto"), s.get("tier"), s.get("replaces")) \
                or info.get("F") != F or rb.end != F:
            _fail(f"{who}: the signed rebuild (a={a}, upto={upto}, tier={tier}, F={info.get('F')}) "
                  f"disagrees with the path log")
        _int(a, f"{who} a")
        _int(upto, f"{who} upto")
        if tier not in ("mirror", "log", "chain"):
            _fail(f"{who}: tier {tier!r}")
        span = self.kinds[lo:hi]
        rule = rebuild_rule(span, self.window, F, upto)
        if a > rule:
            _fail(f"{who}: starts at {a}, after the rebuild rule's {rule}: rows it needs were never fed")
        if rb.subs.get("in") is None or rb.subs["in"].lo != a // GRID * GRID:
            _fail(f"{who}: inputs must start at {a // GRID * GRID}")
        if set(rb.subs) - {"out"} != {"in"} | ({"kvi", "kv"} if self.has_kv(lo, hi) else set()):
            _fail(f"{who}: keeps logs {sorted(rb.subs)}")
        n_swa = sum(k == "swa" for k in self.kinds[lo:hi])
        a_fwd = None
        if "out" in rb.subs:
            succ = next((x for x in self.spans[r] if x["lo"] == hi), None)
            seated = {x["pubkey"]: x for x in (self.rtes[r].get("seated") or [])}
            slog = self.logs.get((succ["pubkey"], r, "rebuild")) if succ is not None else None
            if slog is None or seated.get(succ["pubkey"], {}).get("tier") != "chain":
                _fail(f"{who}: forwards rows that no chained successor consumes")
            need_lo = slog.subs["in"].lo
            a_fwd = max(0, need_lo - (self.window - 1) * n_swa)
            if a > a_fwd:
                _fail(f"{who}: its successor needs exact rows from {need_lo} but it started at {a}")
        _seg_check(rb, a, P=self.P, codec_of=lambda kind: None, where=who)

        # inputs: a chained rebuild's come from its seated predecessor's forwarded rows (which may
        # differ from the committed ones: other numerics, other codec); any other rebuild's are
        # the committed activations at boundary lo (at the head: the token logs)
        if tier == "chain":
            pred = next((x for x in self.spans[r] if x["hi"] == lo), None)
            plg = self.logs.get((pred["pubkey"], r, "rebuild")) if pred is not None else None
            if plg is None or "out" not in plg.subs:
                _fail(f"{who}: a chained rebuild needs a seated predecessor that forwarded its rows")
            ref = _Ref(f"the rows {pred['pubkey'][:10]}.. forwarded")
            ref.add(plg.subs["out"])
        else:
            ref = self._boundary(lo, r, f"the committed activations at layer {lo}")
        ref.check(rb.subs["in"], f"{who}: forged rebuild, its inputs")

        upto_pay = upto
        if tier == "mirror":
            if not self.has_kv(lo, hi) or upto == 0:
                _fail(f"{who}: a mirror takeover of a span without global KV")
            x = self.hops[r - 1].get(repl)
            if x is None or (x["lo"], x["hi"]) != (lo, hi):
                _fail(f"{who}: replaces {str(repl)[:10]}.., which did not serve [{lo}, {hi}) in rte {r - 1}")
            g = self.ends[r - 1][repl][1]
            if upto > g:
                _fail(f"{who}: imported KV to {upto}, past the replaced hop's last delivered checkpoint "
                      f"{g} (rows past g have no signed provenance)")
            self._kv_history(repl, lo, hi, r - 1).check(rb.subs["kvi"], f"{who}: KV lie, imported KV")
        elif upto != 0:
            _fail(f"{who}: tier {tier} imports no KV (upto must be 0)")
        # Pay from the rule's start given the KV the replacement could have held: the leased
        # backup of this span that acked mirror rows to g_ack is paid as if it had imported them,
        # whatever it claims to replace (withholding a mirror earns nothing). A forwarder is owed
        # the rows its successor needed exact. A key re-seated after its own death is owed nothing.
        bh = self.backups[r - 1].get(pk)
        if bh is not None and (bh["lo"], bh["hi"]) == (lo, hi):
            g_rep = self.ends[r - 1][bh["pubkey"]][1]
            plog = self.serve(r - 1, bh["pubkey"])
            held = max(self.mirror_ok.get((pk, r - 1), 0), plog.mirrored if plog is not None else 0)
            upto_pay = max(upto_pay, min(g_rep, held))
        need = rebuild_rule(span, self.window, F, upto_pay)
        if a_fwd is not None:
            need = min(need, a_fwd)
        owed = max(0, F - max(a, need))
        if pk in (self.rtes[r - 1].get("dead") or []):
            owed = 0
        self.work.append({"pubkey": pk, "rte": r, "lo": lo, "hi": hi, "kind": "rebuild",
                          "positions": owed})
