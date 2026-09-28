"""The drafter plug-in (ARCHITECTURE-v2 §4.5, design-pipelined-spec §4): protocol, fused taps, the
tail drafter host, and CPU reference drafters (MiMo MTP-3 and DFlash at tiny random dims).

THE INVARIANT. A drafter's state only ever advances over COMMITTED positions, so it never rolls
back (V4's test_cache_never_speculative, tests/test_v4_dspark.py:267). `propose` is a pure read:
every K/V row a draft block needs is computed into temporaries and dropped. A rejected frame
therefore cannot leak into a later proposal, and the tail's drafter needs none of the checkpoint
machinery a stage's KV needs.

POSITIONS (the tail's convention, pspec.accept.Frontier). A session holds the committed ids
ids[0..c-1] and one context row per position whose INPUT is committed and whose row the tail has
computed: ctx[0..c-2] (len(ctx) == len(ids) - 1 always). The newest committed id ids[c-1] is the
model's token at c-1 (the Frontier's `mfront`); its row has not been fed yet. `propose` returns
drafts for positions c, c+1, ..., i.e. q+2.. with q = c-2 the Frontier's `cfront`, which is V4's
block convention and the scripted tail's `blocks(q)`.

  MTP-3 ctx row  = the target's last hidden (pre-final-norm) at that position.
  DFlash ctx row = fc(concat taps) at that position, i.e. the fused `fc_partial` accumulator; the
                   drafter applies `hidden_norm` itself (dflash.py:253).

FUSED TAPS (TapSpec). DFlash's `fc` is linear with no bias, so fc(concat(h_l0..h_lk)) =
sum_i W_i . h_li with W_i the i-th column block of fc.weight. The stage owning layer l_i adds
W_i . h_li into a running [n, H] accumulator that rides the frame's `tap` lane: one H-vector per
position instead of k taps. Two properties matter for receipts (the wire bytes are hashed):
  * ORDER-FIXED: contributions are added in path order = ascending layer order, one add per stage.
  * ROW-LOCAL: each row's bytes are a function of that row's inputs alone. `partial` computes a
    fixed-order per-row reduction (no BLAS gemm, whose blocking depends on the batch), so the
    accumulator of a row is bitwise independent of which frame, batch or chunk carried it.
The wire dtype is bf16 (shard.frame `tap`); `bf16_round` is the exact round-to-nearest-even the
wire applies between hops.

LAZY LICENCE (the reply field S1 left open). A tail drafter may skip drafting when the entry has
PROVEN it will not consume this reply's block (the V4 lazy-draft hint: `dnxt` rides in the eid
plaintext, `lic` in the frame hint). The reply then says so explicitly with `lz` (shard.frame),
so "I skipped" is distinguishable from "I had nothing to propose" (a reply with neither `blk` nor
`lz`). The entry raises if it ever needs a block from an `lz` reply: that is an unsound licence,
not a slow path. See pspec.coord `Session(lazy=True)` for the hint rule and its proof.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

import numpy as np

from .accept import Frontier, wants_block

F32 = np.float32


# ── numerics (reference, row-local and fixed-order) ─────────────────────────────────────────────
def bf16_round(x):
    """float32 -> the nearest bf16 value (round-to-nearest-even), returned as float32."""
    a = np.ascontiguousarray(x, dtype=F32)
    u = a.view(np.uint32).astype(np.uint64)
    u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000
    u = np.where(np.isnan(a), 0x7FC00000, u)            # a NaN stays one quiet NaN (never inf)
    return u.astype(np.uint32).view(F32).reshape(a.shape)


def mv(W, x):
    """W [out, in] . x [in] as a fixed-order reduction (numpy pairwise sum over the contiguous
    input axis). No BLAS: a gemv/gemm's blocking can depend on the batch, this cannot."""
    return np.add.reduce(np.asarray(W, F32) * np.asarray(x, F32)[None, :], axis=1, dtype=F32)


def rows_mv(W, X):
    """mv applied row by row: row i of the result is mv(W, X[i]), bitwise, for any len(X)."""
    X = np.asarray(X, F32)
    return np.stack([mv(W, r) for r in X]) if len(X) else np.zeros((0, W.shape[0]), F32)


def rmsnorm(x, w, eps=1e-6):
    x = np.asarray(x, F32)
    return (x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True, dtype=F32) + F32(eps))) * w


def silu(x):
    return x / (F32(1) + np.exp(-x))


def rope(x, pos, rot, theta=10000.0):
    """Rotate the first `rot` dims of each head row x [..., hd] at absolute position `pos`
    (partial rotary, rotate_half convention)."""
    if rot == 0:
        return x
    half = rot // 2
    inv = F32(1.0) / (F32(theta) ** (np.arange(0, half, dtype=F32) * F32(2.0 / rot)))
    ang = F32(pos) * inv
    cos, sin = np.cos(ang).astype(F32), np.sin(ang).astype(F32)
    cos, sin = np.concatenate([cos, cos]), np.concatenate([sin, sin])
    r = x[..., :rot]
    rh = np.concatenate([-r[..., half:], r[..., :half]], axis=-1)
    return np.concatenate([r * cos + rh * sin, x[..., rot:]], axis=-1).astype(F32)


def attend(q, K, V, sink, scale):
    """One query row per head against K/V rows [m, kvh, hd] (GQA), with a per-head attention sink
    logit in the softmax denominator (MiMo `attention_sink_bias`). -> [heads, hd]."""
    heads, kvh = q.shape[0], K.shape[1]
    g = heads // kvh
    out = np.empty((heads, V.shape[2]), F32)
    for h in range(heads):
        k, v = K[:, h // g, :], V[:, h // g, :]
        s = np.add.reduce(k * q[h][None, :], axis=1, dtype=F32) * F32(scale)
        mx = max(float(s.max()) if len(s) else -math.inf, float(sink[h]))
        e = np.exp(s - F32(mx))
        z = e.sum(dtype=F32) + F32(math.exp(float(sink[h]) - mx))
        out[h] = np.add.reduce(v * (e / z)[:, None], axis=0, dtype=F32)
    return out


# ── TapSpec: fused taps (fc_partial) ─────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class TapSpec:
    """Target layers a drafter taps and how they fuse. `layers` are absolute target layer ids
    (tap L = the OUTPUT of layer L, pre-final-norm, runtime.api convention) in fc column order;
    `hidden` is H. mode "fc_partial": the stage holding layer l adds W_l . h_l into the frame's
    running accumulator; "raw": taps travel as-is (the concat reference)."""
    layers: tuple[int, ...]
    hidden: int
    mode: Literal["fc_partial", "raw"] = "fc_partial"

    def __post_init__(self):
        if list(self.layers) != sorted(set(self.layers)) or not self.layers:
            raise ValueError(f"TapSpec: layers must be distinct and ascending, got {self.layers}")

    def split(self, fc):
        """fc.weight [H, k*H] -> {layer: W_l [H, H]}, the per-stage slices."""
        fc = np.asarray(fc, F32)
        H, k = self.hidden, len(self.layers)
        if fc.shape != (H, k * H):
            raise ValueError(f"TapSpec: fc is {fc.shape}, want ({H}, {k * H})")
        return {l: np.ascontiguousarray(fc[:, i * H:(i + 1) * H]) for i, l in enumerate(self.layers)}

    def held(self, lo, hi):
        """The tap layers a stage holding [lo, hi) contributes, in accumulation order."""
        return tuple(l for l in self.layers if lo <= l < hi)

    def partial(self, acc, W_l, h_rows, wire="bf16"):
        """One stage's contribution: acc + W_l . h per row (acc None = zeros), rounded to the wire
        dtype. Row-local: row i depends on acc[i] and h_rows[i] only."""
        h_rows = np.asarray(h_rows, F32)
        add = rows_mv(W_l, h_rows)
        out = add if acc is None else np.asarray(acc, F32) + add
        return bf16_round(out) if wire == "bf16" else out.astype(F32)

    def fuse(self, taps_by_layer, W, spans=None, wire="bf16"):
        """Run the accumulator through the path: `spans` = [(lo, hi)] per stage in path order (None:
        one stage per tap). -> the fused rows the tail receives."""
        spans = spans or [(l, l + 1) for l in self.layers]
        acc = None
        for lo, hi in spans:
            for l in self.held(lo, hi):
                acc = self.partial(acc, W[l], taps_by_layer[l], wire=wire)
        return acc

    def concat(self, taps_by_layer, fc):
        """The unfused reference: fc(concat(taps)) in float32."""
        cat = np.concatenate([np.asarray(taps_by_layer[l], F32) for l in self.layers], axis=-1)
        return rows_mv(np.asarray(fc, F32), cat)


# ── the protocol ─────────────────────────────────────────────────────────────────────────────────
@runtime_checkable
class Drafter(Protocol):
    """ARCHITECTURE-v2 §4.5. Every method but `advance`/`open`/`rebuild` is a pure read of state
    built from COMMITTED positions; see the module docstring for the position convention."""
    location: Literal["tail", "head", "entry"]
    block: int                                    # drafts per proposal
    taps: TapSpec | None                          # None = tap-free (MTP-3 reads the last hidden)
    rebuild_rows: int                             # ctx rows a takeover needs for an exact rebuild

    def open(self, sid, ids, ctx) -> None: ...

    def advance(self, sid, pos, ids, ctx) -> None: ...

    def propose(self, sid, frontier) -> list[int]: ...

    def propose_batch(self, reqs) -> list[list[int]]: ...

    def rebuild(self, sid, start, ids, ctx) -> None: ...

    def close(self, sid) -> None: ...


class _Sessions:
    """Shared session bookkeeping: position discipline, which is asserted, never trusted."""
    location = "tail"

    def __init__(self):
        self._s = {}

    def _new(self, sid, start, ids, ctx):
        ids, ctx = [int(t) for t in ids], np.asarray(ctx, F32).reshape(-1, self.ctx_dim)
        if sid in self._s:
            raise RuntimeError(f"drafter: session {sid} already open")
        if len(ids) < 1 or len(ctx) != len(ids) - 1:
            raise RuntimeError(f"drafter: {len(ids)} ids need {len(ids) - 1} ctx rows, got {len(ctx)}")
        s = self._state(start)
        s["ids"], s["start"] = [], start
        self._s[sid] = s
        s["ids"].append(ids[0])
        self._extend(s, ids[1:], ctx)
        return s

    def open(self, sid, ids, ctx):
        self._new(sid, 0, ids, ctx)

    def rebuild(self, sid, start, ids, ctx):
        """Open `sid` from the log tail: ids at positions start..c-1 and ctx rows start..c-2. Exact
        (bitwise) when c-1-start >= rebuild_rows, by the drafter's receptive field."""
        self._new(sid, int(start), ids, ctx)

    def advance(self, sid, pos, ids, ctx):
        s = self._s[sid]
        c = s["start"] + len(s["ids"])
        ids, ctx = [int(t) for t in ids], np.asarray(ctx, F32).reshape(-1, self.ctx_dim)
        if pos != c - 1:
            raise RuntimeError(f"drafter: advance at {pos}, want the newest committed id's row {c - 1}")
        if len(ids) != len(ctx) or not ids:
            raise RuntimeError(f"drafter: advance needs one ctx row per new id ({len(ids)}/{len(ctx)})")
        self._extend(s, ids, ctx)

    def _extend(self, s, ids, ctx):
        for t, row in zip(ids, ctx):
            s["ids"].append(t)
            self._commit_row(s, row)

    def frontier(self, sid):
        s = self._s[sid]
        return s["start"] + len(s["ids"]) - 2

    def propose(self, sid, frontier):
        if frontier != self.frontier(sid):
            raise RuntimeError(f"drafter: propose at frontier {frontier}, state is at {self.frontier(sid)}")
        return [int(t) for t in np.argmax(self.draft_logits(sid), axis=-1)]

    def propose_batch(self, reqs):
        return [self.propose(sid, f) for sid, f in reqs]

    def close(self, sid):
        self._s.pop(sid, None)

    def state_digest(self, sid):
        """Bytes of every persistent array of the session (tests: propose must not change it)."""
        import hashlib
        h = hashlib.sha256()

        def walk(v):
            if isinstance(v, np.ndarray):
                h.update(v.tobytes())
            elif isinstance(v, dict):
                for k in sorted(v, key=repr):
                    h.update(repr(k).encode())
                    walk(v[k])
            elif isinstance(v, (list, tuple)):
                h.update(str(len(v)).encode())
                for x in v:
                    walk(x)
            else:
                h.update(repr(v).encode())
        walk(self._s[sid])
        return h.hexdigest()


def _rng_w(rng, *shape, scale=None):
    scale = scale if scale is not None else 1.0 / math.sqrt(shape[-1])
    return (rng.standard_normal(shape) * scale).astype(F32)


class _Block:
    """One pre-norm transformer block with GQA attention (partial rope, per-head k/q norm optional,
    attention sink) and a SiLU-gated MLP, weights random. Row-local projections throughout."""

    def __init__(self, rng, H, heads, kvh, hd, inter, rot, qk_norm, v_scale=1.0):
        self.H, self.heads, self.kvh, self.hd, self.rot = H, heads, kvh, hd, rot
        self.v_scale = F32(v_scale)
        self.ln1, self.ln2 = (1 + 0.1 * rng.standard_normal(H)).astype(F32), (1 + 0.1 * rng.standard_normal(H)).astype(F32)
        self.wq, self.wk, self.wv = _rng_w(rng, heads * hd, H), _rng_w(rng, kvh * hd, H), _rng_w(rng, kvh * hd, H)
        self.wo = _rng_w(rng, H, heads * hd)
        self.qn = (1 + 0.1 * rng.standard_normal(hd)).astype(F32) if qk_norm else None
        self.kn = (1 + 0.1 * rng.standard_normal(hd)).astype(F32) if qk_norm else None
        self.sink = (0.5 * rng.standard_normal(heads)).astype(F32)
        self.wg, self.wu, self.wd = _rng_w(rng, inter, H), _rng_w(rng, inter, H), _rng_w(rng, H, inter)

    def kv(self, x_normed, pos):
        """K/V row [kvh, hd] x2 of an already-normed input row at absolute `pos`."""
        k = mv(self.wk, x_normed).reshape(self.kvh, self.hd)
        if self.kn is not None:
            k = rmsnorm(k, self.kn)
        v = mv(self.wv, x_normed).reshape(self.kvh, self.hd) * self.v_scale
        return rope(k, pos, self.rot), v

    def q(self, x_normed, pos):
        q = mv(self.wq, x_normed).reshape(self.heads, self.hd)
        if self.qn is not None:
            q = rmsnorm(q, self.qn)
        return rope(q, pos, self.rot)

    def finish(self, resid, attn):
        h = resid + mv(self.wo, attn.reshape(-1))
        m = rmsnorm(h, self.ln2)
        return h + mv(self.wd, silu(mv(self.wg, m)) * mv(self.wu, m))


# ── MiMo MTP-3 (bundled `model.mtp.layers.{0,1,2}`) ──────────────────────────────────────────────
class MTPDrafter(_Sessions):
    """MiMo's bundled multi-token-prediction head, 3 chained depths. Depth k (1..D) at position i:
        row   = eh_proj(concat(enorm(embed(tok[i+k])), hnorm(h^{k-1}_i)))      (h^0 = target hidden)
        h^k_i = block_k(row), attending depth-k rows (i-w, i]   (SWA, w = 128 on MiMo)
        tok[i+k+1] ~ lm_head(final_layernorm_k(h^k_i))            (embed / lm_head: the target's)
    A depth-k row is COMMITTED (cached) when tok[i+k] is committed and h^{k-1}_i is; proposing at
    q = c-2 computes the rows that involve drafts (depth k at (q-k+1, q]) as temporaries and drops
    them. So proposing reads the cache and writes nothing (`test_cache_never_speculative`).
    Weight names follow the checkpoint (model.mtp.layers.N.{enorm,hnorm,eh_proj,input_layernorm,
    self_attn,pre_mlp_layernorm,mlp,final_layernorm}); the MoE-free dense MLP and attention sink
    match it. D2 loads the real tensors and pins parity; this is the structure at tiny dims.

    rebuild_rows = D*(w-1) + 1: each depth's output is exact only w-1 rows after its first exact
    row (MiMo: 3*127 + 1 = 382 positions of ids + last hidden)."""
    taps = None

    def __init__(self, *, vocab=128, hidden=16, heads=4, kv_heads=2, head_dim=8, inter=32,
                 depth=3, window=128, rot=4, seed=0):
        super().__init__()
        rng = np.random.default_rng(seed)
        self.V, self.H, self.D, self.w = vocab, hidden, depth, window
        self.block, self.ctx_dim = depth, hidden
        self.rebuild_rows = depth * (window - 1) + 1
        self.embed = _rng_w(rng, vocab, hidden, scale=1.0)
        self.lm_head = _rng_w(rng, vocab, hidden)
        self.layers = []
        for _ in range(depth):
            L = {"enorm": (1 + 0.1 * rng.standard_normal(hidden)).astype(F32),
                 "hnorm": (1 + 0.1 * rng.standard_normal(hidden)).astype(F32),
                 "eh_proj": _rng_w(rng, hidden, 2 * hidden),
                 "final": (1 + 0.1 * rng.standard_normal(hidden)).astype(F32),
                 "blk": _Block(rng, hidden, heads, kv_heads, head_dim, inter, rot, qk_norm=False)}
            self.layers.append(L)

    def _state(self, start):
        # per depth: committed K/V/out rows keyed by absolute position; h0 = target hidden rows
        return {"h0": {}, "K": [dict() for _ in range(self.D)], "V": [dict() for _ in range(self.D)],
                "out": [dict() for _ in range(self.D)]}

    def _row(self, k, i, h_prev, tok):
        L = self.layers[k]
        x = mv(L["eh_proj"], np.concatenate([rmsnorm(self.embed[tok], L["enorm"]),
                                             rmsnorm(h_prev, L["hnorm"])]))
        xn = rmsnorm(x, L["blk"].ln1)
        K, V = L["blk"].kv(xn, i)
        return x, xn, K, V

    def _out(self, k, i, x, xn, Kd, Vd):
        """depth-k output at i: attend rows (i-w, i] from Kd/Vd (dicts pos -> row)."""
        blk = self.layers[k]["blk"]
        ps = [p for p in range(max(0, i - self.w + 1), i + 1) if p in Kd]
        K = np.stack([Kd[p] for p in ps])
        V = np.stack([Vd[p] for p in ps])
        return blk.finish(x, attend(blk.q(xn, i), K, V, blk.sink, blk.hd ** -0.5))

    def _commit_row(self, s, row):
        base = s["start"]
        c = base + len(s["ids"])                     # committed ids through c-1
        i0 = c - 2                                   # the ctx row just arrived is at c-2
        s["h0"][i0] = np.asarray(row, F32)
        ids = s["ids"]
        for k in range(self.D):                      # depth k+1 rows now committed: i + k + 1 == c - 1
            i = c - 2 - k
            hp = s["h0"].get(i) if k == 0 else s["out"][k - 1].get(i)
            if i < base or hp is None:
                continue
            x, xn, K, V = self._row(k, i, hp, ids[i + k + 1 - base])
            s["K"][k][i], s["V"][k][i] = K, V
            s["out"][k][i] = self._out(k, i, x, xn, s["K"][k], s["V"][k])
        lo = c - 1 - self.rebuild_rows - self.w      # bound memory: nothing below is ever read
        for d in (s["h0"], *s["K"], *s["V"], *s["out"]):
            for p in [p for p in d if p < lo]:
                del d[p]

    def draft_logits(self, sid):
        s = self._s[sid]
        base, ids = s["start"], s["ids"]
        c = base + len(ids)
        q = c - 2
        tok = {base + j: t for j, t in enumerate(ids)}     # committed, then drafts appended
        prev_out = None                                    # temp outputs of depth k-1: pos -> h
        logits = []
        for k in range(self.D):
            Kd, Vd, out = dict(s["K"][k]), dict(s["V"][k]), {}
            for i in range(q - k + 1, q + 1):              # temp rows: tok[i+k+1] is a draft
                hp = prev_out[i] if prev_out and i in prev_out else (
                    s["h0"][i] if k == 0 else s["out"][k - 1][i])
                x, xn, Kd[i], Vd[i] = self._row(k, i, hp, tok[i + k + 1])
                out[i] = self._out(k, i, x, xn, Kd, Vd)
            if q not in out:                               # depth 1: row q is committed
                out[q] = s["out"][k][q]
            for i in range(q - k, q - k + 1):              # the committed output next depth reads
                if i >= base and i not in out and i in s["out"][k]:
                    out[i] = s["out"][k][i]
            lg = mv(self.lm_head, rmsnorm(out[q], self.layers[k]["final"]))
            logits.append(lg)
            tok[q + k + 2] = int(np.argmax(lg))
            prev_out = out
        return np.stack(logits)


# ── MiMo DFlash (`dflash/`, block 8, taps [0, 11, 23, 35, 47]) ──────────────────────────────────
class DFlashDrafter(_Sessions):
    """MiMo's DFlash block drafter (dflash.py + dflash/config.json), tiny random dims.
        ctx_p   = hidden_norm(fc(concat taps_p))            (fc arrives FUSED: the ctx row is fc(.))
        layer l: K/V of ctx rows projected DIRECTLY from ctx_p (no chaining across layers,
                 dflash.py:97-108), plus K/V of the noise block; the block's 8 queries attend
                 non-causally to the block and to ctx rows inside the sliding window.
        block   = [tok at c-1] + 7 x mask embedding (mask_embedding.pt), positions c-1..c+6
        drafts  = argmax lm_head(norm(h))[1:]                 (embed / lm_head: the target's)
    K/V of committed ctx rows are cached per layer; the block's K/V are temporaries (dflash.py
    crops the draft cache back to `start` after every block: `past_key_values_draft.crop(start)`).
    Partial rotary (0.5), q/k norm, attention sink bias and the value scale (0.612) follow the
    config; exact v_scale placement and the mask embedding are D2's to pin against the vendor.

    rebuild_rows = window - 1: the block's anchor query at c-1 reads ctx rows (c-1-w, c-2]."""

    def __init__(self, *, vocab=128, hidden=16, heads=4, kv_heads=2, head_dim=8, inter=32,
                 n_layers=5, block=8, window=1024, taps=(0, 11, 23, 35, 47), rot=4,
                 v_scale=0.612, seed=1):
        super().__init__()
        rng = np.random.default_rng(seed)
        self.V, self.H, self.w, self.B = vocab, hidden, window, block
        self.block, self.ctx_dim = block - 1, hidden
        self.rebuild_rows = window - 1
        self.taps = TapSpec(tuple(taps), hidden)
        self.fc = _rng_w(rng, hidden, len(taps) * hidden)
        self.hidden_norm = (1 + 0.1 * rng.standard_normal(hidden)).astype(F32)
        self.norm = (1 + 0.1 * rng.standard_normal(hidden)).astype(F32)
        self.embed = _rng_w(rng, vocab, hidden, scale=1.0)
        self.lm_head = _rng_w(rng, vocab, hidden)
        self.mask_embedding = _rng_w(rng, hidden, scale=1.0)
        self.layers = [_Block(rng, hidden, heads, kv_heads, head_dim, inter, rot, qk_norm=True,
                              v_scale=v_scale) for _ in range(n_layers)]

    def _state(self, start):
        return {"K": [dict() for _ in self.layers], "V": [dict() for _ in self.layers]}

    def _commit_row(self, s, row):
        p = s["start"] + len(s["ids"]) - 2
        tn = rmsnorm(np.asarray(row, F32), self.hidden_norm)
        for l, blk in enumerate(self.layers):
            s["K"][l][p], s["V"][l][p] = blk.kv(tn, p)     # no input_layernorm on ctx (dflash.py:97)
            s["K"][l].pop(p - self.w, None)
            s["V"][l].pop(p - self.w, None)

    def draft_logits(self, sid):
        s = self._s[sid]
        c = s["start"] + len(s["ids"])
        a = c - 1                                           # the anchor: newest committed id
        pos = list(range(a, a + self.B))
        h = np.stack([self.embed[s["ids"][-1]]] + [self.mask_embedding] * (self.B - 1))
        for l, blk in enumerate(self.layers):
            xn = np.stack([rmsnorm(r, blk.ln1) for r in h])
            kb = [blk.kv(xn[j], pos[j]) for j in range(self.B)]
            Kb, Vb = np.stack([k for k, _ in kb]), np.stack([v for _, v in kb])
            nh = []
            for j, p in enumerate(pos):
                cps = [cp for cp in range(max(s["start"], p - self.w + 1), a) if cp in s["K"][l]]
                K = np.concatenate([np.stack([s["K"][l][cp] for cp in cps]), Kb]) if cps else Kb
                V = np.concatenate([np.stack([s["V"][l][cp] for cp in cps]), Vb]) if cps else Vb
                nh.append(blk.finish(h[j], attend(blk.q(xn[j], p), K, V, blk.sink, blk.hd ** -0.5)))
            h = np.stack(nh)
        return np.stack([mv(self.lm_head, rmsnorm(r, self.norm)) for r in h[1:]])


# ── the tail drafter host ────────────────────────────────────────────────────────────────────────
class TailDrafterHost:
    """Runs a Drafter on the tail next to the shipped accept rule (pspec.accept.Frontier).

    Per session: the Frontier judges each frame (`acc`), the drafter advances over exactly the
    rows the Frontier committed (rows pos..pos+k fed committed ids; their outputs toks[0..k] are the
    new committed ids), and a block is proposed unless the lazy licence proves the entry will not
    consume it. A frame off the committed path moves nothing: no advance, no proposal."""

    def __init__(self, drafter):
        self.d = drafter
        self.front, self.last = {}, {}
        self.skipped = self.drafted = 0

    def prefill(self, sid, ids, ctx, token):
        """Rows 0..n-1 (the prompt, ctx for each) answered `token` at n. -> the first block (the
        entry streams [token] + block right away), or None when the drafter has nothing."""
        self.front[sid] = f = Frontier()
        f.prefill(len(ids), token)
        self.d.open(sid, list(ids) + [int(token)], ctx)
        blk = self.d.propose(sid, f.cfront)
        self.last[sid] = (f.cfront, blk)
        self.drafted += 1
        return blk or None

    def on_frame(self, sid, pos, fed, toks, ctx, dnxt=None, lic=False):
        """-> (acc, block | None, lz). `ctx` holds one row per fed row (the tail computed them all;
        only the committed ones are handed to the drafter)."""
        fr = self.front[sid]
        acc, k = fr.judge(pos, fed, toks)
        if not acc:
            return False, None, False
        self.d.advance(sid, pos, [int(t) for t in toks[:k + 1]], np.asarray(ctx)[:k + 1])
        n = len(fed)
        last_row = pos + n - 1
        if k == n - 1 and not wants_block(last_row, int(toks[-1]), dnxt=dnxt, lic=lic,
                                          last=self.last[sid]):
            self.last[sid] = None
            self.skipped += 1
            return True, None, True
        blk = self.d.propose(sid, fr.cfront)
        self.last[sid] = (fr.cfront, blk)
        self.drafted += 1
        return True, (blk or None), False

    def close(self, sid):
        self.front.pop(sid, None)
        self.last.pop(sid, None)
        self.d.close(sid)
