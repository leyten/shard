"""CPU FAKE-RING harness for the m25_pipe coordinators — no GPU, no model, no network ring.

Runs coordinate_pipe / coordinate_pipe_tree against a TEACHER-FORCED ORACLE ring played by a
background thread over two real sockets (socketpair), speaking the exact head+tail wire protocol
serve() speaks (reset -> "ok", receipt -> [], verify -> per-position argmax [+ EAGLE aux]) through
the same send_msg/recv_msg codec the coordinator uses (SHARD_TRANSPORT=libp2p -> shard/transport.py's
JSON+tensor frames, no PSK).

THE ORACLE: a fixed target token sequence T indexed by ABSOLUTE position. For any verify slot at
absolute position p (plain chunk: start+i; tree node: pos_ids[i]) the reply token is T[p+1] — the
model's greedy argmax depends only on the position, never on the (possibly wrong) draft token in
that slot. This preserves speculative-accept accounting exactly: a draft token is accepted iff it
equals the true continuation, and every committed stream must therefore be a prefix of T after the
prompt. Losslessness == `output_ids == T[P:P+n]`, assertable to the token.

EAGLE aux convention (mirrors serve()'s tail return {"toks":..., "aux":{str(li): [s,H]}}):
aux[li][i] is filled with the ABSOLUTE POSITION of slot i, as float32 (the real tail sends bf16,
but bf16 only holds integers exactly up to 256; fp32 keeps the position encoding assertable at any
test length — the coordinator and codec are dtype-agnostic). This makes the EAGLE extend-pairing
contract (auxes[i] = hidden at base_pos+i, tokens[i] = T[base_pos+i+1]) directly checkable via
RecordingDrafter.

Import bootstrap (must run BEFORE m25_pipe): a minimal fake M25_DIR (real M2.5 dims) satisfies
m25_stage's module-level AutoConfig.from_pretrained + safetensors index read; SHARD_TRANSPORT=libp2p
selects the PSK-free codec; phase0/ and shard/ go on sys.path (node_kv does a flat
`import transport`).
"""
import json
import os
import select
import socket
import sys
import tempfile
import threading
import time

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(_REPO, "phase0"), os.path.join(_REPO, "shard"), os.path.dirname(os.path.abspath(__file__))):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ["SHARD_TRANSPORT"] = "libp2p"          # JSON+tensor codec, no PSK (set before node_kv import)


def _fake_model_dir():
    """Minimal M25_DIR that satisfies m25_stage's module-level reads: a minimax_m2 config.json with
    the real M2.5 dims + an empty safetensors weight index (weights are only read lazily via raw())."""
    d = tempfile.mkdtemp(prefix="m25_fake_")
    json.dump({
        "model_type": "minimax_m2", "hidden_size": 3072, "num_attention_heads": 48,
        "num_key_value_heads": 8, "head_dim": 128, "num_hidden_layers": 62,
        "rms_norm_eps": 1e-6, "num_local_experts": 256, "num_experts_per_tok": 8,
        "intermediate_size": 1536, "moe_intermediate_size": 1536, "rope_theta": 5000000,
        "vocab_size": 200064, "max_position_embeddings": 196608,
    }, open(os.path.join(d, "config.json"), "w"))
    json.dump({"weight_map": {}}, open(os.path.join(d, "model.safetensors.index.json"), "w"))
    return d


if not os.path.exists(os.path.join(os.environ.get("M25_DIR", "/nonexistent"), "config.json")):
    os.environ["M25_DIR"] = _fake_model_dir()

import torch                                       # noqa: E402
import m25_pipe as MP                              # noqa: E402  (imports m25_stage on CPU)
from node_kv import send_msg, recv_msg             # noqa: E402  (libp2p codec — same one the coordinator uses)

S = MP.S


class FakeTok:
    """The minimal tokenizer surface coordinate_pipe touches: eos (an id NOT in T so generation never
    stops early), decode() for the result text, and apply_chat_template() so render_ids() yields
    exactly the chosen prompt prefix of T."""

    def __init__(self, prompt_ids, eos_id=10 ** 6):
        self._ids = [int(t) for t in prompt_ids]
        self.eos_token_id = eos_id

    def apply_chat_template(self, messages, tools=None, add_generation_prompt=True, return_dict=True, **kw):
        return {"input_ids": list(self._ids)}

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(int(i)) for i in ids)

    def __call__(self, text, add_special_tokens=False):   # render_ids' reasoning=False path (unused but cheap)
        return {"input_ids": []}


class RecordingDrafter:
    """Transparent wrapper over a drafter (EagleDrafter, typically) that records every
    extend(tokens, auxes, base_pos) call before delegating — the probe for the EAGLE left-shift
    pairing contract. Everything else (propose/propose_tree/reset/cancel/request/fetch/matched)
    forwards to the wrapped drafter untouched."""

    def __init__(self, inner):
        self.inner = inner
        self.extends = []                          # [(tokens list, aux [n,3,H] fp32 clone, base_pos int)]

    def extend(self, tokens, auxes, base_pos):
        toks = tokens.tolist() if torch.is_tensor(tokens) else list(tokens)
        aux = auxes if torch.is_tensor(auxes) else torch.stack([torch.as_tensor(a) for a in auxes], 0)
        self.extends.append(([int(t) for t in toks], aux.detach().clone().float(), int(base_pos)))
        return self.inner.extend(tokens, auxes, base_pos)

    def __getattr__(self, name):                   # only called for attributes NOT defined above
        return getattr(self.inner, name)


class FakeRing(threading.Thread):
    """Head+tail combined, one thread: recv each coordinator message off the pipe socket, reply the
    oracle answer on the return socket — exactly one reply per message, FIFO, mirroring serve()'s
    per-message reply discipline. Exits on EOF (coordinator closed its ends). Fully deterministic."""

    def __init__(self, pipe_sock, ret_sock, T, eagle=False, aux_h=32, aux_layer_ids=None, stage_dt=None,
                 stall_decode=None, stall_prefill=None):
        super().__init__(daemon=True)
        self.pipe = pipe_sock
        self.ret = ret_sock
        self.stall_decode = stall_decode            # (n, seconds): sleep before the first n decode replies —
        self.stall_prefill = stall_prefill          # (n, seconds): sleep before the first n PREFILL replies —
        self.stalled_pf = 0                         # the F6 heartbeat exempts prefill, so this must NOT trip it
        self.stalled = 0                            # a pipelining coordinator fills its depth window during the
        self.T = [int(t) for t in T]                # stall (deterministic backlog); a sync one cannot send at all
        self.eagle = eagle                          # reply {"toks","aux"} like an M25_EAGLE tail, else plain list
        self.aux_h = aux_h                          # aux hidden width (32 = the synthetic EAGLE head's H)
        self.aux_ids = list(aux_layer_ids if aux_layer_ids is not None else S.EAGLE_AUX_LAYER_IDS)
        self.stage_dt = stage_dt                    # [[stage, span_ms, comp_ms], ...] on every verify reply,
        self.log = []                               # mirroring stages under M25_STAGE_TIMING (bare list -> dict)
        self.error = None
        self._job_graph = None                      # the reset's graph field (per-job A/B arm) — the tail's job_graph
        # KV dirty-frontier model: `clean` = first slot NOT yet written by a causal frame. A chain/prefill
        # frame writes [start, start+n) and requires start <= clean (a gap means the coordinator relied on
        # KV the ring never wrote — the cross-mode hybrid bug class). A TREE frame's causal trunk advances
        # clean; its tree NODES do not (their rows are scattered/dirty until re-fed — why pending_path exists).
        self.clean = 0
        self.written = {}                           # slot -> LAST token written there (KV content model)
        self._viol = []                             # (frame_idx, junk_slot): a frame read context whose last
        self._starts = []                           # write != committed text. LEGAL for stale in-flight frames
                                                    # after a divergence (their replies are discarded and a
                                                    # recovery frame re-writes the slot later — FIFO makes it
                                                    # arrive after), so the sound invariant is POST-HOC: every
                                                    # junk-read slot must be re-written by a later frame with
                                                    # start <= that slot, else a real ring served a stale row
                                                    # forever (adversarial-review gaps G1/G2)
        self.backlog = 0                            # times a verify arrived with MORE frames already buffered
                                                    # on the pipe socket == direct evidence of depth>1 pipelining

    # oracle: the target's greedy argmax AT absolute position p is T[p+1] (clamped past the end so
    # deep in-flight speculation never crashes the ring — those replies get discarded anyway)
    def _tok_at(self, p):
        return self.T[p + 1] if p + 1 < len(self.T) else self.T[-1]

    def _aux(self, positions):
        n = len(positions)
        col = torch.tensor([float(p) for p in positions], dtype=torch.float32).unsqueeze(1)
        a = col.expand(n, self.aux_h).contiguous()
        return {str(li): a.clone() for li in self.aux_ids}

    def run(self):
        try:
            while True:
                msg = recv_msg(self.pipe)
                op = msg.get("op")
                if op == "reset":
                    self._job_graph = msg.get("graph")      # per-job A/B toggle (None = absent)
                    self.log.append({"op": "reset", "graph": self._job_graph, "keepwarm_ms": msg.get("keepwarm_ms")})
                    # mirror the tail's ack contract: a graph-stamped reset acks the APPLIED route +
                    # counters (the fake always applies as asked); plain resets keep the bare "ok"
                    send_msg(self.ret, "ok" if self._job_graph is None else
                             {"ok": 1, "graph": bool(self._job_graph), "graph_captured": 0, "graph_skipped": 0})
                elif op == "noop":                  # cwnd keep-warm frame: leg-local, never answered —
                    self.log.append({"op": "noop"})  # mirrors serve()'s skip; logged so tests can see it
                elif op == "receipt":
                    self.log.append({"op": "receipt"})
                    rec = msg.get("receipts", [])           # graph-A/B jobs get the dict-promoted reply, like the tail
                    send_msg(self.ret, rec if self._job_graph is None else
                             {"receipts": rec, "graph": bool(self._job_graph),
                              "graph_captured": 0, "graph_skipped": 0})
                elif op == "verify":
                    s = int(msg["start"])
                    if s > self.clean:
                        raise AssertionError(
                            f"KV GAP: frame start={s} past clean frontier {self.clean} — the coordinator "
                            f"assumed KV the ring never wrote (dirty pending_path not re-fed?)")
                    idx = len(self._starts); self._starts.append(s)
                    for sl in range(min(s, len(self.T))):   # context below start should be committed text
                        w = self.written.get(sl)
                        if w is not None and w != self.T[sl]:
                            self._viol.append((idx, sl))    # judged post-hoc (stale frames may read junk)
                    for i, t in enumerate(msg["token_ids"]):   # both frame kinds write flat-order at start+i
                        self.written[s + i] = int(t)
                    if msg.get("tree"):
                        pos = [int(p) for p in msg["pos_ids"]]
                        # leading causal run = re-fed trunk (+ a chain-shaped tree prefix, causally
                        # identical to more trunk — indistinguishable by design, and equally clean-KV).
                        # No content check: trunk tokens can't be told from speculation here, and after
                        # a divergence rollback re-writing junk slots is legitimate; output correctness
                        # is what the end-to-end losslessness assertion (out == T-prefix) pins.
                        trunk = 0
                        while trunk < len(msg["parents"]) and msg["parents"][trunk] == trunk - 1:
                            trunk += 1
                        self.clean = max(self.clean, s + trunk)
                    else:
                        pos = [s + i for i in range(len(msg["token_ids"]))]
                        self.clean = max(self.clean, s + len(pos))
                    if self.stall_decode and not msg.get("prefill") and self.stalled < self.stall_decode[0]:
                        self.stalled += 1
                        time.sleep(self.stall_decode[1])
                    if self.stall_prefill and msg.get("prefill") and self.stalled_pf < self.stall_prefill[0]:
                        self.stalled_pf += 1
                        time.sleep(self.stall_prefill[1])
                    if select.select([self.pipe], [], [], 0)[0]:
                        self.backlog += 1           # coordinator sent ahead: >1 frame in flight right now
                    self.log.append({"op": "verify", "start": int(msg["start"]),
                                     "n": len(msg["token_ids"]), "tree": bool(msg.get("tree")),
                                     "prefill": bool(msg.get("prefill")),
                                     "token_ids": [int(t) for t in msg["token_ids"]], "pos": pos})
                    toks = [self._tok_at(p) for p in pos]
                    o = {"toks": toks, "aux": self._aux(pos)} if self.eagle else toks
                    if self.stage_dt is not None:   # M25_STAGE_TIMING tail: timing rows promote a bare list to a dict
                        o = o if isinstance(o, dict) else {"toks": o}
                        o["stage_dt"] = [list(row) for row in self.stage_dt]
                    send_msg(self.ret, o)
                else:
                    raise ValueError(f"fake ring got unexpected op {op!r}")
        except (OSError, EOFError):                 # coordinator closed its ends — normal shutdown
            # POST-HOC: every junk-read must be healed by a later frame re-writing that slot. The final
            # <=tail_slack frames are exempt: when the job ends on a divergence, the drained in-flight
            # frames read junk with no frame ever following — their replies are discarded by design.
            cutoff = len(self._starts) - max(getattr(self, "tail_slack", 4), 1)
            for idx, sl in self._viol:
                if idx >= cutoff:
                    continue
                if not any(st <= sl for st in self._starts[idx + 1:]):
                    self.error = AssertionError(
                        f"KV CONTENT: frame #{idx} (start={self._starts[idx]}) read stale slot {sl} "
                        f"(held {self.written.get(sl)} vs committed T[{sl}]={self.T[sl]}) and NO later "
                        f"frame ever re-wrote it — a real ring serves that stale row forever")
                    return
            return
        except Exception as e:                      # anything else is a harness bug — surface it
            self.error = e
            for s_ in (self.pipe, self.ret):        # close our ends so the coordinator fails NOW,
                try:                                # not after its full recv timeout
                    s_.close()
                except OSError:
                    pass


class FakeRingB(threading.Thread):
    """Batched (de-lockstep) head+tail oracle for coordinate_pipe_rows: per-STREAM target sequences
    Ts, the solo FakeRing's KV dirty-frontier model held PER STREAM (each stream owns a KV row —
    frames must never assume KV their row never got), and decode replies tagged with the stream (the
    rows coordinator's LOUD FIFO-pairing guard requires it; prefill replies stay untagged, mirroring
    serve()'s batched-prefill branch). Chain/prefill row frames write causally and advance that
    stream's clean frontier; TREE frames advance it by their causal trunk only (tree nodes stay
    dirty until re-fed — the per-stream pending_path contract). aux = position-encoded fp32 exactly
    like FakeRing, so a rows stream and a solo reference run against FakeRing(T=Ts[b]) see
    BYTE-IDENTICAL aux -> byte-identical drafter evolution -> the equivalence gate is exact."""

    def __init__(self, pipe_sock, ret_sock, Ts, aux_h=32, aux_layer_ids=None):
        super().__init__(daemon=True)
        self.pipe = pipe_sock
        self.ret = ret_sock
        self.Ts = [[int(t) for t in T] for T in Ts]
        self.aux_h = aux_h
        self.aux_ids = list(aux_layer_ids if aux_layer_ids is not None else S.EAGLE_AUX_LAYER_IDS)
        self.st = [{"clean": 0, "written": {}, "viol": [], "starts": []} for _ in Ts]
        self.log = []
        self.error = None

    def _tok_at(self, b, p):
        T = self.Ts[b]
        return T[p + 1] if p + 1 < len(T) else T[-1]

    def _aux(self, positions):
        n = len(positions)
        col = torch.tensor([float(p) for p in positions], dtype=torch.float32).unsqueeze(1)
        a = col.expand(n, self.aux_h).contiguous()
        return {str(li): a.clone() for li in self.aux_ids}

    def run(self):
        try:
            while True:
                msg = recv_msg(self.pipe)
                op = msg.get("op")
                if op == "reset_batch":
                    self.log.append({"op": "reset_batch", "B": msg.get("B")})
                    send_msg(self.ret, "ok")
                elif op == "noop":
                    self.log.append({"op": "noop"})
                elif op == "receipt":
                    self.log.append({"op": "receipt"})
                    send_msg(self.ret, msg.get("receipts", []))
                elif op == "verify":
                    b = msg["stream"]                   # rows frames ALWAYS tag (untagged = solo-path bug)
                    s = int(msg["start"])
                    st = self.st[b]
                    if s > st["clean"]:
                        raise AssertionError(
                            f"KV GAP stream {b}: frame start={s} past clean frontier {st['clean']} — "
                            f"the rows coordinator assumed row KV the ring never wrote "
                            f"(dirty pending_path not re-fed?)")
                    idx = len(st["starts"]); st["starts"].append(s)
                    for sl in range(min(s, len(self.Ts[b]))):
                        w = st["written"].get(sl)
                        if w is not None and w != self.Ts[b][sl]:
                            st["viol"].append((idx, sl))
                    for i, t in enumerate(msg["token_ids"]):
                        st["written"][s + i] = int(t)
                    if msg.get("tree"):
                        pos = [int(p) for p in msg["pos_ids"]]
                        trunk = 0                       # leading causal run = the re-fed trunk (see FakeRing)
                        while trunk < len(msg["parents"]) and msg["parents"][trunk] == trunk - 1:
                            trunk += 1
                        st["clean"] = max(st["clean"], s + trunk)
                    else:
                        pos = [s + i for i in range(len(msg["token_ids"]))]
                        st["clean"] = max(st["clean"], s + len(pos))
                    self.log.append({"op": "verify", "stream": b, "start": s,
                                     "n": len(msg["token_ids"]), "tree": bool(msg.get("tree")),
                                     "prefill": bool(msg.get("prefill")),
                                     "token_ids": [int(t) for t in msg["token_ids"]], "pos": pos})
                    toks = [self._tok_at(b, p) for p in pos]
                    o = {"toks": toks, "aux": self._aux(pos)}
                    if not msg.get("prefill"):
                        o["stream"] = b                 # decode replies carry the FIFO-pairing tag
                    send_msg(self.ret, o)
                else:
                    raise ValueError(f"fake rows ring got unexpected op {op!r}")
        except (OSError, EOFError):                     # coordinator closed its ends — normal shutdown
            for b, st in enumerate(self.st):            # POST-HOC healing per stream (FakeRing's rule);
                cutoff = len(st["starts"]) - 1          # rows is depth-1 per stream -> slack of 1 frame
                for idx, sl in st["viol"]:
                    if idx >= cutoff:
                        continue
                    if not any(s2 <= sl for s2 in st["starts"][idx + 1:]):
                        self.error = AssertionError(
                            f"KV CONTENT stream {b}: frame #{idx} read stale slot {sl} "
                            f"(held {st['written'].get(sl)} vs committed {self.Ts[b][sl]}) and no "
                            f"later frame re-wrote it")
                        return
            return
        except Exception as e:
            self.error = e
            for s_ in (self.pipe, self.ret):
                try:
                    s_.close()
                except OSError:
                    pass


class FakeTokB:
    """FakeTok for the rows coordinator: ONE tokenizer, per-stream prompts — a message's content is
    the stream index and apply_chat_template returns that stream's prompt prefix."""

    def __init__(self, prompts, eos_id=10 ** 6):
        self._prompts = [list(map(int, p)) for p in prompts]
        self.eos_token_id = eos_id

    def apply_chat_template(self, messages, tools=None, add_generation_prompt=True, return_dict=True, **kw):
        return {"input_ids": list(self._prompts[int(messages[-1]["content"])])}

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(int(i)) for i in ids)

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": []}


def run_rows_coordinator(Ts, prompt_lens, drafters, *, K=8, max_new=160, prefill_chunk=24, timeout=30):
    """One coordinate_pipe_rows job against a fresh FakeRingB. S.M25_TREE (monkeypatched by the
    caller) arms the per-stream tree route, same as production. Returns (result, ring)."""
    c_pipe, r_pipe = socket.socketpair()
    c_ret, r_ret = socket.socketpair()
    c_ret.settimeout(timeout)
    ring = FakeRingB(r_pipe, r_ret, Ts)
    ring.start()
    tok = FakeTokB([T[:pl] for T, pl in zip(Ts, prompt_lens)])
    msgs = [[{"role": "user", "content": str(b)}] for b in range(len(Ts))]
    try:
        try:
            res = MP.coordinate_pipe_rows(c_pipe, tok, msgs, K, max_new, timeout, c_ret, drafters,
                                          prefill_chunk=prefill_chunk)
        except Exception:
            ring.join(2)
            if ring.error is not None:
                raise AssertionError(
                    f"fake rows ring crashed: {type(ring.error).__name__}: {ring.error}") from None
            raise
    finally:
        for s_ in (c_pipe, c_ret):
            try:
                s_.close()
            except OSError:
                pass
        ring.join(10)
        for s_ in (r_pipe, r_ret):
            try:
                s_.close()
            except OSError:
                pass
    if ring.error is not None:
        raise AssertionError(f"fake rows ring crashed: {type(ring.error).__name__}: {ring.error}")
    return res, ring


# ---- target-sequence builders ------------------------------------------------------------------
# Token ids stay < 100 (the synthetic EAGLE head's target vocab) so the real EagleDrafter's embed
# lookups are always in range on every path.

def novel_T(n, seed=1234):
    """Pseudo-random tokens — no n-gram structure, the drafter is blind."""
    import random
    rng = random.Random(seed)
    return [rng.randrange(2, 100) for _ in range(n)]


def repetitive_T(n, seed=99, phrase_len=30):
    """A ~30-token phrase repeated verbatim — the n-gram drafter matches long runs (rag-quote-like)."""
    import random
    rng = random.Random(seed)
    phrase = [rng.randrange(2, 100) for _ in range(phrase_len)]
    out = []
    while len(out) < n:
        out.extend(phrase)
    return out[:n]


def trap_T(n, seed=7, break_at=168, novel_len=30):
    """Repetition that BREAKS once: phrase A repeats through `break_at`, then a novel run, then a
    different phrase B repeats. The n-gram drafter confidently speculates A's continuation across the
    break -> a mid-stream divergence with chunks in flight (the discard-bookkeeping trap)."""
    import random
    rng = random.Random(seed)
    a = [rng.randrange(2, 100) for _ in range(26)]
    b = [rng.randrange(2, 100) for _ in range(26)]
    out = []
    while len(out) < break_at:
        out.extend(a)
    out = out[:break_at]
    out += [rng.randrange(2, 100) for _ in range(novel_len)]
    while len(out) < n:
        out.extend(b)
    return out[:n]


# ---- runner --------------------------------------------------------------------------------------

def run_coordinator(T, prompt_len, drafter, *, K=8, depth=4, max_new=160, prefill_chunk=4096,
                    eagle_ring=False, timeout=30, on_commit=None, stage_dt=None, stall_decode=None,
                    stall_prefill=None):
    """One coordinate_pipe job against a fresh FakeRing. S.M25_TREE (monkeypatched by the caller)
    routes to coordinate_pipe_tree inside coordinate_pipe, same as production. Returns (result, ring);
    ring.log is the wire-level ground truth."""
    c_pipe, r_pipe = socket.socketpair()
    c_ret, r_ret = socket.socketpair()
    c_ret.settimeout(timeout)                       # mirror coord(): bound return-channel recv
    ring = FakeRing(r_pipe, r_ret, T, eagle=eagle_ring, stage_dt=stage_dt, stall_decode=stall_decode,
                    stall_prefill=stall_prefill)
    ring.tail_slack = depth                         # end-of-job drain window exempt from the healing check
    ring.start()
    tok = FakeTok(T[:prompt_len])
    try:
        try:
            res = MP.coordinate_pipe(c_pipe, tok, [{"role": "user", "content": "fake"}], K, max_new,
                                     timeout, depth, c_ret, drafter,
                                     prefill_chunk=prefill_chunk, on_commit=on_commit)
        except Exception:
            ring.join(2)                            # a ring-side assert kills the ring first and the
            if ring.error is not None:              # coordinator only sees the dead socket — surface
                raise AssertionError(               # the ROOT CAUSE, not the TransportError symptom
                    f"fake ring crashed: {type(ring.error).__name__}: {ring.error}") from None
            raise
    finally:
        for s_ in (c_pipe, c_ret):
            try:
                s_.close()
            except OSError:
                pass
        ring.join(10)
        for s_ in (r_pipe, r_ret):
            try:
                s_.close()
            except OSError:
                pass
    if ring.error is not None:
        raise AssertionError(f"fake ring crashed: {type(ring.error).__name__}: {ring.error}")
    return res, ring
