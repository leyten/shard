"""shard-receipt/2 (shard/receipt2.py): every scripted mesh scenario verifies, every lie fails closed.

The scenarios (tests/receipt_mesh.py) run real ToyRuntime stages over encoded shard-frame/2
messages: plain decode, pipelined speculation with rejections, two batched sessions, fp8 wire,
two route changes (a fatter node, then a re-split with a chained rebuild), a KV-mirror takeover
in other numerics (valid, not bit-exact), and a dead tail rebuilt from its predecessor's log.
The mutation suite then makes one party lie at a time and requires a rejection that names it;
the regressions at the end are the adversarial review's attacks (each was accepted, crashed, or
falsely rejected a legitimate recovery before the fix).

Run: OMP_NUM_THREADS=2 python3 -m pytest tests/test_receipt2.py -q
"""
import copy
import hashlib
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import receipt_mesh as M  # noqa: E402
from shard import receipt2 as R2  # noqa: E402
from shard.receipt import ReceiptError  # noqa: E402

_CACHE = {}


def gen(scenario, **kw):
    """Scenario bundles are deterministic; generate each variant once per session."""
    k = (scenario, tuple(sorted(kw.items())))
    if k not in _CACHE:
        _CACHE[k] = M.scenario(scenario, **kw)
    return copy.deepcopy({kk: v for kk, v in _CACHE[k].items() if kk != "job"}) | {"job": _CACHE[k]["job"]} \
        if isinstance(_CACHE[k], dict) else _CACHE[k]


def rejects(b, *needles):
    with pytest.raises(ReceiptError) as ei:
        M.verify(b)
    msg = str(ei.value)
    for n in needles:
        assert n in msg, f"expected {n!r} in the rejection, got: {msg}"
    return msg


def hop_ckpts(b, lo, rte=0, log="serve"):
    return [c for c in b["checkpoints"] if c["lo"] == lo and c["rte"] == rte and c["log"] == log]


def node_of(b, lo, rte=0):
    job = b["job"]
    return next(n for n in job.paths[rte] if n.lo == lo)


def replace(b, old, new):
    ids = {id(c) for c in old}
    b["checkpoints"] = [c for c in b["checkpoints"] if id(c) not in ids] + new


def relink(priv, ckpts, edit=lambda ck: None):
    """A hop re-signs its own checkpoint sequence after editing it (prev links recomputed)."""
    out, prev = [], ""
    for ck in sorted(ckpts, key=lambda c: (c["pos"], c["kind"] != "grid")):
        ck = copy.deepcopy(ck)
        edit(ck)
        ck["prev_ckpt"] = prev
        ck = R2.resign(priv, ck)
        prev = R2.ckpt_id(ck)
        out.append(ck)
    return out


def rechain_seg(ckpts):
    c = None
    for ck in ckpts:
        s = ck["seg"]
        c = c or R2._seg0(s["lo"])
        for rec in s["recs"]:
            c = R2._seg_step(c, rec)
        s["chain"] = c.hex()


def resign_path(b, edit):
    pl = copy.deepcopy(b["path_log"])
    edit(pl)
    b["path_log"] = R2.resign(b["job"].entry, pl, key_field="entry")


# ── every scenario verifies ───────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("name", ["plain", "spec", "fp8", "reroute", "takeover", "dead"])
def test_scenario_verifies(name):
    b = gen(name)
    v = M.verify(b)
    assert v["ok"] and v["pos_end"] == b["job"].F
    assert v["tokens"] == len(b["tokens"]), "the last tail's tok_next attests every generated token"
    # serve work tiles layers x positions exactly once, minus a dead hop's unchained tail
    job = b["job"]
    for r, e in enumerate(job.rtes):
        units = sum((w["hi"] - w["lo"]) * (w["prefill"] + w["decode"]) for w in v["work"]
                    if w["kind"] == "serve" and w["rte"] == r)
        dead_gap = sum((w["hi"] - w["lo"]) * (e["pos_end"] - e["pos_start"] - w["prefill"] - w["decode"])
                       for w in v["work"] if w["kind"] == "serve" and w["rte"] == r and w["pubkey"] in e["dead"])
        assert units + dead_gap == M.L * (e["pos_end"] - e["pos_start"])


def test_batched_sessions_verify_independently():
    a, b = gen("batch")
    for x in (a, b):
        assert M.verify(x)["ok"]
    # the same hops served both sessions in shared steps; receipts never mix sessions
    assert {c["pubkey"] for c in a["checkpoints"]} == {c["pubkey"] for c in b["checkpoints"]}
    b2 = copy.deepcopy({k: v for k, v in b.items() if k != "job"}) | {"job": b["job"]}
    b2["checkpoints"] = b2["checkpoints"][:-1] + [hop_ckpts(a, 9)[-1]]
    rejects(b2, "stale or replayed")          # another session's checkpoint: other job, other nonce


def test_speculation_rows_are_recomputed_but_folded_once():
    b = gen("spec")
    close = [c for c in b["checkpoints"] if c["kind"] == "close"]
    job = b["job"]
    assert job.sep > 50, "the drafter must miss often enough to exercise rejections"
    for c in close:
        assert c["n_rows"] > 2 * c["pos"], "rejected speculation is computed, counted as telemetry..."
    recs = [r for c in hop_ckpts(b, 6) for r in c["seg"]["recs"]]
    assert sum(1 for x, y in zip(recs, recs[1:]) if y[0] < x[0] + x[1]) > 50, \
        "...and only the committed prefix of each segment is folded"


def test_mixed_numerics_on_one_path():
    b = gen("takeover")
    nids = {c["numerics_id"] for c in b["checkpoints"] if c["rte"] == 0 and c["log"] == "serve"}
    assert len(nids) == 2, "chunk-invariant and shape-dependent numerics share the path"
    rb = [c for c in b["checkpoints"] if c["log"] == "rebuild"][0]
    prim = hop_ckpts(b, 6)[0]
    assert rb["numerics_id"] != prim["numerics_id"], "the takeover crosses numerics classes"


def test_takeover_is_valid_not_bit_exact():
    b = gen("takeover")
    job = b["job"]
    path = M.path_a("takeover")
    twin = M.Job(path, name="takeover", K=3, depth=2, miss=4, n_new=330,
                 backups={path[2]: M.Node("takeover/backup", 6, 9, "adversarial")})
    M.run([twin])
    F = job.rtes[0]["pos_end"]
    assert twin.ids[:F + 1] == job.ids[:F + 1]
    assert twin.ids[F + 1:] != job.ids[F + 1:], "other numerics after the takeover: not bit-exact..."
    assert M.verify(b)["ok"]                                  # ...and still verifiable


def test_fp8_wire_is_lossy_and_still_chained():
    b = gen("fp8")
    twin = M.Job(M.path_a("fp8"), name="fp8", K=3, depth=2, miss=4)
    M.run([twin])
    assert twin.ids != b["job"].ids, "fp8 changes the continuation (lossy codec)"
    recs = [r for c in hop_ckpts(b, 6) for r in c["seg"]["recs"]]
    assert {r[2] for r in recs} == {"fp8"}
    assert M.verify(b)["ok"]


def test_dead_hop_paid_to_its_last_delivered_checkpoint():
    b = gen("dead")
    v = M.verify(b)
    dead = node_of(b, 9)
    g = max(c["pos"] for c in hop_ckpts(b, 9))
    w = [w for w in v["work"] if w["pubkey"] == dead.pub]
    assert len(w) == 1 and w[0]["prefill"] + w[0]["decode"] == g
    assert b["job"].rtes[0]["pos_end"] - g > 0, "positions [g, F) of the dead tail are unchained, unpaid"
    rb = [w for w in v["work"] if w["kind"] == "rebuild"]
    assert rb and rb[0]["positions"] == b["job"].rtes[0]["pos_end"], "a fresh tail rebuilds [0, F)"


def test_takeover_rebuild_credit_and_mirror_pay():
    b = gen("takeover")
    v = M.verify(b)
    e = b["job"].rtes[1]["seated"][0]
    F = b["job"].rtes[1]["pos_start"]
    assert e["upto"] > 0 and e["a"] == R2.rebuild_rule(M.KINDS[6:9], M.WINDOW, F, e["upto"])
    rb = [w for w in v["work"] if w["kind"] == "rebuild"][0]
    assert rb["positions"] == F - e["a"]
    mirror = [w for w in v["work"] if w["kind"] == "mirror"][0]
    assert mirror["positions"] == e["upto"], "the backup is paid for the mirror rows it proved"


def test_backup_withholding_its_mirror_earns_no_extra_rebuild_credit():
    """The leased backup acked mirror rows, then seated as a from-scratch rebuild (tier log,
    upto=0) to claim credit for [0, F). It is paid only from the rule's a given what it acked."""
    b = gen("takeover", tier="log")
    v = M.verify(b)
    F = b["job"].rtes[1]["pos_start"]
    e = b["job"].rtes[1]["seated"][0]
    assert e["upto"] == 0 and e["a"] == 0
    acked = [w for w in v["work"] if w["kind"] == "mirror"][0]["positions"]
    rb = [w for w in v["work"] if w["kind"] == "rebuild"][0]
    assert acked > 0 and rb["positions"] == F - R2.rebuild_rule(M.KINDS[6:9], M.WINDOW, F, acked) < F


# ── the mutation suite: each lie fails closed with a specific reason ──────────────────────────
def test_mutation_dropped_checkpoint():
    b = gen("spec")
    grid = [c for c in hop_ckpts(b, 3) if c["pos"] == 256]
    b["checkpoints"] = [c for c in b["checkpoints"] if c not in grid]
    rejects(b, "dropped checkpoint")


def test_mutation_dropped_checkpoint_relinked():
    """The hop re-signs its sequence around the gap: the grid is still missing a position."""
    b = gen("spec")
    old = hop_ckpts(b, 3)
    replace(b, old, relink(node_of(b, 3).priv, [c for c in old if c["pos"] != 256]))
    rejects(b, "next grid checkpoint is at 256")


def test_mutation_withheld_checkpoint():
    b = gen("spec")
    close = [c for c in hop_ckpts(b, 6) if c["kind"] == "close"]
    b["checkpoints"] = [c for c in b["checkpoints"] if c not in close]
    rejects(b, "withheld checkpoint")


def test_mutation_reused_nonce_checkpoint():
    """A hop replays its checkpoints signed for an earlier job (other nonce)."""
    b = gen("spec")
    old = hop_ckpts(b, 6)
    replace(b, old, relink(node_of(b, 6).priv, old, lambda ck: ck.update(nonce="00" * 16)))
    rejects(b, "not the job's (stale or replayed)")


def test_mutation_reused_nonce_path_log():
    b = gen("spec")
    resign_path(b, lambda pl: pl.update(nonce="00" * 16))
    rejects(b, "path log nonce")


def test_mutation_invented_hop():
    """The entry splits the tail's span and puts its own key on half of it."""
    b = gen("spec")
    fake = R2._pub(M.key("fake"))

    def edit(pl):
        hops = pl["rtes"][0]["hops"]
        tail = hops[-1]
        hops[-1] = dict(tail, lo=10)
        hops.insert(3, {"pubkey": fake, "lo": 9, "hi": 10, "role": "mid"})
    resign_path(b, edit)
    rejects(b, "no issued lease", "invented hop")


def test_mutation_invented_backup():
    b = gen("spec")
    resign_path(b, lambda pl: pl["rtes"][0]["hops"][2].update(backup=R2._pub(M.key("fake"))))
    rejects(b, "invented backup")


def test_mutation_seg_lie_codec():
    """A hop claims an fp8 input for a verify segment on a bf16 path (to excuse lossy output)."""
    b = gen("spec")
    old = hop_ckpts(b, 6)

    def edit(ck):
        for r in ck["seg"]["recs"]:
            if r[3] == "verify":
                r[2] = "fp8"
                break
    new = [copy.deepcopy(c) for c in old]
    for c in new:
        edit(c)
    rechain_seg(new)
    replace(b, old, relink(node_of(b, 6).priv, new))
    rejects(b, "SEG lie", "fp8")


def test_mutation_seg_lie_dropped_segment():
    """Dropping a record that still tiles re-attributes rows to an overlapping segment (a shape
    lie a structural check cannot see); the hop's schedule then differs from its neighbours'."""
    b = gen("spec")
    old = hop_ckpts(b, 6)
    new = [copy.deepcopy(c) for c in old]
    recs = new[1]["seg"]["recs"]
    i = next(i for i in range(1, len(recs) - 1) if recs[i + 1][0] <= recs[i - 1][0] + recs[i - 1][1])
    recs.pop(i)
    rechain_seg(new)
    replace(b, old, relink(node_of(b, 6).priv, new))
    rejects(b, "SEG lie", "ran segment")


def test_mutation_seg_lie_gap():
    b = gen("spec")
    old = hop_ckpts(b, 6)
    new = [copy.deepcopy(c) for c in old]
    recs = new[1]["seg"]["recs"]
    i = next(i for i in range(1, len(recs) - 1) if recs[i + 1][0] > recs[i - 1][0] + recs[i - 1][1])
    recs.pop(i)
    rechain_seg(new)
    replace(b, old, relink(node_of(b, 6).priv, new))
    rejects(b, "SEG lie", "does not continue")


def test_mutation_kv_lie_import():
    b = gen("takeover", bugs_backup=("lie_kv",))
    rejects(b, "KV lie", "imported KV")


def test_mutation_kv_lie_mirror_ack():
    """A backup acks a kv_chain it did not verify (to collect mirror pay)."""
    b = gen("takeover")
    old = [c for c in b["checkpoints"] if c["log"] == "mirror"]
    bk = next(n for n in b["job"].paths[1] if n.name == "takeover/backup")

    def edit(ck):
        d = ck["logs"]["kv"]
        roots = [[k, hashlib.sha256(bytes.fromhex(r)).hexdigest()] for k, r in d["roots"]]
        c = R2._chain0(d["lo"])
        for k, r in roots:
            c = R2._chain(c, k, bytes.fromhex(r))
        d["roots"], d["chain"] = roots, c.hex()
    replace(b, old, relink(bk.priv, old[:1], edit))
    rejects(b, "KV lie", "acked")


def test_mutation_token_closure_lie_entry():
    """The entry feeds the head a token the tail never produced (fabricated output)."""
    job = M.Job(M.path_a("lie-e"), name="lie-e", K=3, depth=2, miss=4, n_new=200)
    job.lie_at = 200
    M.run([job])
    rejects(job.bundle(), "token closure broken")


def test_mutation_token_closure_lie_tail():
    """The tail signs a token it did not return."""
    path = M.path_a("lie-t")
    path[3].lie_at = 230
    job = M.Job(path, name="lie-t", K=3, depth=2, miss=4, n_new=200)
    M.run([job])
    rejects(job.bundle(), "token closure broken")


def test_mutation_draft_fed_row_folded_as_committed():
    """[3,6) folds at c instead of F = c-1: the row at F still holds its draft-fed input."""
    b = gen("spec", bugs=((1, ("fold_at_c",)),), name="foldc", n_new=200)
    rejects(b, "chain break", "[3, 6)")


def _inject(job):
    path = job.path
    return {"node": path[1], "seq": 40, "sender": M.pub("rogue")}


def test_mutation_frame_from_non_predecessor_refused_by_the_node():
    with pytest.raises(M.FrameRefused):
        gen("spec", inject=_inject, name="inj-ok", n_new=200)


def test_mutation_frame_from_non_predecessor_accepted_by_a_buggy_node():
    b = gen("spec", bugs=((2, ("accept_any_sender",)),), inject=_inject, name="inj-bug", n_new=200)
    rejects(b, "chain break", "[6, 9)")


def test_mutation_forged_rebuild_inputs():
    b = gen("dead", bugs_new=("forge_rebuild",))
    rejects(b, "forged rebuild")


def test_mutation_forged_rebuild_start():
    """The replacement claims it could start after the rule's a (skipping rows it needed)."""
    b = gen("takeover")
    job = b["job"]
    rb = [c for c in b["checkpoints"] if c["log"] == "rebuild"]
    bk = next(n for n in job.paths[1] if n.name == "takeover/backup")
    a = rb[0]["rebuild"]["a"]
    assert (a + 64) // R2.GRID == a // R2.GRID      # same aligned input log, so only `a` moves

    def late(ck):
        ck["rebuild"]["a"] = a + 64
        ck["seg"]["lo"] = a + 64
        ck["seg"]["recs"] = [r for r in ck["seg"]["recs"] if r[0] >= a + 64]
        c = R2._seg0(a + 64)
        for r in ck["seg"]["recs"]:
            c = R2._seg_step(c, r)
        ck["seg"]["chain"] = c.hex()
    new = relink(bk.priv, rb, late)
    replace(b, rb, new)
    resign_path(b, lambda pl: pl["rtes"][1]["seated"][0].update(a=a + 64))
    rejects(b, "after the rebuild rule")


# ── the entry cannot frame a hop ──────────────────────────────────────────────────────────────
def test_entry_false_death_claim():
    b = gen("spec")
    resign_path(b, lambda pl: pl["rtes"][0]["dead"].append(node_of(b, 6).pub))
    rejects(b, "false death claim")


def test_entry_withholds_a_live_hops_checkpoints_and_declares_it_dead():
    b = gen("spec")
    keep = [c for c in hop_ckpts(b, 6) if c["pos"] <= 128]
    b["checkpoints"] = [c for c in b["checkpoints"] if c not in hop_ckpts(b, 6)] + keep
    resign_path(b, lambda pl: pl["rtes"][0]["dead"].append(node_of(b, 6).pub))
    rejects(b, "a live hop would have checkpointed")


def test_entry_cannot_forge_a_hops_checkpoint():
    b = gen("spec")
    ck = copy.deepcopy(hop_ckpts(b, 6)[0])
    ck["n_rows"] += 1
    b["checkpoints"] = [c for c in b["checkpoints"] if c["prev_ckpt"] != "" or c["lo"] != 6] + [ck]
    rejects(b, "bad signature")


def test_entry_cannot_move_the_prompt_boundary():
    """prompt_len splits prefill (cheap) from decode pay: the head and tail sign it."""
    for d in (40, -40):
        b = gen("spec")
        resign_path(b, lambda pl, d=d: pl.update(prompt_len=pl["prompt_len"] + d))
        rejects(b, "signed prompt_len")


def test_seg_pins_the_prompt_boundary_too():
    """Defence in depth: truth/spec segments never lie inside the prompt, prefill never past it."""
    lg = type("L", (), {"seg_recs": [[0, 32, "bf16", "prefill"], [32, 4, "bf16", "verify"]],
                        "seg_at": {36: 2}, "log": "serve"})()
    with pytest.raises(ReceiptError, match="below the prompt"):
        R2._seg_check(lg, 0, P=40, codec_of=lambda k: None, where="t")
    with pytest.raises(ReceiptError, match="past the prompt"):
        R2._seg_check(lg, 0, P=20, codec_of=lambda k: None, where="t")
    R2._seg_check(lg, 0, P=32, codec_of=lambda k: None, where="t")


def test_seated_record_must_name_a_hop():
    b = gen("takeover")
    fake = R2._pub(M.key("fake"))
    resign_path(b, lambda pl: pl["rtes"][1]["seated"].append(
        {"pubkey": fake, "tier": "log", "a": 0, "upto": 0, "replaces": None}))
    rejects(b, "seated", "not on the path")


def test_orchestrator_leases_are_required():
    b = gen("spec")
    b["leases"] = [le for le in b["leases"] if le["lo"] != 3]
    rejects(b, "no issued lease")


def test_unknown_signer_checkpoint_rejected():
    b = gen("spec")
    stranger = relink(M.key("stranger"), hop_ckpts(b, 6)[:1])
    b["checkpoints"].append(stranger[0])
    rejects(b, "not a hop of rte 0")


# ── ledger unit behaviour ─────────────────────────────────────────────────────────────────────
def _ledger(**kw):
    base = dict(job_id="j", sid="0" * 16, nonce="00" * 8, manifest_cid="m", numerics_id="n", rte=0,
                lo=3, hi=6, role="mid", pos_lo=0)
    return R2.HopLedger(M.key("unit"), **dict(base, **kw))


def _rows(p0, n, tag=b"a"):
    return [tag + bytes([p0 + j]) for j in range(n)]


def test_ledger_folds_only_below_the_frontier_last_write_wins():
    lg = _ledger(pos_lo=128)
    lg.record(128, 4, kind="verify", codec="bf16", in_rows=_rows(0, 4), out_rows=_rows(0, 4))
    lg.commit(130)                          # rows 128, 129 final; 130, 131 speculative
    lg.record(130, 4, kind="verify", codec="bf16", in_rows=_rows(2, 4, b"b"), out_rows=_rows(2, 4, b"b"))
    ck = lg.close(133, final=False)[-1]      # rte_close ships the partial interval's leaves
    leaves = ck["logs"]["in"]["leaves"][0][2]
    want = [R2.leaf_row(128, b"a\x00"), R2.leaf_row(129, b"a\x01")] + \
        [R2.leaf_row(130 + j, b"b" + bytes([2 + j])) for j in range(3)]
    assert leaves == [x.hex() for x in want]
    assert ck["seg"]["recs"] == [[128, 4, "bf16", "verify"], [130, 4, "bf16", "verify"]]


def test_ledger_refuses_rewriting_a_folded_row_and_gaps():
    lg = _ledger()
    lg.record(0, 4, kind="prefill", codec="bf16", in_rows=_rows(0, 4), out_rows=_rows(0, 4))
    lg.commit(3)
    with pytest.raises(ReceiptError, match="rewrites folded rows"):
        lg.record(2, 2, kind="verify", codec="bf16", in_rows=_rows(0, 2), out_rows=_rows(0, 2))
    with pytest.raises(ReceiptError, match="gap"):
        lg.record(6, 1, kind="decode", codec="bf16", in_rows=_rows(0, 1), out_rows=_rows(0, 1))
    with pytest.raises(ReceiptError, match="never written"):
        lg.close(6, final=True)


def test_tail_needs_a_drawn_token_for_every_produced_slot():
    salt = bytes(32)
    lg = _ledger(role="tail", hi=12, salt=salt, prompt_len=2)
    lg.record(0, 3, kind="prefill", codec="bf16", in_rows=_rows(0, 3), ids=[5, 6, 7], toks=[None, None, None])
    with pytest.raises(ReceiptError, match="drew no token"):
        lg.commit(3)


# ── regressions from the adversarial review (each was accepted, crashed or falsely rejected) ──
def _forge_grid(b, lo, upto):
    """The dead hop of rte 0 signs further grid checkpoints: roots copied from its neighbours
    (leaves are untagged) where they exist, invented past them."""
    mine = sorted(hop_ckpts(b, lo), key=lambda c: (c["pos"], c["kind"] != "grid"))
    last = mine[-1]
    pred = {c["pos"]: c for c in b["checkpoints"] if c["rte"] == 0 and c["log"] == "serve" and c["hi"] == lo}
    succ = {c["pos"]: c for c in b["checkpoints"] if c["rte"] == 0 and c["log"] == "serve" and c["lo"] == last["hi"]}
    chains = {k: bytes.fromhex(v["chain"]) for k, v in last["logs"].items()}
    prev, out = R2.ckpt_id(last), []
    for pos in range(last["pos"] + R2.GRID, upto + 1, R2.GRID):
        k = pos // R2.GRID - 1
        logs = {}
        for name in last["logs"]:
            src = {"in": (pred, "out"), "out": (succ, "in")}.get(name)
            if src and pos in src[0]:
                root = src[0][pos]["logs"][src[1]]["roots"][0][1]
            else:
                root = hashlib.sha256(f"{name}/{k}".encode()).hexdigest()
            chains[name] = R2._chain(chains[name], k, bytes.fromhex(root))
            logs[name] = {"lo": 0, "hi": pos, "roots": [[k, root]], "chain": chains[name].hex()}
        body = {kk: vv for kk, vv in last.items() if kk not in ("sig", "pubkey")}
        body.update(kind="grid", pos=pos, prev_ckpt=prev, logs=logs,
                    seg={"lo": 0, "recs": [], "chain": last["seg"]["chain"]})
        ck = R2._sign(node_of(b, lo).priv, body)
        prev = R2.ckpt_id(ck)
        out.append(ck)
    return out


def test_dead_hop_cannot_sign_past_the_rte_end():
    b = gen("takeover")
    b["checkpoints"] += _forge_grid(b, 6, 640)
    rejects(b, "past the rte end")


def test_entry_cannot_zero_a_hop_by_dropping_all_its_checkpoints():
    for lo in (6, 9):                              # a mid hop, and the tail (closure skipped)
        b = gen("spec")
        victim = node_of(b, lo).pub
        b["checkpoints"] = [c for c in b["checkpoints"] if c["pubkey"] != victim]
        resign_path(b, lambda pl, v=victim: pl["rtes"][0]["dead"].append(v))
        rejects(b, "declared dead with no checkpoint")


def test_death_bound_uses_the_leased_window_not_the_path_log():
    b = gen("spec")
    victim = node_of(b, 6).pub
    keep = [c for c in hop_ckpts(b, 6) if c["pos"] <= 128]
    b["checkpoints"] = [c for c in b["checkpoints"] if c["pubkey"] != victim] + keep
    resign_path(b, lambda pl: (pl["rtes"][0]["dead"].append(victim), pl.update(W=10 ** 9)))
    rejects(b, "a live hop would have checkpointed")


def test_kv_import_range_is_pinned():
    b = gen("takeover", bugs_backup=("lie_kv",))
    rb = [c for c in b["checkpoints"] if c["log"] == "rebuild"]
    upto = rb[0]["rebuild"]["upto"]
    bk = next(n for n in b["job"].paths[1] if n.lo == 6)

    def empty(ck):
        ck["logs"]["kvi"] = {"lo": upto, "hi": upto, "roots": [], "chain": R2._chain0(upto).hex()}
    replace(b, rb, relink(bk.priv, rb, empty))
    rejects(b, "sub-log kvi starts at")


@pytest.mark.parametrize("kw", [{"which": "mid"}, {"which": "tail"},
                                {"which": "tail", "cx": "fp8", "cx_prefill": "fp8"}])
def test_chained_recovery_after_a_double_death(kw):
    """Two adjacent hops die; the first replacement (other numerics; an SWA-only span in "mid")
    forwards its rebuilt rows, which differ from the committed ones. The chained successor is
    checked against those rows only, and the forwarder is owed the rows it had to rebuild."""
    b = gen("double", **kw)
    v = M.verify(b)
    F = b["job"].rtes[1]["pos_start"]
    assert sorted(w["positions"] for w in v["work"] if w["kind"] == "rebuild") == [F, F]


def test_integers_past_u32_are_rejected_not_crashed():
    b = gen("spec")
    old = hop_ckpts(b, 6)
    replace(b, old, relink(node_of(b, 6).priv, old[:1], lambda ck: ck.update(pos_lo=2 ** 32)))
    rejects(b, "2^32")
    b = gen("spec")
    old = hop_ckpts(b, 6)

    def big(ck):
        ck["logs"]["in"]["lo"] = 2 ** 40
    replace(b, old, relink(node_of(b, 6).priv, old[:1], big))
    with pytest.raises(ReceiptError):
        M.verify(b)


def test_duplicate_checkpoint_is_a_retransmit_not_a_fork():
    b = gen("spec")
    b["checkpoints"].append(copy.deepcopy(hop_ckpts(b, 3)[1]))
    assert M.verify(b)["ok"]
    g = [c for c in hop_ckpts(b, 9) if c["kind"] == "grid"][0]
    b["checkpoints"].append(R2.resign(node_of(b, 9).priv, dict(g, n_rows=g["n_rows"] + 1)))
    rejects(b, "forked sequence")


def test_backup_rebuild_credit_does_not_depend_on_replaces():
    b = gen("takeover", tier="log", replaces=False)
    v = M.verify(b)
    F = b["job"].rtes[1]["pos_start"]
    acked = [w for w in v["work"] if w["kind"] == "mirror"][0]["positions"]
    rb = [w for w in v["work"] if w["kind"] == "rebuild"][0]
    assert rb["positions"] == F - R2.rebuild_rule(M.KINDS[6:9], M.WINDOW, F, acked) < F


def test_seg_schedule_is_checked_past_an_early_death():
    """The head died early; a mid hop's shape lie after that point is still compared."""
    b = gen("takeover", victim=0)
    assert M.verify(b)["ok"]
    old = hop_ckpts(b, 6)
    new = [copy.deepcopy(c) for c in sorted(old, key=lambda c: (c["pos"], c["kind"] != "grid"))]
    for ck in new:
        recs = ck["seg"]["recs"]
        i = [i for i in range(1, len(recs) - 1)
             if recs[i][0] > 200 and recs[i + 1][0] <= recs[i - 1][0] + recs[i - 1][1]]
        if i:
            recs.pop(i[0])
            break
    rechain_seg(new)
    replace(b, old, relink(node_of(b, 6).priv, new))
    rejects(b, "SEG lie")


def test_tail_tok_next_is_validated():
    b = gen("spec")
    close = [c for c in hop_ckpts(b, 9) if c["kind"] == "close"]
    replace(b, close, [R2.resign(node_of(b, 9).priv, dict(close[0], tok_next=""))])
    rejects(b, "tok_next")


def test_rejoin_after_own_death_earns_no_rebuild_credit():
    b = gen("dead", rejoin=True)
    v = M.verify(b)
    tail = node_of(b, 9)
    assert [w["positions"] for w in v["work"] if w["kind"] == "rebuild"] == [0]
    assert any(w["pubkey"] == tail.pub and w["rte"] == 1 and w["decode"] > 0 for w in v["work"])


def test_tokens_no_live_tail_signed_are_reported():
    b = gen("dead")
    v = M.verify(b)
    e = b["job"].rtes[0]
    g = max(c["pos"] for c in hop_ckpts(b, 9))
    assert v["tokens_unattested"] == e["pos_end"] + 1 - max(g, b["job"].P) > 0
    assert gen("spec") and M.verify(gen("spec"))["tokens_unattested"] == 0


def test_lease_pins_numerics_and_the_orchestrator_pins_the_manifest():
    b = gen("spec")
    old = hop_ckpts(b, 6)
    replace(b, old, relink(node_of(b, 6).priv, old, lambda ck: ck.update(numerics_id="nid1-other")))
    rejects(b, "its lease says")
    b = gen("spec")
    with pytest.raises(ReceiptError, match="manifest"):
        R2.verify_coverage_v2(b["path_log"], b["checkpoints"], leases=b["leases"],
                              entry_pubkey=b["entry_pubkey"], nonce=b["nonce"], job_id=b["job_id"],
                              **dict(M.MODEL_INFO, manifest_cid="toy-other"))
