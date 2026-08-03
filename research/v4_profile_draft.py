#!/usr/bin/env python3
"""Where one DSpark DRAFT actually goes — the 9.16 ms that binds the V4 ring, split per component.

The tail's on_box is 14.64 ms/frame on the 07-31 ring and 9.16 ms of it is `advance_and_draft` —
63% of the bottleneck stage, and therefore the ring's frame ceiling. This is the drafter's version
of research/v4_profile_stage.py (branch perf/v4-profile): the same discipline — WRAP the vendored
reference's own callables, never rewrite them, attribute to a component stack, resolve nesting to
SELF time — pointed at one `DSparkTail.advance_and_draft(n=1)` instead of one `Stage.forward`.

WHAT A CPU BOX CAN AND CANNOT MEASURE, stated before any number. There is no GPU here, and the tail
is CPU-LAUNCH-BOUND (v4_dspark_fast measured the GPU idle through nearly all of a draft), so the
quantity that PRICES a component on the ring is not its CPU-toy wall — it is how many kernel
launches and host round-trips it forces the dispatch thread to make. Those are STRUCTURAL: they
depend on the config (5 rows x 6 experts x 3 mtp blocks x 256-expert MoE), not on the toy dims, so
this harness runs the toy at the SHIPPED STRUCTURE (256 experts, topk 6, block 5, 3 mtp stages) and
counts, per component:

    calls      invocations of the vendored callable (linear / act_quant / sparse_attn / ...)
    aten       aten ops dispatched OUTSIDE the fused-on-GPU callables (each ~1 launch)
    launches   the GPU-launch model: aten + a fixed per-callable cost (a quantized `linear` is
               act_quant + gemm = 2; `hc_split_sinkhorn`/`sparse_attn`/`act_quant` are 1 fused
               tilelang kernel each on the GPU, whatever the CPU stand-in loops look like)
    syncs      host round-trips (`bincount().tolist()`, `torch.where(eq)`'s nonzero) — each one
               DRAINS the launch queue on the ring; v4_moe_multi's header prices one at ~0.15-0.25 ms
    cpu_ms     toy wall, for completeness; toy dims make it a shape check, not a price

and then projects the split onto the measured 9.16 ms by solving the launch/sync model against the
live anchor. The model, and every weight in it, is printed with the table.

Run:      python3 research/v4_profile_draft.py            # reference dispatch + V4_MOE_MULTI legs
          python3 research/v4_profile_draft.py --fp4      # + fp4 drafter MoEs, V4_DSPARK_MOE A/B
"""
import argparse
import collections
import os
import sys
import time

import torch
from torch.utils._python_dispatch import TorchDispatchMode

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "phase0"))

# aten ops that are metadata only — no kernel behind them on a GPU.
_NO_KERNEL = {
    "view", "_unsafe_view", "reshape", "as_strided", "slice", "select", "unsqueeze", "squeeze",
    "expand", "permute", "transpose", "t", "detach", "alias", "split", "chunk", "narrow", "empty",
    "empty_like", "empty_strided", "unfold", "lift_fresh", "_local_scalar_dense",
}
# aten ops that are a host round-trip on a GPU (the queue drains before Python continues).
_SYNC = {"nonzero", "_local_scalar_dense", "equal"}


class _Counter(TorchDispatchMode):
    """Attribute every dispatched aten op to the top of the component stack."""

    def __init__(self, prof):
        super().__init__()
        self.prof = prof

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        p = self.prof
        name = func.overloadpacket.__name__
        c = p.counts[p.top()]
        c["aten_all"] += 1
        if p.fused == 0 and name not in _NO_KERNEL:
            c["aten"] += 1
        if p.fused == 0 and name in _SYNC:      # a fused GPU kernel has no host sync inside it
            c["syncs"] += 1
        return func(*args, **(kwargs or {}))


class DraftProfile:
    """One component stack + per-component counters, shared by all the wrappers."""

    def __init__(self):
        self.stack = ["other"]
        self.fused = 0                                  # >0: inside a callable that is 1 GPU kernel
        self.counts = collections.defaultdict(lambda: collections.defaultdict(float))

    def top(self):
        return self.stack[-1]

    class _Scope:
        def __init__(self, prof, name, fused_launches=0, calls_key=None):
            self.p, self.name, self.fl, self.ck = prof, name, fused_launches, calls_key

        def __enter__(self):
            p, c = self.p, self.p.counts[self.name]
            p.stack.append(self.name)
            if self.ck:
                c[self.ck] += 1
            if self.fl:
                c["fused_launches"] += self.fl
                p.fused += 1
            self.t0 = time.perf_counter()

        def __exit__(self, *exc):
            self.p.counts[self.name]["cpu_ms"] += (time.perf_counter() - self.t0) * 1e3
            if self.fl:
                self.p.fused -= 1
            self.p.stack.pop()

    def scope(self, name, fused_launches=0, calls_key=None):
        return self._Scope(self, name, fused_launches, calls_key)


def install_wrappers(mod, prof, shared_ids, pair_moes=()):
    """Wrap the vendored callables with component attribution. Returns an undo list.

    `mod` is the loaded dsv4_model. Wrapping the MODULE'S bindings (not kernel.py's) for the same
    reason v4_profile_stage does: `from kernel import ...` froze the names into model.py's globals,
    and those globals are what the forward reads. `shared_ids` are id()s of shared-expert modules,
    so Expert.forward can be split routed-vs-shared."""
    undo = []

    def patch(owner, attr, new):
        old = getattr(owner, attr)
        setattr(owner, attr, new)
        undo.append((owner, attr, old))
        return old

    # fused-on-GPU callables: 1 kernel each however many aten the CPU stand-in loops
    for name, comp_of in (("act_quant", lambda: None), ("sparse_attn", lambda: "attn"),
                          ("hc_split_sinkhorn", lambda: "hc")):
        def make(nm, co, ref=getattr(mod, name)):
            def wrapped(*a, **k):
                comp = co() or prof.top()
                with prof.scope(comp, fused_launches=1, calls_key=f"{nm}_calls"):
                    return ref(*a, **k)
            return wrapped
        patch(mod, name, make(name, comp_of))

    # linear: on the REAL model every quantized Linear is act_quant + gemm = 2 launches; the fp32
    # head/markov and the gate's F.linear are 1. The toy may be bf16 — the launch model prices the
    # SHIPPED dtypes, which is why the weight is decided by call-site component, printed below.
    ref_linear = getattr(mod, "linear")

    def linear_wrapped(x, weight, bias=None):
        q = weight.dtype in (torch.float4_e2m1fn_x2, torch.float8_e4m3fn)
        shipped_q = q or prof.top() not in ("head", "moe.gate")   # head/gate are fp32/F.linear
        with prof.scope(prof.top(), fused_launches=2 if shipped_q else 1, calls_key="linear_calls"):
            return ref_linear(x, weight, bias)
    patch(mod, "linear", linear_wrapped)

    def comp_wrap(cls, attr, comp, calls_key=None):
        ref = getattr(cls, attr)

        def wrapped(self, *a, **k):
            name = comp(self) if callable(comp) else comp
            with prof.scope(name, calls_key=calls_key):
                return ref(self, *a, **k)
        patch(cls, attr, wrapped)

    comp_wrap(mod.DSparkAttention, "forward", "attn", "calls")
    comp_wrap(mod.Block, "hc_pre", "hc")
    comp_wrap(mod.Block, "hc_post", "hc")
    comp_wrap(mod.DSparkBlock, "forward_embed", "embed", "calls")
    comp_wrap(mod.DSparkBlock, "forward_head", "head", "calls")
    comp_wrap(mod.MoE, "forward", "moe.dispatch", "calls")
    comp_wrap(mod.Gate, "forward", "moe.gate")
    comp_wrap(mod.Expert, "forward",
              lambda self: "moe.shared" if id(self) in shared_ids else "moe.experts")

    # .tolist() is the reference dispatch's host drain (bincount().tolist(), indices.tolist()) —
    # invisible to the dispatch mode (it happens on a CPU tensor here), so counted at the source.
    ref_tolist = torch.Tensor.tolist

    def tolist_wrapped(self):
        if prof.fused == 0:
            prof.counts[prof.top()]["syncs"] += 1
            prof.counts[prof.top()]["tolist_calls"] += 1
        return ref_tolist(self)
    patch(torch.Tensor, "tolist", tolist_wrapped)

    # The per-instance pair path (V4_DSPARK_MOE): the instance binding shadows the class wrapper,
    # so wrap each bound forward for attribution, and price the pair-GEMM block at its CUDA branch
    # launch count (2 row gathers + 2 bank gathers + 1 grouped GEMM = 5, then 2 gathers + 1 GEMM
    # = 3 for w2) — the CPU branch is a reference-shaped emulation whose own op count is not what
    # a GPU pays (v4_dspark_moe._pair_gemms_cpu's docstring).
    if pair_moes:
        import v4_dspark_moe as DM
        for moe in pair_moes:
            if "forward" not in moe.__dict__:
                continue
            ref_fwd = moe.forward

            def fwd_wrapped(x, ids, _ref=ref_fwd):
                with prof.scope("moe.dispatch", calls_key="pair_calls"):
                    return _ref(x, ids)
            patch(moe, "forward", fwd_wrapped)
        ref_pair = DM._pair_gemms_cpu

        def pair_cpu(xq, xs, flat, bank, sd, inter):
            with prof.scope("moe.experts", fused_launches=5):
                both, w2 = ref_pair(xq, xs, flat, bank, sd, inter)

            def w2_wrapped(hq, hs):
                with prof.scope("moe.experts", fused_launches=3):
                    return w2(hq, hs)
            return both, w2_wrapped
        patch(DM, "_pair_gemms_cpu", pair_cpu)
    return undo


def build_harness(args_overrides=None, fp4_drafter=False, seed=0):
    """The v4_dspark_draft toy harness at the SHIPPED STRUCTURE (see module docstring).

    `fp4_drafter=True` swaps the drafter blocks' MoEs for fp4 ones with REAL quantized weights
    (through the reference's own fp4_act_quant), so the op mix through `linear` is the shipped one
    (act_quant + fp4_gemm per expert matrix) and V4_DSPARK_MOE has a bank to claim."""
    import v4_ref_cpu
    import v4_stage
    import v4_dspark_draft as D

    over = dict(n_routed_experts=256, n_activated_experts=6, dspark_block_size=5, n_mtp_layers=3,
                compress_ratios=(0, 0, 4, 8, 4, 8, 4, 0, 0, 0, 0))
    over.update(args_overrides or {})
    args = v4_ref_cpu.cpu_args(**over)
    oracle = v4_ref_cpu.build_oracle(args, seed)
    st = v4_stage.Stage(0, args.n_layers, args, head=True, tail=True, dspark=True, device="cpu")
    for li in range(args.n_layers):
        st.layers[li].load_state_dict(oracle.layers[li].state_dict(), strict=True)
    st.embed_tokens.load_state_dict(oracle.embed.state_dict(), strict=True)
    st.norm.load_state_dict(oracle.norm.state_dict(), strict=True)
    st.lm_head.load_state_dict(oracle.head.state_dict(), strict=True)
    with torch.no_grad():
        for n in ("hc_head_fn", "hc_head_base", "hc_head_scale"):
            getattr(st, n).data.copy_(getattr(oracle, n).data)
    st._dspark = True
    dr = D.DSparkTail(st)
    for k, blk in enumerate(dr.mtp):
        sd = {n: v for n, v in oracle.mtp[k].state_dict().items() if n not in D.ALIAS_KEYS}
        blk.load_state_dict(sd, strict=False)
    if fp4_drafter:
        import v4_dspark_moe
        v4_dspark_moe.swap_in_fp4_moes(dr, moe_inter_dim=128, seed=seed)
    return args, st, dr


def drive_rounds(args, st, dr, rounds=8, prompt=13, seed=0):
    """Prefill + `rounds` single-position advances — the PIPELINED shape, the one that ships.
    Returns the (ids, main_hidden, start_pos) list so an A/B can replay identical rounds."""
    torch.manual_seed(seed)
    ids = torch.randint(0, args.vocab_size, (1, prompt))
    tok = st.logits_all(st.forward(st.embed(ids), ids, 0), full_logits=False).argmax(-1)
    dr.prefill(tok, st.tail_main_hidden())
    recorded = []
    for i in range(prompt, prompt + rounds):
        h = st.forward(st.embed(tok.unsqueeze(1)), tok.unsqueeze(1), i)
        tok = st.logits_all(h, full_logits=False).argmax(-1)
        recorded.append((tok.unsqueeze(1).clone(), st.tail_main_hidden().clone(), i))
    return recorded


def profile_leg(tag, fp4=False, rounds=8, lever_env=()):
    """One measured leg: build, record rounds, profile `advance_and_draft` over them."""
    for k, v in lever_env:
        os.environ[k] = v
    # a fresh import universe per leg so module-scope env reads and installs land per leg
    for m in [m for m in sys.modules if m.startswith("v4_") or m == "dsv4_model"]:
        del sys.modules[m]
    import v4_ref_cpu
    mod = v4_ref_cpu.load_ref()
    args, st, dr = build_harness(fp4_drafter=fp4)
    if any(k == "V4_DSPARK_MOE" for k, _ in lever_env):
        import v4_dspark_moe
        took = v4_dspark_moe.install_drafter(dr)
        print(f"[leg {tag}] V4_DSPARK_MOE install_drafter -> {took}/{len(dr.mtp)}")
    recorded = drive_rounds(args, st, dr, rounds=rounds)

    shared_ids = set()
    for blk in dr.mtp:
        shared_ids.add(id(blk.ffn.shared_experts))
    prof = DraftProfile()
    undo = install_wrappers(mod, prof, shared_ids,
                            pair_moes=[blk.ffn for blk in dr.mtp])
    triples = []
    try:
        with _Counter(prof):
            for ids_seq, main, pos in recorded:
                t0 = time.perf_counter()
                blk, conf = dr.advance_and_draft(ids_seq, main, start_pos=pos)
                prof.counts["TOTAL"]["cpu_ms"] += (time.perf_counter() - t0) * 1e3
                triples.append(tuple(t.clone() for t in dr.last_spec))
    finally:
        for owner, attr, old in reversed(undo):
            setattr(owner, attr, old)
    for k, v in lever_env:
        os.environ.pop(k, None)
    return prof, rounds, triples


ORDER = ("embed", "attn", "hc", "moe.gate", "moe.dispatch", "moe.experts", "moe.shared",
         "head", "other")


def report(tag, prof, rounds, anchor_ms=None):
    print(f"\n── {tag}: one advance_and_draft (pipelined n=1), per-round means over {rounds} rounds")
    print(f"{'component':<14}{'calls':>7}{'linear':>8}{'aten':>8}{'fused':>7}"
          f"{'launches':>10}{'syncs':>7}{'cpu_ms':>9}")
    tot = collections.defaultdict(float)
    rowvals = {}
    for name in ORDER:
        c = prof.counts.get(name)
        if not c:
            continue
        launches = (c.get("aten", 0) + c.get("fused_launches", 0)) / rounds
        row = (c.get("calls", 0) / rounds, c.get("linear_calls", 0) / rounds,
               c.get("aten", 0) / rounds, c.get("fused_launches", 0) / rounds,
               launches, c.get("syncs", 0) / rounds, c.get("cpu_ms", 0) / rounds)
        rowvals[name] = row
        for k, v in zip(("calls", "linear", "aten", "fused", "launches", "syncs", "cpu_ms"), row):
            tot[k] += v
        print(f"{name:<14}{row[0]:>7.1f}{row[1]:>8.1f}{row[2]:>8.1f}{row[3]:>7.1f}"
              f"{row[4]:>10.1f}{row[5]:>7.1f}{row[6]:>9.2f}")
    print(f"{'TOTAL':<14}{tot['calls']:>7.1f}{tot['linear']:>8.1f}{tot['aten']:>8.1f}"
          f"{tot['fused']:>7.1f}{tot['launches']:>10.1f}{tot['syncs']:>7.1f}{tot['cpu_ms']:>9.2f}")
    if anchor_ms:
        # Solve the model against the live anchor: anchor = L*launch_us + S*drain_us, with the
        # drain priced from v4_moe_multi's header (0.2 ms mid) and the launch cost as the residual.
        drain_us = 200.0
        launch_us = max((anchor_ms * 1e3 - tot["syncs"] * drain_us) / max(tot["launches"], 1), 0.0)
        print(f"\n   anchored to the live {anchor_ms:.2f} ms draft: drain={drain_us:.0f} us (prior) "
              f"-> launch={launch_us:.1f} us/launch")
        print(f"   {'component':<14}{'est_ms':>8}   (launches x {launch_us:.1f} us + syncs x {drain_us:.0f} us)")
        for name, row in rowvals.items():
            est = (row[4] * launch_us + row[5] * drain_us) / 1e3
            print(f"   {name:<14}{est:>8.2f}")
        return launch_us
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--anchor-ms", type=float, default=9.16,
                    help="the live per-frame draft cost this splits (tail, 07-31 ring)")
    ap.add_argument("--fp4", action="store_true",
                    help="also run the fp4-drafter legs (reference vs V4_DSPARK_MOE)")
    args = ap.parse_args()

    prof, r, t_ref = profile_leg("reference dispatch", rounds=args.rounds)
    report("REFERENCE (what the vendored loop pays)", prof, r, anchor_ms=args.anchor_ms)

    prof, r, t_multi = profile_leg("V4_MOE_MULTI", rounds=args.rounds,
                                   lever_env=(("V4_MOE_MULTI", "1"),))
    # The V4_MOE_MULTI leg is the 07-31 ring's recipe, so ITS anchored launch cost is the one the
    # projection below prices the post-lever draft at.
    lus = report("V4_MOE_MULTI=1 (drains collapsed, expert GEMMs untouched)", prof, r,
                 anchor_ms=args.anchor_ms)
    for a, b in zip(t_ref, t_multi):
        assert all(torch.equal(x, y) for x, y in zip(a, b)), "multi leg diverged from reference"

    if args.fp4:
        prof, r, t0 = profile_leg("fp4 reference", fp4=True, rounds=args.rounds)
        report("FP4 DRAFTER, reference dispatch", prof, r, anchor_ms=args.anchor_ms)
        prof, r, t1 = profile_leg("fp4 dspark-moe", fp4=True, rounds=args.rounds,
                                  lever_env=(("V4_DSPARK_MOE", "1"),))
        report("FP4 DRAFTER, V4_DSPARK_MOE=1 (one grouped launch per matrix kind)", prof, r)
        for i, (a, b) in enumerate(zip(t0, t1)):
            for x, y, what in zip(a, b, ("output_ids", "logits", "confidence")):
                assert torch.equal(x, y), f"V4_DSPARK_MOE changed the drafter's {what} at round {i}"
        print("\n[A/B] V4_DSPARK_MOE: output_ids, logits, confidence bit-identical across "
              f"{len(t0)} rounds")
        if lus is not None:
            tot_l = sum(prof.counts[n].get("aten", 0) + prof.counts[n].get("fused_launches", 0)
                        for n in ORDER) / r
            tot_s = sum(prof.counts[n].get("syncs", 0) for n in ORDER) / r
            print(f"[projection] post-lever draft at the anchored launch cost: "
                  f"{(tot_l * lus + tot_s * 200.0) / 1e3:.2f} ms  "
                  f"({tot_l:.0f} launches, {tot_s:.0f} syncs)")
    print("\ndone")


if __name__ == "__main__":
    main()
