"""ONE REGISTRY OF V4 LEVERS, and the rule that "requested" and "observed" can never disagree quietly.

WHY THIS FILE EXISTS. Six times on the V4 engine a lever has read as ON and not been on, and every
one of them cost more than the lever was worth:

  1. the fp8 wire was exported on the launcher and never reached a stage (ENG_ENV was born);
  2. `moe=grouped/N` counted layers BANKED at load, not layers that ever grouped at run time;
  3. the grouped kernel then really did fire on only 2 of 6 layers, and `grouped/6` said otherwise;
  4. the DSpark drafter's MoE block shape fell through every s==1 lever onto the vendored loop;
  5. `V4_LAZY_DRAFT` was set on the stages for hours — it is read by the COORDINATOR;
  6. and this one: with `V4_MOE_MULTI=1` the stage repr printed `moe=ref` on a ring where grouped
     WAS installed, because the status function classified only the two markers it knew and fell
     through to the literal string "ref" for the third.

Every one of those is the same bug wearing a different hat, and the expensive part is never the wrong
answer — there usually isn't one. It is the MEANINGLESS MEASUREMENT: a ring that ran the baseline
while the bench reported the lever, or ran the lever while the operator chased a ghost. Both directions
have now happened.

THE RULE THIS FILE ENFORCES. For every lever there are two facts and they must be compared out loud:

  requested   what THIS PROCESS resolved the flag to — the owning module's own parsed value, not
              os.environ, because the module is what the run actually obeys. (A default-ON lever like
              V4_MOE_DECODE is "requested" with no env at all; a typo'd env is "requested=off".)
  observed    what is LIVE in the process right now — which function is bound to `MoE.forward`,
              whether the stage holds captured graphs, what the frame builder actually packs. Never
              the config that asked for it.

`audit()` computes both for every registered lever, `report()` prints them, and any disagreement is a
finding. `V4_LEVERS_STRICT=1` turns a finding into a refusal to serve. The stage repr carries
`levers=` so the same fact is on the line an operator already reads.

THE VERDICTS, all of which have actually happened:
  MISMATCH        requested and observed disagree (bug 1, 3, 4, 6), OR the environment says on and
                  the owning module parsed it off (an env set after import — every in-process ring
                  and bench does that, and those are what produce numbers).
  UNKNOWN         a V4_* var is set that no module in the engine reads at all — a typo, or a lever
                  that was documented and never implemented (V4_TOPK_STABLE is exactly this today).
  OTHER SIDE      the var is set here and this lever is read by the other side (bug 5). A NOTICE and
                  never an alarm: ENG_ENV ships the coordinator levers to every stage on purpose, so
                  a stage carrying V4_LAZY_DRAFT is the normal state of a healthy ring, and a process
                  cannot see whether the other side has it too. The line that ends bug 5 is on the
                  side that CAN judge it -- a coordinator reading `V4_LAZY_DRAFT requested=off`.
  UNJUDGED        the lever cannot be judged from here (a stage-bound lever with no stage, a MoE
                  lever in a process that never loaded the reference). Reported, never hidden.
  VALUE           a knob: the resolved value, reported so a wrong one is visible, never called OK.

ADDING A LEVER. Register it here. `tests/test_v4_levers.py` re-derives the lever set from the SOURCE
of every module the engine imports, so a new `os.environ.get("V4_...")` anywhere in phase0 fails the
suite until it appears in LEVERS or in NON_LEVER_ENV with a reason. That is deliberate: the registry
cannot silently fall behind the code, which is how five of the six bugs above survived a green suite.

self-test:  python3 phase0/v4_levers.py
"""
import os
import re
import sys

# ── the side table: which PROCESS must carry which var ────────────────────────────────────────────
# STAGE        read by a module in the stage's import closure and acted on inside the stage process.
#              Set it on the stages. `v4_pipe.stage_launch_cmd` propagates ENG_ENV from the launcher.
# COORDINATOR  read by v4_pipe in the process that DRIVES the ring (`v4_pipe.py coord`). Setting it
#              on the stages does nothing at all — bug 5, which cost a night.
# BOTH         genuinely read on both sides, for different jobs.
STAGE = "stage"
COORDINATOR = "coordinator"
BOTH = "both"

# Loud by default, fatal on request. Default LOUD rather than fatal because a lever can decline for a
# legitimate, box-shaped reason (the grouped fp4 kernel is CUDA-only and the CPU parity suite arms it
# anyway), and bricking a warm ring over that would be worse than the bug. A ring that is about to
# produce a NUMBER should set this: an unaudited measurement is the thing being prevented.
V4_LEVERS_STRICT = os.environ.get("V4_LEVERS_STRICT", "0") not in ("", "0")


class Ctx:
    """What a lever's check may look at. Built by `audit`, never by a lever."""

    def __init__(self, side, stage=None):
        self.side = side
        self.stage = stage
        # The reference module the ENGINE IS BOUND TO, if this process has one, and nothing else.
        # THREE WAYS TO GET THIS WRONG, all of them real:
        #   * `deepseek_v4_ref.inference.model` by package path is a SECOND, unpatched class object
        #     (load_ref execs model.py as `dsv4_model`) — a probe that did exactly that is how bug 6
        #     was "confirmed";
        #   * `sys.modules["dsv4_model"]` is only a NAME, and v4_moe_grouped._load_model_module()
        #     re-execs model.py and rebinds it while v4_ref_cpu._REF keeps the original;
        #   * calling `load_ref()` here would cause the very load the audit is meant to observe, and
        #     a coordinator legitimately never loads it at all.
        # So: read `v4_ref_cpu._REF` — the object `v4_stage.ref()` hands the layers — and never import.
        rc = sys.modules.get("v4_ref_cpu")
        self.mod = getattr(rc, "_REF", None) if rc is not None else None
        if self.mod is None:
            self.mod = sys.modules.get("dsv4_model")


def _mod(name):
    """The named phase0 module IF this process imported it, else None. Never imports."""
    return sys.modules.get(name)


def _flag(modname, attr, on="on", off="off"):
    """The owning module's own parsed flag — the value the run obeys. 'absent' if never imported."""
    m = _mod(modname)
    if m is None:
        return "absent"
    return on if getattr(m, attr, False) else off


# ── the MoE forward chain: the observation bug 6 was hiding in ────────────────────────────────────
# Three levers rebind `MoE.forward`, each capturing the previous binding as its fallback, so the live
# state is a CHAIN and not a single value: multi -> grouped -> decode -> ref. `Stage._moe_status` used
# to classify only the top function by the two markers it knew, so `multi` on top read as "ref" — the
# lever underneath it was installed and serving, and the repr said the reference was. Walk the whole
# chain instead: each installer keeps its own captured fallback in a module global, so the chain is
# reconstructible from LIVE objects with no bookkeeping added to the installers.
_MOE_LAYERS = (
    ("multi", "v4_moe_multi", "_v4_multi"),
    ("grouped", "v4_moe_grouped", "_v4_grouped"),
    ("decode", "v4_moe_decode", "_v4_decode_fast"),
)


def moe_chain(mod):
    """`MoE.forward`'s live fallback chain, top first, ending in 'ref'. [] if the class is missing.

    Terminates on a repeat as well as on the reference: a chain that cycles is a real (and once
    warned-about) install-order bug, and an audit that hangs on it would be worse than one that
    reports it. Duck-typed on `mod.MoE` so a test can pass a stand-in module."""
    moe = getattr(mod, "MoE", None)
    fn = getattr(moe, "forward", None)
    out, seen = [], set()
    while fn is not None:
        if id(fn) in seen:
            out.append("CYCLE")             # a chain that eats itself: RecursionError on first use
            break
        seen.add(id(fn))
        for label, modname, marker in _MOE_LAYERS:
            if getattr(fn, marker, False):
                out.append(label)
                owner = _mod(modname)
                if owner is None:
                    out.append("ORPHAN")    # marker present, installer gone: the walk cannot finish
                    fn = None
                else:
                    fn = getattr(owner, "_REF_FORWARD", None)
                break
        else:
            out.append("ref")
            fn = None
    return out


def _moe_check(label, modname, attr):
    """A `check` for one of the three MoE levers: is its link present in the LIVE chain?"""
    def check(ctx):
        req = _flag(modname, attr)
        if ctx.mod is None:
            return req, "unloaded", None
        chain = moe_chain(ctx.mod)
        broken = [w for w in ("CYCLE", "ORPHAN") if w in chain]
        if broken:
            return req, ">".join(chain), False        # nobody can be judged on a broken chain
        obs = "on" if label in chain else "off"
        if label == "grouped" and obs == "on" and ctx.stage is not None:
            # INSTANCE 3 OF THE BUG, closed at the only moment it is knowable before a token: the
            # kernel can be installed and bound while NOT ONE LAYER took the bank, which is the
            # `grouped/0` a ring ran on all night. Installed with an empty bank is not engaged.
            banked = getattr(ctx.stage, "_moe_banked", 0)
            return req, f"on/{banked}", (_agree(req, obs) and banked > 0)
        return req, obs, _agree(req, obs)
    return check


def _agree(req, obs):
    """Compare a switch's requested and observed state. 'absent' (the owning module was never
    imported in this process) can only agree with 'off' -- there is no flag to obey, so anything
    bound would be a genuine surprise, and a lever nothing imported is not a mismatch."""
    if req == "absent":
        return None if obs == "off" else False
    return req == obs


# ── per-lever checks ──────────────────────────────────────────────────────────────────────────────

def _check_cuda_graph(ctx):
    """Requested = the resolved MODE; observed = whether the stage really holds captured graphs.

    `V4_CUDA_GRAPH` is a mode, not a bool, and the stage refuses it loudly on a box that cannot
    capture (see `Stage._graph_refusal`), so `whole` requested against `off` observed on a CPU box is
    a true and expected finding rather than a false alarm — it is reported, and it is why the default
    is loud-not-fatal.

    ARMED, NOT CAPTURED, and the difference is real: capture is LAZY (`_BlockGraphs.run` builds on
    the first decode step, `WholeBlockGraphs.__init__` only allocates buffers), so this cannot be
    judged before a token and does not claim to. `V4_GRAPH_MAX=0`, an exhausted budget or a per-layer
    capture failure would still read as armed here — v4_stage counts those in `_GRAPH_SKIPPED`, which
    is appended when it is non-zero so the number is at least in front of the operator."""
    st = _mod("v4_stage")
    req = st._graph_mode() if st is not None else "absent"
    if ctx.stage is None:
        return req, "no-stage", None
    obs = ctx.stage._graph_mode if getattr(ctx.stage, "_block_graphs", None) is not None else "off"
    st = _mod("v4_stage")
    skipped = getattr(st, "_GRAPH_SKIPPED", 0) if st is not None else 0
    return req, (f"{obs}(-{skipped})" if skipped else obs), _agree(req, obs)


def _check_moe_in_graph(ctx):
    """Requested = the module's parsed flag; observed = how many layers CAPTURED their routed MoE.

    Rides on V4_CUDA_GRAPH=whole and is refused per layer (v4_whole_layer_graph._moe_refusal), so
    every honest outcome has to be distinguishable and none of them may read as OK by default:
      no-stage / off        no stage here, or this stage never built whole-layer graphs at all.
      armed                 the layers asked and none has been judged — capture is LAZY, so this is
                            the state before the first decode token and it is UNJUDGED, not OK.
      on/N-of-M             N layers captured it, M asked. Anything less than M is reported and, at
                            N == 0, is the finding: the lever is on and not one layer took it, which
                            on a default chain (V4_MOE_GROUPED unset) is exactly what happens.
    The refusals PRINT their reason as they happen; this is the one-line version an operator reads."""
    wl = _mod("v4_whole_layer_graph")
    req = _flag("v4_whole_layer_graph", "V4_MOE_IN_GRAPH")
    if ctx.stage is None or wl is None:
        return req, "no-stage" if wl is not None else "unloaded", None
    if getattr(ctx.stage, "_block_graphs", None) is None:
        return req, "off", _agree(req, "off")
    got, refused, undecided = wl.moe_graph_coverage(ctx.stage)
    asked = got + refused + undecided
    if not asked:
        return req, "off", _agree(req, "off")
    if undecided == asked:
        return req, f"armed/{asked}", None
    return req, f"on/{got}-of-{asked}", (_agree(req, "on") and got > 0)


def _check_fp8_gemv(ctx):
    """The occupancy-tiled fp8 GEMV path, whose observation is its own run-time proof.

    v4_fp8_gemv gates every tile on a per-(N,K) probe — tuned kernel torch.equal the vendored
    fp8_gemm, or the shape declines — so `gemv_status()` IS the live state: 'armed' before the
    first claimed GEMM (probe is lazy; unjudged, never OK), 'on/k-of-n' once serving,
    'declined/n' when every probe said no, 'invalid(...)' when the string never parsed. The
    module-global `fp8_gemm` binding is checked too: requested with nothing rebound (a CPU box,
    or an install that never ran) is a true finding, like V4_CUDA_GRAPH on a box that cannot
    capture — reported, and why the default is loud-not-fatal."""
    m = _mod("v4_fp8_gemv")
    if m is None:
        return "absent", "unloaded", None
    raw = getattr(m, "V4_FP8_GEMV", "")
    obs = m.gemv_status()
    if not raw or raw == "0":
        return "off", obs, _agree("off", obs)
    if getattr(m, "_GEMV_MODE", None) is None:
        return raw, obs, False                             # set but never parsed: a dead knob
    if ctx.mod is not None and not getattr(ctx.mod.fp8_gemm, "_v4_fp8_gemv", False):
        return "on", "not-installed", False
    if obs == "armed":
        return "on", obs, None
    return "on", obs, obs.startswith("on/")


def _check_fp8_shared(ctx):
    """The fused shared-expert lever: the Expert.forward rebind, the per-stage bank count, and the
    fused-launch verdicts, told apart the way _moe_check tells `grouped` apart — installed with an
    empty bank is not engaged (instance 3 of the recurring bug, in this lever's costume)."""
    m = _mod("v4_fp8_gemv")
    req = _flag("v4_fp8_gemv", "V4_FP8_SHARED")
    if m is None or ctx.mod is None:
        return req, "unloaded", None
    if not getattr(ctx.mod.Expert.forward, "_v4_fp8_shared", False):
        return req, "off", _agree(req, "off")
    obs = m.shared_status()
    if ctx.stage is not None:
        banked = getattr(ctx.stage, "_shared_banked", 0)
        return req, f"{obs}/banked-{banked}", (None if obs == "armed"
                                               else (_agree(req, "on") and banked > 0
                                                     and obs.startswith("on/")))
    if obs == "armed":
        return req, obs, None
    return req, obs, (_agree(req, "on") and obs.startswith("on/"))


def _check_fast_verify(ctx):
    st = _mod("v4_stage")
    req = "on" if (st is not None and st.V4_FAST_VERIFY) else ("absent" if st is None else "off")
    if ctx.stage is None:
        return req, "no-stage", None
    obs = "on" if getattr(ctx.stage, "_fast", False) else "off"
    return req, obs, _agree(req, obs)


def _check_spec_depth(ctx):
    """Read on BOTH sides for two different jobs — the coordinator's in-flight window and the stage's
    rollback ring — and they have to be the same number or a rejection rewinds past what a stage kept."""
    if ctx.side == COORDINATOR:
        vp = _mod("v4_pipe")
        req = str(vp.V4_SPEC_DEPTH) if vp is not None else "absent"
        fact = _NOTES.get("V4_SPEC_DEPTH")         # the W a real round streamed at
        if fact is None:
            return req, "no-job-yet", None
        return req, fact, _agree(req, fact)
    req = os.environ.get("V4_SPEC_DEPTH", "16")
    if ctx.stage is None:
        return req, "no-stage", None
    obs = str(getattr(ctx.stage, "_spec_depth", ""))
    return req, obs, req == obs


def _check_fp8_wire(ctx):
    """Observed by BUILDING A FRAME, not by reading the flag back.

    The fp8 wire is the worst case for a flag-shaped check: a ring where only some processes pack is
    merely inefficient, never wrong (`_recv_hids` dispatches on the frame), so nothing downstream can
    tell you it did not happen. Running the real builder on a 1-element tensor costs microseconds and
    answers the question the flag cannot — does a frame leaving THIS process carry `h8`.

    WHAT IT CANNOT CATCH, stated so nobody over-reads it: the builder branches on the same module
    global this reads, so a flag that never REACHED the process reads off on both sides and passes.
    No in-process check can catch that — the process has no idea what the operator intended — which
    is exactly why ENG_ENV and test_v4_pipe's propagation gate exist. What this catches is a builder
    that stopped honouring the flag, and the env/module gate in `audit` catches a flag set too late."""
    vp = _mod("v4_pipe")
    req = _flag("v4_pipe", "V4_FP8_WIRE")
    if vp is None:
        return req, "unloaded", None
    try:
        import torch
        frame, _ = vp._make_step_frame(torch.zeros(1, 1, 4, 8), [[0]], 0, None)
        obs = "on" if "h8" in frame else "off"
    except Exception as e:                          # noqa: BLE001 — an audit never breaks the serve
        return req, f"unprobed({type(e).__name__})", None
    return req, obs, _agree(req, obs)


def _check_dspark_fast(ctx):
    """The drafter lever, and the one that is TAIL-ONLY: only the tail builds a DSparkTail, so a
    non-tail stage legitimately has nothing rebound. Judged only where the class exists."""
    req = _flag("v4_dspark_fast", "V4_DSPARK_FAST")
    fact = _NOTES.get("V4_DSPARK_FAST")               # recorded by build_tail, which is where it lands
    if fact is not None:
        return req, fact, _agree(req, fact)
    d = _mod("v4_dspark_draft")
    if d is None:
        return req, "unloaded", None
    obs = "on" if getattr(d.DSparkTail.advance_and_draft, "_v4_dspark_fast", False) else "off"
    if ctx.stage is not None and not getattr(ctx.stage, "tail", False):
        return req, obs, None                       # not the tail: nothing here drafts
    return req, obs, _agree(req, obs)


def _check_dspark_moe(ctx):
    """The drafter's grouped-MoE lever — tail-only, like V4_DSPARK_FAST, and judged the same way:
    from the fact `ring_drafter` records at build (took == every drafter MoE banked + bound). The
    drafter is built lazily at the first dspark job, so before one there is nothing to observe and
    this says so rather than guessing; the class chain is untouched by design (the bind is per
    INSTANCE), so no module-level rebind exists to inspect."""
    req = _flag("v4_dspark_moe", "V4_DSPARK_MOE")
    fact = _NOTES.get("V4_DSPARK_MOE")
    if fact is not None:
        return req, fact, _agree(req, fact)
    if ctx.stage is not None and not getattr(ctx.stage, "tail", False):
        return req, "not-tail", None
    return req, "no-drafter-yet", None


def _check_dspark_block(ctx):
    """The drafter's inference-time WIDTH — tail-only and judged like the two levers above, from the
    fact `ring_drafter` records off the live DSparkTail at build. Requested is the owning module's
    parsed override ('off' = the trained width); observed is the width the drafter actually carries,
    so a run that drafted at the wrong width is a MISMATCH and not a slow mystery. Before the first
    dspark job there is no drafter and nothing honest to claim."""
    d = _mod("v4_dspark_draft")
    if d is None:
        req = "absent"
    else:
        v = getattr(d, "V4_DSPARK_BLOCK", 0)
        req = str(v) if v else "off"
    fact = _NOTES.get("V4_DSPARK_BLOCK")
    if fact is not None:
        return req, fact, (None if fact == "off" else False) if req == "absent" else req == fact
    if ctx.stage is not None and not getattr(ctx.stage, "tail", False):
        return req, "not-tail", None
    return req, "no-drafter-yet", None


def _check_draft_top2(ctx):
    """The tree-gate measurement — tail-only, judged like the drafter levers above, from the fact
    `ring_drafter` records off the live tail at build. The one wrinkle it reports honestly: an armed
    flag over a SAMPLING drafter observes as off (there is no runner-up when top-1 is not the
    draft), which is a mismatch worth seeing, not a silent no-op."""
    req = _flag("v4_dspark_draft", "V4_DRAFT_TOP2")
    fact = _NOTES.get("V4_DRAFT_TOP2")
    if fact is not None:
        return req, fact, _agree(req, fact)
    if ctx.stage is not None and not getattr(ctx.stage, "tail", False):
        return req, "not-tail", None
    return req, "no-drafter-yet", None


def _ref_slim_check(attr, marker_of):
    def check(ctx):
        req = _flag("v4_ref_slim", attr)
        if ctx.mod is None:
            return req, "unloaded", None
        obs = "on" if marker_of(ctx.mod) else "off"
        return req, obs, _agree(req, obs)
    return check


def _check_lazy_draft(ctx):
    """COORDINATOR-side, and verified from the RUN rather than the flag.

    `coordinate_dspark_pipelined` records the `lazy` it actually ran with (`note()` below), so once a
    job has gone through, the observation is a fact off the coordinator loop and not a re-read of the
    same env that was already believed. Before the first job there is nothing to observe and it says
    so. This is the lever whose var was set on six stages for hours; the WRONG PROCESS check below is
    what makes that visible in one line."""
    req = _flag("v4_pipe", "V4_LAZY_DRAFT")
    fact = _NOTES.get("V4_LAZY_DRAFT")
    if fact is None:
        return req, "no-job-yet", None
    return req, fact, _agree(req, fact)


def _check_pipelined(ctx):
    req = _flag("v4_pipe", "V4_PIPELINED_SPEC")
    fact = _NOTES.get("V4_PIPELINED_SPEC")
    if fact is None:
        return req, "no-job-yet", None
    # a JOB may ask for pipelining without the env, so `on` observed against `off` requested is legal
    return req, fact, not (req == "on" and fact == "off")


def _check_conf_gate(ctx):
    req = _flag("v4_pipe", "V4_DSPARK_CONF_GATE")
    fact = _NOTES.get("V4_DSPARK_CONF_GATE")
    if fact is None:
        return req, "no-job-yet", None
    return req, fact, _agree(req, fact)


def _check_refill_floor(ctx):
    """COORDINATOR-side, and verified from the RUN, like V4_LAZY_DRAFT: the loop notes the floor it
    actually refilled at, so the observation is a fact off a real round and not a re-read of the env.

    A job may pass `floor=` without the env — like `pipelined` itself — so requested and observed
    disagreeing is only a finding when the environment explicitly named a floor and the run then
    refilled at another one, which is bug 5's shape (the operator set the lever somewhere it does
    not act) wearing this lever's costume."""
    vp = _mod("v4_pipe")
    req = str(vp.V4_REFILL_FLOOR) if vp is not None else "absent"
    fact = _NOTES.get("V4_REFILL_FLOOR")
    if fact is None:
        return req, "no-job-yet", None
    set_here = os.environ.get("V4_REFILL_FLOOR", "") not in ("", "0")
    return req, fact, (req == fact) if set_here else True


def _value_check(modname, attr):
    """A KNOB, not a switch: prove the module resolved the value the operator set, not that something
    was rebound. Catches the other half of the propagation bug — a flag that arrives as the wrong
    string is the same meaningless measurement in a different costume."""
    def check(ctx):
        m = _mod(modname)
        if m is None:
            return "-", "unloaded", None
        return str(getattr(m, attr, "?")), str(getattr(m, attr, "?")), None
    return check


class Lever:
    """One lever: its env var, which process reads it, and how to see it LIVE."""

    def __init__(self, env, side, owner, check, doc, kind="switch"):
        self.env = env
        self.side = side
        self.owner = owner
        self.check = check
        self.doc = doc
        # "switch": something is installed and can be seen installed. "knob": a VALUE the engine had
        # to resolve, with no separate install -- reported as VALUE, never as a silent OK, so nobody
        # mistakes "we printed the number back" for "we verified an effect".
        self.kind = kind

    def wanted_here(self, side):
        return self.side in (side, BOTH)


LEVERS = (
    Lever("V4_MOE_GROUPED", STAGE, "v4_moe_grouped",
          _moe_check("grouped", "v4_moe_grouped", "V4_MOE_GROUPED"),
          "grouped fp4 MoE kernel for the s==1 score-routed decode step (CUDA only)"),
    Lever("V4_FP8_GEMV", STAGE, "v4_fp8_gemv", _check_fp8_gemv,
          "occupancy-tiled fp8 GEMM at decode shapes (M<=32), self-gated torch.equal per (N,K)"),
    Lever("V4_FP8_SHARED", STAGE, "v4_fp8_gemv", _check_fp8_shared,
          "the shared expert's w1+w3 as one banked fp8 launch (+ one act_quant), gated per half"),
    Lever("V4_MOE_DECODE", STAGE, "v4_moe_decode",
          _moe_check("decode", "v4_moe_decode", "V4_MOE_DECODE"),
          "sync-free MoE dispatch at s==1 (DEFAULT ON)"),
    Lever("V4_MOE_MULTI", STAGE, "v4_moe_multi",
          _moe_check("multi", "v4_moe_multi", "V4_MOE_MULTI"),
          "sync-free MoE dispatch at the DSpark drafter's small block shape"),
    Lever("V4_CUDA_GRAPH", STAGE, "v4_stage", _check_cuda_graph,
          "decode-step CUDA graphs: off / island / whole"),
    Lever("V4_MOE_IN_GRAPH", STAGE, "v4_whole_layer_graph", _check_moe_in_graph,
          "capture the routed MoE INSIDE the whole-layer graph (needs whole mode + grouped)"),
    Lever("V4_FAST_VERIFY", STAGE, "v4_stage", _check_fast_verify,
          "chunked verify path (one pass per layer over a speculation chunk)"),
    Lever("V4_REF_SLIM", STAGE, "v4_ref_slim",
          _ref_slim_check("V4_REF_SLIM", lambda m: getattr(m.Indexer.forward, "_v4_ref_slim", False)),
          "skip the Indexer's scoring while every compressed slot is selected"),
    Lever("V4_REF_SLIM_NOQAT", STAGE, "v4_ref_slim",
          _ref_slim_check("V4_REF_SLIM_NOQAT", lambda m: getattr(m.act_quant, "_v4_ref_slim", False)),
          "skip the inplace KV/Q QAT quant-simulation (APPROXIMATE — not in the ring recipe)"),
    Lever("V4_DSPARK_FAST", STAGE, "v4_dspark_fast", _check_dspark_fast,
          "tail only: cache-advance-only drafter forwards"),
    Lever("V4_DSPARK_MOE", STAGE, "v4_dspark_moe", _check_dspark_moe,
          "tail only: the drafter's block MoE as one grouped fp4 launch per matrix kind"),
    Lever("V4_DSPARK_BLOCK", STAGE, "v4_dspark_draft", _check_dspark_block,
          "tail only: draft the block at this width instead of the trained dspark_block_size — "
          "deeper proposals from the same tap, lifting the pipelined in-flight cap to width+1"),
    Lever("V4_DRAFT_TOP2", STAGE, "v4_dspark_draft", _check_draft_top2,
          "tail only: ship the drafter's runner-up token per block slot, so the coordinators can "
          "count the rescue rate that gates tree speculation (docs/V4_TREE_VERDICT.md)"),
    Lever("V4_FP8_WIRE", STAGE, "v4_pipe", _check_fp8_wire,
          "fp8-pack h on the forward leg (every non-tail stage packs its own output)"),
    Lever("V4_SPEC_DEPTH", BOTH, "v4_pipe", _check_spec_depth,
          "pipelined speculation depth: the coordinator's window AND the stage's rollback ring"),
    Lever("V4_PIPELINED_SPEC", COORDINATOR, "v4_pipe", _check_pipelined,
          "stream s=1 frames without waiting for their replies"),
    Lever("V4_LAZY_DRAFT", COORDINATOR, "v4_pipe", _check_lazy_draft,
          "hint the tail to skip drafting a block the round will not consume"),
    Lever("V4_DSPARK_CONF_GATE", COORDINATOR, "v4_pipe", _check_conf_gate,
          "serial DSpark only: trim the tail's offered block length by confidence"),
    Lever("V4_REFILL_FLOOR", COORDINATOR, "v4_pipe", _check_refill_floor,
          "pipelined refill floor: consume a reply's block at or below this in-flight level "
          "(1 = drain-only, the shipped round)"),
    # Knobs: a value the engine must have resolved, with no separate install to observe.
    Lever("V4_MOE_MULTI_MAX", STAGE, "v4_moe_multi", _value_check("v4_moe_multi", "V4_MOE_MULTI_MAX"),
          "widest block the multi-dispatch path claims", kind="knob"),
    Lever("V4_FAST_VERIFY_MAX", STAGE, "v4_stage", _value_check("v4_stage", "V4_FAST_VERIFY_MAX"),
          "chunk positions reserved per layer for the fast verify scratch", kind="knob"),
    Lever("V4_GRAPH_MAX", STAGE, "v4_stage", _value_check("v4_stage", "V4_GRAPH_MAX"),
          "process-wide captured-graph budget", kind="knob"),
    Lever("V4_KERNELS", STAGE, "v4_kernels_cpu", _value_check("v4_kernels_cpu", "V4_KERNELS"),
          "kernel backend selection (tilelang / cpu)", kind="knob"),
)

LEVERS_BY_ENV = {lv.env: lv for lv in LEVERS}

# Read by the engine but not levers: nothing installs, nothing can be observed, and a wrong value
# fails loudly on its own (a missing V4_DIR cannot be mistaken for a slow ring). Listed rather than
# pattern-matched so the registry test stays total — an unlisted new name fails the suite.
NON_LEVER_ENV = {
    "V4_DIR": "checkpoint directory (emitted by stage_launch_cmd)",
    "V4_DEV": "torch device (emitted by stage_launch_cmd)",
    "V4_DTYPE": "default construction dtype",
    "V4_MAX_SEQ": "stage build-out: kv cache length",
    "V4_MAX_BATCH": "stage build-out: batch width",
    "V4_KEEPWARM": "transport keep-warm",
    "V4_KEEPWARM_MS": "transport keep-warm period",
    "V4_DIAL_CONNECT_TIMEOUT": "inter-stage dial timeout",
    "V4_DIAL_RETRY_S": "inter-stage dial retry window",
    "V4_TIMING": "instrumentation",
    "V4_TIMING_EVERY": "instrumentation period",
    "V4_DSPARK_CONF_MIN": "conf-gate knob, consumed with V4_DSPARK_CONF_GATE",
    "V4_DSPARK_CONF_THRESH": "conf-gate knob, consumed with V4_DSPARK_CONF_GATE",
    "V4_DSPARK_GRAPH": "drafter head graph, rides on V4_DSPARK_FAST and is CUDA-only",
    "V4_LEVERS_STRICT": "this file: turn a finding into a refusal to serve",
}


# ── facts recorded by the code that actually runs a lever ─────────────────────────────────────────
# A coordinator lever has no rebound method to inspect: `lazy` is an ARGUMENT to a loop, so the only
# honest observation is what the loop ran with. The loop says so here, once, and the audit reads the
# fact instead of re-reading the env it already believed.
_NOTES = {}


def note(env, value):
    """Record what a run ACTUALLY used for `env`. Called by the coordinator loops; never by a check."""
    _NOTES[env] = ("on" if value else "off") if isinstance(value, bool) else str(value)


def notes():
    return dict(_NOTES)


# ── the audit ─────────────────────────────────────────────────────────────────────────────────────

class Finding:
    __slots__ = ("env", "side", "requested", "observed", "verdict", "why")

    def __init__(self, env, side, requested, observed, verdict, why=""):
        self.env, self.side = env, side
        self.requested, self.observed = requested, observed
        self.verdict, self.why = verdict, why

    @property
    def bad(self):
        # OTHER SIDE is deliberately NOT here: ENG_ENV propagates coordinator levers to every stage
        # by design, so counting that would fire on every healthy ring and, under STRICT, stop the
        # ring forming. It is printed and explained; it is not an alarm.
        return self.verdict in ("MISMATCH", "UNKNOWN")

    def __repr__(self):
        return f"<{self.env} req={self.requested} obs={self.observed} {self.verdict}>"


def _stray_env():
    """V4_* vars set in this process that no lever and no known knob claims. Typos, and levers that
    were documented and never built — `V4_TOPK_STABLE` is written up in docs/V4_FLASH_ENGINE.md as
    "the real acceptance fix" and is read by nothing in phase0 on any branch. Setting it configures
    a bench banner and nothing else, which is the purest form of this bug."""
    known = set(LEVERS_BY_ENV) | set(NON_LEVER_ENV)
    return sorted(k for k in os.environ if k.startswith("V4_") and k not in known)


def audit(side=STAGE, stage=None):
    """Every registered lever's requested vs observed state, plus the side, env and stray checks."""
    ctx = Ctx(side, stage)
    out = []
    for lv in LEVERS:
        set_here = os.environ.get(lv.env, "") not in ("", "0")
        if set_here and not lv.wanted_here(side):
            # NEITHER DIRECTION IS A PROBLEM, AND THAT IS DELIBERATE. ENG_ENV ships the coordinator
            # levers to every stage ON PURPOSE (v4_pipe:ENG_ENV, and the note above V4_LAZY_DRAFT
            # says so) because v4_pipe runs on both sides and an operator exporting a lever on the
            # launcher should get it wherever it is read. So a stage seeing V4_LAZY_DRAFT is the
            # NORMAL state of a healthy ring -- calling it a problem would fire on every launch and,
            # under STRICT, would stop the ring forming at all. It is a NOTICE: named, explained, and
            # not counted, because a process cannot see whether the other side has it too.
            # The actionable version is on the side that CAN judge it: a coordinator whose
            # V4_LAZY_DRAFT reads `requested=off` is the line that ends instance 5.
            where = COORDINATOR if lv.side == COORDINATOR else STAGE
            out.append(Finding(lv.env, lv.side, os.environ[lv.env], "n/a", "OTHER SIDE",
                               f"{lv.side}-side lever: THIS {side} process never reads it. It is "
                               f"carried here because ENG_ENV propagates it; it configures the "
                               f"{where}, and only the {where}'s own audit can judge it."))
            continue
        if not lv.wanted_here(side):
            continue
        try:
            req, obs, ok = lv.check(ctx)
        except Exception as e:                      # noqa: BLE001 — an audit never breaks the serve
            out.append(Finding(lv.env, lv.side, "?", "?", "UNJUDGED", f"check raised {type(e).__name__}: {e}"))
            continue
        # THE ENV/MODULE GATE, ahead of the observation. `requested` is the owning module's PARSED
        # value, which is what the run obeys -- but that means an env set AFTER the module was
        # imported reads as "off" on both sides and passes as OK. Every in-process ring, bench and
        # research harness sets env in the same interpreter, and those are what produce numbers. So
        # the two are compared directly: an environment that says on against a module that parsed
        # off is a MISMATCH, whatever the live state then shows.
        if set_here and req in ("off", "0", "False", "absent"):
            out.append(Finding(lv.env, lv.side, os.environ[lv.env], f"module parsed {req}", "MISMATCH",
                               f"{lv.env} is set in this process's environment but {lv.owner} parsed "
                               f"it {req} — it was almost certainly set after the module was imported"))
            continue
        verdict = ("OK" if ok else
                   ("VALUE" if lv.kind == "knob" else "UNJUDGED") if ok is None else "MISMATCH")
        out.append(Finding(lv.env, lv.side, req, obs, verdict,
                           "requested and live state disagree" if verdict == "MISMATCH" else ""))
    for name in _stray_env():
        out.append(Finding(name, "?", os.environ[name], "nothing", "UNKNOWN",
                           "set in this process but no v4 module reads this name"))
    return out


def summary(stage=None, side=STAGE):
    """The one token the stage repr carries. Never raises — a broken audit must not hide a stage."""
    try:
        bad = [f for f in audit(side, stage) if f.bad]
    except Exception as e:                          # noqa: BLE001
        return f"audit-failed({type(e).__name__})"
    if not bad:
        return "ok"
    return "!" + ",".join(f"{f.env}:{f.verdict.split()[0].lower()}" for f in bad)


def report(side=STAGE, stage=None, strict=None, out=None):
    """Print the audit and return it as a string. Raises under strict if anything is wrong.

    Written to STDERR by default, which is where both callers want it: a stage's stdout and stderr are
    both redirected into its per-port log, and the coordinator's stdout is the SHARD_JOB_* contract the
    node daemon parses and must not be polluted."""
    findings = audit(side, stage)
    bad = [f for f in findings if f.bad]
    w = max((len(f.env) for f in findings), default=10)
    lines = [f"{'=' * 26} V4 LEVER AUDIT ({side}) {'=' * 26}"]
    for f in findings:
        line = f"  {f.env:<{w}}  requested={f.requested:<10} observed={f.observed:<12} {f.verdict}"
        lines.append(line + (f" — {f.why}" if f.why else ""))
    if bad:
        lines.append(f"V4 LEVER AUDIT: {len(bad)} PROBLEM(S) — "
                     + ", ".join(f"{f.env}({f.verdict})" for f in bad))
        if not (V4_LEVERS_STRICT if strict is None else strict):
            lines.append("V4 LEVER AUDIT: continuing anyway (set V4_LEVERS_STRICT=1 to refuse to serve)")
    else:
        lines.append(f"V4 LEVER AUDIT: all {len(findings)} clean")
    lines.append("=" * 74)
    text = "\n".join(lines)
    print(text, file=(out if out is not None else sys.stderr), flush=True)
    if bad and (V4_LEVERS_STRICT if strict is None else strict):
        raise RuntimeError(
            "V4_LEVERS_STRICT: refusing to serve with " + ", ".join(f"{f.env}={f.verdict}" for f in bad)
            + ". Every number this process produced would be about a configuration nobody asked for.")
    return text


# ── the source-derived closure, shared with the tests ─────────────────────────────────────────────
# The registry is only trustworthy if it cannot fall behind the code, so the set of levers is
# re-derived from the SOURCE of every module the engine imports rather than maintained by hand
# anywhere. `tests/test_v4_levers.py` asserts the derived set is exactly LEVERS + NON_LEVER_ENV.
ENGINE_MODULES = (
    "v4_pipe.py", "v4_stage.py", "v4_levers.py", "v4_moe_grouped.py", "v4_moe_decode.py",
    "v4_moe_multi.py", "v4_fp8_gemv.py", "v4_dspark_fast.py", "v4_dspark_moe.py",
    "v4_dspark_draft.py", "v4_ref_slim.py", "v4_ref_cpu.py", "v4_whole_layer_graph.py",
    "v4_kernels_cpu.py", "v4_sparse_attn_sm120.py",
)

_ENV_RE = re.compile(r"""environ(?:\.get)?[.(\[]+["'](V4_[A-Z0-9_]+)["']""")


def env_names_in_source(root=None):
    """Every V4_* env name read anywhere in ENGINE_MODULES, scraped from the source on disk.

    Tolerant of a missing file so a partial deploy reports what it has rather than dying inside an
    audit — the caller (the test) asserts the files are there."""
    root = root or os.path.dirname(os.path.abspath(__file__))
    found = set()
    for name in ENGINE_MODULES:
        p = os.path.join(root, name)
        if os.path.exists(p):
            with open(p) as f:
                found |= set(_ENV_RE.findall(f.read()))
    return found


def _selftest():
    """No GPU, no reference: prove the chain walk and the four verdicts on stand-ins."""
    class _F:
        pass

    class _MoE:
        pass

    class _Mod:
        MoE = _MoE

    def ref_fwd(self, x, ids):
        return x

    _MoE.forward = ref_fwd
    assert moe_chain(_Mod) == ["ref"], moe_chain(_Mod)

    # a two-deep chain, built the way the installers build it
    import types
    dec = types.ModuleType("v4_moe_decode")
    dec._REF_FORWARD = ref_fwd
    dec.V4_MOE_DECODE = True

    def dec_fwd(self, x, ids):
        return x
    dec_fwd._v4_decode_fast = True
    sys.modules["v4_moe_decode"] = dec
    _MoE.forward = dec_fwd
    assert moe_chain(_Mod) == ["decode", "ref"], moe_chain(_Mod)

    mul = types.ModuleType("v4_moe_multi")
    mul._REF_FORWARD = dec_fwd
    mul.V4_MOE_MULTI = True

    def mul_fwd(self, x, ids):
        return x
    mul_fwd._v4_multi = True
    sys.modules["v4_moe_multi"] = mul
    _MoE.forward = mul_fwd
    chain = moe_chain(_Mod)
    assert chain == ["multi", "decode", "ref"], chain
    print(f"chain      {'>'.join(chain)}   (the shape that used to report 'ref')")

    # a cycle must terminate AND be named -- it is a RecursionError on the first prefill, not a
    # slow path, and a walk that merely stopped early would have reported it as three clean levers
    mul._REF_FORWARD = mul_fwd
    assert moe_chain(_Mod) == ["multi", "CYCLE"], moe_chain(_Mod)
    ctx_cyc = Ctx(STAGE)
    ctx_cyc.mod = _Mod
    assert _moe_check("multi", "v4_moe_multi", "V4_MOE_MULTI")(ctx_cyc)[2] is False
    mul._REF_FORWARD = dec_fwd
    # and a marker whose installer is gone cannot reach `ref`, so it must not claim to
    del sys.modules["v4_moe_decode"]
    assert moe_chain(_Mod) == ["multi", "decode", "ORPHAN"], moe_chain(_Mod)
    sys.modules["v4_moe_decode"] = dec

    ctx = Ctx(STAGE)
    ctx.mod = _Mod
    for label, modname, flag in (("multi", "v4_moe_multi", "V4_MOE_MULTI"),
                                 ("decode", "v4_moe_decode", "V4_MOE_DECODE")):
        req, obs, ok = _moe_check(label, modname, flag)(ctx)
        assert ok is True, (label, req, obs, ok)
    # and the finding this whole file exists for: flag on, link absent from the live chain
    dec.V4_MOE_DECODE = True
    _MoE.forward = ref_fwd
    req, obs, ok = _moe_check("decode", "v4_moe_decode", "V4_MOE_DECODE")(ctx)
    assert (req, obs, ok) == ("on", "off", False), (req, obs, ok)
    print(f"finding    V4_MOE_DECODE requested={req} observed={obs} -> MISMATCH")
    print(f"registry   {len(LEVERS)} levers, {len(NON_LEVER_ENV)} non-lever knobs")
    print(f"sides      stage={sum(l.side == STAGE for l in LEVERS)} "
          f"coordinator={sum(l.side == COORDINATOR for l in LEVERS)} "
          f"both={sum(l.side == BOTH for l in LEVERS)}")
    scraped = env_names_in_source()
    missing = scraped - set(LEVERS_BY_ENV) - set(NON_LEVER_ENV)
    assert not missing, f"unregistered levers in the source: {sorted(missing)}"
    print(f"source     {len(scraped)} V4_* names read by the engine, all registered")
    print("OK")


if __name__ == "__main__":
    _selftest()
