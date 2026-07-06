"""hf_runtime_test — $0 proof of the generic ModelRuntime adapter selection + contract (M2, recut).

  python3 phase0/hf_runtime_test.py

No torch, no GPU, no model. Proves:
  - make_runtime maps registry adapter ids -> the right ModelRuntime (generic-hf/hf -> HF;
    generic-vllm -> HF fallback until VllmRuntime lands; unknown -> fail closed)
  - GenericHFRuntime conforms to the upstream shard/node.py ModelRuntime contract (subclass,
    right method set, LayerRange role flags)
  - a FAKE ModelRuntime drives the head/forward/tail boundary law (embed head-only, logits
    tail-only) exactly as coordinate_pipe relies on

The real load_shard/forward (torch + a model) is the metered GPU spike (MODEL_RUNTIME.md
step 2) — not offline-provable. This pins the seam: shard's generic adapter is ONE impl
behind the in-house ModelRuntime firewall, not a second competing interface.
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "phase0"))
from shard.node import LayerRange, ModelRuntime   # noqa: E402
import hf_runtime as hr                            # noqa: E402

passed = failed = 0


def ok(cond, name, detail=""):
    global passed, failed
    if cond:
        passed += 1
        print(f"  OK {name}{(' ' + detail) if detail else ''}")
    else:
        failed += 1
        print(f"  FAIL {name} {detail}")


def raises(exc, fn, name):
    try:
        fn()
    except exc as e:
        ok(True, name, f"-> {type(e).__name__}")
        return
    except Exception as e:
        ok(False, name, f"raised {type(e).__name__}, expected {exc.__name__}")
        return
    ok(False, name, "did not raise")


# A torch-free fake that satisfies the SAME ModelRuntime contract, to exercise the boundary
# law without loading anything. Mirrors GenericHFRuntime's role semantics.
class FakeRuntime(ModelRuntime):
    def __init__(self, model, layer_range, **kw):
        super().__init__(model, layer_range, is_head=kw.get("is_head", False),
                         is_tail=kw.get("is_tail", False), device=kw.get("device", "cpu"))
        self.loaded = False
        self.kv_len = 0

    def load_shard(self):
        self.loaded = True

    def reset(self):
        self.kv_len = 0

    def heartbeat(self):
        return {"lo": self.layer_range.start, "hi": self.layer_range.end, "loaded": self.loaded}

    def embed(self, token_ids):
        if not self.is_head:
            raise NotImplementedError("embed is a head-only op")
        return {"h": list(token_ids), "from": "embed"}

    def forward(self, hidden_states, start_pos: int):
        self.kv_len = start_pos + 1
        return {"h": hidden_states, "ran": (self.layer_range.start, self.layer_range.end)}

    def logits(self, hidden_states):
        if not self.is_tail:
            raise NotImplementedError("logits is a tail-only op")
        return {"logits": True}


def main():
    # ── GenericHFRuntime IS a ModelRuntime (subclass of the upstream firewall, not a new iface)
    ok(issubclass(hr.GenericHFRuntime, ModelRuntime),
       "GenericHFRuntime subclasses shard.node.ModelRuntime")
    for m in ("load_shard", "reset", "heartbeat", "embed", "forward", "logits"):
        ok(hasattr(hr.GenericHFRuntime, m), f"implements {m}()")

    # construct one (no load_shard -> no torch touched) and check role/range plumbing
    g = hr.GenericHFRuntime("some/model", LayerRange(4, 8), is_head=False, is_tail=False,
                            nstages=3, stage=1, device="cpu")
    ok(g.layer_range.start == 4 and g.layer_range.end == 8, "layer_range carried")
    ok(g.stage == 1 and g.nstages == 3, "ring position carried")

    # ── make_runtime selection (the registry adapter -> runtime map) ──
    for ad in ("generic-hf", "hf"):
        rt = hr.make_runtime(ad, "m", LayerRange(0, 4), device="cpu")
        ok(isinstance(rt, hr.GenericHFRuntime), f"adapter {ad!r} -> GenericHFRuntime")
    # generic-vllm falls back to HF until VllmRuntime lands (still servable, native precision)
    rt = hr.make_runtime("generic-vllm", "m", LayerRange(0, 4), device="cpu")
    ok(isinstance(rt, hr.GenericHFRuntime), "generic-vllm -> HF fallback (no VllmRuntime yet)")
    raises(KeyError, lambda: hr.make_runtime("nonsense", "m", LayerRange(0, 4)),
           "unknown adapter fails closed")

    # ── boundary law via the fake (head embeds, tail logits, middle neither) ──
    head = FakeRuntime("m", LayerRange(0, 4), is_head=True, stage=0)
    mid = FakeRuntime("m", LayerRange(4, 8), stage=1)
    tail = FakeRuntime("m", LayerRange(8, 12), is_tail=True, stage=2)
    for rt in (head, mid, tail):
        rt.load_shard()
    ok(head.loaded and mid.loaded and tail.loaded, "all stages load_shard")

    h0 = head.embed([1, 2, 3])
    ok(h0["from"] == "embed", "head embed works")
    raises(NotImplementedError, lambda: mid.embed([1]), "embed on non-head rejected")

    out = mid.forward({"x": 1}, start_pos=10)
    ok(out["ran"] == (4, 8), "forward ran this block")
    ok(mid.kv_len == 11, "forward advanced kv to start_pos+1")
    mid.reset()
    ok(mid.kv_len == 0, "reset drops kv")

    lg = tail.logits({"x": 1})
    ok(lg["logits"] is True, "tail logits works")
    raises(NotImplementedError, lambda: mid.logits({"x": 1}), "logits on non-tail rejected")

    # coverage tiles [0:12]
    spans = sorted((r.layer_range.start, r.layer_range.end) for r in (head, mid, tail))
    cur, tiled = 0, True
    for lo, hi in spans:
        tiled = tiled and lo == cur
        cur = hi
    ok(tiled and cur == 12, "blocks tile [0:12]", f"({spans})")

    print(f"\n{'ALL' if failed == 0 else 'SOME FAILURES:'} {passed} PASS"
          + (f", {failed} FAIL" if failed else ""))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
