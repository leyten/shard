"""The sync-free MoE decode path is BIT-EXACT, or it is a silent accuracy regression.

v4_moe_decode replaces the reference's expert-dispatch loop on the hot path of every decoded token
— it is installed by default, so a rounding difference here would not fail anything loudly, it
would just quietly make the ring stop being the model. So the bar is `torch.equal`, not a
tolerance, on both routing flavours (hash-routed and top-k-routed layers), and every shape it does
NOT claim must fall through to the reference's own code untouched.

Run: python3 -m pytest tests/test_v4_moe_decode.py -q
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

torch = pytest.importorskip("torch")
REFCPU = pytest.importorskip("v4_ref_cpu")
MOE = pytest.importorskip("v4_moe_decode")

SEED = 5
DRAWS = 12


@pytest.fixture(scope="module")
def args():
    return REFCPU.cpu_args()


@pytest.fixture(scope="module")
def oracle(args):
    return REFCPU.build_oracle(args, SEED)


def _x(args, s, g, oracle):
    """An activation in the dtype the MoE is really fed — `Block.forward` hands it `ffn_norm(x)`,
    which carries the model dtype, and the expert Linears refuse anything else."""
    dt = oracle.layers[0].ffn.gate.weight.dtype
    return torch.randn(1, s, args.dim, generator=g).to(dt)


def _ids(args, s, g):
    return torch.randint(0, args.vocab_size, (1, s), generator=g)


def test_install_took(oracle):
    """The default path under test IS the fast one — otherwise every assert below is vacuous."""
    assert MOE._REF_FORWARD is not None, "v4_moe_decode.install() never ran"
    assert type(oracle.layers[0].ffn).forward is MOE.decode_forward


def test_decode_is_bit_exact_every_layer(args, oracle):
    """One token through every layer's MoE: the fast path and the reference agree EXACTLY.

    Both routing flavours are covered by sweeping all layers — `layer_id < n_hash_layers` routes by
    hashing the token id (`tid2eid[input_ids]`, which may repeat an expert), the rest by a top-k of
    the gate scores (which cannot)."""
    g = torch.Generator().manual_seed(SEED)
    for li, layer in enumerate(oracle.layers):
        for _ in range(DRAWS):
            x, ids = _x(args, 1, g, oracle), _ids(args, 1, g)
            with torch.no_grad():
                ref = MOE._REF_FORWARD(layer.ffn, x, ids)
                fast = MOE.decode_forward(layer.ffn, x, ids)
            assert ref.shape == fast.shape
            assert torch.equal(ref, fast), f"layer {li}: max |d| = {(ref - fast).abs().max()}"


def test_multi_token_falls_back(args, oracle):
    """Prefill and a speculation chunk are s > 1 — the fast path must not claim them."""
    g = torch.Generator().manual_seed(SEED + 1)
    for s in (2, 5, 13):
        x, ids = _x(args, s, g, oracle), _ids(args, s, g)
        with torch.no_grad():
            ref = MOE._REF_FORWARD(oracle.layers[-1].ffn, x, ids)
            fast = MOE.decode_forward(oracle.layers[-1].ffn, x, ids)
        assert torch.equal(ref, fast)


def test_repeated_expert_falls_back(args, oracle, monkeypatch):
    """A gate that routes the same expert twice: `y[idx] += ...` with duplicate indices is the
    reference's own semantics, and the fast path must defer to it rather than invent an answer."""
    ffn = oracle.layers[-1].ffn
    g = torch.Generator().manual_seed(SEED + 2)
    x, ids = _x(args, 1, g, oracle), _ids(args, 1, g)
    real_gate = ffn.gate.forward

    def dup_gate(xx, input_ids=None):
        w, idx = real_gate(xx, input_ids)
        return w, idx[:, :1].expand_as(idx).contiguous()      # every slot -> the same expert

    monkeypatch.setattr(ffn.gate, "forward", dup_gate)
    with torch.no_grad():
        ref = MOE._REF_FORWARD(ffn, x, ids)
        fast = MOE.decode_forward(ffn, x, ids)
    assert torch.equal(ref, fast)


def test_tensor_parallel_falls_back(args, oracle, monkeypatch):
    """Under world_size > 1 the reference all_reduces the routed sum across ranks before adding the
    shared expert. The fast path has no all_reduce, so skipping another rank's experts would drop
    them silently — it must defer instead."""
    MOE_ = sys.modules["v4_moe_decode"]
    g = torch.Generator().manual_seed(SEED + 3)
    x, ids = _x(args, 1, g, oracle), _ids(args, 1, g)
    monkeypatch.setattr(MOE_, "_WORLD_SIZE", 2)
    calls = []
    real = MOE_._REF_FORWARD
    monkeypatch.setattr(MOE_, "_REF_FORWARD",
                        lambda s, xx, ii: (calls.append(1), real(s, xx, ii))[1])
    with torch.no_grad():
        MOE_.decode_forward(oracle.layers[-1].ffn, x, ids)
    assert calls, "world_size > 1 must take the reference path"


def test_install_is_idempotent(oracle):
    """load_ref memoizes, but a second install must not chain the fast path onto itself and lose
    the reference it needs for the fallbacks."""
    mod = sys.modules["dsv4_model"]
    before = MOE._REF_FORWARD
    assert MOE.install(mod) is False
    assert MOE._REF_FORWARD is before
    assert mod.MoE.forward is MOE.decode_forward
