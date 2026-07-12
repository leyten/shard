"""Offline gates for MlxRuntime (shard/mlx_runtime.py) — no Mac, no GPU, no mlx.

MlxRuntime is written blind (no Apple silicon reachable), so these tests pin everything
that does NOT need a Metal device:

  - the module imports (and fails HONESTLY at call time) without mlx installed;
  - the range-loader's weight-key selection is a pure function, tested against a
    synthetic minimax-shaped weight_map (experts, gate, norms, embed, lm_head, quant
    .scales/.biases companions, tied-embedding fallback, prefix-collision safety);
  - the bf16 wire-bit bridge round-trips (numpy carries bf16 as uint16 bit patterns);
  - the ModelRuntime call/shape/dtype contract, EAGLE aux capture, and the spec-decode
    crop-to-start_pos KV ROLLBACK are exercised byte-for-byte on MlxRuntimeStub
    (shard/mlx_stub.py) — a stateful fake whose output CHANGES if the crop is wrong,
    so rollback == fresh-runtime equality is a real check, not a stateless tautology.

The real-mlx parity smoke at the bottom self-skips off this box; the full Mac-gated
checklist lives in docs/MLX_RUNTIME.md.

Run: python3 -m pytest tests/test_mlx_runtime.py -q
"""
import importlib.util
import os
import sys

import numpy as np
import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from shard.mlx_runtime import (MlxRuntime, aux_layers_from_env, bf16_bits_to_f32,  # noqa: E402
                               f32_to_bf16_bits, files_for_keys, select_weight_keys)
from shard.mlx_stub import H, N_LAYERS, VOCAB, MlxRuntimeStub  # noqa: E402
from shard.node import LayerRange  # noqa: E402

_HAS_MLX = importlib.util.find_spec("mlx") is not None


def _stub(lo, hi, head=False, tail=False, aux=()):
    r = MlxRuntimeStub(layer_range=LayerRange(lo, hi), is_head=head, is_tail=tail,
                       aux_layers=aux)
    r.load_shard()
    return r


def _h(seed, s):
    return (np.random.default_rng(seed).standard_normal((1, s, H)) * 0.1).astype(np.float32)


# ---- 1. importable + honest failure without mlx -----------------------------------------------------

def test_module_imports_and_fails_honestly_without_mlx():
    """The module must be importable on ANY box (the scheduler/CI see the class without
    Apple silicon); only USING the runtime may demand mlx — and then with a clear error."""
    import shard.mlx_runtime  # noqa: F401 — already imported above; re-import is the point
    rt = MlxRuntime("/nonexistent/model", LayerRange(0, 4))
    if _HAS_MLX:
        pytest.skip("mlx installed here; the no-mlx failure path is untestable")
    assert "mlx" not in sys.modules, "mlx_runtime must not import mlx at module import"
    with pytest.raises(ImportError, match="mlx"):
        rt.load_shard()
    with pytest.raises(RuntimeError, match="load_shard"):   # lifecycle guard fires before mlx
        rt.forward(np.zeros((1, 1, 8), dtype=np.float32), 0)


def test_heartbeat_never_raises_without_mlx():
    hb = MlxRuntime("/nonexistent/model", LayerRange(2, 5), is_tail=True).heartbeat()
    assert hb["backend"] == "mlx" and hb["layers"] == [2, 5] and hb["loaded"] is False
    if not _HAS_MLX:
        assert hb["alive"] is False and "mlx" in hb["error"]


# ---- 2. stub conformance: pipeline shapes/dtypes/determinism ----------------------------------------

def test_stub_pipeline_shapes_and_dtypes():
    """head embed -> forward -> (wire) -> tail forward -> logits; numpy in = numpy out."""
    head = _stub(0, 2, head=True)
    tail = _stub(2, N_LAYERS, tail=True)
    ids = np.array([[1, 5, 9, 3]])
    h0 = head.embed(ids)
    assert h0.shape == (1, 4, H) and h0.dtype == np.float32
    h1 = head.forward(h0, 0)
    assert h1.shape == (1, 4, H) and h1.dtype == np.float32 and isinstance(h1, np.ndarray)
    h2 = tail.forward(h1, 0)
    lg = tail.logits(h2)
    assert lg.shape == (1, 4, VOCAB) and lg.dtype == np.float32
    assert np.isfinite(lg).all()


def test_stub_is_deterministic_across_instances():
    """Two fresh stubs with the same range are bit-identical — determinism is a receipt
    property, so the contract double must have it too."""
    a, b = _stub(0, N_LAYERS), _stub(0, N_LAYERS)
    x = _h(0, 5)
    np.testing.assert_array_equal(a.forward(x, 0), b.forward(x, 0))
    y = _h(1, 1)
    np.testing.assert_array_equal(a.forward(y, 5), b.forward(y, 5))


def test_stub_role_guards_and_lifecycle():
    mid = _stub(1, 3)
    with pytest.raises(RuntimeError, match="head-only"):
        mid.embed([[1]])
    with pytest.raises(RuntimeError, match="tail-only"):
        mid.logits(np.zeros((1, 1, H), dtype=np.float32))
    fresh = MlxRuntimeStub(layer_range=LayerRange(0, 2))
    with pytest.raises(RuntimeError, match="load_shard"):
        fresh.forward(np.zeros((1, 1, H), dtype=np.float32), 0)
    assert mid.heartbeat()["alive"] is True and mid.heartbeat()["backend"] == "stub"


def test_forward_mirrors_bf16_bits_representation():
    """A bf16-bits (uint16) frame comes back as bf16 bits, numerically tracking the f32
    path within bf16 rounding — the wire representation is the caller's choice per frame."""
    x = _h(2, 3)
    f32_out = _stub(0, N_LAYERS).forward(x, 0)
    bits_out = _stub(0, N_LAYERS).forward(f32_to_bf16_bits(x), 0)
    assert bits_out.dtype == np.uint16 and bits_out.shape == (1, 3, H)
    np.testing.assert_allclose(bf16_bits_to_f32(bits_out), f32_out, atol=0.02, rtol=0.05)


def test_bf16_bit_bridge_roundtrip():
    """bits -> f32 is exact; f32 -> bits -> f32 is idempotent (second pass changes nothing)."""
    x = (np.random.default_rng(4).standard_normal((3, 7)) * 5).astype(np.float32)
    bits = f32_to_bf16_bits(x)
    once = bf16_bits_to_f32(bits)
    np.testing.assert_array_equal(f32_to_bf16_bits(once), bits)
    assert np.abs(once - x).max() <= np.abs(x).max() * 2 ** -8  # bf16 has 8 mantissa bits


# ---- 3. KV rollback: crop-to-start_pos == m25's spec-decode semantics -------------------------------

def test_kv_rollback_reproduces_fresh_runtime():
    """The m25 rollback contract: after a speculative block is partially rejected, a
    re-forward at an EARLIER start_pos must overwrite the stale speculative KV — outputs
    byte-equal to a fresh runtime that only ever saw the accepted prefix."""
    prompt, spec, fix = _h(3, 6), _h(4, 3), _h(5, 2)

    a = _stub(0, N_LAYERS)
    a.forward(prompt, 0)
    a.forward(spec, 6)                    # speculative KV at positions 6..8
    out_a = a.forward(fix, 7)             # rollback: only position 6 was accepted

    b = _stub(0, N_LAYERS)
    b.forward(prompt, 0)
    b.forward(spec[:, :1], 6)             # the accepted token only
    out_b = b.forward(fix, 7)
    # near-equality, not byte-equality: the STUB's numpy attention accumulates in a
    # different order between the two paths under macOS Accelerate BLAS (~1 ULP, 3e-08
    # measured) — platform summation order, not a crop bug. A no-op trim still fails this
    # by ~7 orders of magnitude (stale rows shift values at the 1e-1 scale). The REAL
    # runtime's contract stays byte-equal and is enforced on-device by the Mac gate
    # (docs/receipts/mlx-mac-gate-20260712.json: kv_rollback_crop, byte-equal).
    np.testing.assert_allclose(out_a, out_b, rtol=1e-6, atol=1e-7)


def test_full_rollback_to_zero_equals_fresh():
    """forward at start_pos=0 after arbitrary history == a fresh runtime's first forward
    (a re-prefill at an earlier start overwrites EVERYTHING stale)."""
    x = _h(6, 4)
    a = _stub(0, N_LAYERS)
    a.forward(_h(7, 5), 0)
    a.forward(_h(8, 2), 5)
    np.testing.assert_array_equal(a.forward(x, 0), _stub(0, N_LAYERS).forward(x, 0))


def test_reset_drops_cache():
    a = _stub(0, N_LAYERS)
    x = _h(9, 4)
    first = a.forward(x, 0)
    a.forward(_h(10, 1), 4)
    a.reset()
    np.testing.assert_array_equal(a.forward(x, 0), first)


def test_kv_gap_is_refused():
    """start_pos past the cached span would mis-position everything (real backend: RoPE
    reads positions off cache.offset) — a hard error, never a garbage forward."""
    a = _stub(0, N_LAYERS)
    a.forward(_h(11, 3), 0)
    with pytest.raises(ValueError, match="gap"):
        a.forward(_h(12, 1), 5)


# ---- 4. EAGLE aux capture ----------------------------------------------------------------------------

def test_aux_captured_in_range_only():
    """aux_layers ∩ [lo:hi) captured as [S,H] after the layer; out-of-range ids ignored —
    m25's 'a stage records ONLY the aux layers in its own range' contract."""
    r = _stub(1, 3, aux={1, 2, 3, 30})    # 3 and 30 are outside [1,3)
    x = _h(13, 4)
    r.forward(x, 0)
    assert set(r.aux) == {1, 2}
    for li in (1, 2):
        assert r.aux[li].shape == (4, H) and r.aux[li].dtype == np.uint16   # ALWAYS bf16 bits
    r.forward(_h(14, 1), 4)               # refreshed per forward, not accumulated
    assert set(r.aux) == {1, 2} and r.aux[1].shape == (1, H)


def test_aux_rides_the_rollback_contract():
    """aux is a snapshot of the SAME forward the rollback semantics govern — after an
    identical rollback, aux matches the fresh runtime's byte-for-byte."""
    prompt, fix = _h(15, 5), _h(16, 2)
    a = _stub(0, N_LAYERS, aux={1})
    a.forward(prompt, 0)
    a.forward(_h(17, 3), 5)
    a.forward(fix, 5)                     # roll all 3 speculative tokens back
    b = _stub(0, N_LAYERS, aux={1})
    b.forward(prompt, 0)
    b.forward(fix, 5)
    np.testing.assert_array_equal(a.aux[1], b.aux[1])


def test_aux_matches_bits_representation():
    r = _stub(0, 2, aux={0})
    r.forward(f32_to_bf16_bits(_h(18, 2)), 0)
    assert r.aux[0].dtype == np.uint16 and r.aux[0].shape == (2, H)


def test_aux_layers_env_default(monkeypatch):
    """m25_stage parity: no capture unless M25_EAGLE=1; then M25_EAGLE_AUX (default 1,30,58)."""
    monkeypatch.delenv("M25_EAGLE", raising=False)
    assert aux_layers_from_env() == frozenset()
    monkeypatch.setenv("M25_EAGLE", "1")
    assert aux_layers_from_env() == frozenset({1, 30, 58})
    monkeypatch.setenv("M25_EAGLE_AUX", "2,5")
    assert aux_layers_from_env() == frozenset({2, 5})


# ---- 5. range-loader key selection (pure; the real loader's selection logic) ------------------------

def _minimax_weight_map(n_layers=4):
    """A synthetic MLX-conversion index for the minimax family: quantized attn projections
    (+ .scales/.biases companions), q/k/norms, MoE gate (8-bit override in real conversions,
    same key shape), e_score_correction_bias, stacked switch_mlp experts, embed, final norm,
    lm_head. Two files, split mid-model, so file resolution is non-trivial."""
    wm = {}
    for j in range(n_layers):
        fn = f"model-{1 if j < n_layers // 2 else 2:05d}.safetensors"
        P = f"model.layers.{j}."
        for name in ("input_layernorm.weight", "post_attention_layernorm.weight",
                     "self_attn.q_norm.weight", "self_attn.k_norm.weight",
                     "block_sparse_moe.e_score_correction_bias"):
            wm[P + name] = fn
        for mod in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                    "self_attn.o_proj", "block_sparse_moe.gate",
                    "block_sparse_moe.switch_mlp.gate_proj",
                    "block_sparse_moe.switch_mlp.up_proj",
                    "block_sparse_moe.switch_mlp.down_proj"):
            for suf in ("weight", "scales", "biases"):
                wm[f"{P}{mod}.{suf}"] = fn
    for suf in ("weight", "scales", "biases"):
        wm[f"model.embed_tokens.{suf}"] = "model-00001.safetensors"
        wm[f"lm_head.{suf}"] = "model-00002.safetensors"
    wm["model.norm.weight"] = "model-00002.safetensors"
    return wm


def test_select_keys_middle_stage_exact():
    wm = _minimax_weight_map()
    keys = select_weight_keys(wm, 1, 3, is_head=False, is_tail=False)
    assert keys == sorted(k for k in wm if k.startswith(("model.layers.1.", "model.layers.2.")))
    assert not any(k.startswith(("model.embed_tokens", "model.norm", "lm_head")) for k in keys)
    assert any(k.endswith("switch_mlp.down_proj.scales") for k in keys)   # experts + quant ride
    assert any(k.endswith("e_score_correction_bias") for k in keys)       # the layer prefix


def test_select_keys_head_and_tail_edges():
    wm = _minimax_weight_map()
    head = set(select_weight_keys(wm, 0, 2, is_head=True, is_tail=False))
    assert {f"model.embed_tokens.{s}" for s in ("weight", "scales", "biases")} <= head
    assert not any(k.startswith(("model.norm", "lm_head")) for k in head)
    tail = set(select_weight_keys(wm, 2, 4, is_head=False, is_tail=True))
    assert "model.norm.weight" in tail
    assert {f"lm_head.{s}" for s in ("weight", "scales", "biases")} <= tail
    assert not any(k.startswith("model.embed_tokens") for k in tail)


def test_select_keys_no_prefix_collision():
    """layer 1 must not drag in layers 10-13 — the trailing dot in the prefix is load-bearing."""
    wm = _minimax_weight_map(n_layers=14)
    keys = select_weight_keys(wm, 1, 2, is_head=False, is_tail=False)
    assert keys and all(k.startswith("model.layers.1.") for k in keys)


def test_select_keys_tied_embeddings_tail():
    """Tiedness is CONFIG-driven (tied=True): the tail pulls embed_tokens (logits via
    embed.as_linear) and needs no lm_head.* in the map."""
    wm = {k: v for k, v in _minimax_weight_map().items() if not k.startswith("lm_head.")}
    tail = set(select_weight_keys(wm, 2, 4, is_head=False, is_tail=True, tied=True))
    assert "model.embed_tokens.weight" in tail and "model.norm.weight" in tail


def test_select_keys_untied_tail_missing_lm_head_fails_loud():
    """An UNTIED model whose index merely lacks lm_head.* must REFUSE, never fall back to
    a random-init substitute — a tail projecting logits through garbage produces plausible
    tokens with valid receipts (the silent-corruption class)."""
    wm = {k: v for k, v in _minimax_weight_map().items() if not k.startswith("lm_head.")}
    with pytest.raises(ValueError, match="lm_head"):
        select_weight_keys(wm, 2, 4, is_head=False, is_tail=True, tied=False)


def test_select_keys_corrupt_index_edges_fail_loud():
    """A head with no embed keys / a tail with no norm keys / a tied tail with no embed
    keys = corrupt or partial index -> hard error, never a random edge module."""
    wm = _minimax_weight_map()
    no_embed = {k: v for k, v in wm.items() if not k.startswith("model.embed_tokens.")}
    with pytest.raises(ValueError, match="embed_tokens"):
        select_weight_keys(no_embed, 0, 2, is_head=True, is_tail=False)
    no_norm = {k: v for k, v in wm.items() if not k.startswith("model.norm.")}
    with pytest.raises(ValueError, match="norm"):
        select_weight_keys(no_norm, 2, 4, is_head=False, is_tail=True, tied=False)
    with pytest.raises(ValueError, match="embed_tokens"):
        select_weight_keys(no_embed, 2, 4, is_head=False, is_tail=True, tied=True)


def test_files_for_keys_resolves_exactly():
    wm = _minimax_weight_map()
    mid = select_weight_keys(wm, 0, 2, is_head=False, is_tail=False)   # first half -> file 1
    assert files_for_keys(wm, mid) == ["model-00001.safetensors"]
    span = select_weight_keys(wm, 1, 3, is_head=False, is_tail=False)  # straddles the split
    assert files_for_keys(wm, span) == ["model-00001.safetensors", "model-00002.safetensors"]


def test_bridge_matches_torch_rne():
    """torch is what produces the wire's bf16 bytes on the CUDA path — the numpy bridge
    must round IDENTICALLY (round-to-nearest-even), or a Mac stage reads different
    activations than a CUDA stage shipped. Adversarial values: denormals, huge/tiny,
    random bit patterns."""
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(7)
    vals = np.concatenate([
        rng.standard_normal(50_000).astype(np.float32),
        (rng.standard_normal(50_000) * 1e30).astype(np.float32),
        (rng.standard_normal(50_000) * 1e-30).astype(np.float32),
        rng.integers(0, 2**31, 50_000, dtype=np.uint32).view(np.float32),
    ])
    vals = vals[np.isfinite(vals)]
    ours = f32_to_bf16_bits(vals)
    theirs = torch.from_numpy(vals.copy()).to(torch.bfloat16).view(torch.uint16).numpy()
    np.testing.assert_array_equal(ours, theirs)


def test_forward_rejects_foreign_dtypes():
    """An fp8-wire frame (uint8 bytes) coerced to float32 becomes the integers 0..255 as
    activations — garbage forward, healthy transport. The contract: float32 or bf16-bits
    ONLY; everything else is refused at the boundary."""
    rt = _stub(0, 2)
    good = np.zeros((1, 3, H), dtype=np.float32)
    rt.forward(good, 0)
    for bad_dtype in (np.uint8, np.float16, np.int32, np.float64):
        with pytest.raises(TypeError, match="dequantize|float32"):
            rt.forward(np.zeros((1, 3, H), dtype=bad_dtype), 0)


def test_aux_always_bf16_bits():
    """m25 captures aux as bf16 UNCONDITIONALLY (the wire aux contract); the runtime must
    too, even when the frame representation is float32."""
    rt = _stub(0, 2, aux=(1,))
    rt.forward(np.random.default_rng(3).standard_normal((1, 4, H)).astype(np.float32), 0)
    assert rt.aux and rt.aux[1].dtype == np.uint16


# ---- 6. real-mlx parity smoke (self-skips off-Mac; part of the Mac checklist) -----------------------

@pytest.mark.skipif(not _HAS_MLX, reason="mlx not installed (Apple-silicon gate)")
def test_mlx_numpy_parity_smoke():
    """Two random linear layers, numpy vs mlx, to 1e-2 — plus the bf16 wire-bit bridge
    against mlx's own bf16 cast (the boundary MlxRuntime ships activations across)."""
    import mlx.core as mx
    rng = np.random.default_rng(0)
    w1 = (rng.standard_normal((16, 8)) * 0.3).astype(np.float32)
    w2 = (rng.standard_normal((8, 16)) * 0.3).astype(np.float32)
    x = (rng.standard_normal((1, 5, 8)) * 0.5).astype(np.float32)
    ref = np.tanh(x @ w1.T) @ w2.T
    got = np.array(mx.matmul(mx.tanh(mx.matmul(mx.array(x), mx.array(w1).T)), mx.array(w2).T))
    np.testing.assert_allclose(got, ref, atol=1e-2)
    ours = f32_to_bf16_bits(x)
    theirs = np.array(mx.view(mx.array(x).astype(mx.bfloat16), mx.uint16))
    np.testing.assert_array_equal(ours, theirs)
