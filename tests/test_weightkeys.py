"""The shared stage-key classification (shard/weightkeys.py) and the LOAD-side site keyed on it.

shard/fetch.py picks a range's FILES out of the signed weight_map by the checkpoint's own tensor
names; these gates cover the other half of the same seam — the module paths a stage materializes:

  - the classification itself: a layer number comes from `layers.<n>.` whatever namespace
    surrounds it, and a per-layer norm is never mistaken for the tail's final norm;
  - `namespace()` maps a checkpoint's tensor names to the MODULE paths a device_map is keyed by,
    keeps the HF-standard layout when there is no index to read, and RAISES on a namespace whose
    decoder it cannot locate rather than returning a map that places nothing on the GPU;
  - phase0/pipeline.load_stage's device_map is IDENTICAL to the hardcoded one it replaces for
    the HF namespace, and follows a foreign text stack instead of naming nothing.

Run: python3 -m pytest tests/test_weightkeys.py -q
"""
import itertools
import json
import os
import shutil
import subprocess
import sys

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from shard.weightkeys import HF_LAYOUT, boundary_of, layer_of, namespace   # noqa: E402


def _foreign_names(n_layers=4):
    """A Kimi-K3-shaped multimodal checkpoint: the text stack under its own prefix, plus a
    shallower vision tower that numbers ITS blocks `layers.<n>.` too."""
    names = [f"language_model.model.layers.{j}.self_attn.q_proj.weight" for j in range(n_layers)]
    names += ["language_model.model.embed_tokens.weight", "language_model.model.norm.weight",
              "language_model.lm_head.weight", "vision_tower.layers.0.attn.qkv.weight"]
    return names


# ---- 1. classification -------------------------------------------------------------------------

def test_layer_number_comes_from_the_tensors_own_name():
    assert layer_of("model.layers.7.block_sparse_moe.experts.3.w1.weight") == 7    # not expert 3
    assert layer_of("language_model.model.layers.61.self_attn.q_proj.weight") == 61
    assert layer_of("transformer.h.4.attn.c_attn.weight") is None                  # no such component


def test_a_layers_own_norm_is_not_the_tails_norm():
    """The regression that would silently put every layer's norms on the tail: a tensor with a
    layer number belongs to its layer, never to a boundary."""
    for k in ("model.layers.0.input_layernorm.weight", "model.layers.3.self_attn.q_norm.weight"):
        assert boundary_of(k) is None and layer_of(k) is not None


def test_boundaries_classified_in_any_namespace():
    for prefix in ("", "language_model.", "text_model."):
        assert boundary_of(f"{prefix}model.embed_tokens.weight") == "embed"
        assert boundary_of(f"{prefix}model.norm.weight") == "norm"
        assert boundary_of(f"{prefix}lm_head.weight") == "lm_head"
    assert boundary_of("vision_tower.patch_embed.proj.weight") is None


# ---- 2. namespace -> module paths --------------------------------------------------------------

def test_namespace_of_the_hf_layout_is_what_the_loaders_hardcoded():
    """M2.5 must not move: for the HF namespace the derived paths ARE the hardcoded ones."""
    names = [f"model.layers.{j}.self_attn.q_proj.weight" for j in range(62)]
    names += ["model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"]
    assert namespace(names) == HF_LAYOUT


def test_namespace_follows_a_foreign_text_stack():
    assert namespace(_foreign_names()) == {
        "inner": "language_model.model", "layers": "language_model.model.layers",
        "embed": "language_model.model.embed_tokens", "norm": "language_model.model.norm",
        "lm_head": "language_model.lm_head"}


def test_namespace_picks_the_tower_the_boundaries_hang_off():
    """A device_map must name ONE stack (the fetch can union files; this cannot). The decoder is
    the tower the embedding and final norm sit next to — NOT the deepest: Qwen2-Audio's Whisper
    encoder is 32 layers against a 28-layer text decoder, so depth alone picks the encoder."""
    names = [f"model.audio_tower.layers.{j}.self_attn.q_proj.weight" for j in range(32)]
    names += [f"model.language_model.layers.{j}.self_attn.q_proj.weight" for j in range(28)]
    names += ["model.language_model.embed_tokens.weight", "model.language_model.norm.weight"]
    ns = namespace(names)
    assert ns["layers"] == "model.language_model.layers"
    assert ns["inner"] == "model.language_model"       # coherent: stack and boundaries agree


def test_namespace_falls_back_to_depth_without_a_boundary_vote():
    names = [f"vision_tower.layers.{j}.attn.qkv.weight" for j in range(2)]     # listed FIRST
    names += [f"language_model.model.layers.{j}.self_attn.q_proj.weight" for j in range(4)]
    assert namespace(names)["layers"] == "language_model.model.layers"


def test_namespace_prefers_the_boundary_inside_the_chosen_stack():
    """A second tower's own final norm is not the tail's — and the pick must not depend on
    where the index happens to list it."""
    names = ["vision_tower.norm.weight"] + _foreign_names()
    assert namespace(names)["norm"] == "language_model.model.norm"


def test_namespace_keeps_a_boundary_the_checkpoint_puts_outside_the_stack():
    """Some multimodal checkpoints keep lm_head at the root: take the path the checkpoint
    actually carries over the convention, which is only the last resort."""
    names = [n for n in _foreign_names() if "lm_head" not in n] + ["lm_head.weight"]
    assert namespace(names)["lm_head"] == "lm_head"


def test_namespace_defaults_a_missing_boundary_under_the_derived_stack():
    """A tied checkpoint carries no lm_head tensor; the module still exists, so its path falls
    back to the convention relative to the DERIVED stack, not to the hardcoded root."""
    names = [n for n in _foreign_names() if "lm_head" not in n]
    assert namespace(names)["lm_head"] == "language_model.lm_head"


def test_namespace_without_an_index_keeps_the_hf_layout():
    """No local index (a bare hub id) = no information, not a namespace miss: keep today's
    behavior instead of failing a load that works."""
    assert namespace([]) == HF_LAYOUT


def test_namespace_of_an_unlocatable_stack_raises():
    """A checkpoint numbering its blocks without a `layers.<n>.` component (GPT-2-shaped
    transformer.h.<n>.) must RAISE — a device_map derived from a namespace we could not read
    would place nothing on the GPU."""
    with pytest.raises(ValueError, match="layer pattern"):
        namespace([f"transformer.h.{j}.attn.c_attn.weight" for j in range(4)])


# ---- 3. the device_map load_stage builds from it -----------------------------------------------

def _pipeline():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    import pipeline                                   # phase0/, on sys.path via conftest
    return pipeline


def _legacy_dmap(n_layers, lo, hi, *, is_head, is_tail, tied, device):
    """The pre-fix device_map verbatim — hardcoded HF module names — as the equivalence oracle."""
    dmap = {"model.embed_tokens": device if (is_head or (is_tail and tied)) else "meta",
            "model.rotary_emb": device,
            "model.norm": device if is_tail else "meta",
            "lm_head": device if is_tail else "meta"}
    for j in range(n_layers):
        dmap[f"model.layers.{j}"] = device if lo <= j < hi else "meta"
    return dmap


@pytest.mark.parametrize("lo,hi", [(0, 2), (2, 4), (1, 3), (0, 6)])
def test_device_map_identical_to_the_hardcoded_one_for_the_hf_layout(lo, hi):
    P = _pipeline()
    for is_head, is_tail, tied in itertools.product((False, True), repeat=3):
        role = dict(is_head=is_head, is_tail=is_tail, tied=tied)
        assert P.device_map_for_block(HF_LAYOUT, 6, lo, hi, device="cuda", **role) == \
            _legacy_dmap(6, lo, hi, device="cuda", **role), f"[{lo}:{hi}] {role}"


def test_device_map_follows_a_foreign_stack():
    """The leak: for a checkpoint whose text stack lives elsewhere, every hardcoded key named a
    module that does not exist (and no key named the layers), so transformers refused the load."""
    P = _pipeline()
    dmap = P.device_map_for_block(namespace(_foreign_names()), 4, 0, 2,
                                  is_head=True, is_tail=False, tied=False, device="cuda")
    assert dmap["language_model.model.layers.0"] == "cuda"
    assert dmap["language_model.model.layers.3"] == "meta"
    assert dmap["language_model.model.embed_tokens"] == "cuda"
    assert dmap["language_model.lm_head"] == "meta"
    assert dmap["language_model.model.rotary_emb"] == "cuda"   # carries no tensor: sits by the layers
    assert not any(k.startswith("model.") for k in dmap)


@pytest.mark.integration
def test_pipeline_imports_in_the_flat_box_layout(tmp_path):
    """LAUNCH.md P0-#2, the landmine class this file could have re-armed: launchers push phase0
    and shard files FLAT into /root/, where there is no `shard` package. pipeline must import
    with no PYTHONPATH in that layout (flat first) and from a repo checkout (package fallback)."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    for src in ("phase0/pipeline.py", "phase0/wire.py", "phase0/node_kv.py",
                "shard/weightkeys.py", "shard/transport.py"):
        shutil.copy(os.path.join(_REPO, src), tmp_path / os.path.basename(src))
    env = {k: v for k, v in os.environ.items()
           if k != "PYTHONPATH" and not k.startswith(("M25_", "SHARD_"))}
    r = subprocess.run([sys.executable, "-c", "import pipeline; pipeline.namespace"],
                       cwd=str(tmp_path), env=env, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, f"flat layout import failed:\n{r.stderr[-2000:]}"


def test_checkpoint_tensor_names_reads_a_local_index(tmp_path):
    P = _pipeline()
    assert P.checkpoint_tensor_names("some-org/some-model") == []          # a hub id: nothing local
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {"language_model.model.layers.0.q.weight": "a.safetensors"}}))
    assert P.checkpoint_tensor_names(str(tmp_path)) == ["language_model.model.layers.0.q.weight"]
