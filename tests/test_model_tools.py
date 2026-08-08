"""The per-model tools seam (phase0/model_tools.py) — and the proof M2.5 did not move.

docs/MODEL_RUNTIME.md leak #4: `m25_tools.py` hardcoded MiniMax's XML and m25_pipe / m25_gateway
imported it by name. This closes that with a lookup, which means the load-bearing test is not "K3
resolves" but "every existing ring still binds exactly the objects it bound before" — asserted by
IDENTITY (`is`), not by behaviour, because a family module that merely behaves the same is a
regression waiting for its first divergence.

Run: python3 -m pytest tests/test_model_tools.py -q
"""
import importlib
import importlib.util
import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PHASE0 = os.path.join(_ROOT, "phase0")
sys.path.insert(0, _PHASE0)

import model_tools as MT                                   # noqa: E402
import m25_tools                                           # noqa: E402


@pytest.fixture(autouse=True)
def _no_family_override(monkeypatch):
    monkeypatch.delenv("SHARD_TOOLS_FAMILY", raising=False)


# ---------------------------------------------------------------- selection

@pytest.mark.parametrize("model_id, family", [
    ("minimax-m2.5", "m25"), ("MiniMax-M2.5", "m25"), ("nvidia/MiniMax-M2.5-NVFP4", "m25"),
    ("m25", "m25"), (None, "m25"), ("", "m25"), ("some-unknown-model", "m25"),
    ("kimi-k3", "k3"), ("moonshotai/Kimi-K3", "k3"), ("KIMI_K3", "k3"), ("k3", "k3"),
    ("unsloth/Kimi-K3-GGUF", "k3"),
])
def test_family_for(model_id, family):
    assert MT.family_for(model_id) == family


def test_an_unknown_id_falls_back_to_m25_not_an_error():
    """Every ring in existence passes an M2.5 id or none. This file must not be the reason one of
    them starts refusing to boot."""
    assert MT.for_model("gpt-oss-120b") is m25_tools


def test_longest_matching_rule_wins():
    """"kimi-k3" and "k3" both match a K3 id; "m25" must not steal an id that also says minimax."""
    assert MT.family_for("k3-minimax-hybrid-m25") == "m25"      # "minimax" (7) beats "k3" (2)
    assert MT.family_for("kimi-k3-m25-quant") == "k3"           # "kimi-k3" (7) beats "m25" (3)


def test_env_override(monkeypatch):
    monkeypatch.setenv("SHARD_TOOLS_FAMILY", "k3")
    assert MT.family_for("minimax-m2.5") == "k3"
    monkeypatch.setenv("SHARD_TOOLS_FAMILY", "nonsense")
    with pytest.raises(ValueError, match="nonsense"):
        MT.family_for("minimax-m2.5")


def test_k3_is_not_imported_unless_it_is_selected():
    """k3_tools pulls in the vendored reference renderer. A box serving M2.5 must not pay for it."""
    src = open(os.path.join(_PHASE0, "model_tools.py")).read()
    assert "import k3_tools" not in src and "from k3_tools" not in src


def test_for_model_is_cached():
    assert MT.for_model("kimi-k3") is MT.for_model("moonshotai/Kimi-K3")


# ---------------------------------------------------------------- the contract each family owes

REQUIRED = ("render_ids", "parse_completion", "to_openai_message",
            "THINK_BEGIN", "THINK_END", "TOOLCALL_BEGIN")


@pytest.mark.parametrize("family", sorted(MT.FAMILIES))
def test_every_family_exports_the_call_sites_names(family):
    mod = importlib.import_module(MT.FAMILIES[family][0])
    missing = [n for n in REQUIRED if not hasattr(mod, n)]
    assert missing == [], f"{family} is missing {missing}"


def test_the_families_do_not_share_a_wire_format():
    k3 = MT.for_model("kimi-k3")
    assert k3.THINK_END != m25_tools.THINK_END
    assert k3.TOOLCALL_BEGIN != m25_tools.TOOLCALL_BEGIN


# ---------------------------------------------------------------- the call sites

def _load_gateway(model_id):
    """A private copy of m25_gateway bound to one model id. Its heavy imports (torch, m25_pipe,
    transformers) all live inside functions, so this costs nothing and touches no global state."""
    prev = os.environ.get("M25_MODEL_ID")
    if model_id is None:
        os.environ.pop("M25_MODEL_ID", None)
    else:
        os.environ["M25_MODEL_ID"] = model_id
    try:
        spec = importlib.util.spec_from_file_location(
            f"_gw_{abs(hash(model_id))}", os.path.join(_ROOT, "engines", "minimax_m25", "m25_gateway.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        os.environ.pop("M25_MODEL_ID", None)
        if prev is not None:
            os.environ["M25_MODEL_ID"] = prev


def test_gateway_default_binds_the_identical_m25_objects():
    """The bit-identity claim, at the seam that carries it: with no M25_MODEL_ID the gateway's
    module-level names ARE m25_tools' objects, exactly as the direct import made them."""
    gw = _load_gateway(None)
    assert gw.render_ids is m25_tools.render_ids
    assert gw.parse_completion is m25_tools.parse_completion
    assert gw.to_openai_message is m25_tools.to_openai_message
    assert (gw.TOOLCALL_BEGIN, gw.THINK_BEGIN, gw.THINK_END) == (
        m25_tools.TOOLCALL_BEGIN, m25_tools.THINK_BEGIN, m25_tools.THINK_END)


def test_gateway_default_keeps_the_m25_splitter_and_mock():
    gw = _load_gateway(None)
    assert isinstance(gw._splitter(True), gw._SSESplitter)
    assert gw.TOOLCALL_BEGIN in gw._mock_generate(
        [{"role": "user", "content": "x"}], [{"function": {"name": "f"}}], 8, None)["text"]


def test_gateway_on_a_k3_id_binds_k3():
    gw = _load_gateway("moonshotai/Kimi-K3")
    k3 = MT.for_model("kimi-k3")
    assert gw.render_ids is k3.render_ids and gw.parse_completion is k3.parse_completion
    assert gw.THINK_END == k3.THINK_END
    assert isinstance(gw._splitter(True), k3.StreamSplitter)


def test_gateway_k3_mock_emits_xtml_that_the_k3_parser_understands():
    """MOCK is the no-GPU api-shape path. Left ungeneralized it would emit MiniMax XML whatever
    model is selected, so the shape test would pass against a format nothing can parse."""
    gw = _load_gateway("kimi-k3")
    k3 = MT.for_model("kimi-k3")
    r = gw._mock_generate([{"role": "user", "content": "weather?"}],
                          [{"type": "function", "function": {"name": "get_weather"}}], 8, None)
    parsed = k3.parse_completion(r["text"])
    assert [c["name"] for c in parsed["tool_calls"]] == ["get_weather"]
    assert "<invoke" not in r["text"]


def test_gateway_k3_streaming_never_leaks_the_response_channel_marker():
    """The reason the splitter is per-family and not one shared state machine: M2.5's three markers
    know nothing about K3's response channel, so _SSESplitter would stream its opener as content."""
    gw = _load_gateway("kimi-k3")
    k3 = MT.for_model("kimi-k3")
    text = gw._mock_generate([{"role": "user", "content": "hi"}], None, 8, None)["text"]
    sp, r, c = gw._splitter(True), "", ""
    for i in range(0, len(text), 3):
        dr, dc = sp.feed(text[i:i + 3])
        r, c = r + dr, c + dc
    dr, dc = sp.end()
    r, c = r + dr, c + dc
    assert (r, c) == (k3.parse_completion(text)["reasoning_content"],
                      k3.parse_completion(text)["content"])
    assert k3.RESPONSE_BEGIN not in c and k3.OPEN not in c


def test_pipe_binds_through_the_seam_not_by_direct_import():
    """m25_pipe imports torch and m25_stage at module scope, so it cannot be imported here. Read the
    source instead: the direct `from m25_tools import ...` must be gone."""
    src = open(os.path.join(_ROOT, "engines", "minimax_m25", "m25_pipe.py")).read()
    assert "from m25_tools import" not in src
    assert "from model_tools import for_model" in src
    assert "render_ids, parse_completion = _TOOLS.render_ids, _TOOLS.parse_completion" in src


def test_model_tools_is_in_the_deploy_push_set():
    """m25_pipe and m25_gateway import it at module scope: a box that does not get this file cannot
    start a stage. m25_scatter_pipe's push list is that box's whole world."""
    src = open(os.path.join(_ROOT, "engines", "minimax_m25", "m25_scatter_pipe.py")).read()
    assert '"phase0/model_tools.py"' in src


def test_deploy_path_strings_exist_in_the_tree():
    """Repo-relative paths in the scatter launchers are STRINGS, not imports — a refactor that moves
    a file leaves them dangling and no import error ever fires (#166 broke 4 of them this way).
    Every quoted phase0/engines/shard path in these files must exist."""
    import re
    for rel in (("engines", "minimax_m25", "m25_scatter_pipe.py"),
                ("engines", "minimax_m25", "m25_scatter.py"),
                ("engines", "deepseek_v4", "v4_ngram_accept.py")):
        src = open(os.path.join(_ROOT, *rel)).read()
        paths = re.findall(r'"((?:phase0|engines|shard)/[^"\s]+\.(?:py|sh|json|md))"', src)
        assert paths, f"no repo-relative path strings found in {'/'.join(rel)} — pattern drifted?"
        missing = [p for p in paths if not os.path.exists(os.path.join(_ROOT, p))]
        assert not missing, f"{'/'.join(rel)} references files that do not exist: {missing}"
