"""Operator --dir must win over M25_DIR (m25_stage L3 audit fix).

m25_stage does its model init (AutoConfig + safetensors index) at MODULE level, consuming M25_DIR
at import time; the old flow parsed --dir in __main__ AFTER that init, so the self-test's --dir was
silently ignored (it always loaded whatever M25_DIR/its default pointed at). The fix pre-parses
--dir before init when run as a script. The subprocess test executes the real file as __main__ with
M25_DIR pointing at model A and --dir at model B (distinguishable hidden_size) and asserts the
module's config constants came from B.

Run: python3 -m pytest tests/test_stage_dir.py -q
"""
import json
import os
import subprocess
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

pytest.importorskip("torch")
fr = pytest.importorskip("fake_ring")                # bootstraps env + imports m25_stage on CPU
S = fr.S

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_STAGE = os.path.join(_REPO, "phase0", "m25_stage.py")


def _model_dir(hidden):
    """fake_ring's minimal M25_DIR, with a distinguishable hidden_size."""
    d = tempfile.mkdtemp(prefix="m25_dirtest_")
    json.dump({
        "model_type": "minimax_m2", "hidden_size": hidden, "num_attention_heads": 48,
        "num_key_value_heads": 8, "head_dim": 128, "num_hidden_layers": 62,
        "rms_norm_eps": 1e-6, "num_local_experts": 256, "num_experts_per_tok": 8,
        "intermediate_size": 1536, "moe_intermediate_size": 1536, "rope_theta": 5000000,
        "vocab_size": 200064, "max_position_embeddings": 196608,
    }, open(os.path.join(d, "config.json"), "w"))
    json.dump({"weight_map": {}}, open(os.path.join(d, "model.safetensors.index.json"), "w"))
    return d


def test_cli_dir_preparse():
    assert S._cli_dir(["--dir", "/x/m25", "--layers", "1", "2"]) == "/x/m25"
    assert S._cli_dir(["--layers", "3"]) is None      # absent -> env/default flow unchanged
    assert S._cli_dir([]) is None


def test_script_dir_flag_wins_over_env():
    a, b = _model_dir(3072), _model_dir(1024)
    # Execute the real file as __main__ (the pre-parse only arms there). _selftest crashes later
    # (no GPU/vllm in the test env) — irrelevant: the config constants are set at module init,
    # which is exactly the code under test.
    code = f"""
import sys, importlib.util
sys.argv = ["m25_stage.py", "--dir", {b!r}, "--layers", "1"]
spec = importlib.util.spec_from_file_location("__main__", {_STAGE!r})
mod = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(mod)
except BaseException:
    pass
print("H=%d DIR=%s" % (mod.H, mod.DIR))
"""
    env = dict(os.environ, M25_DIR=a)
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True,
                         text=True, timeout=300)
    assert f"DIR={b}" in out.stdout, f"--dir ignored: {out.stdout}\n{out.stderr[-2000:]}"
    assert "H=1024" in out.stdout                      # config actually LOADED from --dir's model
