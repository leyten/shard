"""The spot-check seam — torch-free sketch comparison + the `python3 -m shard.challenge` CLI.

The control plane (c0mpute) judges spot-check sketches without a CUDA stack: compare_sketches is
pure python and the CLI mirrors the shard.plan/shard.verify stdio pattern. What must hold: (1) the
verdict semantics match the torch compare() on the same sketches, (2) malformed / mismatched
sketches FAIL CLOSED (a cheater can't dodge the check with a sketch the verifier can't line up),
(3) the module imports and the CLI runs with no torch in the process.

Run: python3 -m pytest tests/test_challenge_seam.py -q
"""
import json
import math
import os
import random
import subprocess
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from shard.challenge import compare_sketches  # noqa: E402


def _sk(vals, norm=None, n=None):
    return {"n": len(vals) if n is None else n,
            "norm": math.sqrt(sum(v * v for v in vals)) if norm is None else norm,
            "proj": list(vals)}


def test_identical_sketches_pass():
    rng = random.Random(7)
    v = [rng.gauss(0, 1) for _ in range(256)]
    r = compare_sketches(_sk(v, n=10_000), _sk(v, n=10_000))
    assert r["passed"] and r["cosine"] > 0.9999 and r["rel_norm"] < 1e-9


def test_ulp_drift_passes_garbage_fails():
    rng = random.Random(11)
    v = [rng.gauss(0, 1) for _ in range(256)]
    drift = [x * (1 + rng.uniform(-1e-4, 1e-4)) for x in v]        # heterogeneous-hardware ULP noise
    garbage = [rng.gauss(0, 1) for _ in range(256)]                # a faked/wrong block output
    assert compare_sketches(_sk(v, n=10_000), _sk(drift, n=10_000))["passed"]
    assert not compare_sketches(_sk(v, n=10_000), _sk(garbage, n=10_000))["passed"]


def test_norm_cheat_fails():
    """right direction, wrong magnitude (e.g. a rescaled cheaper computation) -> rel_norm guard."""
    rng = random.Random(13)
    v = [rng.gauss(0, 1) for _ in range(256)]
    scaled = _sk(v, n=10_000)
    scaled["norm"] *= 1.2
    assert not compare_sketches(_sk(v, n=10_000), scaled)["passed"]


def test_malformed_and_mismatched_fail_closed():
    good = _sk([1.0, 2.0, 3.0], n=100)
    for bad in ({}, {"proj": [1, 2]}, {"n": 3, "norm": "x", "proj": [1, 2, 3]},
                {"n": 3, "norm": 1.0, "proj": None}, {"n": 3, "norm": 1.0, "proj": []},
                _sk([1.0, 2.0], n=100), _sk([1.0, 2.0, 3.0], n=999)):
        r = compare_sketches(good, bad)
        assert not r["passed"], f"malformed sketch passed: {bad}"
    assert not compare_sketches(_sk([]), _sk([]))["passed"]        # empty projections prove nothing


def test_matches_torch_compare_semantics():
    """same verdict as the torch path on the same sketch dicts (compare() now delegates)."""
    try:
        import torch  # noqa: F401
    except ImportError:
        return
    from shard.challenge import compare, sketch
    t = torch.randn(1, 8, 64)
    honest, garbage = sketch(t), sketch(torch.randn(1, 8, 64))
    assert compare(honest, dict(honest))["passed"] is True
    assert compare(honest, garbage)["passed"] is False
    r_pure, r_ref = compare_sketches(honest, garbage), compare(honest, garbage)
    assert abs(r_pure["cosine"] - r_ref["cosine"]) < 1e-5


def test_cli_torch_free_roundtrip():
    """the stdio seam works in a process where torch is unimportable (control-plane host)."""
    rng = random.Random(17)
    v = [rng.gauss(0, 1) for _ in range(256)]
    req = {"a": _sk(v, n=5000), "b": _sk(v, n=5000)}
    env = dict(os.environ, PYTHONPATH=_REPO)
    code = ("import sys; sys.modules['torch'] = None\n"           # torch import would raise instantly
            "from shard.challenge import _main; sys.exit(_main())")
    p = subprocess.run([sys.executable, "-c", code], input=json.dumps(req),
                       capture_output=True, text=True, env=env, cwd=_REPO)
    out = json.loads(p.stdout)
    assert p.returncode == 0 and out["passed"] is True
    # malformed request -> JSON error, exit 2 (matches the plan/verify seam contract)
    p = subprocess.run([sys.executable, "-c", code], input="not json",
                       capture_output=True, text=True, env=env, cwd=_REPO)
    assert p.returncode == 2 and "error" in json.loads(p.stdout)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("\nALL challenge-seam tests passed.")
