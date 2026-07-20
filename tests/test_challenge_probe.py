"""Node-side challenge probe (P0-#1) — the loopback door a stage answers spot-checks on.

Drives the REAL serve() on CPU (compute stubbed to identity, the frame-stall harness pattern) and
speaks the probe protocol as the daemon would. The properties pinned here are the adversarial
pass's survivors:

  * probes run IN the serve thread (select-integrated) — so an idle stage answers, a MID-JOB probe
    is refused `busy` (a probe forward would overwrite KV[0:n) of a paying job that still emits
    valid receipts — the unforgivable failure), and every job-termination path re-opens the door;
  * the door is fail-closed: no SHARD_PROBE_TOKEN -> no listener at all; wrong token -> bad_token;
  * a probe never disturbs the ring: job frames flow identically after one;
  * the sketch a stage returns lines up with the verifier's own compare (commit-first proj_seed).

Run: python3 -m pytest tests/test_challenge_probe.py -q
"""
import os
import socket
import sys
import threading
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

torch = pytest.importorskip("torch")
fr = pytest.importorskip("fake_ring")               # bootstraps env + m25_pipe on CPU

MP = fr.MP
send_msg, recv_msg = fr.send_msg, fr.recv_msg
from challenge import compare_sketches, derive_challenge, sketch, sketch_seed  # noqa: E402

TOKEN = "probe-secret-for-tests"


class _FakeLayer:
    def reset(self):
        pass


def _start_middle(monkeypatch, timeout=1, token=TOKEN):
    """Real serve() middle stage on CPU: engine compute stubbed to identity (job path AND probe
    path), probe token injected. Returns (pred_port, probe_port, nxt_srv)."""
    monkeypatch.setattr(MP, "dev", "cpu")
    monkeypatch.setattr(MP, "RECEIPTS", False)
    monkeypatch.setattr(MP, "PROBE_TOKEN", token)
    monkeypatch.setattr(MP, "_load",
                        lambda stage, nstages, lo, hi: {"layers": [_FakeLayer()], "head": False, "tail": False})
    monkeypatch.setattr(MP, "_block", lambda grs, layers, start, x, vcfg: x)
    monkeypatch.setattr(MP, "_probe_forward", lambda layers, x, vcfg: x)   # identity block
    monkeypatch.setattr(MP.S, "_CTX", (None, None), raising=False)
    monkeypatch.setattr(MP.S, "M25_EAGLE", False, raising=False)
    monkeypatch.setattr(MP.S, "M25_STAGE_TIMING", False, raising=False)
    nxt_srv = socket.socket()
    nxt_srv.bind(("127.0.0.1", 0)); nxt_srv.listen(2)
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0)); port = srv.getsockname()[1]; srv.close()
    pp = socket.socket()                             # explicit free probe port: port+3 can collide
    pp.bind(("127.0.0.1", 0)); probe_port = pp.getsockname()[1]; pp.close()
    monkeypatch.setenv("SHARD_PROBE_PORT", str(probe_port))
    t = threading.Thread(target=MP.serve,
                         args=(1, 3, 12, 24, port, f"127.0.0.1:{nxt_srv.getsockname()[1]}", timeout),
                         daemon=True)
    t.start()
    return port, probe_port, nxt_srv


def _dial(port, timeout=3):
    deadline = time.monotonic() + timeout
    while True:
        try:
            s = socket.create_connection(("127.0.0.1", port), timeout=timeout)
            s.settimeout(timeout)
            return s
        except OSError:
            if time.monotonic() > deadline:
                raise
            time.sleep(0.02)


def _probe(port, req):
    s = _dial(port)
    try:
        send_msg(s, req)
        return recv_msg(s)
    finally:
        s.close()


def _req(seed=None, proj_seed=None, n=4, lo=12, hi=24, token=TOKEN, **extra):
    r = {"op": "challenge", "token": token, "proj_seed": proj_seed or sketch_seed(),
         "n_tokens": n, "lo": lo, "hi": hi}
    if seed is not None:
        r["seed"] = seed
    r.update(extra)
    return r


# ---- the honest path: an idle stage answers, and the sketch lines up verifier-side --------------

def test_idle_stage_answers_and_sketch_verifies(monkeypatch):
    _, probe_port, nxt = _start_middle(monkeypatch)
    seed, proj = "chal-seed-1", sketch_seed()
    out = _probe(probe_port, _req(seed=seed, proj_seed=proj))
    assert out.get("ok") == 1 and out["lo"] == 12 and out["hi"] == 24
    # identity block -> expected output IS the derived input; the verifier's own sketch must agree
    expected = sketch(derive_challenge(seed, 4, MP.S.H, device="cpu"), seed=proj)
    assert compare_sketches(out["sketch"], expected)["passed"]
    nxt.close()


def test_explicit_x_input_bank_mode(monkeypatch):
    _, probe_port, nxt = _start_middle(monkeypatch)
    proj = sketch_seed()
    x = torch.randn(2, MP.S.H)
    out = _probe(probe_port, _req(proj_seed=proj, n=2, x=x.tolist()))
    assert out.get("ok") == 1
    expected = sketch(x.unsqueeze(0).to(torch.bfloat16), seed=proj)
    assert compare_sketches(out["sketch"], expected)["passed"]
    nxt.close()


# ---- busy semantics: the KV-poison gate ---------------------------------------------------------

def test_mid_job_probe_refused_then_reopens_on_receipt(monkeypatch):
    port, probe_port, nxt = _start_middle(monkeypatch)
    # idle first: door open
    assert _probe(probe_port, _req(seed="s0")).get("ok") == 1
    pred = _dial(port)
    fwd, _ = nxt.accept(); fwd.settimeout(5)
    send_msg(pred, {"op": "reset"})
    assert recv_msg(fwd)["op"] == "reset"            # job armed on this stage
    assert _probe(probe_port, _req(seed="s1")) == {"error": "busy"}
    send_msg(pred, {"op": "receipt", "receipts": []})
    assert recv_msg(fwd)["op"] == "receipt"          # job done
    assert _probe(probe_port, _req(seed="s2")).get("ok") == 1
    pred.close(); fwd.close(); nxt.close()


def test_job_error_reopens_the_door(monkeypatch):
    port, probe_port, nxt = _start_middle(monkeypatch)
    pred = _dial(port)
    fwd, _ = nxt.accept(); fwd.settimeout(5)
    send_msg(pred, {"op": "reset"})
    assert recv_msg(fwd)["op"] == "reset"
    assert _probe(probe_port, _req(seed="s1")) == {"error": "busy"}
    send_msg(pred, {"op": "job_error", "error": {"code": "kv_overflow", "message": "x"}})
    assert recv_msg(fwd)["op"] == "job_error"        # relayed down-ring; job dead here
    assert _probe(probe_port, _req(seed="s2")).get("ok") == 1
    pred.close(); fwd.close(); nxt.close()


# ---- the probe never disturbs the ring ----------------------------------------------------------

def test_job_frames_flow_identically_after_a_probe(monkeypatch):
    port, probe_port, nxt = _start_middle(monkeypatch)
    assert _probe(probe_port, _req(seed="s0")).get("ok") == 1
    pred = _dial(port)
    fwd, _ = nxt.accept(); fwd.settimeout(5)
    h = torch.zeros(1, 2, 4)
    send_msg(pred, {"op": "verify", "h": h, "start": 0})
    out = recv_msg(fwd)
    assert out["op"] == "verify" and out["start"] == 0
    assert torch.equal(out["h"], h)                  # identity stage: bytes unchanged post-probe
    pred.close(); fwd.close(); nxt.close()


# ---- fail-closed edges --------------------------------------------------------------------------

def test_no_token_means_no_listener(monkeypatch):
    port, probe_port, nxt = _start_middle(monkeypatch, token=None)
    _dial(port).close()                              # engine port IS up (stage serving normally)
    with pytest.raises(OSError):
        socket.create_connection(("127.0.0.1", probe_port), timeout=0.5)
    nxt.close()


def test_wrong_token_rejected(monkeypatch):
    _, probe_port, nxt = _start_middle(monkeypatch)
    assert _probe(probe_port, _req(seed="s", token="wrong")) == {"error": "bad_token"}
    nxt.close()


def test_range_mismatch_rejected(monkeypatch):
    _, probe_port, nxt = _start_middle(monkeypatch)
    out = _probe(probe_port, _req(seed="s", lo=0, hi=12))
    assert out == {"error": "range_mismatch", "lo": 12, "hi": 24}
    nxt.close()


def test_malformed_challenges_rejected_stage_survives(monkeypatch):
    _, probe_port, nxt = _start_middle(monkeypatch)
    bad_n = _req(seed="s"); bad_n["n_tokens"] = 0
    assert "bad_challenge" in _probe(probe_port, bad_n)["error"]
    no_seed = _req(); no_seed.pop("seed", None)
    assert "bad_challenge" in _probe(probe_port, no_seed)["error"]
    no_proj = _req(seed="s"); no_proj["proj_seed"] = ""
    assert "bad_challenge" in _probe(probe_port, no_proj)["error"]
    # junk op: silent close, and the door still answers a good probe after all of it
    s = _dial(probe_port); send_msg(s, {"op": "verify", "h": [1]}); s.close()
    assert _probe(probe_port, _req(seed="s2")).get("ok") == 1
    nxt.close()
