"""python -m shard.stage — the daemon-facing stage entrypoint (c0mpute NODE_DAEMON.md §4).

Pins the two things Leg 7 needs from the engine side:
1. LAYOUT PORTABILITY (LAUNCH.md P0-#2): a stage must come up from a plain repo checkout with NO
   hand-set PYTHONPATH — the flat `import transport` in node_kv (needs shard/transport.py) and the
   /root/.hf_token hardcode both broke the first residential join. The subprocess tests here run
   with PYTHONPATH stripped, so a regression to flat-only imports fails them.
2. THE STDOUT CONTRACT the node daemon supervises: SHARD_STAGE_OK (preflight) / SHARD_STAGE_READY
   (weights loaded + listening, emitted by serve()) / SHARD_STAGE_FATAL + nonzero exit.

Run: python3 -m pytest tests/test_shard_stage.py -q
"""
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

torch = pytest.importorskip("torch")
fr = pytest.importorskip("fake_ring")               # bootstraps fake M25_DIR + m25_pipe on CPU

MP = fr.MP
send_msg, recv_msg = fr.send_msg, fr.recv_msg

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _clean_env():
    """The stranger's-box environment: no PYTHONPATH, no engine env — only what the entrypoint
    itself sets up. (A pass under this env is exactly what the flat vast layout never proved.)"""
    env = {k: v for k, v in os.environ.items()
           if k != "PYTHONPATH" and not k.startswith(("M25_", "SHARD_"))}
    return env


def _run_stage(args, timeout=120):
    return subprocess.run([sys.executable, "-m", "shard.stage", *args],
                          cwd=_REPO, env=_clean_env(), capture_output=True, text=True,
                          timeout=timeout)


def _tagged(out, tag):
    for line in out.splitlines():
        if line.startswith(tag + " "):
            return json.loads(line[len(tag) + 1:])
    raise AssertionError(f"no {tag} line in output:\n{out}")


@pytest.mark.integration
def test_check_standalone_no_pythonpath():
    """--check from a repo checkout, clean env: the full engine import (m25_pipe -> m25_stage ->
    node_kv -> transport) must succeed with no PYTHONPATH — node_kv's flat `import transport`
    falls back to shard.transport. This is the P0-#2 regression gate."""
    r = _run_stage(["--check", "--dir", fr._fake_model_dir()])
    assert r.returncode == 0, f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}"
    ok = _tagged(r.stdout, "SHARD_STAGE_OK")
    assert ok["transport"] == "libp2p"              # the mode default applied without operator env
    assert isinstance(ok["cuda"], bool)             # reported, never required, for --check


@pytest.mark.integration
def test_fatal_contract_bad_model_dir():
    """A useless model dir must fail LOUD and machine-readable, not deep in a torch stack."""
    r = _run_stage(["--check", "--dir", tempfile.mkdtemp(prefix="m25_empty_")])
    assert r.returncode != 0
    f = _tagged(r.stdout, "SHARD_STAGE_FATAL")
    assert "config.json" in f["error"]


@pytest.mark.integration
def test_fatal_contract_serve_needs_assignment():
    """Serving without the assignment flags is a supervisor bug — surfaced via the contract."""
    r = _run_stage(["--dir", fr._fake_model_dir()])
    assert r.returncode != 0
    f = _tagged(r.stdout, "SHARD_STAGE_FATAL")
    assert "--stage" in f["error"]


def test_ready_line_then_frames_flow(monkeypatch, capsys):
    """The CLI drives the REAL serve() loop (compute stubbed like test_stage_frame_stall): the
    READY line must carry the assignment, and the port it names must accept + forward a frame."""
    monkeypatch.setattr(MP, "dev", "cpu")
    monkeypatch.setattr(MP, "RECEIPTS", False)
    monkeypatch.setattr(MP, "_load", lambda stage, nstages, lo, hi:
                        {"layers": [], "head": False, "tail": False})
    monkeypatch.setattr(MP, "_block", lambda grs, layers, start, x, vcfg: x)
    monkeypatch.setattr(MP.S, "_CTX", (None, None), raising=False)
    monkeypatch.setattr(MP.S, "M25_EAGLE", False, raising=False)
    monkeypatch.setattr(MP.S, "M25_STAGE_TIMING", False, raising=False)
    # pre-set the env the CLI would fill, so the run leaks nothing across tests
    monkeypatch.setenv("SHARD_TRANSPORT", "libp2p")
    monkeypatch.setenv("M25_ENGINE_BIND", "127.0.0.1")
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import shard.stage as stage_cli
    nxt_srv = socket.socket()
    nxt_srv.bind(("127.0.0.1", 0)); nxt_srv.listen(2)
    srv = socket.socket(); srv.bind(("127.0.0.1", 0))
    port = srv.getsockname()[1]; srv.close()
    t = threading.Thread(target=stage_cli.main, daemon=True, args=([
        "--stage", "1", "--nstages", "3", "--lo", "3", "--hi", "5",
        "--port", str(port), "--next", f"127.0.0.1:{nxt_srv.getsockname()[1]}",
        "--dir", os.environ["M25_DIR"], "--timeout", "5"],))
    t.start()

    out, deadline = "", time.monotonic() + 15
    while "SHARD_STAGE_READY " not in out:
        assert time.monotonic() < deadline, f"no READY line; output so far:\n{out}"
        time.sleep(0.05)
        out += capsys.readouterr().out
    ready = json.loads(out.split("SHARD_STAGE_READY ", 1)[1].splitlines()[0])
    assert (ready["stage"], ready["nstages"], ready["lo"], ready["hi"]) == (1, 3, 3, 5)
    assert ready["port"] == port and ready["tail"] is False

    pred = socket.create_connection(("127.0.0.1", port), timeout=5)
    fwd, _ = nxt_srv.accept()                       # serve() dialed its forward leg at boot
    fwd.settimeout(5)
    send_msg(pred, {"op": "verify", "h": torch.zeros(1, 2, 4), "start": 0})
    fwd_msg = recv_msg(fwd)
    assert fwd_msg["op"] == "verify" and fwd_msg["start"] == 0
    for s in (pred, fwd, nxt_srv):
        try:
            s.close()
        except OSError:
            pass


def test_hf_token_resolution_portable(monkeypatch, tmp_path):
    """m25_pull_range auth order: env HF_TOKEN wins; else ~/.hf_token; else NOTHING (no crash —
    the old code open()'d /root/.hf_token unconditionally and died on every non-vast box)."""
    import m25_pull_range as pr
    monkeypatch.setenv("HF_TOKEN", "env-wins")
    (tmp_path / ".hf_token").write_text("from-home-file\n")
    monkeypatch.setenv("HOME", str(tmp_path))
    pr._hf_token_env()
    assert os.environ["HF_TOKEN"] == "env-wins"

    monkeypatch.delenv("HF_TOKEN")
    pr._hf_token_env()
    assert os.environ["HF_TOKEN"] == "from-home-file"

    monkeypatch.delenv("HF_TOKEN")
    monkeypatch.setenv("HOME", str(tmp_path / "nowhere"))
    pr._hf_token_env()                              # falls through to hf's own login chain
    assert "HF_TOKEN" not in os.environ
