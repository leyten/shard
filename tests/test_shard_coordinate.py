"""python -m shard.coordinate — the daemon's serving exec target (leg 8, node half).

In-process: serve_jobs() against the fake-ring oracle — the stdout contract, the delta-stream/
response equality the server half relies on (joined swarm:job_token deltas == swarm:job_complete
response), settlement-nonce threading into the reset op (what stages sign), and job-fault
isolation (a bad line never kills the loop). Subprocess: the --check preflight contract with a
clean env (no PYTHONPATH, no M25_*/SHARD_* leakage), mirroring test_shard_stage.py."""
import argparse
import json
import os
import socket
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fake_ring as FR                                    # noqa: E402  (bootstraps env + sys.path)
from fake_ring import FakeRing, FakeTok, repetitive_T    # noqa: E402

import shard.coordinate as C                              # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _args(**kw):
    d = dict(K=8, depth=4, ngram_n=3, max_ctx=0, timeout=30)
    d.update(kw)
    return argparse.Namespace(**d)


def _run_jobs(T, prompt_len, lines, max_ring_jobs=4):
    """Drive serve_jobs over a fresh fake ring; returns (emits, ring)."""
    c_pipe, r_pipe = socket.socketpair()
    c_ret, r_ret = socket.socketpair()
    c_ret.settimeout(30)
    ring = FakeRing(r_pipe, r_ret, T)
    ring.tail_slack = 4
    ring.start()
    tok = FakeTok(T[:prompt_len])
    emits = []

    def emit(tag, **fields):
        emits.append((tag, fields))

    try:
        rc = C.serve_jobs(FR.MP, tok, c_pipe, c_ret, _args(), iter(lines), emit=emit)
    finally:
        for s in (c_pipe, c_ret):
            try: s.close()
            except OSError: pass
        ring.join(2)
    return rc, emits, ring


def test_serve_jobs_streams_and_completes():
    T = repetitive_T(400)
    job = {"jobId": "j-1", "swarmId": "sw-1", "nonce": "aa" * 16, "maxNew": 64,
           "messages": [{"role": "user", "content": "fake"}]}
    rc, emits, ring = _run_jobs(T, 40, [json.dumps(job)])
    assert rc == 0
    done = [f for t, f in emits if t == "SHARD_JOB_DONE"]
    assert len(done) == 1 and done[0]["ok"] and done[0]["jobId"] == "j-1"
    assert done[0]["tokensGenerated"] == 64
    assert done[0]["nonce"] == "aa" * 16
    # the server-half contract: joined swarm:job_token deltas == the complete response
    deltas = [f["delta"] for t, f in emits if t == "SHARD_JOB_TOKEN"]
    assert deltas and "".join(deltas) == done[0]["response"]
    # losslessness against the oracle: the response is exactly T's continuation
    assert done[0]["response"] == FakeTok(T).decode(T[40:40 + 64])


def test_settlement_nonce_threads_into_the_reset():
    T = repetitive_T(300)
    job = {"jobId": "job-77", "swarmId": "swarm-9", "nonce": "beef" * 8, "maxNew": 16,
           "messages": [{"role": "user", "content": "fake"}]}
    rc, emits, ring = _run_jobs(T, 30, [json.dumps(job)])
    assert rc == 0
    resets = [e for e in ring.log if e.get("op") == "reset"]
    assert resets and resets[0]["nonce"] == "beef" * 8      # stages sign THE SERVER'S nonce
    assert resets[0]["swarm_id"] == "swarm-9" and resets[0]["job_id"] == "job-77"


def test_bad_job_line_does_not_kill_the_loop():
    T = repetitive_T(300)
    good = {"jobId": "j-2", "nonce": "cc" * 16, "maxNew": 8,
            "messages": [{"role": "user", "content": "fake"}]}
    rc, emits, ring = _run_jobs(T, 30, ["{not json", json.dumps(good)])
    assert rc == 0
    tags = [t for t, _ in emits]
    assert "SHARD_JOB_FATAL" in tags                        # the bad line reported...
    done = [f for t, f in emits if t == "SHARD_JOB_DONE"]
    assert len(done) == 1 and done[0]["jobId"] == "j-2"     # ...and the next job still served


def _clean_env():
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("M25_", "SHARD_")) and k != "PYTHONPATH"}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def test_cli_check_contract():
    r = subprocess.run([sys.executable, "-m", "shard.coordinate", "--check",
                        "--dir", os.environ["M25_DIR"]],
                       capture_output=True, text=True, cwd=REPO, env=_clean_env(), timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
    ok = [l for l in r.stdout.splitlines() if l.startswith("SHARD_COORD_OK ")]
    assert ok and json.loads(ok[0].split(" ", 1)[1])["transport"] == "libp2p"


def test_cli_missing_model_dir_is_fatal():
    r = subprocess.run([sys.executable, "-m", "shard.coordinate", "--check"],
                       capture_output=True, text=True, cwd=REPO, env=_clean_env(), timeout=120)
    assert r.returncode == 1
    assert any(l.startswith("SHARD_JOB_FATAL ") for l in r.stdout.splitlines())
