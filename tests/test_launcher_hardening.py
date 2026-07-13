"""Launcher hardening (audit cluster: scatter) — pure-python, no GPU, no network.

C1  — a PeerId is REMOTE stdout interpolated into root shell commands: strict base58btc
      validation + two-level shlex quoting of every multiaddr-carrying sidecar arg.
H1  — the launcher negotiates ONE context limit (min of operator ceiling and every stage's
      KV cap) and passes it to the gateway/coordinator; stage KV caps are always pinned.
C2  — sidecar neighbour allowlist (-allow), per-swarm token, loopback engine bind.
M4  — launch failures exit nonzero; readiness checks are nonce-gated (a stale log from a
      previous run can never satisfy them); one-shot pipelines preserve the engine's rc.
"""
import os
import shlex
import sys
import types

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "phase0"))
os.environ.setdefault("SHARD_PSK", "test-psk-not-real")   # launch_oss reads it at import (launch_libp2p dep)

import m25_scatter as msc                                  # noqa: E402
import m25_scatter_pipe as msp                             # noqa: E402
import launch_libp2p as llp                                # noqa: E402

VALID_PID = "12D3KooW" + "a" * 44                          # 52 chars, strict base58btc
INJ_PID = "12D3KooW';id>/tmp/pwned;'"                      # the audit exploit string
HOSTILE_MADDR = f"/ip4/1.2.3.4/tcp/29600/p2p/{INJ_PID}"


# ---------------------------------------------------------------- C1: PeerId validation

@pytest.mark.parametrize("mod", [msp, msc, llp])
def test_check_peerid_rejects_injection(mod):
    for bad in (INJ_PID, "", None, "short", "12D3KooW aaaa", "12D3KooW$(id)" + "a" * 40,
                "0" * 52, "O" * 52, "I" * 52, "l" * 52, "a" * 65):
        with pytest.raises(ValueError):
            mod.check_peerid(bad)


@pytest.mark.parametrize("mod", [msp, msc, llp])
def test_check_peerid_accepts_valid(mod):
    assert mod.check_peerid(VALID_PID) == VALID_PID


def _res(stdout):
    return types.SimpleNamespace(stdout=stdout, stderr="", returncode=0)


def test_peerid_parse_validates_remote_stdout(monkeypatch):
    hostile = f"PEERID {INJ_PID}\n"
    monkeypatch.setattr(msp, "sh", lambda *a, **k: _res(hostile))
    with pytest.raises(ValueError):
        msp.peerid("h", 22)
    monkeypatch.setattr(msc, "sh", lambda *a, **k: _res(hostile))
    with pytest.raises(ValueError):
        msc.peerid("h", 22)
    monkeypatch.setattr(llp, "rssh", lambda *a, **k: _res(hostile))
    with pytest.raises(ValueError):
        llp.peerid({"id": 1})


# ---------------------------------------------------------------- C1: sidecar cmd quoting

def _bash_c_payload(cmd):
    """extract the single argv the outer shell hands to `bash -c`."""
    toks = shlex.split(cmd)
    return toks[toks.index("-c") + 1]


def _assert_contained(cmd, hostile_value):
    inner = _bash_c_payload(cmd)
    toks = shlex.split(inner)
    # the hostile value survives as EXACTLY one argv token of the sidecar — never re-parsed
    assert any(t.endswith(hostile_value) for t in toks), toks
    # and nothing in the injection became a separate shell word
    assert "id>/tmp/pwned" not in " ".join(t for t in toks if hostile_value not in t)


def test_scatter_pipe_sidecar_cmd_quotes_hostile_maddr():
    fwd = f"127.0.0.1:29611={HOSTILE_MADDR}"
    cmd = msp.sidecar_cmd("/ip4/1.2.3.4/tcp/29600", "127.0.0.1:29610", [fwd])
    _assert_contained(cmd, HOSTILE_MADDR)


def test_scatter_sidecar_cmd_quotes_hostile_maddr():
    fwd = f"127.0.0.1:29611={HOSTILE_MADDR}"
    cmd = msc.sidecar_cmd("/ip4/1.2.3.4/tcp/29600", "127.0.0.1:29610", fwd)
    _assert_contained(cmd, HOSTILE_MADDR)


def test_libp2p_sidecar_cmd_quotes_hostile_maddr():
    fwd = f"127.0.0.1:29611={HOSTILE_MADDR}"
    cmd = llp.sidecar_cmd("/ip4/1.2.3.4/tcp/29600", "127.0.0.1:29610", [fwd],
                          dht_bootstrap=[HOSTILE_MADDR])
    _assert_contained(cmd, HOSTILE_MADDR)


# ---------------------------------------------------------------- main() drive harness

class Drive:
    """run msp.main() with every remote op stubbed; record what the launcher would execute."""

    def __init__(self, monkeypatch, argv, sh_rc=0, sidecar_ok=True, warm_ok=True):
        self.sh_cmds = []            # (host, cmd) of every sh() the launcher runs
        self.stage_calls = []        # (args, kwargs) per launch_stage
        self.sidecar_calls = []      # (args, kwargs) per launch_sidecar
        self.exit_code = None
        pids = {}

        def fake_vinst(iid):
            return {"ssh_host": f"h{iid}", "ssh_port": 22, "public_ipaddr": "1.2.3.4",
                    "ports": {"29600/tcp": [{"HostPort": "29600"}]}}

        def fake_peerid(host, port):
            return pids.setdefault(host, "12D3KooW" + host.ljust(44, "x"))

        def fake_sh(host, port, cmd, timeout=120):
            self.sh_cmds.append((host, cmd))
            return _res("GPU\n" if "nvidia-smi" in cmd else "")

        monkeypatch.setattr(msp, "vinst", fake_vinst)
        monkeypatch.setattr(msp, "push_code", lambda h, p: None)
        monkeypatch.setattr(msp, "peerid", fake_peerid)
        monkeypatch.setattr(msp, "sh", fake_sh)
        monkeypatch.setattr(msp, "launch_sidecar",
                            lambda *a, **k: (self.sidecar_calls.append((a, k)), sidecar_ok)[1])
        monkeypatch.setattr(msp, "launch_stage",
                            lambda *a, **k: (self.stage_calls.append((a, k)), "stubnonce")[1])
        monkeypatch.setattr(msp, "warm", lambda *a, **k: warm_ok)
        monkeypatch.setattr(msp.time, "sleep", lambda s: None)
        if hasattr(msp, "fresh_count"):
            monkeypatch.setattr(msp, "fresh_count", lambda *a, **k: "1" if sh_rc == 0 else "0")
        monkeypatch.setattr(sys, "argv", ["m25_scatter_pipe.py"] + argv)
        monkeypatch.setenv("M25_CWND_KEEPWARM_MS", "150")   # keep main() from mutating real env
        try:
            msp.main()
        except SystemExit as e:
            self.exit_code = e.code

    def gw_cmd(self):
        return next(c for _, c in self.sh_cmds if "m25_gateway.py" in c)

    def coord_cmd(self):
        return next(c for _, c in self.sh_cmds if "m25_pipe.py coord" in c)


ORDER3 = ["--order", "A:1:0:20", "B:2:20:41", "C:3:41:62"]


# ---------------------------------------------------------------- H1: one negotiated ctx limit

def test_negotiated_max_ctx():
    assert msp.negotiated_max_ctx(131072, [40960, 40960]) == 40960
    assert msp.negotiated_max_ctx(16384, [40960, 40960]) == 16384
    assert msp.negotiated_max_ctx(131072, [0, 40960]) == 40960     # zero caps filtered
    assert msp.negotiated_max_ctx(131072, []) == 131072


def test_launcher_pins_kv_and_gateway_max_ctx(monkeypatch):
    d = Drive(monkeypatch, ORDER3 + ["--serve"])
    # every stage gets the pinned KV cap (40960 = m25_stage default, now EXPLICIT config)
    for args, kwargs in d.stage_calls:
        assert 40960 in args or kwargs.get("kv_maxlen") == 40960
    assert "--max-ctx 40960" in d.gw_cmd()


def test_launcher_negotiates_operator_ceiling(monkeypatch):
    d = Drive(monkeypatch, ORDER3 + ["--serve", "--max-ctx", "16384"])
    assert "--max-ctx 16384" in d.gw_cmd()
    d = Drive(monkeypatch, ORDER3 + ["--serve", "--kv-maxlen", "12288"])
    assert "--max-ctx 12288" in d.gw_cmd()


def test_oneshot_coord_gets_max_ctx(monkeypatch):
    d = Drive(monkeypatch, ORDER3)
    assert "--max-ctx 40960" in d.coord_cmd()


def test_sidecar_cmd_benign_values_unchanged():
    """for legit values the built command is the same shape as the old literal one-quote form."""
    fwd = f"127.0.0.1:29611=/ip4/5.6.7.8/tcp/29600/p2p/{VALID_PID}"
    cmd = msp.sidecar_cmd("/ip4/1.2.3.4/tcp/29600", "127.0.0.1:29610", [fwd])
    assert f"-forward {fwd} " in cmd            # benign values pass through unquoted
    assert cmd.count("bash -c '") == 1
