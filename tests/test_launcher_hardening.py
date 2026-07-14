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
        self.pushed_assignments = None
        self.exit_code = None
        pids = {}

        def fake_vinst(iid):
            return {"ssh_host": f"h{iid}", "ssh_port": 22, "public_ipaddr": "1.2.3.4",
                    "ports": {"29600/tcp": [{"HostPort": "29600"}]}}

        def fake_peerid(host, port):
            return pids.setdefault(host, "12D3KooW" + host.ljust(44, "x"))

        def fake_sh(host, port, cmd, timeout=120):
            self.sh_cmds.append((host, cmd))
            r = _res("GPU\n" if "nvidia-smi" in cmd else "")
            if "m25_pipe.py coord" in cmd:
                r.returncode = sh_rc
            return r

        monkeypatch.setattr(msp, "vinst", fake_vinst)
        monkeypatch.setattr(msp, "push_code", lambda h, p: None)
        monkeypatch.setattr(msp, "peerid", fake_peerid)
        monkeypatch.setattr(msp, "sh", fake_sh)
        monkeypatch.setattr(msp, "launch_sidecar",
                            lambda *a, **k: (self.sidecar_calls.append((a, k)), sidecar_ok)[1])
        monkeypatch.setattr(msp, "launch_stage",
                            lambda *a, **k: (self.stage_calls.append((a, k)), "stubnonce")[1])
        monkeypatch.setattr(msp, "warm", lambda *a, **k: warm_ok)
        if hasattr(msp, "collect_assignments"):
            monkeypatch.setattr(msp, "collect_assignments", lambda nodes: {"PUB=": [0, 62]})
            monkeypatch.setattr(msp, "push_assignments",
                                lambda head, m: setattr(self, "pushed_assignments", (head["host"], m)))
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


# ---------------------------------------------------------------- C2: allowlist + token + bind

def test_sidecar_allowlist_is_neighbour_pinned(monkeypatch):
    d = Drive(monkeypatch, ORDER3 + ["--serve"])
    allows = [k.get("allow") for _, k in d.sidecar_calls]
    pid = {h: "12D3KooW" + h.ljust(44, "x") for h in ("h1", "h2", "h3")}
    assert allows[0] is None                              # head has no -inbound, no allowlist
    assert allows[1] == [pid["h1"]]                       # middle: predecessor only
    assert allows[2] == [pid["h2"], pid["h1"]]            # tail: predecessor + HEAD (return tunnel)


def test_sidecar_cmd_renders_allow_flags():
    cmd = msp.sidecar_cmd("/ip4/1.2.3.4/tcp/29600", "127.0.0.1:29610", [], allow=[VALID_PID])
    assert f"-allow {VALID_PID}" in _bash_c_payload(cmd)
    cmd = msp.sidecar_cmd("/ip4/1.2.3.4/tcp/29600", "127.0.0.1:29610", [])
    assert "-allow" not in cmd                            # unset = open (legacy)


def test_sidecar_cmd_renders_nat_flags():
    """Home/NAT'd-GPU reach: -quic (udp/quic-v1 listener), -relay (be a public relay), -relays <csv>
    (reserve on relays when NAT'd) surface only when asked. Token-exact checks (split, not substring)
    so -relays never masquerades as -relay; an untouched call stays the legacy tcp-only form."""
    toks = lambda **kw: _bash_c_payload(
        msp.sidecar_cmd("/ip4/1.2.3.4/tcp/29600", "", [], **kw)).split()
    assert "-quic" in toks(quic=True)
    assert "-relay" in toks(relay=True) and "-relays" not in toks(relay=True)
    relays = "/ip4/9.9.9.9/tcp/29600/p2p/12D3KooWrelay,/ip4/8.8.8.8/tcp/29600/p2p/12D3KooWx"
    t = toks(relays=relays)
    assert "-relays" in t and t[t.index("-relays") + 1] == relays and "-relay" not in t
    base = toks()
    assert "-quic" not in base and "-relay" not in base and "-relays" not in base


def test_stage_cmd_token_and_loopback_bind():
    cmd = msp.stage_cmd(1, 3, 20, 41, False, token="deadbeef")
    assert "SHARD_SWARM_TOKEN=deadbeef " in cmd
    assert "M25_ENGINE_BIND=127.0.0.1 " in cmd
    assert "SHARD_SWARM_TOKEN" not in msp.stage_cmd(1, 3, 20, 41, False)   # unset = legacy


def test_swarm_token_reaches_stages_and_gateway_unprinted(monkeypatch, capsys):
    d = Drive(monkeypatch, ORDER3 + ["--serve"])
    toks = {k.get("token") for _, k in d.stage_calls}
    assert len(toks) == 1
    tok = toks.pop()
    assert tok and len(tok) == 32                         # secrets.token_hex(16)
    assert f"SHARD_SWARM_TOKEN={tok} " in d.gw_cmd()      # gateway gets the SAME epoch token
    assert tok not in capsys.readouterr().out             # never printed in the banner


def test_swarm_token_reaches_oneshot_coord(monkeypatch):
    d = Drive(monkeypatch, ORDER3)
    tok = d.stage_calls[0][1]["token"]
    assert f"SHARD_SWARM_TOKEN={tok} " in d.coord_cmd()


def test_warm_only_stays_tokenless(monkeypatch):
    d = Drive(monkeypatch, ORDER3 + ["--warm-only"])
    assert all(k.get("token") is None for _, k in d.stage_calls)


# ---------------------------------------------------------------- M4: fail loud, fresh readiness

def test_sidecar_failure_exits_nonzero(monkeypatch):
    d = Drive(monkeypatch, ORDER3 + ["--serve"], sidecar_ok=False)
    assert d.exit_code not in (None, 0)


def test_warm_failure_exits_nonzero(monkeypatch):
    d = Drive(monkeypatch, ORDER3 + ["--serve"], warm_ok=False)
    assert d.exit_code not in (None, 0)


def test_coord_failure_propagates_rc(monkeypatch):
    d = Drive(monkeypatch, ORDER3, sh_rc=1)
    assert d.exit_code not in (None, 0)
    assert d.coord_cmd().startswith("set -o pipefail;")   # engine rc survives the tee|grep pipeline


def test_gateway_failure_exits_nonzero(monkeypatch):
    d = Drive(monkeypatch, ORDER3 + ["--serve"], sh_rc=1)  # fresh_count stub reports never-up
    assert d.exit_code not in (None, 0)


def test_success_exits_zero(monkeypatch):
    d = Drive(monkeypatch, ORDER3 + ["--serve"])
    assert d.exit_code in (None, 0)
    d = Drive(monkeypatch, ORDER3)
    assert d.exit_code in (None, 0)


@pytest.mark.parametrize("mod", [msp, msc])
def test_readiness_checks_are_nonce_gated(mod, monkeypatch):
    """the readiness probe must refuse a matching log line unless THIS launch's nonce is on the box
    — a stale log from a previous run (launch ssh flaked before the rm) can no longer read WARM."""
    seen = []
    monkeypatch.setattr(mod, "sh", lambda h, p, cmd, t=20: (seen.append(cmd), _res("7"))[1])
    assert mod.fresh_count("h", 22, "/root/stage.nonce", "abcd1234", "/root/stage.log", "WARM") == "7"
    q = seen[0]
    assert '[ "$(cat /root/stage.nonce 2>/dev/null)" = "abcd1234" ] &&' in q


@pytest.mark.parametrize("mod", [msp, msc])
def test_launch_cmds_write_nonce_after_log_reset(mod):
    if mod is msp:
        sc = mod.sidecar_cmd("/ip4/1.2.3.4/tcp/29600", "", [], nonce="n1")
        st = mod.stage_cmd(0, 3, 0, 20, False, nonce="n2")
    else:
        sc = mod.sidecar_cmd("/ip4/1.2.3.4/tcp/29600", "", "", nonce="n1")
        st = None
    assert "rm -f /root/sidecar.log; echo n1 > /root/sidecar.nonce; " in sc
    if st is not None:
        assert "rm -f /root/stage.log; echo n2 > /root/stage.nonce; " in st


# ---------------------------------------------------------------- H4: frame-timeout config

def test_sidecar_cmd_frame_timeout():
    cmd = msp.sidecar_cmd("/ip4/1.2.3.4/tcp/29600", "", [])
    assert "-frame-timeout 60" in _bash_c_payload(cmd)          # defaults ON at 60s
    cmd = msp.sidecar_cmd("/ip4/1.2.3.4/tcp/29600", "", [], frame_timeout=0)
    assert "-frame-timeout 0" in _bash_c_payload(cmd)           # 0 = explicit opt-out


def test_frame_timeout_env_forwarded_to_engines(monkeypatch):
    assert "M25_FRAME_TIMEOUT" in msp.ENG_ENV                   # stages+gateway+coord get the same number
    monkeypatch.setenv("M25_FRAME_TIMEOUT", "45")
    assert "M25_FRAME_TIMEOUT=45 " in msp.eng_env()


def test_launch_sidecar_reads_operator_frame_timeout(monkeypatch):
    seen = []
    monkeypatch.setattr(msp, "sh", lambda h, p, cmd, t=30: (seen.append(cmd), _res(""))[1])
    monkeypatch.setattr(msp, "fresh_count",
                        lambda h, p, nf, n, log, pattern: "1" if "tunnel" in pattern else "0")
    monkeypatch.setattr(msp.time, "sleep", lambda s: None)
    monkeypatch.setenv("M25_FRAME_TIMEOUT", "45")
    assert msp.launch_sidecar("h", 22, "/ip4/1.2.3.4/tcp/29600", "", [])
    assert "-frame-timeout 45" in seen[0]


# ---------------------------------------------------------------- H2: assignment-map plumbing

def _pub(i):
    return "A" * 42 + f"k{i}="                                  # base64-shaped, distinct per node


def test_collect_assignments(monkeypatch):
    monkeypatch.setattr(msp, "sh", lambda h, p, cmd, t=60: _res(f"{_pub(h)}\n"))
    nodes = [{"region": "A", "host": 1, "port": 22, "lo": 0, "hi": 31},
             {"region": "B", "host": 2, "port": 22, "lo": 31, "hi": 62}]
    assert msp.collect_assignments(nodes) == {_pub(1): [0, 31], _pub(2): [31, 62]}


def test_collect_assignments_rejects_hostile_pubkey(monkeypatch):
    monkeypatch.setattr(msp, "sh", lambda h, p, cmd, t=60: _res("PUB'; id #\n"))
    with pytest.raises(RuntimeError, match="bad receipt pubkey"):
        msp.collect_assignments([{"region": "A", "host": 1, "port": 22, "lo": 0, "hi": 62}])


def test_collect_assignments_rejects_duplicate_signer(monkeypatch):
    monkeypatch.setattr(msp, "sh", lambda h, p, cmd, t=60: _res(f"{_pub(0)}\n"))
    nodes = [{"region": "A", "host": 1, "port": 22, "lo": 0, "hi": 31},
             {"region": "B", "host": 2, "port": 22, "lo": 31, "hi": 62}]
    with pytest.raises(RuntimeError, match="duplicate receipt pubkey"):
        msp.collect_assignments(nodes)


def test_receipts_run_pins_settlement(monkeypatch):
    d = Drive(monkeypatch, ORDER3 + ["--serve", "--receipts"])
    assert d.pushed_assignments == ("h1", {"PUB=": [0, 62]})    # map lands on the HEAD box
    assert "SHARD_ASSIGNMENTS=/root/assignments.json " in d.gw_cmd()
    d = Drive(monkeypatch, ORDER3 + ["--receipts"])
    assert "SHARD_ASSIGNMENTS=/root/assignments.json " in d.coord_cmd()


def test_no_receipts_run_stays_unpinned(monkeypatch):
    d = Drive(monkeypatch, ORDER3 + ["--serve"])
    assert d.pushed_assignments is None
    assert "SHARD_ASSIGNMENTS" not in d.gw_cmd()


def test_sidecar_cmd_benign_values_unchanged():
    """for legit values the built command is the same shape as the old literal one-quote form."""
    fwd = f"127.0.0.1:29611=/ip4/5.6.7.8/tcp/29600/p2p/{VALID_PID}"
    cmd = msp.sidecar_cmd("/ip4/1.2.3.4/tcp/29600", "127.0.0.1:29610", [fwd])
    assert f"-forward {fwd} " in cmd            # benign values pass through unquoted
    assert cmd.count("bash -c '") == 1
