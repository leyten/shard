"""A peer that cannot beat the throughput floor is abandoned for the mirror.

Regression for the 2026-07-25 live incident: a 5090 with a 48.9 MB/s path to the mirror pulled a
5 GB shard from a residential seeder at ~2.5 MB/s instead — ~33 minutes against a 30-minute
deadline, with progress-gated retry keeping the tarpit alive. Peers-before-mirror is only a win
when the peer is competitive, and nothing measured that.

No GPUs, no network: a stub "sidecar" script writes into the .part file at a controlled rate.
"""
import os
import stat
import subprocess
import textwrap

import pytest

from shard.fetch import Libp2pProvider, ProviderUnavailable


def _stub_sidecar(tmp_path, mbps: float):
    """A fake sidecar that streams into '<dest>.p2p.<pid>.<peer>.part' at ~mbps, then exits 0."""
    p = tmp_path / "stub_sidecar"
    p.write_text(textwrap.dedent(f"""\
        #!/usr/bin/env python3
        import os, sys, time
        dest = sys.argv[sys.argv.index("-fetch-out") + 1]
        total = int(sys.argv[sys.argv.index("-fetch-size") + 1])
        part = f"{{dest}}.p2p.{{os.getpid()}}.STUBPEER.part"
        chunk = max(1, int({mbps} * 1e6 / 4))     # 4 writes per second
        wrote = 0
        with open(part, "wb") as f:
            while wrote < total:
                n = min(chunk, total - wrote)
                f.write(b"\\0" * n); f.flush(); wrote += n
                time.sleep(0.25)
        os.replace(part, dest)
        """))
    p.chmod(p.stat().st_mode | stat.S_IEXEC)
    return str(p)


def _shard(size):
    return {"shard_id": "bafkreistub", "path": "model-00001.safetensors", "size": size}


def test_slow_peer_is_abandoned_for_the_mirror(tmp_path):
    """A ~2.5 MB/s residential-shaped seeder must NOT hold the pull."""
    prov = Libp2pProvider(bootstrap=["/ip4/127.0.0.1/tcp/1/p2p/stub"],
                          sidecar_bin=_stub_sidecar(tmp_path, mbps=2.5))
    prov.MIN_PEER_MBPS = 8.0
    prov._RAMP_S = 1.0          # keep the test quick; the production ramp is 25s
    prov._POLL_S = 0.25
    dest = str(tmp_path / "model-00001.safetensors")

    with pytest.raises(ProviderUnavailable) as e:
        prov.fetch(_shard(200 * 1024 * 1024), dest)
    assert "too slow" in str(e.value)
    # the abandoned partial must not be left behind to leak disk or fake progress
    assert not [f for f in os.listdir(tmp_path) if f.endswith(".part")]


def test_fast_peer_is_kept(tmp_path):
    """A competitive peer must still be used — the floor must not kill the torrent path."""
    prov = Libp2pProvider(bootstrap=["/ip4/127.0.0.1/tcp/1/p2p/stub"],
                          sidecar_bin=_stub_sidecar(tmp_path, mbps=40.0))
    prov.MIN_PEER_MBPS = 8.0
    prov._RAMP_S = 1.0
    prov._POLL_S = 0.25
    dest = str(tmp_path / "model-00001.safetensors")

    prov.fetch(_shard(20 * 1024 * 1024), dest)
    assert os.path.getsize(dest) == 20 * 1024 * 1024


def test_floor_can_be_disabled(tmp_path):
    """SHARD_MIN_PEER_MBPS=0 restores the old always-wait behaviour for a deliberate operator."""
    prov = Libp2pProvider(bootstrap=["/ip4/127.0.0.1/tcp/1/p2p/stub"],
                          sidecar_bin=_stub_sidecar(tmp_path, mbps=30.0))
    prov.MIN_PEER_MBPS = 0.0
    prov._RAMP_S = 1.0
    prov._POLL_S = 0.25
    dest = str(tmp_path / "model-00001.safetensors")
    prov.fetch(_shard(8 * 1024 * 1024), dest)
    assert os.path.exists(dest)
