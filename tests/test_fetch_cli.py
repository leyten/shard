"""`python -m shard.fetch` — the daemon's verified peers-first weight-pull entrypoint.

Pins the CLI contract the node daemon (c0mpute shard-runner.pullRange) supervises: a signed
manifest + a layer range -> only that range's shards, re-hashed against the manifest, into --dir,
with a machine-readable stdout line (SHARD_FETCH_DONE / SHARD_FETCH_FATAL + nonzero exit). Reuses
the synthetic signed repo the fetch trust-root tests use — no network.

Run: python3 -m pytest tests/test_fetch_cli.py -q
"""
import json
import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shard import manifest as mf                     # noqa: E402
from shard.fetch import (ChainProvider, LocalDirProvider, MirrorProvider,  # noqa: E402
                         build_chain_provider)
from test_fetch import _repo                          # noqa: E402  (the synthetic signed repo builder)

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _tagged(out, tag):
    for line in out.splitlines():
        if line.startswith(tag + " "):
            return json.loads(line[len(tag) + 1:])
    raise AssertionError(f"no {tag} line in:\n{out}")


def _run(args):
    return subprocess.run([sys.executable, "-m", "shard.fetch", *args],
                          cwd=_REPO, capture_output=True, text=True, timeout=60)


@pytest.fixture
def signed_repo(tmp_path):
    priv = mf.gen_key()
    src, manifest = _repo(tmp_path, priv)
    mpath = tmp_path / "manifest.json"
    mpath.write_text(json.dumps(manifest))
    return {"src": src, "manifest": str(mpath), "pub": mf.pub_b64(priv), "dest": str(tmp_path / "model")}


def test_cli_fetches_verified_range_via_local_seed(signed_repo):
    """Tail range [2:4) over a LocalDirProvider seed — the CLI verifies + selects norm/lm_head and
    prints the DONE contract with the right file count (2 weights shards + config + index)."""
    r = _run(["--manifest", signed_repo["manifest"], "--dir", signed_repo["dest"],
              "--lo", "2", "--hi", "4", "--tail", "--role", "stage",
              "--local-dir", signed_repo["src"], "--pubkey", signed_repo["pub"]])
    assert r.returncode == 0, f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}"
    done = _tagged(r.stdout, "SHARD_FETCH_DONE")
    got = set(os.listdir(signed_repo["dest"]))
    assert "model-00002.safetensors" in got          # the tail's weights shard
    assert "model-00001.safetensors" not in got      # the head's shard was NOT pulled (selective)
    assert done["files"] == len(got)


def test_cli_fatal_on_wrong_publisher(signed_repo):
    """A pinned pubkey that isn't the signer must fail closed, machine-readable — a stranger's box
    must never load weights under an unpinned/forged manifest."""
    other = mf.pub_b64(mf.gen_key())
    r = _run(["--manifest", signed_repo["manifest"], "--dir", signed_repo["dest"],
              "--lo", "0", "--hi", "2", "--head", "--local-dir", signed_repo["src"],
              "--pubkey", other])
    assert r.returncode != 0
    _tagged(r.stdout, "SHARD_FETCH_FATAL")
    assert not os.path.isdir(signed_repo["dest"]) or not os.listdir(signed_repo["dest"])


def test_build_chain_provider_orders_peers_before_mirror():
    """The default source chain is peers-first, mirror-last (the torrent property with a
    guaranteed fallback) — every source is re-hashed, so order is speed, not trust."""
    prov = build_chain_provider(mirror="https://example.com/repo/resolve/main/",
                                bootstrap=["/ip4/1.2.3.4/tcp/29600/p2p/Qm"],
                                sidecar_bin=None, key=None, local_dir="/seed")
    assert isinstance(prov, ChainProvider)
    kinds = [type(p).__name__ for p in prov.providers]
    assert kinds == ["LocalDirProvider", "Libp2pProvider", "MirrorProvider"]


def test_build_chain_provider_single_is_bare():
    assert isinstance(build_chain_provider(mirror="https://m/r/resolve/main/", bootstrap=[],
                                           sidecar_bin=None, key=None), MirrorProvider)
    with pytest.raises(ValueError):
        build_chain_provider(mirror=None, bootstrap=[], sidecar_bin=None, key=None)
