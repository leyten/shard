"""P2P shard propagation (sidecar blockx + shard.fetch Libp2pProvider/ChainProvider) — the torrent half.

Two REAL sidecar processes on localhost: peer A seeds a signed manifest's shards (DHT provide +
blockx serve), peer B fetches its layer block by CID through the full verified path
(fetch_block_range -> ChainProvider[Libp2pProvider, LocalDirProvider]). No GPU, no internet.

What must hold (each is a test):
  1. PEER PATH — B pulls every byte from A (the mirror dir is EMPTY, so success proves peer transfer),
     and the manifest re-hash accepts the files.
  2. PROPAGATION — C fetches from B's seed of what B just pulled (A -> B -> C, the torrent moment).
  3. HOSTILE SEEDER — a seeder serving corrupted bytes cannot wedge the pull: ChainProvider verifies
     per source, drops the garbage, and the mirror serves the good copy.
  4. DEAD PEER — no seeder up -> Libp2pProvider is unavailable -> chain falls back to the mirror.

Needs the sidecar binary: $SHARD_SIDECAR_BIN, else /tmp/sidecar_new, else `go build` (skips if
neither works — CI without Go still runs the rest of the suite).

Run: python3 -m pytest tests/test_blockx.py -q
"""
import json
import os
import shutil
import socket
import subprocess
import sys
import time

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "phase0"))

from shard import manifest as mf                                        # noqa: E402
from shard.fetch import (ChainProvider, FetchError, Libp2pProvider,     # noqa: E402
                         LocalDirProvider, fetch_block_range)
import publish_manifest as pub                                          # noqa: E402


# ---- sidecar binary (build once per session, skip when impossible) --------------------------------
def _sidecar_bin():
    env = os.environ.get("SHARD_SIDECAR_BIN")
    if env and os.path.exists(env):
        return env
    cached = "/tmp/sidecar_new"
    src = os.path.join(_REPO, "sidecar")
    if os.path.exists(cached) and os.path.getmtime(cached) >= max(
            os.path.getmtime(os.path.join(src, f)) for f in ("main.go", "blockx.go")):
        return cached
    try:
        r = subprocess.run(["go", "build", "-o", cached, "."], cwd=src, timeout=300,
                           capture_output=True, text=True,
                           env={**os.environ, "GOTOOLCHAIN": os.environ.get("GOTOOLCHAIN", "auto")})
        if r.returncode == 0:
            return cached
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


BIN = _sidecar_bin()
pytestmark = pytest.mark.skipif(BIN is None, reason="sidecar binary unavailable (no Go toolchain)")


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


class Seeder:
    """One sidecar -seed process on localhost; .addr is its dialable /p2p multiaddr."""

    def __init__(self, tmp, name, manifest_path, model_dir, bootstrap=None):
        self.log = os.path.join(tmp, f"{name}.log")
        addrfile = os.path.join(tmp, f"{name}.addr")
        cmd = [BIN, "-key", os.path.join(tmp, f"{name}.key"),
               "-listen", f"/ip4/127.0.0.1/tcp/{_free_port()}",
               "-addrfile", addrfile, "-seed", f"{manifest_path}={model_dir}"]
        for b in bootstrap or []:
            cmd += ["-dht-bootstrap", b]
        self.proc = subprocess.Popen(cmd, stdout=open(self.log, "w"), stderr=subprocess.STDOUT)
        deadline = time.time() + 30
        self.addr = None
        while time.time() < deadline:
            if os.path.exists(addrfile) and "SEEDING" in open(self.log).read():
                self.addr = open(addrfile).read().strip()
                break
            if self.proc.poll() is not None:
                raise RuntimeError(f"seeder {name} died:\n{open(self.log).read()[-800:]}")
            time.sleep(0.2)
        if not self.addr:
            self.stop()
            raise RuntimeError(f"seeder {name} never came up:\n{open(self.log).read()[-800:]}")

    def stop(self):
        self.proc.kill()
        self.proc.wait(timeout=10)


@pytest.fixture()
def net(tmp_path):
    """A signed 2-file checkpoint + one live seeder (peer A) + an OFFLINE mirror copy."""
    tmp = str(tmp_path)
    ckpt = pub_checkpoint(tmp_path)
    priv = mf.gen_key()
    cfg, weight_map, shards = pub.build_from_dir(ckpt)
    manifest = mf.sign_manifest(
        {"schema": mf.SCHEMA, "model_id": "test/m25", "arch": cfg["architectures"][0],
         "layer_count": cfg["num_hidden_layers"], "tied_embeddings": False,
         "tokenizer": "test/m25", "weight_map": weight_map, "shards": shards}, priv)
    man_path = os.path.join(tmp, "manifest.json")
    json.dump(manifest, open(man_path, "w"))
    mirror = os.path.join(tmp, "mirror")            # the origin copy, used only by fallback tests
    shutil.copytree(ckpt, mirror)
    a = Seeder(tmp, "peerA", man_path, ckpt)
    yield dict(tmp=tmp, manifest=manifest, man_path=man_path, ckpt=ckpt, mirror=mirror,
               seederA=a, pub=mf.pub_b64(priv))
    a.stop()


def pub_checkpoint(tmp_path, layers=4):
    """Synthetic checkpoint matching publish_manifest's loader rules (mirrors test_publish_manifest)."""
    d = str(tmp_path / "ckpt")
    os.makedirs(d, exist_ok=True)
    open(os.path.join(d, "model-00001.safetensors"), "wb").write(os.urandom(1 << 20))
    open(os.path.join(d, "model-00002.safetensors"), "wb").write(os.urandom(2 << 20))
    json.dump({"num_hidden_layers": layers, "architectures": ["MiniMaxM2ForCausalLM"],
               "tie_word_embeddings": False}, open(os.path.join(d, "config.json"), "w"))
    wm = {}
    for j in (0, 1):
        wm[f"model.layers.{j}.q.weight"] = "model-00001.safetensors"
    for j in (2, 3):
        wm[f"model.layers.{j}.q.weight"] = "model-00002.safetensors"
    wm["model.embed_tokens.weight"] = "model-00001.safetensors"
    wm["model.norm.weight"] = "model-00002.safetensors"
    wm["lm_head.weight"] = "model-00002.safetensors"
    json.dump({"weight_map": wm}, open(os.path.join(d, "model.safetensors.index.json"), "w"))
    json.dump({"tok": 1}, open(os.path.join(d, "tokenizer.json"), "w"))
    return d


def _pull(net_, dest, bootstrap, mirror_dir, lo=0, hi=4):
    """The deploy-shaped pull: peers first, mirror as origin fallback, full verification."""
    provider = ChainProvider([
        Libp2pProvider(bootstrap=bootstrap, sidecar_bin=BIN, timeout=60),
        LocalDirProvider(mirror_dir),
    ])
    return fetch_block_range(net_["manifest"], dest, lo, hi, is_head=True, is_tail=True,
                             role="stage", provider=provider, expected_pubkey=net_["pub"])


# ---- 1. the peer path: every byte from peer A (mirror EMPTY -> peer transfer is proven) ------------
def test_fetch_from_peer(net, tmp_path):
    empty = str(tmp_path / "empty_mirror")
    os.makedirs(empty)
    dest = str(tmp_path / "nodeB")
    paths = _pull(net, dest, [net["seederA"].addr], empty)
    assert len(paths) >= 3                                   # 2 weights + config/index (+tokenizer)
    for p in paths:
        rel = os.path.relpath(p, dest)
        src = os.path.join(net["ckpt"], rel)
        assert open(p, "rb").read() == open(src, "rb").read(), f"{rel} differs from the seed"
    assert "SERVED" in open(net["seederA"].log).read()       # peer A actually served blocks


# ---- 2. propagation: C pulls from B's seed of what B pulled (A -> B -> C) --------------------------
def test_propagation_chain(net, tmp_path):
    empty = str(tmp_path / "empty_mirror")
    os.makedirs(empty)
    destB = str(tmp_path / "nodeB")
    _pull(net, destB, [net["seederA"].addr], empty)
    b = Seeder(net["tmp"], "peerB", net["man_path"], destB, bootstrap=[net["seederA"].addr])
    try:
        net["seederA"].stop()                                # A leaves; only B's copy remains
        destC = str(tmp_path / "nodeC")
        paths = _pull(net, destC, [b.addr], empty)
        assert paths and "SERVED" in open(b.log).read()      # C's bytes came from B
    finally:
        b.stop()


# ---- 3. hostile seeder: corrupted bytes are dropped, the mirror serves the good copy ---------------
def test_hostile_seeder_falls_back(net, tmp_path):
    bad_dir = str(tmp_path / "bad_ckpt")
    shutil.copytree(net["ckpt"], bad_dir)
    f = os.path.join(bad_dir, "model-00001.safetensors")
    blob = bytearray(open(f, "rb").read())
    blob[100] ^= 0xFF                                        # same size, wrong byte
    open(f, "wb").write(bytes(blob))
    evil = Seeder(net["tmp"], "peerEvil", net["man_path"], bad_dir)
    net["seederA"].stop()                                    # only the hostile seeder is on the DHT
    try:
        dest = str(tmp_path / "nodeB")
        paths = _pull(net, dest, [evil.addr], net["mirror"])
        for p in paths:                                      # pull SUCCEEDS despite the hostile peer
            rel = os.path.relpath(p, dest)
            assert open(p, "rb").read() == open(os.path.join(net["mirror"], rel), "rb").read()
    finally:
        evil.stop()


# ---- 4. dead peer: nothing seeding -> ProviderUnavailable -> mirror fallback ------------------------
def test_dead_peer_mirror_fallback(net, tmp_path):
    net["seederA"].stop()
    dest = str(tmp_path / "nodeB")
    paths = _pull(net, dest, [], net["mirror"])              # no bootstrap at all
    assert len(paths) >= 3


# ---- 5. all sources dead -> fail closed --------------------------------------------------------------
def test_all_sources_dead_fails_closed(net, tmp_path):
    net["seederA"].stop()
    empty = str(tmp_path / "empty_mirror")
    os.makedirs(empty)
    with pytest.raises(FetchError):
        _pull(net, str(tmp_path / "nodeB"), [], empty)
