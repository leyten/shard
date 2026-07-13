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


pytestmark = pytest.mark.integration


# ---- sidecar binary (built lazily in a session fixture: collect-only must run NO subprocess;
#      Go absent -> clean skip; Go present but build FAILS -> the suite FAILS, never a silent skip)
def _sidecar_bin():
    """Returns (path, err): (path, None) usable; (None, None) Go absent; (None, msg) build failed."""
    env = os.environ.get("SHARD_SIDECAR_BIN")
    if env and os.path.exists(env):
        return env, None
    cached = "/tmp/sidecar_new"
    src = os.path.join(_REPO, "sidecar")
    if os.path.exists(cached) and os.path.getmtime(cached) >= max(
            os.path.getmtime(os.path.join(src, f)) for f in ("main.go", "blockx.go")):
        return cached, None
    try:
        r = subprocess.run(["go", "build", "-o", cached, "."], cwd=src, timeout=300,
                           capture_output=True, text=True,
                           env={**os.environ, "GOTOOLCHAIN": os.environ.get("GOTOOLCHAIN", "auto")})
    except FileNotFoundError:
        return None, None
    except subprocess.TimeoutExpired:
        return None, "go build timed out (300s)"
    if r.returncode == 0:
        return cached, None
    return None, ((r.stderr or "") + (r.stdout or ""))[-800:] or f"go build exit {r.returncode}"


BIN = None            # set by the sidecar_bin fixture; Seeder/_pull read it
_BUILD = None


@pytest.fixture(scope="session")
def sidecar_bin():
    global BIN, _BUILD
    if _BUILD is None:
        _BUILD = _sidecar_bin()
    path, err = _BUILD
    if path is None:
        if err is None:
            pytest.skip("sidecar binary unavailable (no Go toolchain)")
        pytest.fail(f"Go is installed but the sidecar build FAILED:\n{err}", pytrace=False)
    BIN = path
    return path


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
def net(tmp_path, sidecar_bin):
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
        # the hostile peer was actually CONTACTED and served its garbage (else this proves nothing) —
        # the corrupted weight file was requested, so fallback, not a silent mirror-direct, saved us.
        assert "SERVED" in open(evil.log).read(), "evil seeder was never contacted — test is vacuous"
        # and no peer partial survives to poison a later run
        leftover = [f for f in os.listdir(dest) if ".p2p." in f and f.endswith(".part")]
        assert not leftover, f"peer partials left behind: {leftover}"
    finally:
        evil.stop()


# ---- 6. hostile seeder cannot WEDGE a pull when an honest peer also holds the shard ----------------
def test_hostile_and_honest_peer(net, tmp_path):
    """The finding-#3 case: a hostile seeder serving size-correct garbage must not be able to
    contaminate an HONEST peer's transfer (the old shared .p2p.part let it force every completion
    to hash-fail). Empty mirror, so success proves the honest PEER delivered."""
    bad_dir = str(tmp_path / "bad_ckpt")
    shutil.copytree(net["ckpt"], bad_dir)
    for name in ("model-00001.safetensors", "model-00002.safetensors"):
        f = os.path.join(bad_dir, name)
        blob = bytearray(open(f, "rb").read())
        blob[100] ^= 0xFF
        open(f, "wb").write(bytes(blob))
    evil = Seeder(net["tmp"], "peerEvil2", net["man_path"], bad_dir, bootstrap=[net["seederA"].addr])
    empty = str(tmp_path / "empty_mirror")
    os.makedirs(empty)
    try:
        dest = str(tmp_path / "nodeB")
        # both the honest seeder A and the hostile seeder are reachable; no mirror to fall back to
        paths = _pull(net, dest, [net["seederA"].addr, evil.addr], empty)
        for p in paths:
            rel = os.path.relpath(p, dest)
            assert open(p, "rb").read() == open(os.path.join(net["ckpt"], rel), "rb").read()
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


# ---- 7. build classification: a FAILED build must never masquerade as "no Go" (silent skip) --------
def test_sidecar_build_failure_is_distinguished(monkeypatch):
    """Go present + broken build -> (None, msg); the fixture turns msg into pytest.fail."""
    monkeypatch.delenv("SHARD_SIDECAR_BIN", raising=False)
    monkeypatch.setattr(os.path, "exists", lambda p: False)   # defeat the /tmp cache
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(
        a, 2, stdout="", stderr="main.go:1:1: syntax error"))
    path, err = _sidecar_bin()
    assert path is None and err and "syntax error" in err


def test_sidecar_no_go_toolchain_is_skip(monkeypatch):
    """Go absent entirely -> (None, None); the fixture turns that into a clean skip."""
    def no_go(*a, **k):
        raise FileNotFoundError("go")
    monkeypatch.delenv("SHARD_SIDECAR_BIN", raising=False)
    monkeypatch.setattr(os.path, "exists", lambda p: False)
    monkeypatch.setattr(subprocess, "run", no_go)
    assert _sidecar_bin() == (None, None)
