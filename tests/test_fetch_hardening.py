"""Audit hardening of the weight trust root (shard/fetch.py + shard/manifest.py +
phase0/publish_manifest.py). Companion to test_fetch.py / test_publish_manifest.py —
these are the adversarial regressions from the external audit:

  M8  — _safe_rel's lexical check let a pre-existing SYMLINK component (or a symlink at
        the destination itself) redirect a shard write outside the model root; the
        publisher key was written world-readable (0644 under umask 022).

Run: python3 -m pytest tests/test_fetch_hardening.py -q
"""
import os
import sys

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "phase0"))

from shard import manifest as mf                     # noqa: E402
from shard.fetch import FetchError, _safe_rel        # noqa: E402


# ---- M8: symlinks must not redirect shard writes outside model_dir ---------------------------------

def test_symlinked_dir_component_refused(tmp_path):
    """A lexically-clean path like 'sub/shard.bin' whose 'sub' is a pre-existing symlink
    pointing OUTSIDE the model root must be refused — the old check compared strings only."""
    outside = tmp_path / "outside"
    outside.mkdir()
    model = tmp_path / "model"
    model.mkdir()
    os.symlink(str(outside), str(model / "sub"))
    with pytest.raises(FetchError, match="symlink"):
        _safe_rel(str(model), "sub/shard.bin")
    assert not os.path.exists(outside / "shard.bin")


def test_symlinked_destination_refused(tmp_path):
    """A symlink AT the destination filename redirects the open() itself."""
    victim = tmp_path / "victim.bin"
    victim.write_bytes(b"precious")
    model = tmp_path / "model"
    model.mkdir()
    os.symlink(str(victim), str(model / "model-00001.safetensors"))
    with pytest.raises(FetchError, match="symlink"):
        _safe_rel(str(model), "model-00001.safetensors")
    assert victim.read_bytes() == b"precious"


def test_symlink_inside_model_dir_still_refused(tmp_path):
    """Even an inside-root symlink is refused at the destination: it would alias one
    verified shard's bytes over another's."""
    model = tmp_path / "model"
    model.mkdir()
    (model / "real.bin").write_bytes(b"a")
    os.symlink(str(model / "real.bin"), str(model / "alias.bin"))
    with pytest.raises(FetchError, match="symlink"):
        _safe_rel(str(model), "alias.bin")


def test_plain_nested_path_still_allowed(tmp_path):
    """No symlinks involved -> unchanged behavior (nested dirs are fine)."""
    model = tmp_path / "model"
    model.mkdir()
    dest = _safe_rel(str(model), "nested/dir/shard.bin")
    assert dest.startswith(os.path.realpath(str(model)) + os.sep)


def test_model_dir_itself_a_symlink_is_fine(tmp_path):
    """Operators legitimately symlink the model dir to a big disk; that must keep working."""
    real = tmp_path / "disk"
    real.mkdir()
    link = tmp_path / "model"
    os.symlink(str(real), str(link))
    dest = _safe_rel(str(link), "shard.bin")
    assert dest == os.path.join(os.path.realpath(str(link)), "shard.bin")


# ---- M8: publisher key hygiene ----------------------------------------------------------------------

def test_save_key_is_0600(tmp_path):
    path = str(tmp_path / "publisher.key")
    old_umask = os.umask(0o022)                       # the deployment default that exposed 0644
    try:
        mf.save_key(mf.gen_key(), path)
    finally:
        os.umask(old_umask)
    assert os.stat(path).st_mode & 0o777 == 0o600
    mf.load_key(path)                                 # round-trips


def test_save_key_never_clobbers_existing(tmp_path):
    path = str(tmp_path / "publisher.key")
    mf.save_key(mf.gen_key(), path)
    with pytest.raises(FileExistsError):
        mf.save_key(mf.gen_key(), path)               # exclusive create: a second write is a bug


def test_load_key_rejects_exposed_perms(tmp_path):
    path = str(tmp_path / "publisher.key")
    mf.save_key(mf.gen_key(), path)
    os.chmod(path, 0o644)
    with pytest.raises(mf.ManifestError, match="group/world"):
        mf.load_key(path)
