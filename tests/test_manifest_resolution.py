"""Network manifest resolution gates (P0-#1 Leg-7 residue) — the ref→verified-manifest seam.

A joiner's assignment carries `manifestRef = mf1:<name>@<cid>`; the daemon resolves the manifest
doc over untrusted transport and hands it to `shard.fetch`. These tests pin the two NEW fail-closed
gates that make that resolution trustworthy, in the engine where the existing verification lives:

  - check_manifest_cid: the manifest FILE BYTES must hash to the ref's CID (substitution among
    validly-signed manifests, rollback to an older release, and a tampered local cache all die
    BEFORE the JSON is parsed);
  - check_expected: a validly-signed, correctly-addressed manifest for the WRONG model (ref/manifest
    mismatch — assignment says m2.5, doc is something else) refuses to drive a pull;
  - the publisher's `version` field is covered by the signature (the rollback guard a resolver
    enforces monotonically).

Run: python3 -m pytest tests/test_manifest_resolution.py -q
"""
import json
import os
import subprocess
import sys

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "tests"))

from shard import manifest as mf                                  # noqa: E402
from shard.fetch import check_expected, check_manifest_cid        # noqa: E402
from shard.manifest import ManifestError                          # noqa: E402
from test_fetch import _repo                                      # noqa: E402


def _signed_repo(tmp_path):
    priv = mf.gen_key()
    src, manifest = _repo(tmp_path, priv)
    mpath = str(tmp_path / "manifest.json")
    with open(mpath, "w") as f:
        json.dump(manifest, f)
    sha, _ = mf.sha256_file(mpath)
    return {"src": src, "manifest": mpath, "cid": mf.cidv1_raw(sha), "pub": mf.pub_b64(priv),
            "dest": str(tmp_path / "model")}


# ---- check_manifest_cid: the ref's CID pins the exact manifest bytes ---------------------------

def test_cid_check_passes_bare_cid_and_mf1_ref(tmp_path):
    r = _signed_repo(tmp_path)
    check_manifest_cid(r["manifest"], r["cid"])                       # bare CIDv1
    check_manifest_cid(r["manifest"], f"mf1:m25-nvfp4-v1@{r['cid']}")  # assignment ref, verbatim


def test_cid_check_rejects_substituted_manifest(tmp_path):
    """A DIFFERENT validly-signed manifest under the pinned ref (mirror substitution, rollback to
    an older release, tampered cache) fails on bytes — before the JSON is ever parsed."""
    r = _signed_repo(tmp_path)
    other = mf.sign_manifest({"schema": mf.SCHEMA, "model_id": "test/m2.5", "layer_count": 4,
                              "weight_map": {}, "shards": []}, mf.gen_key())
    with open(r["manifest"], "w") as f:
        json.dump(other, f)
    with pytest.raises(ManifestError, match="content id mismatch"):
        check_manifest_cid(r["manifest"], r["cid"])


def test_cid_check_refuses_refs_without_a_cid(tmp_path):
    """The legacy bare-name ref (`mf:m25-nvfp4-v1`) and a malformed mf1 ref carry no content id —
    refused outright, never treated as 'no check'."""
    r = _signed_repo(tmp_path)
    for bad in ("mf:m25-nvfp4-v1", "mf1:m25-nvfp4-v1", "", "not-a-cid"):
        with pytest.raises(ManifestError, match="no content id"):
            check_manifest_cid(r["manifest"], bad)


# ---- check_expected: the assignment cross-check -----------------------------------------------

def test_expected_passes_on_match_and_skips_none():
    m = {"model_id": "test/m2.5", "layer_count": 4}
    check_expected(m, "test/m2.5", 4)
    check_expected(m)                                 # no expectations given -> no check


def test_expected_rejects_wrong_model_id():
    with pytest.raises(ManifestError, match="model_id"):
        check_expected({"model_id": "evil/other", "layer_count": 4}, "test/m2.5", 4)


def test_expected_rejects_wrong_layer_count():
    with pytest.raises(ManifestError, match="layer_count"):
        check_expected({"model_id": "test/m2.5", "layer_count": 36}, "test/m2.5", 4)


# ---- the CLI carries the gates (the daemon's actual call shape) --------------------------------

def _run(args):
    return subprocess.run([sys.executable, "-m", "shard.fetch", *args],
                          cwd=_REPO, capture_output=True, text=True, timeout=60)


def _tagged(out, tag):
    for line in out.splitlines():
        if line.startswith(tag + " "):
            return json.loads(line[len(tag) + 1:])
    raise AssertionError(f"no {tag} line in:\n{out}")


def test_cli_verified_pull_under_full_assignment_shape(tmp_path):
    """The launch call shape: --manifest-cid (mf1 ref verbatim) + --pubkey + --expect-* — a joiner
    pulls its range with every gate green."""
    r = _signed_repo(tmp_path)
    res = _run(["--manifest", r["manifest"], "--manifest-cid", f"mf1:m25-nvfp4-v1@{r['cid']}",
                "--pubkey", r["pub"], "--expect-model-id", "test/m2.5",
                "--expect-layer-count", "4", "--dir", r["dest"], "--lo", "2", "--hi", "4",
                "--tail", "--local-dir", r["src"]])
    assert res.returncode == 0, f"stdout:\n{res.stdout}\nstderr:\n{res.stderr}"
    _tagged(res.stdout, "SHARD_FETCH_DONE")
    assert "model-00002.safetensors" in os.listdir(r["dest"])


def test_cli_fatal_on_cid_mismatch_before_any_fetch(tmp_path):
    """A manifest that doesn't hash to the pinned ref dies machine-readably and NOTHING lands in
    the model dir — the loader never sees a byte selected by an unpinned manifest."""
    r = _signed_repo(tmp_path)
    wrong = mf.cidv1_raw("00" * 32)
    res = _run(["--manifest", r["manifest"], "--manifest-cid", wrong, "--pubkey", r["pub"],
                "--dir", r["dest"], "--lo", "0", "--hi", "2", "--head",
                "--local-dir", r["src"]])
    assert res.returncode != 0
    fatal = _tagged(res.stdout, "SHARD_FETCH_FATAL")
    assert "content id mismatch" in fatal["error"]
    assert not os.path.isdir(r["dest"]) or not os.listdir(r["dest"])


def test_cli_fatal_on_wrong_expected_model(tmp_path):
    r = _signed_repo(tmp_path)
    res = _run(["--manifest", r["manifest"], "--manifest-cid", r["cid"], "--pubkey", r["pub"],
                "--expect-model-id", "minimax/m2.5-nvfp4", "--dir", r["dest"],
                "--lo", "0", "--hi", "2", "--head", "--local-dir", r["src"]])
    assert res.returncode != 0
    fatal = _tagged(res.stdout, "SHARD_FETCH_FATAL")
    assert "model_id" in fatal["error"]


# ---- the publisher's version field is signed (the rollback guard's substrate) ------------------

def test_version_is_covered_by_the_signature():
    """A resolver enforces version monotonicity; that only means something if version can't be
    forged on an otherwise-valid manifest. Flipping it must break the signature."""
    signed = mf.sign_manifest({"schema": mf.SCHEMA, "model_id": "test/m2.5", "version": 2,
                               "layer_count": 4, "weight_map": {}, "shards": []}, mf.gen_key())
    mf.verify_manifest(signed)
    signed["version"] = 1
    with pytest.raises(ManifestError):
        mf.verify_manifest(signed)


def test_publish_cli_stamps_version(tmp_path):
    from test_publish_manifest import _checkpoint
    ckpt = _checkpoint(tmp_path)
    out = str(tmp_path / "m.json")
    res = subprocess.run([sys.executable, os.path.join(_REPO, "phase0", "publish_manifest.py"),
                          "--dir", ckpt, "--key", str(tmp_path / "pub.key"), "--out", out,
                          "--version", "7"], capture_output=True, text=True, timeout=60)
    assert res.returncode == 0, res.stderr
    m = json.load(open(out))
    assert m["version"] == 7
    mf.verify_manifest(m)                              # version rides inside the signed body
