"""proof_receipt attestation honesty (M6) — the verdict must never overclaim.

Every field in a proof receipt is SELF-REPORTED by whoever ran the swarm (node identity,
topology, WAN latencies, model, commit, perf — all unsigned), so a passing checklist proves the
record agrees with itself and nothing more. What must hold: (1) build() emits a versioned record
labelled self-reported with an envelope binding the raw artifact hashes + assignments + code/model
identity + stage receipts (signature None until a signing path exists), (2) verify() on a passing
receipt says SELF-CONSISTENT / SELF-REPORTED and never the old "distributed, real-WAN, correct,
reproducible" claim, (3) an independently-supplied reference (--ref-tokens) is the only externally
verified fact and is called out as such, (4) tampering still fails the checklist.

Run: python3 -m pytest tests/test_proof_receipt.py -q
"""
import argparse
import hashlib
import json
import os
import sys

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "phase0"))

import proof_receipt as pr  # noqa: E402

OVERCLAIM = "distributed, real-WAN, correct, reproducible"


def _write(path, obj):
    json.dump(obj, open(path, "w"))
    return str(path)


def _build(tmp_path, **over):
    nodes = [{"role": "head", "layer_range": [0, 31], "public_ip": "1.2.3.4", "geo": "NL",
              "gpu_uuid": "GPU-a", "gpu_name": "5090"},
             {"role": "tail", "layer_range": [31, 62], "public_ip": "5.6.7.8", "geo": "DE",
              "gpu_uuid": "GPU-b", "gpu_name": "5090"}]
    edges = [{"from": "head", "to": "tail", "rtt_ms": 12.5}]
    run = {"prompt": "p", "output_text": "t", "output_token_ids": [1, 2, 3], "tok_s_warm": 20.0}
    a = argparse.Namespace(
        nodes=_write(tmp_path / "nodes.json", nodes),
        edges=_write(tmp_path / "edges.json", edges),
        run=_write(tmp_path / "run.json", run),
        model="minimax-m2.5", quant="nvfp4", out=str(tmp_path / "receipt.json"),
        run_id="r1", utc="", assignments=None, stage_receipts=None)
    for k, v in over.items():
        setattr(a, k, v)
    pr.build(a)
    return a


def _verify(receipt, ref_tokens=None):
    with pytest.raises(SystemExit) as e:
        pr.verify(argparse.Namespace(receipt=receipt, ref_tokens=ref_tokens))
    return e.value.code


def test_build_emits_versioned_self_reported_envelope(tmp_path, capsys):
    a = _build(tmp_path)
    r = json.load(open(a.out))
    assert r["format"] == "shard-proof-receipt/v2"
    assert r["attestation"] == "self-reported"
    env = r["envelope"]
    assert env["signature"] is None                              # unsigned until a signing path exists
    assert env["model_identity"] == {"model": "minimax-m2.5", "quant": "nvfp4"}
    assert env["code_identity"]["shard_commit"] == r["shard_commit"]
    assert env["artifact_sha256"]["nodes"] == hashlib.sha256(open(a.nodes, "rb").read()).hexdigest()
    assert env["artifact_sha256"]["run"] == hashlib.sha256(open(a.run, "rb").read()).hexdigest()
    assert env["assignments"] is None and env["stage_receipts"] is None


def test_build_binds_assignments_and_stage_receipts(tmp_path):
    asg = {"pubA==": [0, 31], "pubB==": [31, 62]}
    a = _build(tmp_path, assignments=_write(tmp_path / "asg.json", asg),
               stage_receipts=_write(tmp_path / "sr.json", [{"sig": "s1"}]))
    env = json.load(open(a.out))["envelope"]
    assert env["assignments"] == asg and env["stage_receipts"] == [{"sig": "s1"}]


def test_verify_never_overclaims(tmp_path, capsys):
    a = _build(tmp_path)
    assert _verify(a.out) == 0
    out = capsys.readouterr().out
    assert OVERCLAIM not in out                                  # the M6 overclaim is gone
    assert "SELF-CONSISTENT" in out and "SELF-REPORTED" in out
    assert "attestation: self-reported" in out


def test_verify_calls_out_independent_reference(tmp_path, capsys):
    a = _build(tmp_path)
    ref = _write(tmp_path / "ref.json", [1, 2, 3])
    assert _verify(a.out, ref_tokens=ref) == 0
    out = capsys.readouterr().out
    assert "independently-supplied reference" in out and OVERCLAIM not in out
    # a wrong reference still fails the checklist
    bad = _write(tmp_path / "bad.json", [9])
    assert _verify(a.out, ref_tokens=bad) == 1
    assert "FAILED" in capsys.readouterr().out


def test_verify_tampered_tokens_fail(tmp_path, capsys):
    a = _build(tmp_path)
    r = json.load(open(a.out))
    r["output_token_ids"] = [6, 6, 6]                            # hash no longer matches
    _write(a.out, r)
    assert _verify(a.out) == 1
    assert "FAILED" in capsys.readouterr().out
