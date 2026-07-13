"""Fault-tolerance checkpoint envelope (specpipe --ft-dump / --resume-file).

The old dump was a bare {"output_ids": [...]}: nothing bound it to the generation it came
from, so a stale or foreign checkpoint (other prompt, other model, other sampling settings)
resumed silently and spliced another job's tokens into a fresh prefill — a hybrid generation.
These tests pin the fix: a versioned envelope binding every generation input plus a digest of
the committed tokens, written atomically, with any mismatch REFUSED (CheckpointError).

Run: pytest tests/test_checkpoint_envelope.py -q   (CPU-only, no GPU / no swarm needed)
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "phase0"))

import specpipe
from specpipe import CheckpointError, checkpoint_env, load_checkpoint, write_checkpoint


def _env(prompt="explain rings", model="gpt-oss-120b", tokenizer="gpt-oss-120b",
         temp=0.0, top_p=1.0, top_k=0, seed=0, reasoning=None):
    return checkpoint_env(prompt=prompt, model=model, tokenizer=tokenizer,
                          settings={"temp": temp, "top_p": top_p, "top_k": top_k,
                                    "seed": seed, "reasoning": reasoning})


def test_roundtrip(tmp_path):
    p = str(tmp_path / "ft.json")
    env = _env()
    d = write_checkpoint(p, env, [3, 1, 4, 1, 5], ok=False, error="edge died")
    assert d["schema"] == specpipe.CKPT_SCHEMA
    assert load_checkpoint(p, env) == [3, 1, 4, 1, 5]
    # extra report fields survive for the control plane (heal.py reads ok/error/text)
    saved = json.load(open(p))
    assert saved["ok"] is False and saved["error"] == "edge died"


def test_legacy_bare_output_ids_refused(tmp_path):
    """The pre-envelope format carries no binding at all — it must never resume."""
    p = str(tmp_path / "ft.json")
    json.dump({"output_ids": [1, 2, 3]}, open(p, "w"))
    with pytest.raises(CheckpointError, match="schema"):
        load_checkpoint(p, _env())


@pytest.mark.parametrize("field,other", [
    ("prompt", dict(prompt="a DIFFERENT prompt")),
    ("model", dict(model="some-other-model")),
    ("tokenizer", dict(tokenizer="some-other-tok")),
    ("settings", dict(temp=0.7)),
    ("settings", dict(seed=42)),
])
def test_cross_generation_reuse_refused(tmp_path, field, other):
    p = str(tmp_path / "ft.json")
    write_checkpoint(p, _env(), [7, 8, 9], ok=False)
    with pytest.raises(CheckpointError, match="mismatch"):
        load_checkpoint(p, _env(**other))


def test_tampered_tokens_refused(tmp_path):
    """Edited committed tokens no longer match the digest -> refused, not spliced in."""
    p = str(tmp_path / "ft.json")
    write_checkpoint(p, _env(), [7, 8, 9], ok=False)
    d = json.load(open(p))
    d["output_ids"] = [7, 8, 999]
    json.dump(d, open(p, "w"))
    with pytest.raises(CheckpointError, match="digest"):
        load_checkpoint(p, _env())


def test_malformed_ids_refused(tmp_path):
    p = str(tmp_path / "ft.json")
    env = _env()
    d = write_checkpoint(p, env, [1, 2], ok=True)
    d["output_ids"] = [1, "two"]
    json.dump(d, open(p, "w"))
    with pytest.raises(CheckpointError, match="malformed"):
        load_checkpoint(p, env)


def test_write_is_atomic_and_leaves_no_tmp(tmp_path):
    p = str(tmp_path / "ft.json")
    env = _env()
    write_checkpoint(p, env, [1], ok=False)
    write_checkpoint(p, env, [1, 2, 3], ok=True)          # overwrite in place (healer polls this path)
    assert load_checkpoint(p, env) == [1, 2, 3]
    assert [f for f in os.listdir(tmp_path) if ".tmp" in f] == []


def test_same_request_resume_matches():
    """Heal+resume of the SAME request derives an identical envelope -> job ids match."""
    assert _env()["job"] == _env()["job"]
    assert _env()["job"] != _env(prompt="other")["job"]
