"""Layer-block challenge (PROVE) — the stage-node analogue of c0mpute's whole-model canary.

c0mpute's anti-cheat sends a math+nonce prompt and checks the answer. That CANNOT probe a
stage node: a node holding layers 12..23 never sees a prompt, it transforms an activation
tensor. This primitive is how you catch a stage node that forwards plausible garbage (or a
cheaper wrong model) instead of actually running its assigned block:

  1. a verifier draws a deterministic challenge input from a seed (derive_challenge),
  2. it feeds that exact input to the suspect node's block and to a TRUSTED replica of the
     same block (block_forward on each),
  3. it compares the two outputs (compare). A node that ran the real block matches to within
     floating-point noise; a node that faked it does not.

WHY TOLERANCE, NOT A HASH (this is the load-bearing design choice): out = block(in) is NOT
bit-reproducible across different GPUs/kernels/quant-runtime — the same class of ULP
non-associativity we measured cross-K and in the windowed read. So a bit-exact out_hash match
would FALSE-POSITIVE every honest node on heterogeneous hardware. We compare with a cosine
threshold instead: an honest recompute lands at cosine ~1.0 (a few ULPs); garbage or a wrong
block lands far below. This is the *economic-now* verifier (strike -> reputation -> eject +
withhold pay); a succinct cryptographic proof-of-compute is the crypto-later upgrade and drops
into the same seam (docs/INTEGRATION.md §6b, §9, §11).

Pure engine (boundary law): knows activations, blocks, tensors — nothing about c0mpute's
reputation policy. shard provides "run this block on this input -> output"; *when* to probe,
*how* to score, *when* to eject is c0mpute policy.

SEAM: `python3 -m shard.challenge` compares two sketches over stdio (JSON in/out), mirroring
`shard.plan`/`shard.verify` — the TS control plane drives the ONE comparison implementation
instead of re-porting the tolerance math. compare_sketches is deliberately TORCH-FREE (pure
python over the 256-dim projections): sketches come FROM the GPU nodes; the control plane's
host needs no CUDA stack to judge them. torch imports are lazy for the same reason.
"""
import hashlib
import json
import math
import secrets
import sys

# The pre-commit fixed projection seed (v0). PREDICTABLE: a cheater who learns the 256 sampled
# coordinates once can forge a passing sketch forever (preserve those coords + the norm, garbage
# everywhere else). Kept ONLY as an explicit back-compat/diagnostics opt-in — never a payment check.
LEGACY_SEED = "fixed-v0"
_LEGACY_PROJ_SEED = 1234567


def sketch_seed() -> str:
    """A fresh UNPREDICTABLE per-challenge projection seed (verifier-side). Commit to it with
    seed_commitment() BEFORE the prover sees the challenge, reveal it with the challenge: the
    prover can't know which coordinates will be sampled, so it must produce the whole correct
    tensor — preserving a fixed subset no longer passes."""
    return secrets.token_hex(16)


def seed_commitment(seed: str) -> str:
    """Publishable commitment to a projection seed (sha256 hex). Posting this before the
    challenge and revealing `seed` with it proves the sampled coordinates weren't chosen after
    seeing the prover's answer."""
    return hashlib.sha256(f"sketch-seed:{seed}".encode()).hexdigest()


def _proj_seed(seed) -> int:
    if seed == LEGACY_SEED:
        return _LEGACY_PROJ_SEED
    return int.from_bytes(hashlib.sha256(str(seed).encode()).digest()[:8], "big")


def derive_challenge(seed: str, n_tokens: int, hidden_size: int,
                     device="cuda", dtype=None):
    """A deterministic [1, n_tokens, hidden_size] activation derived from `seed`. The verifier
    and the challenged node both derive the SAME input from the same seed, so only the block's
    transform is under test. Seeded CPU generation -> identical bytes on any host (the input is
    reproducible even though the block's OUTPUT is not), then moved to device."""
    import torch
    h = hashlib.sha256(seed.encode()).digest()
    g = torch.Generator()                                  # CPU generator: host-independent draw
    g.manual_seed(int.from_bytes(h[:8], "big"))
    x = torch.randn(1, n_tokens, hidden_size, generator=g, dtype=torch.float32)
    return x.to(device=device, dtype=torch.bfloat16 if dtype is None else dtype)


def block_forward(parts, x, start: int = 0):
    """Run ONE forward of a node's loaded block (pipeline.load_stage parts) on input x at
    absolute position `start`, eager + causal, no cache write. Returns the block output hidden
    states. This is exactly the transform a stage applies on the hot path, isolated for probing."""
    import torch
    from pipeline import run_block
    from transformers import DynamicCache
    cache = DynamicCache()
    with torch.no_grad():
        return run_block(x, parts, cache, start)


def sketch(h, seed: str = None, full: bool = False) -> dict:
    """A compact, transport-friendly fingerprint of a block output: shape, L2 norm, and a
    low-dim random projection (enough to compare via cosine without shipping the full tensor).

    COMMIT-FIRST projection: the sampled coordinates derive from `seed`. Default (seed=None)
    draws a fresh unpredictable seed via sketch_seed() and records it in the sketch — the
    challenged node must sketch with the seed the VERIFIER supplies, and compare_sketches
    fails closed on a seed mismatch. seed=LEGACY_SEED restores the old fixed projection
    (predictable → forgeable; back-compat/diagnostics only). full=True ships the entire
    flattened activation for an occasional full-tensor audit — nothing to hide behind."""
    import torch
    hf = h.detach().to(torch.float32).flatten()
    if full:
        return {"n": int(hf.numel()), "norm": float(hf.norm()), "proj": hf.cpu().tolist(),
                "full": True}
    if seed is None:
        seed = sketch_seed()
    g = torch.Generator(device=hf.device if hf.is_cuda else "cpu")
    g.manual_seed(_proj_seed(seed))
    dim = min(256, hf.numel())
    idx = torch.randint(0, hf.numel(), (dim,), generator=g, device=hf.device)
    out = {"n": int(hf.numel()), "norm": float(hf.norm()), "proj": hf[idx].cpu().tolist()}
    if seed != LEGACY_SEED:
        out["seed"] = seed                             # legacy sketches keep their v0 shape
    return out


def compare_sketches(a: dict, b: dict, cos_thresh: float = 0.99) -> dict:
    """Torch-free sketch comparison — the control-plane side of the spot-check. Same tolerance
    semantics as compare(): cosine over the seed-derived projections + relative L2 norm.
    Malformed, shape-mismatched or seed-mismatched sketches FAIL CLOSED (a cheater must not be
    able to dodge the check by sending a sketch the verifier can't line up with its own — or
    one projected with a seed of its OWN choosing)."""
    try:
        if a.get("seed") != b.get("seed"):
            return {"cosine": 0.0, "rel_norm": 1.0, "passed": False,
                    "error": "projection seed mismatch"}
        va, vb = list(map(float, a["proj"])), list(map(float, b["proj"]))
        na, nb = float(a["norm"]), float(b["norm"])
        n_a, n_b = int(a["n"]), int(b["n"])
    except (KeyError, TypeError, ValueError, AttributeError):
        return {"cosine": 0.0, "rel_norm": 1.0, "passed": False, "error": "malformed sketch"}
    if not va or len(va) != len(vb) or n_a != n_b:
        return {"cosine": 0.0, "rel_norm": 1.0, "passed": False, "error": "sketch shape mismatch"}
    dot = sum(x * y for x, y in zip(va, vb))
    pa = math.sqrt(sum(x * x for x in va)) or 1e-9
    pb = math.sqrt(sum(y * y for y in vb)) or 1e-9
    cos = dot / (pa * pb)
    rel_norm = abs(na - nb) / max(na, nb, 1e-9)
    passed = cos >= cos_thresh and rel_norm < 0.05
    return {"cosine": cos, "rel_norm": rel_norm, "passed": passed}


def compare(a, b, cos_thresh: float = 0.99) -> dict:
    """Compare two block outputs (full tensors or sketches) by cosine similarity + relative norm.
    PASS = the node ran the real block (cosine ~1, a few ULPs); FAIL = garbage/wrong block.
    cos_thresh 0.99 sits far above honest ULP drift (cosine ~0.9999) and far above any
    independent/garbage output (cosine ~0)."""
    if isinstance(a, dict) and isinstance(b, dict):
        return compare_sketches(a, b, cos_thresh)
    import torch
    va = torch.tensor(a["proj"]) if isinstance(a, dict) else a.detach().to(torch.float32).flatten()
    vb = torch.tensor(b["proj"]) if isinstance(b, dict) else b.detach().to(torch.float32).flatten()
    na = a["norm"] if isinstance(a, dict) else float(va.norm())
    nb = b["norm"] if isinstance(b, dict) else float(vb.norm())
    cos = float(torch.nn.functional.cosine_similarity(va.float(), vb.float(), dim=0))
    rel_norm = abs(na - nb) / max(na, nb, 1e-9)
    passed = cos >= cos_thresh and rel_norm < 0.05
    return {"cosine": cos, "rel_norm": rel_norm, "passed": passed}


def challenge_block(suspect_parts, trusted_parts, seed: str, n_tokens: int, hidden_size: int,
                    start: int = 0, device="cuda", cos_thresh: float = 0.99) -> dict:
    """End-to-end on one host (both blocks local) — the trusted-redundant-recompute check:
    feed the same seeded input to both blocks, compare. In production the suspect block runs on
    the remote node and only its sketch() crosses the wire; verify it against the trusted local
    recompute with compare()."""
    x = derive_challenge(seed, n_tokens, hidden_size, device=device)
    h_suspect = block_forward(suspect_parts, x, start)
    h_trusted = block_forward(trusted_parts, x, start)
    return compare(h_suspect, h_trusted, cos_thresh)


def _main() -> int:
    """`python3 -m shard.challenge` — JSON in ({a: sketch, b: sketch, cos_thresh?}), JSON out
    (the compare_sketches verdict). `a` = the suspect node's sketch, `b` = the trusted replica's.
    Torch-free by design: the control plane judges sketches, the GPU nodes produce them."""
    try:
        req = json.load(sys.stdin)
    except Exception as e:  # noqa: BLE001 — a malformed request is a caller error, report it as JSON
        json.dump({"error": f"bad request json: {e}"}, sys.stdout)
        return 2
    try:
        out = compare_sketches(req["a"], req["b"], float(req.get("cos_thresh", 0.99)))
    except KeyError as e:
        json.dump({"error": f"missing field: {e}"}, sys.stdout)
        return 2
    json.dump(out, sys.stdout)
    return 0


if __name__ == "__main__":
    sys.exit(_main())
