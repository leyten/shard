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
"""
import hashlib

import torch


def derive_challenge(seed: str, n_tokens: int, hidden_size: int,
                     device="cuda", dtype=torch.bfloat16) -> torch.Tensor:
    """A deterministic [1, n_tokens, hidden_size] activation derived from `seed`. The verifier
    and the challenged node both derive the SAME input from the same seed, so only the block's
    transform is under test. Seeded CPU generation -> identical bytes on any host (the input is
    reproducible even though the block's OUTPUT is not), then moved to device."""
    h = hashlib.sha256(seed.encode()).digest()
    g = torch.Generator()                                  # CPU generator: host-independent draw
    g.manual_seed(int.from_bytes(h[:8], "big"))
    x = torch.randn(1, n_tokens, hidden_size, generator=g, dtype=torch.float32)
    return x.to(device=device, dtype=dtype)


def block_forward(parts, x: torch.Tensor, start: int = 0) -> torch.Tensor:
    """Run ONE forward of a node's loaded block (pipeline.load_stage parts) on input x at
    absolute position `start`, eager + causal, no cache write. Returns the block output hidden
    states. This is exactly the transform a stage applies on the hot path, isolated for probing."""
    from pipeline import run_block
    from transformers import DynamicCache
    cache = DynamicCache()
    with torch.no_grad():
        return run_block(x, parts, cache, start)


def sketch(h: torch.Tensor) -> dict:
    """A compact, transport-friendly fingerprint of a block output: shape, L2 norm, and a
    low-dim random projection (enough to compare via cosine without shipping the full tensor).
    The projection seed is fixed so verifier and node project identically."""
    hf = h.detach().to(torch.float32).flatten()
    g = torch.Generator(device=hf.device if hf.is_cuda else "cpu")
    g.manual_seed(1234567)
    dim = min(256, hf.numel())
    idx = torch.randint(0, hf.numel(), (dim,), generator=g, device=hf.device)
    return {"n": int(hf.numel()), "norm": float(hf.norm()), "proj": hf[idx].cpu().tolist()}


def compare(a, b, cos_thresh: float = 0.99) -> dict:
    """Compare two block outputs (full tensors or sketches) by cosine similarity + relative norm.
    PASS = the node ran the real block (cosine ~1, a few ULPs); FAIL = garbage/wrong block.
    cos_thresh 0.99 sits far above honest ULP drift (cosine ~0.9999) and far above any
    independent/garbage output (cosine ~0)."""
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
