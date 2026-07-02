"""Pure tree-verify primitives for EAGLE tree speculative decoding (no model deps; torch only).

A "tree" is what EagleDrafter.propose_tree(m) returns: M nodes, each with a token id, a parent index
(within the drafted set; -1 = the anchor = the last committed slot) and a depth (>=1, parents BEFORE children
— best-first pop order guarantees it). The target verifies the WHOLE tree in one forward with an ancestor-only
attention mask (a node sees the committed prefix + its own root->node chain, never its siblings), then we
greedily commit the longest accepted path. Lossless under greedy decoding: the accepted path is the longest
prefix of the true greedy continuation that the tree happens to contain.

  build_tree_mask    -> the additive attention bias + per-node RoPE positions for the verify pass.
  tree_greedy_walk   -> the lossless greedy accept walk over the target's per-node argmax.
  _rope_gather       -> per-position partial RoPE (siblings share a position).
  _gqa_masked_attend -> the manual broadcast-GQA masked-attention kernel attn_tree runs.

The ring (run_block_tree / coordinate_pipe tree branch) imports these; they carry no GPU/model deps, so the
offline CPU tests (tests/test_tree_spec.py) certify the exact math the stages execute.
"""
import torch


def _rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), -1)


def _rope_gather(t, cos, sin, pos_ids, rd):
    """Partial RoPE on the first `rd` dims via PER-POSITION gather (not a contiguous start:start+s slice): t is
    [1,heads,N,HD], pos_ids [N] long = the ABSOLUTE RoPE position of each token. Tree siblings share a position,
    so they get an identical rotation; the pass-through tail [...,rd:] is left unrotated. Same gather as
    attn_decode_b's per-stream RoPE, specialised to one row of N tokens."""
    cu = cos[pos_ids].unsqueeze(0).unsqueeze(0)                  # [1,1,N,rd]
    su = sin[pos_ids].unsqueeze(0).unsqueeze(0)
    tr, tp = t[..., :rd], t[..., rd:]
    return torch.cat([tr * cu + _rotate_half(tr) * su, tp], -1)


def _gqa_masked_attend(q, k, v, mask, grp):
    """Manual GQA attention with an ADDITIVE mask: softmax(QK^T / sqrt(HD) + mask) @ V, fp32 softmax
    (matches the stage's attn()). q [1,NH,N,HD], k/v [1,NKV,T,HD], mask [1,1,N,T] (0 = attend, -inf = block).
    BROADCAST-GQA: K/V stay at NKV heads and the GRP query groups broadcast against them — no
    repeat_interleave copy of the whole context (at tree-verify context lengths that copy is ~6x the K/V
    bytes). Manual is the right kernel for the tree's small-N dense mask: SDPA-with-dense-mask falls off
    flash on sm_120 (8-14x, see m25_stage.attn()'s graphed-decode note) while N<=~16 rows of matmul are
    flash-adjacent — and it is bit-reproducible (no SDPA backend variance). NKV-major head order ==
    repeat_interleave order."""
    b, nh, N, hd = q.shape
    nkv = k.shape[1]
    qg = q.view(b, nkv, grp, N, hd)                              # NH split (NKV, GRP), NKV-major
    a = (qg @ k.unsqueeze(2).transpose(-1, -2)) * (hd ** -0.5) + mask.unsqueeze(1)
    o = torch.softmax(a.float(), -1).to(v.dtype) @ v.unsqueeze(2)
    return o.view(b, nh, N, hd)


def build_tree_mask(parents, depths, start, N):
    """Additive attention bias [1,1,N,start+N] (0 = attend, -inf = block) + per-node RoPE positions [N] for
    verifying N tree nodes that continue a committed prefix of length `start`:
      * every tree node attends the WHOLE committed prefix      cols [0:start]   -> 0;
      * inside the tree block cols [start:start+N], node i attends exactly its ancestors-or-self (root->i),
        never its siblings.
    Ancestor inheritance (EAGLE cnets.py L770-782): attend = eye(N); attend[i] |= attend[parent[i]] -- since
    parents come before children, one forward pass over i propagates the whole root->i chain. Per-node RoPE
    position = (anchor pos = start-1) + depth[i] (the anchor is the last committed slot at position start-1, so
    a depth-1 node lands at the first new position `start`). Returns (bias, positions) so the caller feeds the
    target each node at the right position."""
    parents = [int(p) for p in parents]
    attend = torch.eye(N, dtype=torch.bool)
    for i in range(N):
        p = parents[i]
        if p >= 0:
            attend[i] |= attend[p]                       # inherit every ancestor the parent sees, + self
    bias = torch.zeros(1, 1, N, start + N)
    block = torch.zeros(N, N)
    block.masked_fill_(~attend, float("-inf"))           # -inf where node i must NOT attend tree node j
    bias[0, 0, :, start:] = block                        # cols [0:start] stay 0 (the whole committed prefix is visible)
    positions = (start - 1) + torch.as_tensor([int(d) for d in depths], dtype=torch.long)
    return bias, positions


def tree_greedy_walk(node_tokens, parents, target_argmax, anchor_target):
    """Lossless greedy accept walk (EAGLE utils.py::evaluate_posterior). Walk from the anchor; at each step
    follow the child of the current node whose token == the target's argmax at the current node (anchor_target
    at the anchor; target_argmax[node] after an accepted node). Stop at the first mismatch (no such child) or
    at a leaf -- the stopping token (the target's argmax there) is committed as the correction/bonus.

    Args:
      node_tokens   : [M] drafted token ids.
      parents       : [M] parent index per node (-1 = anchor).
      target_argmax : [M] the target model's greedy argmax AT each tree node (= its next token).
      anchor_target : the target's greedy argmax at the anchor (= the first token it wants); the first
                      accepted child must equal it.
    Returns (accepted_path_indices, committed_tokens): the accepted tree-node indices root->...->leaf and the
    committed tokens = accepted tokens + 1 (always the first len+1 tokens of the true greedy continuation, so a
    length-k accepted path yields k+1 correct tokens)."""
    children = {}
    for i, p in enumerate(parents):
        children.setdefault(int(p), []).append(i)
    path = []
    cur = -1                                             # current node (-1 = the anchor)
    want = anchor_target                                 # the token the target wants next
    while True:
        nxt = next((c for c in children.get(cur, []) if node_tokens[c] == want), None)
        if nxt is None:
            break                                        # mismatch / leaf -> `want` is the correction/bonus
        path.append(nxt)
        cur = nxt
        want = target_argmax[nxt]                        # the target's greedy token after this accepted node
    committed = [int(node_tokens[i]) for i in path] + [int(want)]
    return path, committed
