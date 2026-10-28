"""The pipelined-speculation accept rule: one rule for every frame length, both ends of a round.

Model-agnostic and stdlib-only (no torch, no engine, no wire), so an entry, a node's tail and a
drafter host on any device family import the same code. Lifted from the V4 engine:

  plan_verify_round   engines/deepseek_v4/v4_dspark_draft.py:189-226, body unchanged
  wants_block         engines/deepseek_v4/v4_dspark_draft.py:762-795, logic unchanged; it takes
                      the v2 names (`pos`, `dnxt`, `lic`) instead of a V4 step-frame dict
  Frontier            the tail's incremental rule of v4_dspark_draft.py:672-708
                      (_on_chunk_pipelined), generalised from s=1 frames to frames of n rows
  accept              the entry's rule of coordinate_dspark_pipelined (v4_pipe.py:2100-2180),
                      generalised the same way

POSITIONS. A frame at `pos` with n rows feeds the tokens x[0..n-1] at positions pos..pos+n-1. The
tail answers toks[j], the model's greedy token AT position pos+j+1 (computed from row pos+j).

ONE RULE FOR EVERY n. Row pos+j's successor was already streamed: it is x[j+1] for j < n-1, and
for the last row it is the next frame's first token, if anything is in flight past this frame.
The rule walks the rows in order and commits toks[j] while that successor equals toks[j]. The
first mismatch commits toks[j] as the correction and ends the speculation there (REJECT). If the
last row's successor is not streamed, the frame commits all n tokens and the round needs a new
block (OPEN). If it is streamed and matches, the speculation runs on into the next frame (RUN).
  n = 1    V4's per-token rule (`fed == m`, v4_pipe.py:2130), frame for frame.
  n = K+1  with nothing in flight behind it: exactly `plan_verify_round(drafts, toks)`, i.e.
           M2.5's chunk round (m25_pipe.py:970-1003 at depth 1, full-accept bonus included).
`accept` runs `plan_verify_round` over the frame's own rows, so there is one implementation of
"longest matching prefix", not two.

Every committed token is the MODEL's token (toks), never a draft that happens to equal it, which
is what makes the round lossless whatever the drafter proposed.
"""

REJECT, OPEN, RUN = "reject", "open", "run"


def plan_verify_round(drafts, replies):
    """The accept rule, as one pure function both the pipe and the tests use. -> (n, committed).

    `drafts` are the g tokens the drafter proposed for chunk positions p+1..p+g. `replies` are the
    model's own greedy tokens for those same positions, one per chunk position -- so `replies[j]` is
    what the model says belongs at p+j+1, and there are g+1 of them because the chunk is
    `[cur] + drafts` and the tail answers at every position including the last.

    Longest matching prefix, then the correction: the first j where the draft and the model disagree
    ends the accept, and `replies[n]` is committed in its place. A FULL accept (n == g) commits
    `replies[g]` too -- the free bonus token, which the model produced from the last accepted draft
    and which cost nothing extra to compute. `len(committed) == n + 1` either way, and every token in
    it came out of the MAIN model, which is what makes the round lossless.

    The degenerate `drafts == []` (round 0, and any round the drafter is skipped) falls out of the
    same rule: n = 0, committed = [replies[0]]. That is plain greedy decode.

    BOTH ENDS RUN THIS: the entry to decide what to commit (`accept`), the tail to know how far its
    frontier and its drafter advance (`Frontier`). Two implementations of "longest matching prefix"
    that ever disagreed would desynchronise the drafter from the committed stream silently."""
    # int(), not list(): drafts and replies may arrive as tensor rows or numpy ints, and a list of
    # 0-dim tensors mixed with ints compares and serialises in ways nobody wants on a wire.
    drafts, replies = [int(t) for t in drafts], [int(t) for t in replies]
    if len(replies) != len(drafts) + 1:
        raise RuntimeError(
            f"pspec: {len(drafts)} drafts need {len(drafts) + 1} replies (one per chunk "
            f"position, the chunk being [cur] + drafts), got {len(replies)}")
    n = 0
    for d, r in zip(drafts, replies):
        if d != r:
            break
        n += 1
    return n, drafts[:n] + [replies[n]]


def accept(toks, fed):
    """The entry's rule for ONE judged frame of n rows. -> (committed, verdict).

    `toks` are the tail's n tokens (toks[j] at pos+j+1). `fed[j]` is the token already streamed at
    pos+j+1 in the current speculation epoch: fed[:n-1] are this frame's own rows x[1:], and
    fed[n-1] is the first row of the frame behind it, or None when nothing is in flight past this
    frame. The caller must already know the frame's first row is committed (the fence and the
    `sent[pos] == ids[pos]` check); the rule only decides how far the commit reaches.

    `committed` is a prefix of `toks` (1..n tokens). `verdict` is REJECT (the last committed token
    is a correction and everything streamed after it is a discarded future), OPEN (all n committed,
    nothing streamed past: the round needs a block) or RUN (all n committed and the next frame
    already carries the last of them: the speculation continues)."""
    toks = [int(t) for t in toks]
    n = len(toks)
    if n < 1 or len(fed) != n:
        raise ValueError(f"pspec: a frame of {n} rows needs {n} successor slots, got {len(fed)}")
    own = fed[:n - 1]
    if any(t is None for t in own):
        raise ValueError("pspec: a frame's own rows are always streamed; fed[:n-1] has a hole")
    k, committed = plan_verify_round(own, toks)
    if k < n - 1:
        return committed, REJECT
    nxt = fed[n - 1]
    if nxt is None:
        return committed, OPEN
    return committed, (RUN if int(nxt) == committed[-1] else REJECT)


class Frontier:
    """The TAIL's half of the rule: is this frame on the committed path? (`acc` on the reply)

    Frames reach the tail in injection order (per-session FIFO), so the tail sees the frame at q
    after everything injected before it. `cfront` is the newest row whose input is committed and
    whose output the tail has produced; `mfront` is that output, the model's token at cfront+1.
    A frame is on the committed path exactly when it starts at cfront+1 and feeds mfront there,
    which is `plan_verify_round`'s test at one position. Its own rows then advance the frontier by
    the same rule, so a chunk frame that diverges at row j leaves cfront at pos+j. A rejected
    frame never moves the frontier, so every frame behind it fails the test too: poisoning of the
    speculative future falls out of the rule instead of needing a flag. The correction the entry
    sends next lands at cfront+1 carrying mfront and re-opens it.

    A drafter on the tail advances over committed positions only (it drafts off (cfront, mfront)
    after an accepted frame), so it never rolls back (tests/test_v4_dspark.py:267's invariant).
    The entry asserts `acc` against its own judgment: two ends that disagree fail the job."""

    def __init__(self):
        self.cfront = None
        self.mfront = None

    def prefill(self, n, token):
        """After a prefill of rows 0..n-1 whose last row answered `token` (the model's token at n)."""
        self.cfront, self.mfront = n - 1, int(token)

    def judge(self, pos, fed, toks):
        """A frame at `pos` that fed `fed` (n ids) and produced `toks` (n tokens). -> (acc, k).

        acc: the frame's first row is on the committed path. k: how many of its drafts the model
        accepted (the frontier moves to pos+k); -1 when acc is False (nothing moves)."""
        if self.cfront is None:
            raise RuntimeError("pspec: a decode frame before the prefill; there is no committed "
                               "frontier to judge it against")
        fed = [int(t) for t in fed]
        if len(fed) != len(toks) or not fed:
            raise ValueError(f"pspec: {len(fed)} fed ids for {len(toks)} tokens")
        if pos != self.cfront + 1 or fed[0] != self.mfront:
            return False, -1
        k, committed = plan_verify_round(fed[1:], toks)
        self.cfront, self.mfront = pos + k, committed[-1]
        return True, k


def wants_block(pos, m, dnxt=None, lic=False, last=None):
    """Will the entry consume a block off this frame's reply? -> draft it, or don't.

    The lazy-draft licence, lifted from v4_dspark_draft.wants_block with its logic unchanged. The
    V4 step-frame fields map as start_pos -> `pos` (for a frame of n rows: its LAST row, pos+n-1),
    dnxt -> `dnxt` (the token already streamed at pos+1; on shard-frame/2 it rides inside the `eid`
    plaintext, never in clear) and dprev -> `lic` (frame hint `{lic}`). `m` is the model's token at
    pos+1 and `last` is `(position, block)` for the block this drafter returned one frame back.

    UNHINTED MEANS DRAFT: only a hint the entry has PROVEN (its hint rule) lets the tail skip, so a
    missing hint costs throughput and never a block the round needed. A mismatch (the round ends
    on this reply) always drafts, because that reply is the one the entry refills from. `lic` says
    "your successor is the head of the block you returned one frame ago"; every condition that
    makes that dereference sound is re-checked here (the block exists, is ours, from EXACTLY the
    previous position, non-empty), and any failure falls back to drafting."""
    nxt = dnxt
    if nxt is None and lic and last is not None:
        at, blk = last
        if blk and at == int(pos) - 1:
            nxt = blk[0]
    return nxt is None or int(nxt) != m
